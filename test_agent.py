"""Test an OpenAI-powered optimization agent on an archetype-4 query.

The agent receives a stakeholder critique (text-only or text+annotation), is
given access to the view tool and lookup tools, decides on a small set of
actions (force_open / force_close / force_assign), and submits a Proposal.
The script then re-solves the MILP under the proposal's fixings and reports
before/after metrics including the score against guards.

Usage:
    export OPENAI_API_KEY=sk-...
    python test_agent.py                              # text-only critique
    python test_agent.py --with_annotation            # text + 3 polygons
    python test_agent.py --model gpt-4o-mini          # cheaper model
    python test_agent.py --max_iters 25 --save_log ./test_agent_cluster_pairs00.json   # longer trajectory + save

Requires: openai>=1.40, gurobipy, the rest of the project files.

Notes on the OpenAI tool-result image pattern
---------------------------------------------
The OpenAI Chat Completions API expects tool messages (`role="tool"`) to have
*string* content. To return an image to the agent, we therefore:
    1. Append a tool message with a brief textual confirmation.
    2. Append a follow-up `user` message containing the rendered PNG via
       `image_url` with a base64 data URL.
This is the conservative pattern that works across SDK versions. If your SDK
version supports multipart tool content, you can collapse the two steps.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from instance import Instance, Solution
from solver import solve_baseline
from metrics import compute_metrics
from agent_tools import (
    list_sites, list_precincts_in_region, get_site_at, get_precinct_at,
    Proposal, apply_proposal, view_solution_png, view_solution_v2_no_markers_png,
    apply_local_assignment, apply_local_swap, get_precinct_adjacency_data,
    get_current_assignments, get_distance_matrix_data, get_precinct_centroids,
    COVERAGE_GAP_AGENT_GUIDANCE, COVERAGE_GAP_TOOL_NOTE,
)
from queries import ARCHETYPE_FACTORIES, ArchetypeQuery
from rendering import render_view

# Load environment variables from .env (e.g. OPENAI_API_KEY) if dotenv is
# available. Optional dependency — falls back gracefully if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------
# Canonical view types — exactly the layer combinations that get saved as
# per-pair PNGs (see rendering.render_all_pair_views). The agent picks one
# (or more) of these names; the dispatcher expands each into the matching
# set of rendering primitives so the rendered image precisely matches what's
# in the dataset's saved view files.
LAYER_ENUM = [
    "plain",              # precincts + candidate sites (no solution)
    "baseline",           # solution + assignments (precinct fill colored by assigned site)
    "population_density", # solution + voter density heatmap
    "v2_no_markers",      # marker-free catchment view (rendering_v2_no_markers)
]

LAYER_TO_PRIMITIVES = {
    "plain":              ["closed_sites"],
    "baseline":           ["closed_sites", "solution", "assignments"],
    "population_density": ["population_density", "solution"],
}

# Marker-free multimodal experiment: only v2_no_markers in the tool schema.
MARKER_FREE_LAYER_ENUM = ["v2_no_markers"]


def _multimodal_solution_png(
    instance: Instance,
    solution: Solution,
    *,
    marker_free_maps: bool,
    region: Optional[Any] = None,
) -> bytes:
    """Initial / follow-up map bytes for multimodal runs."""
    if marker_free_maps:
        return view_solution_v2_no_markers_png(instance, solution)
    return view_solution_png(
        instance, solution,
        layers=["closed_sites", "solution", "assignments"],
        show_precinct_labels=False,
        region=region,
    )

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_solution",
            "description": (
                "Render the current solution as a map image. The `layers` "
                "argument selects canonical view types. For `plain`, "
                "`baseline`, and `population_density`, the composition matches "
                "the per-pair PNGs saved in the dataset's `views/` folder. "
                "`v2_no_markers` is generated live (marker-free catchments; see "
                "description below).\n"
                "  - 'plain': precincts + candidate sites + named landmarks "
                "(NO solution overlay; useful to see the static context).\n"
                "  - 'baseline': the current solution — opened sites + "
                "precinct→site assignments, with each precinct color-filled "
                "by its assigned polling place (so service areas are "
                "visible at a glance). Use as a default if no other "
                "context is needed; especially useful when the critique "
                "refers to non-contiguous service areas or split "
                "catchments.\n"
                "  - 'population_density': solution overlaid on voters/km². "
                "Useful as visual context for where the demand is.\n"
                "  - 'v2_no_markers': marker-free map from rendering_v2_no_markers — "
                "saturated catchment-colored fills and **black catchment outlines** "
                "only (no site labels, no closed-site dots, no assignment lines). "
                "Best for **coverage-gap** and **shape-niceness** spatial reasoning "
                "about service-area shape and gaps; **always** pair with "
                "list_sites / get_distance_matrix for indices and distances.\n"
                "  **Do not** combine 'v2_no_markers' with other layer names in the "
                "same call — if present, only the v2 renderer is used.\n"
                "Site index labels are shown on plain/baseline/population_density; "
                "precinct index labels are optional via `show_precinct_labels` "
                "(ignored for v2_no_markers).\n"
                "The `view_purpose` field (required) tags which role this "
                "inspection plays in your reasoning cycle — used for audit "
                "logging and is your commitment about what you will state "
                "after seeing the image:\n"
                "  'baseline_vs_critique': first structured look — RELATE "
                "WHAT YOU OBSERVE TO THE STAKHOLDER'S CRITIQUE.\n"
                "  'post_action_vs_baseline': after a resolve or local edit "
                "— COMPARE the NEW map to the PREVIOUS state, state whether "
                "the targeted issue IMPROVED and whether any guards look "
                "compromised.\n"
                "  'other_context': supplementary view for extra context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "layers": {
                        "type": "array",
                        "items": {"type": "string", "enum": LAYER_ENUM},
                        "description": (
                            "Which view type(s) to render. Pass one "
                            "element (e.g. ['baseline']) for a single "
                            "view. For ['plain','baseline'], primitives merge. "
                            "Pass **only** ['v2_no_markers'] for the marker-free "
                            "catchment view — it cannot merge with other tokens."
                        ),
                    },
                    "view_purpose": {
                        "type": "string",
                        "enum": ["baseline_vs_critique",
                                 "post_action_vs_baseline",
                                 "other_context"],
                        "description": (
                            "Role of this inspection in your reasoning cycle. "
                            "'baseline_vs_critique': initial look before any "
                            "changes. "
                            "'post_action_vs_baseline': after a resolve or "
                            "local edit — compare new vs. previous state. "
                            "'other_context': supplementary context view."),
                    },
                    "show_precinct_labels": {
                        "type": "boolean",
                        "description": "If true, overlay precinct index labels (default false).",
                    },
                },
                "required": ["layers", "view_purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sites",
            "description": (
                "Return structured info on candidate sites. Each entry has index, "
                "x, y, type, capacity, opened, and (if a solution exists) load."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "opened_only": {"type": "boolean",
                                    "description": "If true, only return opened sites."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_precincts_in_region",
            "description": (
                "Return precincts whose centroid lies inside the given polygon. "
                "Each entry has index, x, y, voters, and demographic shares. "
                "Use this to translate a region you've identified visually into "
                "a concrete list of precinct indices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "polygon": {
                        "type": "array",
                        "description": "List of [x, y] vertices forming a closed polygon (km coords).",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2, "maxItems": 2,
                        },
                        "minItems": 3,
                    },
                },
                "required": ["polygon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_site_at",
            "description": "Return the candidate site nearest to (x, y) within max_distance km, or null.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "max_distance": {"type": "number", "description": "default 0.6"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_precinct_at",
            "description": "Return the precinct containing the point (x, y), or null if out of bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_precinct_centroids",
            "description": (
                "Return precinct centroid coordinates in km. Each entry "
                "has precinct index, x, y, and voters. Use precinct_indices "
                "to request a focused subset, or omit it to return all "
                "precinct centroids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional precinct indices to include. If omitted, "
                            "all precinct centroids are returned."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_precinct_adjacency",
            "description": (
                "Return precinct adjacency: for each precinct, the list "
                "of other precincts that share a boundary with it "
                "(4-connected on the rasterised Voronoi map). Use this to "
                "check whether an opened site's catchment is contiguous: "
                "collect the precincts currently assigned to that site, "
                "then run a connected-components search. Two or more "
                "components means the catchment is split — a "
                "non-contiguous service area. Especially useful for "
                "split-catchment-style critiques in the tools-only "
                "condition where you can't perceive disjoint patches "
                "from the rendered map."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_assignments",
            "description": (
                "Return the current solution's precinct-to-polling-place "
                "assignments. Each entry includes precinct index, centroid, "
                "voters, assigned site index, assigned travel distance, "
                "weighted distance, and assigned site coordinates/type. "
                "Use precinct_indices to request a focused subset, or omit "
                "it to return all precinct assignments. "
                + COVERAGE_GAP_TOOL_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional precinct indices to include. If omitted, "
                            "all current assignments are returned."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_distance_matrix",
            "description": (
                "Return the precinct-to-site travel distance matrix in km, "
                "or a requested slice of it. By default this returns all "
                "precincts and all candidate sites. Use precinct_indices, "
                "site_indices, or opened_only to reduce output size. "
                + COVERAGE_GAP_TOOL_NOTE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional precinct row indices. If omitted, all "
                            "precincts are included."
                        ),
                    },
                    "site_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional site column indices. If omitted, all "
                            "sites are included unless opened_only is true."
                        ),
                    },
                    "opened_only": {
                        "type": "boolean",
                        "description": (
                            "If true and site_indices is omitted, return "
                            "columns for currently opened sites only."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "force_assign",
            "description": (
                "LOCAL EDIT — pin a precinct's assignment to a specific "
                "opened site, holding the rest of the solution constant. "
                "The objective and per-site capacity are recomputed; the "
                "MILP is NOT re-solved. The set of opened sites (x) is "
                "unchanged — if you need to open or close sites, use "
                "the resolve tool instead. The target site must already "
                "be opened in the current solution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_index": {
                        "type": "integer",
                        "description": "Precinct to reassign.",
                    },
                    "site_index": {
                        "type": "integer",
                        "description": "Target site (must be opened).",
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "WHY this reassignment addresses the critique "
                            "(required for audit logging)."),
                    },
                },
                "required": ["precinct_index", "site_index", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swap_assignments",
            "description": (
                "LOCAL EDIT — swap the assigned sites of two precincts, "
                "holding the rest of the solution constant. The "
                "objective and per-site capacity are recomputed; the "
                "MILP is NOT re-solved. Useful for fixing non-contiguous "
                "catchments by trading a stray precinct for a same-"
                "pocket one. The two precincts must currently be "
                "assigned to different opened sites."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_a_index": {
                        "type": "integer",
                        "description": "First precinct to swap.",
                    },
                    "precinct_b_index": {
                        "type": "integer",
                        "description": "Second precinct to swap.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": (
                            "WHY this swap addresses the critique "
                            "(required for audit logging)."),
                    },
                },
                "required": ["precinct_a_index", "precinct_b_index", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve",
            "description": (
                "MILP RE-SOLVE — submit a set of fixings (force_open, "
                "force_close, force_assign, precinct_weight_multipliers) "
                "and have the optimizer re-solve. The result becomes the "
                "new 'current solution'. Use this when the change "
                "requires opening or closing sites, or when you want the "
                "MILP to globally re-balance the assignment around your "
                "fixings. Each call REPLACES any prior fixings — they "
                "do not stack. After resolving, inspect the new state "
                "(view_solution, list_sites, etc.) and either refine "
                "with another resolve, make local edits with "
                "force_assign / swap_assignments, or call submit_proposal "
                "to end the reasoning loop. "
                "If the MILP is INFEASIBLE under your fixings, the "
                "current solution is left unchanged and you should "
                "revise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "force_open": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Site indices to fix as opened (x_j = 1).",
                    },
                    "force_close": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Site indices to fix as closed (x_j = 0).",
                    },
                    "force_assign": {
                        "type": "array",
                        "description": "List of [precinct_index, site_index] pairs to fix (y_ij = 1).",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2, "maxItems": 2,
                        },
                    },
                    "precinct_weight_multipliers": {
                        "type": "array",
                        "description": (
                            "SOFT action — per-precinct objective weight. "
                            "List of [precinct_index, multiplier] pairs. "
                            "Multiplies that precinct's contribution to "
                            "the voter-weighted distance objective by "
                            "the given factor (default 1.0). Values >1 "
                            "make the optimizer care more about "
                            "minimising that precinct's travel distance. "
                            "Values clamped to [0, 100]."
                        ),
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2, "maxItems": 2,
                        },
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief justification.",
                    },
                },
                "required": ["rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_proposal",
            "description": (
                "FINAL SUBMISSION — indicate that your reasoning loop is "
                "done. This ends the session; the benchmark will score the "
                "best feasible solution explored during the session, not "
                "the arguments to this tool. Call this only after "
                "you've inspected the current solution (via "
                "view_solution / list_sites / etc.) and you're "
                "satisfied that it addresses the stakeholder's critique. "
                "If you stop calling tools without explicitly "
                "submitting, the current solution is taken as final at "
                "the end of the iteration budget — but you should "
                "always submit explicitly when you believe you're done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Brief justification of the submitted "
                            "solution: what you changed and why."
                        ),
                    },
                },
                "required": ["rationale"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt(
    enable_visual: bool = True,
    marker_free_maps: bool = False,
) -> str:
    """System prompt, parameterised on whether the agent has the view tool.

    enable_visual=True (multimodal): the agent can call view_solution and
        receives an initial rendered map in the user message.
    enable_visual=False (tools-only): no view_solution tool, no rendered
        image — all reasoning happens via the structured tools. Used to
        isolate the contribution of multimodal access in experiments.

    marker_free_maps : when True with multimodal, every map (initial,
        view_solution, and auto-attached after resolves/local edits) uses
        rendering_v2_no_markers only — no site/candidate markers.
    """
    if enable_visual:
        if marker_free_maps:
            critical_tail = (
                "The map is for perceiving spatial PATTERNS (shape, compactness, "
                "gaps, clustering). This session uses MARKER-FREE maps only "
                "(no site labels, no closed-site dots). The map is NOT for "
                "reading indices — use structured tools for precision.\n"
                "\n"
                "VISUAL INSPECTION CYCLE (required every time you make a change):\n"
                "  Step A — before any resolve or local edit: call view_solution "
                "with layers=['v2_no_markers'] and view_purpose='baseline_vs_critique'. "
                "State WHAT pattern you observe and HOW it maps to the critique.\n"
                "  Step B — after EACH resolve or local edit: the pipeline "
                "auto-attaches a fresh marker-free map (same style as v2_no_markers). "
                "Call view_solution with view_purpose='post_action_vs_baseline' "
                "and state: (1) whether the targeted issue VISIBLY IMPROVED? "
                "(2) do travel or coverage look compromised?\n"
                "Skipping either step means your proposal is not grounded "
                "in the actual solution state."
            )
            workflow_step_3 = (
                "3. CALL view_solution with layers=['v2_no_markers'] and "
                "view_purpose='baseline_vs_critique' before making changes. "
                "Only the marker-free catchment view is available — saturated "
                "fills and black outlines (rendering_v2_no_markers). Ground every "
                "pattern to indices via list_sites, get_current_assignments, "
                "get_precinct_centroids, get_distance_matrix, etc. After EACH "
                "action, call view_solution again with layers=['v2_no_markers'] "
                "and view_purpose='post_action_vs_baseline' to verify the change."
            )
        else:
            critical_tail = (
                "The map is for perceiving spatial PATTERNS (clustering, "
                "demographic relationships, shape, density). The map is NOT "
                "for reading off precise positions or identifying which "
                "labeled entity sits where. Whenever a question requires "
                "precision, lean on the structured tools.\n"
                "\n"
                "VISUAL INSPECTION CYCLE (required every time you make a change):\n"
                "  Step A — before any resolve or local edit: call view_solution "
                "with view_purpose='baseline_vs_critique'. In your reply, state "
                "explicitly WHAT pattern you observe and HOW it maps to the "
                "stakeholder's critique.\n"
                "  Step B — after EACH resolve or local edit: the pipeline "
                "auto-attaches a fresh rendered map. Call view_solution with "
                "view_purpose='post_action_vs_baseline' and state: (1) whether the "
                "targeted issue VISIBLY IMPROVED? (2) does total distance or "
                "coverage look compromised?\n"
                "Skipping either step means your proposal is not grounded "
                "in the actual solution state."
            )
            workflow_step_3 = (
                "3. CALL view_solution with view_purpose='baseline_vs_critique' "
                "to anchor your analysis before making any changes. "
                "Pick the layer most relevant to the critique:\n"
                "     - 'baseline' (default): solution + assignments, each "
                "precinct color-filled by its assigned polling place. Use for "
                "cluster, coverage_gap, contiguity, and shape-niceness critiques.\n"
                "     - 'population_density': solution overlaid on voters/km².\n "
                "Useful to see where demand is concentrated.\n"
                "     - 'plain': precincts + candidate sites only — static "
                "context with no solution overlay.\n"
                "     - 'v2_no_markers': catchment shapes without site markers "
                "(rendering_v2_no_markers). Especially useful for **coverage_gap** "
                "(worst-served pockets, corridor gaps) and **shape_niceness** "
                "(compactness, jagged catchments). No site labels on the image — "
                "ground indices with list_sites / get_distance_matrix after viewing.\n"
                "   After viewing, ground the visible pattern to specific indices "
                "via the structured tools (list_sites, list_precincts_in_region, "
                "get_precinct_centroids, get_current_assignments, "
                "get_distance_matrix, etc.). After EACH action, call "
                "view_solution again with view_purpose='post_action_vs_baseline' "
                "to verify the change is effective."
            )
        enumerate_clause = (
            "ENUMERATE the relevant entities and their coordinates "
            "explicitly in your reasoning. Do not estimate from the image."
        )
    else:
        critical_tail = (
            "NOTE: You do NOT have access to a rendered map for this "
            "session. The view_solution tool is unavailable. All reasoning "
            "must come from the structured tools (list_sites, "
            "list_precincts_in_region, get_site_at, get_precinct_at, "
            "get_precinct_centroids, get_current_assignments, "
            "get_distance_matrix, get_precinct_adjacency). When a query references a region "
            "or an emergent visual pattern, you must hypothesise candidate "
            "polygons or coordinate ranges and probe them with the tools."
        )
        workflow_step_3 = (
            "3. If a spatial pattern matters (a region, a cluster, a gap, "
            "a starburst), construct hypotheses from the structured tools "
            "— probe candidate polygons with list_precincts_in_region, "
            "retrieve precinct coordinates with get_precinct_centroids, "
            "compare opened-site coordinates and loads from list_sites, "
            "inspect current precinct assignments with get_current_assignments, "
            "compare travel distances with get_distance_matrix, "
            "and use get_precinct_adjacency for connectivity questions "
            "(e.g. is a site's catchment contiguous?) — to localise and "
            "characterise the pattern."
        )
        enumerate_clause = (
            "ENUMERATE the relevant entities and their coordinates "
            "explicitly in your reasoning, drawn from the structured tool "
            "outputs."
        )

    return f"""You are an optimization agent helping a stakeholder analyze and refine a polling place location plan for a county.

PROBLEM SETUP
- The county has roughly 80 precincts and 40 candidate polling place sites.
- The baseline solution opens 18 sites and assigns each precinct to one open site.
- The baseline minimises total voter-weighted travel distance subject to capacity and a site-budget constraint.
- The baseline does NOT account for demographic equity or other stakeholder preferences.

INDEX NAMESPACES — READ CAREFULLY
- Site indices (roughly 0–39 for candidate polling places) and precinct indices (roughly 0–79 for voter districts) are completely separate namespaces. The same integer can refer to a site OR a precinct — they are NOT interchangeable.
- When the stakeholder names a specific "polling place", "site", or "polling site" by number, that number is a SITE index unless they explicitly say "precinct". Confirm the entity with list_sites (and coordinates) before using it in any tool argument.
- Never pass a site index as precinct_index (or precinct_a_index / precinct_b_index) to any tool. Never pass a precinct index as site_index, force_open, force_close, or any argument that expects a site.
- get_distance_matrix takes precinct_indices and site_indices as separate arguments — do not mix them up.

TWO KINDS OF REQUEST
Respond to the stakeholder's request as one of:
  (a) FACTUAL / DIAGNOSTIC question  ("which sites are at the bottom?", "how many precincts are in this region?", "why is this region underserved?"). Answer the question directly using tools, then reply with a clear textual answer. DO NOT call submit_proposal for factual questions.
  (b) CHANGE REQUEST / CRITIQUE  ("fix the disparity in the south side"). Inspect, decide on a SMALL set of actions, then call submit_proposal. The system will re-solve the MILP under your variable fixings and report new metrics.

CRITICAL: VERIFY POSITIONS AND INDICES WITH STRUCTURED TOOLS.
For ANY question that depends on
  - which site/precinct is at a particular location,
  - the coordinates or index of a labeled entity,
  - which entities lie in a region or share a property (highest, lowest, leftmost, bottom-most, in a line, etc.),
  - which polling place currently serves a precinct,
  - how far a precinct is from its assigned site or from alternative sites,
  - whether a reassignment improves or worsens travel distance,
you MUST first call the relevant structured tool and reason from its output:
  - list_sites(opened_only=True)        -> index, (x, y), type, capacity, load of every opened site.
  - list_precincts_in_region(polygon)   -> precincts with centroid inside a polygon.
  - get_site_at(x, y) / get_precinct_at(x, y) -> the entity at a coordinate.
  - get_precinct_centroids(precinct_indices?) -> precinct centroid coordinates in km.
  - get_current_assignments(precinct_indices?) -> current precinct→site assignments.
  - get_distance_matrix(precinct_indices?, site_indices?, opened_only?) -> precinct→site distances in km.
Do NOT infer exact current assignments or distances from the rendered map.
Use get_current_assignments for the current y[i,j] assignment state, and
get_distance_matrix for distance comparisons against assigned, opened, or
candidate sites.
{critical_tail}

REASONING WORKFLOW
1. Read the request and decide whether it's factual (a) or a change request (b).
2. If positions or indices matter: call the structured tools FIRST to get ground truth.
{workflow_step_3}
4. If assignments, travel distances, nearest alternatives, or distance tradeoffs matter,
   call get_current_assignments and/or get_distance_matrix before deciding.
5. Before answering or proposing, {enumerate_clause}
6. (a) For factual questions, give a direct textual answer with the indices/values from the tool output. (b) For change requests, call submit_proposal with the smallest set of actions that addresses the critique.

YOUR ACTIONS (only for change requests)
There are THREE classes of solution-modifying action:

(1) RESOLVE — MILP re-solve under fixings. The optimizer re-balances
    the rest of the solution around what you pin. Use this when the
    change requires opening or closing sites, OR when you want the
    optimizer to globally re-optimise around your constraints.
    Fixings supported by resolve:
    - force_open[j]:                require site j to be opened (x_j = 1).
    - force_close[j]:               require site j to be closed (x_j = 0).
    - force_assign[i, j]:           require precinct i assigned to site j.
    - precinct_weight_multipliers[i]: SOFT objective-weight multiplier on
      precinct i (default 1.0, clamped to [0, 100]). Values > 1 make
      the optimizer care more about that precinct's travel distance.

(2) LOCAL EDITS — modify the current solution directly, no MILP solve.
    The set of opened sites (x) is unchanged.
    - force_assign(precinct_index, site_index): pin a precinct's
      assignment to a different OPEN site. Objective and capacity are
      recomputed; the rest of the solution stays put.
    - swap_assignments(precinct_a_index, precinct_b_index): swap two
      precincts' assigned sites. Useful for fixing non-contiguous
      catchments by trading a stray precinct for a same-pocket one.

(3) SUBMIT — submit_proposal signals that the reasoning loop is done
    and ends the session. Call this once you've explored enough
    candidate solutions. It takes no fixings; the benchmark scores the
    best feasible solution explored during the session.

WHEN TO USE WHICH
- Use resolve when the fix needs site openings/closings OR when you
  want the optimizer to globally re-balance around your constraints.
- Use force_assign / swap_assignments for surgical fixes — reassigning
  a single precinct or trading two — when you do NOT want the rest of
  the solution disturbed. This is especially valuable for contiguity
  critiques where you want to move one stray precinct from a far site
  to a same-pocket site without re-routing every other precinct.
- Use soft weights inside a resolve when the stakeholder describes a
  group of precincts that should be prioritised but doesn't dictate
  the exact configuration.
- get_current_assignments: returns the current solution's precinct→site
  assignment state. Use it to verify which precincts are assigned to a
  site, to inspect the current catchment before local edits, and to
  confirm the assignment state after any resolve or local edit.
- get_distance_matrix: returns precinct→site distances in km. Use it to
  compare assigned distance against nearby opened sites or candidate
  sites before proposing force_assign, force_open, or force_close actions.
- get_precinct_adjacency: returns each precinct's spatial neighbours.
  Use this to compute connected components of an opened site's
  assigned precincts — two or more components ⇒ non-contiguous
  service area. Especially valuable in tools-only mode where you
  can't see disjoint patches on the map.

The current solution shown to view_solution / list_sites / etc.
ADVANCES whenever you call resolve (feasibly) OR make a local edit.
You can chain: do a local edit, view the result, do another local
edit, call resolve, then submit_proposal. Every feasible local edit
and feasible resolve is retained for superscoring.

ITERATIVE FLOW
You can call resolve and the local-edit tools MULTIPLE TIMES per
session before finally submitting:
  - Call resolve to apply fixings via the MILP. The system updates the
    "current solution" you see, tells you which sites opened/closed and
    how many precincts were reassigned, and (in multimodal mode) shows
    you a fresh rendered map. Each resolve REPLACES any prior fixings.
  - Call force_assign / swap_assignments to make local edits without
    re-solving. Useful for surgical fixes.
  - Inspect the updated state — is it feasible, and does it actually
    address the critique? Use view_solution / list_sites /
    get_current_assignments / get_distance_matrix /
    get_precinct_adjacency on the CURRENT solution to verify.
  - When you believe you have explored enough candidate solutions, call
    submit_proposal to end the loop. submit_proposal takes only a
    rationale; scoring uses the best feasible solution explored during
    the session.
You should usually iterate at least once — make a change, look at the
result, and only submit once you've verified the change genuinely
addresses the stakeholder's concern. A common failure mode is making
changes that LOOK reasonable but don't actually fix the emergent
property the stakeholder described; iterating catches this.
When a feasible resolve's tool result includes primary target feedback
showing the metric did not strictly improve (unchanged or wrong-way),
do not treat the task as done: re-diagnose with tools and attempt at
least one materially different resolve or local edit before ending.

GUARDS (only relevant for change requests)
The system rejects responses that significantly degrade total voter-weighted travel distance or the global p90 voter distance, so prefer minimal interventions."""


def get_tools_for_run(
    enable_visual: bool = True,
    marker_free_maps: bool = False,
) -> List[Dict[str, Any]]:
    """Return the OpenAI tools list for this run. When enable_visual is False,
    view_solution is filtered out so the agent has only structured tools.

    When marker_free_maps is True (multimodal only), view_solution's layer
    enum is restricted to v2_no_markers so the model cannot request marked maps.
    """
    if not enable_visual:
        return [t for t in TOOLS
                if t.get("function", {}).get("name") != "view_solution"]
    if not marker_free_maps:
        return TOOLS
    tools = copy.deepcopy(TOOLS)
    for t in tools:
        if t.get("function", {}).get("name") != "view_solution":
            continue
        fn = t["function"]
        fn["description"] = (
            "Marker-free multimodal protocol: render the current solution using "
            "ONLY the v2_no_markers view (rendering_v2_no_markers) — saturated "
            "catchment-colored fills and **black catchment outlines** only; no "
            "site labels, no closed-site dots, no assignment lines. The initial "
            "session map and any auto-attached maps after resolves/local edits "
            "use this same renderer. Always pass layers=['v2_no_markers'].\n"
            "Use list_sites / get_current_assignments / get_distance_matrix for "
            "indices and distances.\n"
            "The `view_purpose` field (required) tags which role this inspection "
            "plays in your reasoning cycle — used for audit logging and is your "
            "commitment about what you will state after seeing the image:\n"
            "  'baseline_vs_critique': first structured look — RELATE "
            "WHAT YOU OBSERVE TO THE STAKHOLDER'S CRITIQUE.\n"
            "  'post_action_vs_baseline': after a resolve or local edit "
            "— COMPARE the NEW map to the PREVIOUS state, state whether "
            "the targeted issue IMPROVED and whether any guards look "
            "compromised.\n"
            "  'other_context': supplementary view for extra context."
        )
        layers_prop = fn["parameters"]["properties"]["layers"]
        layers_prop["items"]["enum"] = list(MARKER_FREE_LAYER_ENUM)
        layers_prop["description"] = (
            "Marker-free mode: pass exactly ['v2_no_markers']."
        )
        break
    return tools


# Backwards-compatibility alias used elsewhere in the code base.
SYSTEM_PROMPT = build_system_prompt(enable_visual=True)


# ---------------------------------------------------------------------------
# Proposal application + factual diff (for the iterative flow)
# ---------------------------------------------------------------------------
def _proposal_from_dict(prop_dict: Dict[str, Any]) -> Proposal:
    fa_raw = prop_dict.get("force_assign", []) or []
    pwm_raw = prop_dict.get("precinct_weight_multipliers", []) or []
    pwm: Dict[int, float] = {}
    for pair in pwm_raw:
        try:
            pwm[int(pair[0])] = float(pair[1])
        except (IndexError, TypeError, ValueError):
            continue
    return Proposal(
        force_open=[int(j) for j in (prop_dict.get("force_open") or [])],
        force_close=[int(j) for j in (prop_dict.get("force_close") or [])],
        force_assign=[tuple(int(v) for v in p) for p in fa_raw],
        precinct_weight_multipliers=pwm,
    )


def _summarise_proposal_outcome(
    prop_dict: Dict[str, Any],
    new_solution: Solution,
    prev_solution: Solution,
    resolve_index: int,
) -> str:
    """Tool-message text returned to the agent after a `resolve` call.
    Reports feasibility + a factual diff (which sites flipped, how many
    precincts were reassigned). Deliberately does NOT reveal the formal
    score — the agent should judge from the structural change whether
    the critique is addressed."""
    feasible = bool(new_solution.metadata.get("feasible", True))
    if not feasible:
        return (
            f"resolve #{resolve_index} could not be applied — the MILP "
            f"is INFEASIBLE under your fixings (likely budget / capacity "
            f"conflict). The current solution shown to other tools "
            f"remains the previous one. Revise and call resolve again."
        )

    prev_open = set(int(j) for j in np.where(prev_solution.x == 1)[0])
    new_open = set(int(j) for j in np.where(new_solution.x == 1)[0])
    newly_opened = sorted(new_open - prev_open)
    newly_closed = sorted(prev_open - new_open)
    n_reassigned = int((prev_solution.y != new_solution.y).any(axis=1).sum())

    parts = [f"resolve #{resolve_index} applied (feasible)."]
    if newly_opened:
        parts.append(f"Newly opened sites: {newly_opened}.")
    if newly_closed:
        parts.append(f"Newly closed sites: {newly_closed}.")
    if not (newly_opened or newly_closed):
        parts.append("No site openings/closings changed.")
    parts.append(f"Precincts whose assignment changed: {n_reassigned}.")
    parts.append(
        "The 'current solution' visible to view_solution / list_sites / "
        "get_current_assignments / get_distance_matrix / "
        "get_precinct_adjacency is now this re-solved solution. Inspect "
        "and either call resolve again to refine (it will REPLACE this "
        "one — include every fixing you want to keep), make local edits "
        "with force_assign / swap_assignments, or call submit_proposal "
        "to end the reasoning loop."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
def dispatch_tool(
    name: str,
    args_str: str,
    instance: Instance,
    solution: Solution,
) -> Tuple[
    str,                        # text_for_tool_message
    Optional[str],              # image_b64 (or None)
    Optional[Dict[str, Any]],   # proposal_dict (when name == "submit_proposal")
    Optional[Solution],         # updated_solution (local edits / MILP one-shots)
]:
    """Run a tool. Return (text, image_b64_or_None, proposal_dict_or_None,
    updated_solution_or_None).

    `updated_solution` is non-None when a local-edit tool (force_assign,
    swap_assignments) directly mutated the solution; the loop adopts it
    as the new current_solution.

    `proposal_dict` is non-None for `resolve` (the loop applies the MILP
    fixings) or for `submit_proposal` (the loop ends; the dict carries
    `_submit: True`).
    """
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError as e:
        return f"Error parsing arguments: {e}", None, None, None

    if name == "view_solution":
        # The agent passes one or more canonical view types
        # ("baseline", "population_density", "plain", "v2_no_markers").
        # v2_no_markers uses rendering_v2_no_markers exclusively and cannot
        # merge with rendering.py primitives.
        view_types = args.get("layers", ["baseline"])
        show_pl = bool(args.get("show_precinct_labels", False))
        if "v2_no_markers" in view_types:
            others = [v for v in view_types if v != "v2_no_markers"]
            note = ""
            if others:
                note = (
                    f" Note: 'v2_no_markers' cannot merge with other view tokens; "
                    f"ignored: {others}."
                )
            png = view_solution_v2_no_markers_png(instance, solution)
            b64 = base64.b64encode(png).decode("ascii")
            opened = list_sites(instance, solution, opened_only=True)
            opened_compact = [
                {"i": s["index"],
                 "x": round(s["x"], 2),
                 "y": round(s["y"], 2),
                 "type": s["type"],
                 "load": s["load"]}
                for s in opened
            ]
            text = (
                f"Map rendered (v2_no_markers). view_types={view_types}, "
                f"show_precinct_labels ignored for this renderer.{note}\n\n"
                "This view has NO site index labels on the image — use the "
                "ground-truth opened sites list for any index/coordinate question:\n"
                f"{json.dumps(opened_compact)}"
            )
            return text, b64, None, None

        primitive_layers: List[str] = []
        seen: set = set()
        for vt in view_types:
            for prim in LAYER_TO_PRIMITIVES.get(vt, [vt]):
                if prim not in seen:
                    primitive_layers.append(prim)
                    seen.add(prim)
        png = view_solution_png(instance, solution, layers=primitive_layers,
                                 show_precinct_labels=show_pl)
        b64 = base64.b64encode(png).decode("ascii")

        # Attach structured ground truth alongside the image so the agent
        # never needs to OCR labels off the map for positional reasoning.
        opened = list_sites(instance, solution, opened_only=True)
        opened_compact = [
            {"i": s["index"],
             "x": round(s["x"], 2),
             "y": round(s["y"], 2),
             "type": s["type"],
             "load": s["load"]}
            for s in opened
        ]
        text = (
            f"Map rendered. view_types={view_types}, "
            f"show_precinct_labels={show_pl}.\n\n"
            "Ground-truth opened sites (use this list, NOT the image, for any "
            "positional question — coordinates, ordering, indices, etc.):\n"
            f"{json.dumps(opened_compact)}"
        )
        return text, b64, None, None

    if name == "list_sites":
        opened_only = bool(args.get("opened_only", False))
        sites = list_sites(instance, solution, opened_only=opened_only)
        return json.dumps(sites), None, None, None

    if name == "list_precincts_in_region":
        polygon = np.array(args["polygon"], dtype=float)
        ps = list_precincts_in_region(instance, polygon)
        return json.dumps(ps), None, None, None

    if name == "get_site_at":
        max_d = float(args.get("max_distance", 0.6))
        s = get_site_at(instance, float(args["x"]), float(args["y"]), max_d)
        return json.dumps(s), None, None, None

    if name == "get_precinct_at":
        p = get_precinct_at(instance, float(args["x"]), float(args["y"]))
        return json.dumps(p), None, None, None

    if name == "get_precinct_centroids":
        data = get_precinct_centroids(instance, args.get("precinct_indices"))
        return json.dumps(data), None, None, None

    if name == "get_precinct_adjacency":
        data = get_precinct_adjacency_data(instance)
        return json.dumps(data), None, None, None

    if name == "get_current_assignments":
        precinct_indices = args.get("precinct_indices")
        data = get_current_assignments(instance, solution, precinct_indices)
        return json.dumps(data), None, None, None

    if name == "get_distance_matrix":
        data = get_distance_matrix_data(
            instance,
            solution=solution,
            precinct_indices=args.get("precinct_indices"),
            site_indices=args.get("site_indices"),
            opened_only=bool(args.get("opened_only", False)),
        )
        return json.dumps(data), None, None, None

    if name == "force_assign":
        try:
            i = int(args["precinct_index"])
            j = int(args["site_index"])
        except (KeyError, TypeError, ValueError) as e:
            return (f"force_assign needs integer precinct_index and "
                     f"site_index ({e})."), None, None, None
        new_sol, summary = apply_local_assignment(instance, solution, i, j)
        updated_sol = new_sol if new_sol is not solution else None
        return summary, None, None, updated_sol

    if name == "swap_assignments":
        try:
            a = int(args["precinct_a_index"])
            b = int(args["precinct_b_index"])
        except (KeyError, TypeError, ValueError) as e:
            return (f"swap_assignments needs integer precinct_a_index "
                     f"and precinct_b_index ({e})."), None, None, None
        new_sol, summary = apply_local_swap(instance, solution, a, b)
        updated_sol = new_sol if new_sol is not solution else None
        return summary, None, None, updated_sol

    if name == "resolve":
        # Return the args as a "resolve dict" — the loop will apply the
        # fixings, advance current_solution, and write a richer summary.
        return "Resolve received.", None, args, None

    if name == "submit_proposal":
        # Final submission — return a sentinel marking loop end. The
        # current_solution is the answer; submit_proposal carries no
        # fixings of its own (just a rationale).
        return "Submission received.", None, {"_submit": True, **args}, None

    return f"Unknown tool: {name}", None, None, None


def _last_feasible_resolve_target_delta(
    log: List[Dict[str, Any]],
) -> Optional[float]:
    """Return target_delta from the most recent feasible resolve, if any."""
    for e in reversed(log):
        if e.get("event") != "resolve_applied" or not e.get("feasible"):
            continue
        td = e.get("target_delta")
        if td is None:
            return None
        try:
            return float(td)
        except (TypeError, ValueError):
            return None
    return None


def _primary_metric_stalled(
    query: ArchetypeQuery,
    target_delta: float,
    *,
    eps: float = 1e-9,
) -> bool:
    """True if the last resolve did not strictly improve the primary target."""
    if query.target_direction == "minimize":
        return target_delta >= -eps
    return target_delta <= eps


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def run_agent(
    instance: Instance,
    baseline: Solution,
    query_text: str,
    query: Optional[ArchetypeQuery] = None,
    annotation_polygons: Optional[List[np.ndarray]] = None,
    model: str = "gpt-4o",
    max_iters: int = 25,
    max_primary_stall_nudges: int = 5,
    save_log_path: Optional[str] = None,
    enable_visual: bool = True,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    system_prompt_suffix: Optional[str] = None,
    marker_free_maps: bool = False,
) -> Tuple[
    Optional[Dict[str, Any]],
    Solution,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Run the agent loop. The agent has three classes of solution-
    modifying tool:
      - resolve: re-solve the MILP under fixings; advance current_solution.
      - force_assign / swap_assignments: local edits on current_solution.
      - submit_proposal: end the reasoning loop.

    The loop ends when:
      - the agent calls submit_proposal, OR
      - the agent stops calling tools after at least one solution-modifying
        action (implicit submit), OR
      - max_iters is exhausted.

    When ``query`` is set, a feasible ``resolve`` that does not strictly
    improve the primary archetype metric triggers up to
    ``max_primary_stall_nudges`` continuation user-messages instead of
    accepting an immediate implicit submit, so the agent is pushed to try
    further interventions.

    When ``marker_free_maps`` is True (multimodal only), every rendered map
    — initial user attachment, ``view_solution`` tool output, and automatic
    follow-ups after feasible resolves/local edits — uses
    ``rendering_v2_no_markers`` only (no site/candidate markers). The
    ``view_solution`` tool schema allows only ``['v2_no_markers']``.

    Parameters
    ----------
    enable_visual : if True (default), the agent has the view_solution tool
        and receives an initial rendered map AND a fresh map after each
        resolve. If False, view_solution is removed and no images are
        sent — tools-only condition.
    marker_free_maps : if True with ``enable_visual``, all maps use
        ``rendering_v2_no_markers`` only and the tool schema restricts
        ``view_solution`` layers to ``['v2_no_markers']``.

    Returns
    -------
    (final_proposal_dict_or_None, final_solution, conversation_log,
     explored_solutions)
        - final_proposal_dict: the last feasible resolve's fixings, or
          None if no feasible resolve was ever called.
        - final_solution: the current_solution at end-of-loop. SCORING
          may choose to superscore against explored_solutions instead.
        - conversation_log: list of trajectory events.
        - explored_solutions: feasible solution states visited during the
          loop, including the baseline and every feasible resolve/local edit.
          submit_proposal only ends the reasoning loop; it does not select
          which explored solution should be scored.
        - when `query` is provided, each feasible resolve logs
          target_before/target_after/delta so the agent can see whether a
          resolve actually improved the archetype metric.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai>=1.40 required. `pip install --upgrade openai`.") from e

    client = OpenAI()  # uses OPENAI_API_KEY env var

    if marker_free_maps and not enable_visual:
        raise ValueError(
            "marker_free_maps=True requires enable_visual=True (multimodal)."
        )
    mf = bool(enable_visual and marker_free_maps)

    # Pick prompt + tool list according to modality.
    system_prompt = build_system_prompt(
        enable_visual=enable_visual, marker_free_maps=mf)
    if query is not None and query.archetype == "coverage_gap":
        system_prompt = system_prompt + "\n\n" + COVERAGE_GAP_AGENT_GUIDANCE
    if system_prompt_suffix:
        system_prompt = system_prompt + "\n\n" + system_prompt_suffix.strip()
    tools_for_run = get_tools_for_run(
        enable_visual=enable_visual, marker_free_maps=mf)

    # Build the initial user content: critique text + (optionally) annotation
    # polygons + (when visual is enabled) an initial map view.
    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": query_text}]
    if annotation_polygons:
        ann_text = "Stakeholder annotation: the user has marked the following region(s)"
        ann_text += " (polygon vertices in km coordinates):\n"
        for i, p in enumerate(annotation_polygons):
            ann_text += f"  Region {i + 1}: {np.asarray(p).tolist()}\n"
        user_parts.append({"type": "text", "text": ann_text})

    if enable_visual:
        # Initial map view. Marker-free mode cannot overlay annotation polygons
        # on the v2_no_markers renderer — vertices are still in the text above.
        init_png = _multimodal_solution_png(
            instance, baseline,
            marker_free_maps=mf,
            region=None if mf else (
                annotation_polygons if annotation_polygons else None),
        )
        init_b64 = base64.b64encode(init_png).decode("ascii")
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{init_b64}"},
        })
        if mf:
            ann_note = ""
            if annotation_polygons:
                ann_note = (
                    " Stakeholder polygon vertices are listed in text only — "
                    "the marker-free map has no polygon overlay."
                )
            user_parts.append({
                "type": "text",
                "text": (
                    "Initial view: marker-free catchment map (v2_no_markers — "
                    "saturated fills, black catchment outlines; no site markers)."
                    f"{ann_note} Call view_solution with layers=['v2_no_markers'] "
                    "to refresh after changes."
                ),
            })
        else:
            user_parts.append({
                "type": "text",
                "text": ("Initial view: baseline solution with assignments. "
                          "Call view_solution again with population_density "
                          "or plain if you want a different background."),
            })
    else:
        user_parts.append({
            "type": "text",
            "text": ("(TOOLS-ONLY MODE: no rendered map is provided and the "
                      "view_solution tool is unavailable. Use the structured "
                      "tools to gather information.)"),
        })

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]

    log: List[Dict[str, Any]] = [
        {"event": "init",
         "log_schema_version": 1,
         "query_text": query_text,
         "enable_visual": bool(enable_visual),
         "marker_free_maps": mf,
         "n_annotations": 0 if not annotation_polygons else len(annotation_polygons),
         "model": model,
         "temperature": temperature,
         "seed": seed},
    ]

    # Iterative-proposal state. The "current solution" is what the agent's
    # subsequent tool calls operate against; it starts as the baseline and
    # advances each time a feasible proposal is applied.
    current_solution: Solution = baseline
    proposal_history: List[Dict[str, Any]] = []
    explored_solutions: List[Dict[str, Any]] = [{
        "source": "baseline",
        "iteration": -1,
        "solution": baseline,
        "proposal": None,
        "n_resolves": 0,
        "n_local_edits": 0,
    }]
    final_proposal: Optional[Dict[str, Any]] = None
    final_submitted: bool = False
    tool_call_made = False
    nudges_used = 0
    primary_stall_nudges_sent = 0

    # Visual-inspection tracking (multimodal only; ignored when enable_visual=False).
    had_baseline_view: bool = False      # agent called view_solution w/ baseline_vs_critique
    pending_post_action_view: bool = False  # a state change occurred; post-action view expected
    n_view_solution_calls: int = 0
    view_purpose_counts: Dict[str, int] = {}

    loop_exhausted = False
    termination_reason: Optional[str] = None
    for it in range(max_iters):
        print(f"\n--- Agent iteration {it + 1} ---", flush=True)
        try:
            create_kwargs: Dict[str, Any] = {
                "model": model, "messages": messages,
                "tools": tools_for_run, "tool_choice": "auto",
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if seed is not None:
                create_kwargs["seed"] = seed
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as e:
            print(f"OpenAI API error: {e}", file=sys.stderr)
            log.append({"event": "api_error", "error": str(e)})
            termination_reason = "api_error"
            break

        msg = resp.choices[0].message
        choice0 = resp.choices[0]
        finish_reason = getattr(choice0, "finish_reason", None)
        usage = getattr(resp, "usage", None)
        usage_event: Dict[str, Any] = {"event": "api_usage", "iteration": it}
        if usage is not None:
            usage_event["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            usage_event["completion_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0)
            usage_event["total_tokens"] = int(getattr(usage, "total_tokens", 0) or 0)
            ptd = getattr(usage, "prompt_tokens_details", None)
            if ptd is not None:
                usage_event["cached_tokens"] = int(
                    getattr(ptd, "cached_tokens", 0) or 0)
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                usage_event["reasoning_tokens"] = int(
                    getattr(ctd, "reasoning_tokens", 0) or 0)
        log.append(usage_event)
        log.append({
            "event": "assistant",
            "content": msg.content,
            "finish_reason": finish_reason,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
        })
        # Append assistant message back into history
        try:
            asst_dict = msg.model_dump(exclude_none=True)
        except Exception:
            asst_dict = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in (msg.tool_calls or [])
                ] if msg.tool_calls else None,
            }
        asst_dict["role"] = "assistant"
        messages.append(asst_dict)

        if msg.content:
            local_edits_so_far = sum(1 for e in log if e.get("event") == "local_edit")
            is_terminal_narrative = (
                not msg.tool_calls
                and (proposal_history or local_edits_so_far > 0)
            )
            if is_terminal_narrative:
                # Replace agent self-report with a factual summary derived from
                # the actual tool call record to avoid hallucinated descriptions.
                parts = []
                for p in proposal_history:
                    prop = p.get("proposal", {})
                    fixings = []
                    for k in ("force_open", "force_close", "force_assign"):
                        if prop.get(k):
                            fixings.append(f"{k}={prop[k]}")
                    if prop.get("precinct_weight_multipliers"):
                        fixings.append(
                            f"weights({len(prop['precinct_weight_multipliers'])} precincts)"
                        )
                    status = "feasible" if p.get("feasible") else "INFEASIBLE"
                    parts.append(
                        f"resolve #{p['index']} "
                        f"[{', '.join(fixings) or 'no fixings'}] -> {status}"
                    )
                local_edit_events = [e for e in log if e.get("event") == "local_edit"]
                if local_edit_events:
                    parts.append(f"{len(local_edit_events)} local edit(s): "
                                 f"{[e.get('tool') for e in local_edit_events]}")
                summary = " | ".join(parts) if parts else "(no solution-modifying actions)"
                print(f"  [factual summary] {summary}", flush=True)
            else:
                preview = (msg.content[:300] + "...") if len(msg.content) > 300 else msg.content
                print(f"  assistant: {preview}", flush=True)

        if not msg.tool_calls:
            # No tool call this turn. Three cases:
            #   - The agent has submitted at least one proposal OR made
            #     local edits — current_solution reflects the agent's
            #     intended final state; accept and end. (Implicit finalize.)
            #   - The agent has called other tools but never modified the
            #     solution; treat the textual reply as a factual answer.
            #   - The agent hasn't inspected yet and is guessing — nudge.
            local_edits = sum(1 for e in log if e.get("event") == "local_edit")
            if proposal_history or local_edits > 0:
                feasible_props = [p for p in proposal_history if p["feasible"]]
                stall_nudge = False
                if (
                    query is not None
                    and feasible_props
                    and primary_stall_nudges_sent < max_primary_stall_nudges
                ):
                    td = _last_feasible_resolve_target_delta(log)
                    if td is not None and _primary_metric_stalled(query, td):
                        stall_nudge = True
                if stall_nudge:
                    primary_stall_nudges_sent += 1
                    print(
                        "  (no tool calls — primary target did not strictly improve "
                        f"after last feasible resolve; continuation nudge "
                        f"{primary_stall_nudges_sent}/{max_primary_stall_nudges})",
                        flush=True,
                    )
                    log.append({
                        "event": "primary_target_stall_nudge",
                        "iteration": it,
                        "nudge_index": primary_stall_nudges_sent,
                        "max_primary_stall_nudges": max_primary_stall_nudges,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "System note: the last feasible `resolve` did not strictly "
                            "improve the primary benchmark target for this task (see the "
                            "primary target feedback on that resolve). The stakeholder "
                            "request is not yet quantitatively addressed.\n\n"
                            "Continue the session: use tools to refine which sites or "
                            "precincts matter, then call `resolve` again with different "
                            "fixings and/or try `force_assign` / `swap_assignments`. "
                            "Only end with `submit_proposal` or a final text turn after "
                            "you have either improved the primary target or you are "
                            "confident no better feasible action exists under the guards."
                        ),
                    })
                    continue
                if feasible_props:
                    final_proposal = feasible_props[-1]["proposal"]
                print(f"  (no tool calls -> implicit submit: "
                       f"{len(proposal_history)} resolve(s), "
                       f"{local_edits} local edit(s))", flush=True)
                break
            if tool_call_made or nudges_used >= 1:
                print("  (no tool calls -> treating as final answer)", flush=True)
                break
            messages.append({
                "role": "user",
                "content": (
                    "Reminder: do not estimate positions or indices from "
                    "the image. Call list_sites / list_precincts_in_region "
                    "/ get_site_at / get_precinct_at / "
                    "get_precinct_centroids first to get ground-truth "
                    "coordinates and indices, then reason from that output. "
                    "If your task is a factual question, give a "
                    "direct textual answer after inspecting. If it is a "
                    "change request, use resolve (MILP-based) and/or "
                    "force_assign / swap_assignments (local edits) to "
                    "modify the current solution, then call submit_proposal "
                    "to end the reasoning loop."
                ),
            })
            nudges_used += 1
            continue

        tool_call_made = True
        # Dispatch each tool call. Tool calls operate against the
        # CURRENT solution (which is updated each time a feasible
        # proposal is applied OR a local edit returns an updated solution).
        image_followups: List[Dict[str, Any]] = []
        n_local_edits_this_turn = 0
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = tc.function.arguments
            # Parse args once; used for structured logging and view tracking.
            try:
                parsed_args: Dict[str, Any] = json.loads(tool_args) if tool_args else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if mf and tool_name == "view_solution":
                if parsed_args.get("layers") != ["v2_no_markers"]:
                    parsed_args = {**parsed_args, "layers": ["v2_no_markers"]}
                    tool_args = json.dumps(parsed_args)
            print(f"  -> {tool_name}({tool_args[:200]}{'...' if len(tool_args) > 200 else ''})",
                  flush=True)

            text, img_b64, prop_dict, updated_sol = dispatch_tool(
                tool_name, tool_args, instance, current_solution)

            # Local-edit tools (force_assign, swap_assignments) return an
            # updated_solution that the loop should adopt as the new
            # current_solution. These don't go through the proposal-
            # application path; they act directly on the solution state.
            if updated_sol is not None and prop_dict is None:
                current_solution = updated_sol
                n_local_edits_this_turn += 1
                total_local_edits = sum(
                    1 for e in log if e.get("event") == "local_edit") + 1
                feasible_after = bool(
                    current_solution.metadata.get("feasible", True))
                # Structured local_edit event — rationale + key indices inline,
                # no raw argument string, for clean log reading.
                edit_event: Dict[str, Any] = {
                    "event": "local_edit",
                    "iteration": it,
                    "tool": tool_name,
                    "rationale": parsed_args.get("rationale", ""),
                    "feasible_after": feasible_after,
                    "solver_status_after":
                        current_solution.metadata.get("solver_status"),
                }
                if tool_name == "force_assign":
                    edit_event["precinct_index"] = parsed_args.get("precinct_index")
                    edit_event["site_index"] = parsed_args.get("site_index")
                elif tool_name == "swap_assignments":
                    edit_event["precinct_a_index"] = parsed_args.get("precinct_a_index")
                    edit_event["precinct_b_index"] = parsed_args.get("precinct_b_index")
                log.append(edit_event)

                if feasible_after:
                    explored_solutions.append({
                        "source": tool_name,
                        "iteration": it,
                        "solution": current_solution,
                        "proposal": None,
                        "n_resolves": len(proposal_history),
                        "n_local_edits": total_local_edits,
                    })
                    # Auto-render after feasible local edit in multimodal mode
                    # so the agent always sees the updated state (mirrors
                    # what resolve already does).
                    if enable_visual:
                        follow_png = _multimodal_solution_png(
                            instance, current_solution,
                            marker_free_maps=mf,
                        )
                        img_b64 = base64.b64encode(follow_png).decode("ascii")
                if enable_visual:
                    pending_post_action_view = True

            # If this was a `resolve` or `submit_proposal` call, handle it.
            #   resolve         -> apply MILP under fixings, advance current_sol
            #   submit_proposal -> end the loop; current_solution is the answer
            if prop_dict is not None:
                is_submit = bool(prop_dict.get("_submit", False))
                if is_submit:
                    log.append({
                        "event": "solution_submitted",
                        "iteration": it,
                        "rationale": prop_dict.get("rationale", ""),
                        "n_resolves_before": len(proposal_history),
                        "n_local_edits_before":
                            sum(1 for e in log if e.get("event")
                                                    == "local_edit"),
                        "feasible": bool(
                            current_solution.metadata.get("feasible", True)),
                    })
                    text = (
                        f"Submission received. Reasoning loop ending. "
                        f"{len(proposal_history)} resolve(s) and "
                        f"{sum(1 for e in log if e.get('event') == 'local_edit')} "
                        f"local edit(s) preceded this submission. The "
                        f"benchmark will score the best feasible solution "
                        f"explored during the session."
                    )
                    final_submitted = True
                    img_b64 = None
                else:
                    # resolve: apply MILP under fixings.
                    proposal_obj = _proposal_from_dict(prop_dict)
                    applied_solution = apply_proposal(instance, proposal_obj)
                    resolve_index = len(proposal_history) + 1
                    feasible = bool(
                        applied_solution.metadata.get("feasible", True))

                    text = _summarise_proposal_outcome(
                        prop_dict, applied_solution, current_solution,
                        resolve_index,
                    )
                    # Primary-target feedback (when query is known): report
                    # whether this resolve moved the archetype metric.
                    target_before: Optional[float] = None
                    target_after: Optional[float] = None
                    target_delta: Optional[float] = None
                    if query is not None and feasible:
                        try:
                            target_before = float(
                                query.target_metric_fn(instance, current_solution))
                            target_after = float(
                                query.target_metric_fn(instance, applied_solution))
                            target_delta = target_after - target_before
                            if query.target_direction == "minimize":
                                direction_note = (
                                    "improved" if target_after < target_before
                                    else "unchanged" if abs(target_after - target_before) < 1e-9
                                    else "worsened"
                                )
                            else:
                                direction_note = (
                                    "improved" if target_after > target_before
                                    else "unchanged" if abs(target_after - target_before) < 1e-9
                                    else "worsened"
                                )
                            text += (
                                f" Primary target feedback ({query.archetype}, "
                                f"{query.target_direction}): "
                                f"{target_before:.3f} -> {target_after:.3f} "
                                f"(delta {target_delta:+.3f}; {direction_note})."
                            )
                        except Exception:
                            # Never fail a resolve due to feedback plumbing.
                            target_before = None
                            target_after = None
                            target_delta = None

                    proposal_history.append({
                        "iteration": it,
                        "index": resolve_index,
                        "proposal": prop_dict,
                        "feasible": feasible,
                    })
                    resolve_ev: Dict[str, Any] = {
                        "event": "resolve_applied",
                        "iteration": it,
                        "resolve_index": resolve_index,
                        "feasible": feasible,
                        "proposal": prop_dict,
                        "target_before": target_before,
                        "target_after": target_after,
                        "target_delta": target_delta,
                        "target_direction": query.target_direction if query else None,
                        "target_name": query.archetype if query else None,
                    }
                    if not feasible:
                        meta = applied_solution.metadata or {}
                        keys = (
                            "gurobi_status", "solver_status", "force_open",
                            "force_close", "force_assign",
                        )
                        resolve_ev["solver_infeasible_meta"] = {
                            k: meta[k] for k in keys if k in meta
                        }
                    log.append(resolve_ev)

                    # Advance current_solution only when feasible.
                    if feasible:
                        current_solution = applied_solution
                        explored_solutions.append({
                            "source": "resolve",
                            "iteration": it,
                            "solution": current_solution,
                            "proposal": prop_dict,
                            "resolve_index": resolve_index,
                            "n_resolves": len(proposal_history),
                            "n_local_edits": sum(
                                1 for e in log
                                if e.get("event") == "local_edit"),
                        })

                    # Auto-render the updated state in multimodal mode.
                    if enable_visual and feasible:
                        follow_png = _multimodal_solution_png(
                            instance, current_solution,
                            marker_free_maps=mf,
                        )
                        img_b64 = base64.b64encode(follow_png).decode("ascii")
                        pending_post_action_view = True
                    else:
                        img_b64 = None

            # Build the tool_call log event. For view_solution, record
            # view_purpose and update tracking counters. For all other tools
            # the event is minimal (name + text preview, capped at 4k chars).
            tool_call_event: Dict[str, Any] = {
                "event": "tool_call",
                "name": tool_name,
                "text_response": text[:4000],
                "had_image": img_b64 is not None,
                "is_proposal": prop_dict is not None,
            }
            if tool_name == "view_solution":
                vp = parsed_args.get("view_purpose", "unspecified")
                tool_call_event["view_purpose"] = vp
                tool_call_event["layers"] = parsed_args.get("layers", [])
                if img_b64:
                    try:
                        tool_call_event["render_png_sha256"] = (
                            hashlib.sha256(base64.b64decode(img_b64)).hexdigest()
                        )
                    except Exception:
                        pass
                n_view_solution_calls += 1
                view_purpose_counts[vp] = view_purpose_counts.get(vp, 0) + 1
                if vp == "baseline_vs_critique":
                    had_baseline_view = True
                elif vp == "post_action_vs_baseline":
                    pending_post_action_view = False
            log.append(tool_call_event)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": text,
            })

            if img_b64 is not None:
                image_followups.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

        if image_followups:
            content_parts = [
                {"type": "text", "text": "Tool result image(s):"},
            ] + image_followups
            messages.append({"role": "user", "content": content_parts})

        if final_submitted:
            # Take the last feasible resolve's fixings as the proposal-of-
            # record (or None if the agent only used local edits).
            feasible_props = [p for p in proposal_history if p["feasible"]]
            if feasible_props:
                final_proposal = feasible_props[-1]["proposal"]
            print(f"  (submit_proposal called after "
                   f"{len(proposal_history)} resolve(s) — ending loop)",
                   flush=True)
            termination_reason = "explicit_submit"
            break
    else:
        loop_exhausted = True
        if termination_reason is None:
            termination_reason = "max_iters"

    # If we ran out of iterations without an explicit submit but the
    # agent did some work, accept the current state.
    if not final_submitted and proposal_history and loop_exhausted:
        feasible_props = [p for p in proposal_history if p["feasible"]]
        if feasible_props:
            final_proposal = feasible_props[-1]["proposal"]
        print(f"  (max_iters reached — taking last resolve "
               f"#{len(proposal_history)} as final)", flush=True)
        if termination_reason == "max_iters":
            termination_reason = "max_iters_implicit_finalize"

    n_local_edits = sum(1 for e in log if e.get("event") == "local_edit")
    n_resolves = len(proposal_history)
    n_submitted = sum(1 for e in log
                       if e.get("event") == "solution_submitted")
    summary_event: Dict[str, Any] = {
        "event": "trajectory_summary",
        "log_schema_version": 1,
        "termination_reason": termination_reason or "unknown",
        "n_resolves": n_resolves,
        "n_local_edits": n_local_edits,
        "n_submitted_explicit": n_submitted,
        "any_resolve_infeasible":
            any(not p["feasible"] for p in proposal_history),
        "final_resolve_index":
            proposal_history[-1]["index"] if proposal_history else None,
        "final_solution_feasible":
            bool(current_solution.metadata.get("feasible", True)),
        "final_solver_status":
            current_solution.metadata.get("solver_status"),
        "n_feasible_solutions_explored": len(explored_solutions),
    }
    usage_events = [e for e in log if e.get("event") == "api_usage"]
    if usage_events:
        summary_event["n_model_calls"] = len(usage_events)
        summary_event["usage"] = {
            "prompt_tokens": int(sum(int(e.get("prompt_tokens", 0))
                                      for e in usage_events)),
            "completion_tokens": int(sum(int(e.get("completion_tokens", 0))
                                          for e in usage_events)),
            "total_tokens": int(sum(int(e.get("total_tokens", 0))
                                     for e in usage_events)),
            "cached_tokens": int(sum(int(e.get("cached_tokens", 0))
                                      for e in usage_events)),
            "reasoning_tokens": int(sum(int(e.get("reasoning_tokens", 0))
                                         for e in usage_events)),
        }
    if enable_visual:
        summary_event["n_view_solution"] = n_view_solution_calls
        summary_event["had_baseline_view"] = had_baseline_view
        summary_event["pending_post_action_view_at_end"] = pending_post_action_view
        if view_purpose_counts:
            summary_event["view_purpose_counts"] = view_purpose_counts
    if primary_stall_nudges_sent:
        summary_event["primary_stall_nudges_sent"] = primary_stall_nudges_sent
    if n_resolves or n_local_edits or n_submitted or n_view_solution_calls:
        log.append(summary_event)

    if save_log_path:
        with open(save_log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"Trajectory log saved: {save_log_path}", flush=True)

    return final_proposal, current_solution, log, explored_solutions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", required=True,
                        help=("Directory with instance.pkl, "
                              "baseline_solution.pkl, query_metadata.json "
                              "(produced by dataset_generator.py)."))
    parser.add_argument("--query_type", default="vague",
                        choices=["vague", "precise"],
                        help=("'vague' = stakeholder-style, no specific "
                              "entity reference. 'precise' = names the "
                              "offending site / region."))
    parser.add_argument("--model", default="gpt-5-mini",
                        help="OpenAI model id (must support vision).")
    parser.add_argument("--max_iters", type=int, default=25)
    parser.add_argument("--save_log", default=None,
                        help="Optional path to save the conversation log.")
    parser.add_argument("--no_visual", action="store_true",
                        help=("Disable view_solution and the initial map. "
                              "Tools-only condition."))
    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature (e.g. 0.0 = deterministic). "
                             "Omit to use the model's API default.")
    parser.add_argument("--seed", type=int, default=None,
                        help="LLM seed for reproducibility (e.g. 42). "
                             "Omit to use the model's API default.")
    parser.add_argument(
        "--marker_free_maps",
        action="store_true",
        help=(
            "Multimodal-only: all session maps use rendering_v2_no_markers; "
            "view_solution accepts only layers=['v2_no_markers']. "
            "Incompatible with --no_visual."
        ),
    )
    args = parser.parse_args()

    if args.marker_free_maps and args.no_visual:
        print(
            "ERROR: --marker_free_maps requires multimodal (omit --no_visual).",
            file=sys.stderr,
        )
        sys.exit(1)

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    pair_dir = os.path.abspath(args.pair_dir)
    instance = Instance.load(os.path.join(pair_dir, "instance.pkl"))
    baseline = Solution.load(os.path.join(pair_dir, "baseline_solution.pkl"))
    with open(os.path.join(pair_dir, "query_metadata.json")) as f:
        meta = json.load(f)

    archetype = meta.get("archetype")
    factory = ARCHETYPE_FACTORIES.get(archetype)
    if factory is None:
        print(f"ERROR: Unknown archetype '{archetype}' in metadata.",
               file=sys.stderr)
        sys.exit(1)

    text_key = "vague_text" if args.query_type == "vague" else "precise_text"
    text = meta.get(text_key) or meta.get("text") or ""

    pair_id = os.path.basename(pair_dir.rstrip("/"))
    query: ArchetypeQuery = factory(
        query_id=f"{archetype}_{pair_id}_{args.query_type}",
        text=text,
        metadata_dict=meta,
    )

    modality = "tools_only" if args.no_visual else "multimodal"
    print("=" * 60)
    print(f"QUERY  archetype={query.archetype}  "
           f"query_type={args.query_type}  modality={modality}")
    print("=" * 60)
    if query.description:
        print(f"description: {query.description}")
    print()
    print(query.text)

    target_baseline_value = float(query.target_metric_fn(instance, baseline))
    print()
    print(f"Baseline target value ({query.target_direction}): "
           f"{target_baseline_value}")

    base_metrics = compute_metrics(instance, baseline)
    print("Reference metrics on baseline solution:")
    for k in ["total_weighted_distance", "p90_distance", "sites_opened",
              "feasible"]:
        print(f"  {k}: {base_metrics[k]}")

    proposal_dict, final_solution, log, explored_solutions = run_agent(
        instance, baseline, query.text,
        query=query,
        annotation_polygons=None,
        model=args.model, max_iters=args.max_iters,
        save_log_path=args.save_log,
        enable_visual=not args.no_visual,
        temperature=args.temperature,
        seed=args.seed,
        marker_free_maps=args.marker_free_maps,
    )

    n_local_edits = sum(1 for e in log if e.get("event") == "local_edit")
    if proposal_dict is None and n_local_edits == 0:
        last_text = None
        for entry in reversed(log):
            if entry.get("event") == "assistant" and entry.get("content"):
                last_text = entry["content"]
                break
        print("\n" + "=" * 60)
        print("AGENT ANSWER (no proposal submitted, no local edits)")
        print("=" * 60)
        print(last_text or
              "(Agent produced no textual response within the iteration budget.)")
        return

    n_resolves = sum(1 for e in log if e.get("event") == "resolve_applied")
    n_submitted = sum(1 for e in log if e.get("event") == "solution_submitted")
    n_infeasible = sum(1 for e in log if e.get("event") == "resolve_applied"
                                            and not e.get("feasible"))

    print("\n" + "=" * 60)
    print(f"AGENT TRAJECTORY  ({n_resolves} resolve(s), "
           f"{n_infeasible} infeasible, {n_local_edits} local edits, "
           f"{n_submitted} explicit submit)")
    print("=" * 60)
    if proposal_dict is not None:
        print("Last resolve fixings:")
        print(json.dumps(proposal_dict, indent=2))
    else:
        print("(No resolve called — final solution comes from local edits.)")

    feasible_explored = []
    for idx, entry in enumerate(explored_solutions):
        sol = entry["solution"]
        if not bool(sol.metadata.get("feasible", True)):
            continue
        feasible_explored.append({
            "index": idx,
            "entry": entry,
            "score": query.score(instance, baseline, sol),
        })
    if feasible_explored:
        def _score_key(item):
            score_item = item["score"]
            return (
                bool(score_item.get("success", False)),
                bool(score_item.get("valid", False)),
                float(score_item.get("fraction_improved", 0.0)),
                float(score_item.get("raw_improvement", 0.0)),
                -float(score_item.get("assignment_distance_delta", 0.0)),
                item["entry"].get("source") != "baseline",  # prefer agent action over baseline on tie
            )
        best_explored = max(feasible_explored, key=_score_key)
        new_solution = best_explored["entry"]["solution"]
    else:
        best_explored = None
        new_solution = final_solution
    new_metrics = compute_metrics(instance, new_solution)
    score = query.score(instance, baseline, new_solution)

    print("\n" + "=" * 60)
    print(f"RESULT  archetype={query.archetype}  "
           f"query_type={args.query_type}")
    print("=" * 60)
    if best_explored is not None:
        print(f"  superscore selected: source={best_explored['entry'].get('source')}  "
              f"iteration={best_explored['entry'].get('iteration')}  "
              f"feasible explored={len(feasible_explored)}")
    print(f"  target ({query.target_direction}): "
           f"{score['target_baseline']:.3f} -> "
           f"{score['target_response']:.3f}  "
           f"(improvement: {score['raw_improvement']:+.3f}, "
           f"{score['fraction_improved']*100:.0f}% of baseline)")
    print(f"  feasible              : {score['feasible']}")
    print(f"  ALL GUARDS PASSED     : {score['all_guards_passed']}")
    print(f"  VALID (feas + guards) : {score['valid']}")
    print()
    print("  Guards:")
    for g in score["guards"]:
        marker = "OK" if g["passed"] else "FAIL"
        print(f"    [{marker}] {g['name']:25s}  {g['baseline']:.3f} -> "
               f"{g['response']:.3f}  (bound {g['bound']:.3f}, "
               f"violation {g['violation']:.3f})")
    print()
    print(f"  Secondary metric — assignment distance:")
    print(f"    baseline: {score['baseline_assignment_distance']:.0f}  ->  "
           f"response: {score['final_assignment_distance']:.0f}  "
           f"(delta {score['assignment_distance_delta']:+.0f})")
    print()
    print("  Reference metrics (baseline -> response):")
    for k in ["total_weighted_distance", "p90_distance", "sites_opened"]:
        print(f"    {k}: {base_metrics[k]}  ->  {new_metrics[k]}")

    # ---- Render before/after solution maps ----
    views_dir = os.path.join(pair_dir, "views")
    os.makedirs(views_dir, exist_ok=True)
    render_layers = ['closed_sites', 'solution', 'assignments']

    slug = f"{query.archetype}__{args.query_type}"
    before_path = os.path.join(views_dir, f"response_{slug}__before.png")
    after_path = os.path.join(views_dir, f"response_{slug}__after.png")

    render_view(
        instance, baseline,
        layers=render_layers,
        title=f"BEFORE — {query.archetype} ({args.query_type})",
        save_path=before_path,
    )
    render_view(
        instance, new_solution,
        layers=render_layers,
        title=(f"AFTER agent action — {query.archetype} ({args.query_type}) "
                f"|  target {score['target_baseline']:.2f} -> "
                f"{score['target_response']:.2f}  "
                f"({score['fraction_improved']*100:.0f}% improved)"),
        save_path=after_path,
    )

    print()
    print("  Rendered comparison maps:")
    print(f"    before: {before_path}")
    print(f"    after : {after_path}")


if __name__ == "__main__":
    main()
