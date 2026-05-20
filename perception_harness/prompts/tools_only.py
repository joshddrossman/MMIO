"""Tools-only prompt: no map, structured probes only.

Mirrors the `--no_visual` mode in instance_generator/run_dataset.py.
The LLM never sees the rendered map; instead it has read-only tools
that expose the same structured data the optimization agent uses in
its tools-only condition. Lets the harness establish a baseline that
quantifies "how much does the visual modality contribute on top of
perfect structured access?".

Expected behaviour by archetype:
  - contiguity     : structured tools should saturate the score
                     (adjacency + assignments → connected components).
                     Failures here mean "the LLM didn't think to run
                     CC," not "the LLM lacked information." A useful
                     check on tool-using competence.
  - shape_niceness : centroids + adjacency are crude proxies for shape.
                     The LLM cannot compute NPI directly without
                     perimeter data. Expected to underperform v2 +
                     vision; the gap is the visual contribution.

Tools exposed (all read-only, mirror agent_tools.py in instance_generator):
    list_sites(opened_only)
    get_precinct_centroids(precinct_indices)
    get_precinct_adjacency()
    get_current_assignments(precinct_indices)
    get_assigned_site_at(x, y)

The build() ignores image_b64 — no map is sent. Pair with
models/openai_vlm_tools.py (or any adapter advertising
SUPPORTS_TOOLS=True). The renderer in --renderers is still produced
and saved for inspection but doesn't reach the LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Reuse the optimization agent's structured-tool implementations so
# semantics stay in sync between the harness and the deployed agent.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))

from agent_tools import (  # noqa: E402
    list_sites as _list_sites,
    get_precinct_centroids as _get_precinct_centroids,
    get_precinct_adjacency_data as _get_precinct_adjacency_data,
    get_current_assignments as _get_current_assignments,
)


NAME = "tools_only"

# Prompt capability — read by eval_perception.py to build view_info.
# False ⇒ the rendered map is not sent to the model (the runner still
# saves the PNG to disk for inspection, but it isn't part of the prompt).
USES_IMAGE = False


SYSTEM_TEXT = (
    "You are a careful analyst examining a polling-place plan for a "
    "county.  Use the structured tools below as needed to answer the user's question.\n\n"
    "Tools available:\n"
    "  list_sites(opened_only) — candidate or opened polling places "
    "(index, location in km, capacity, type, opened/closed status, "
    "and current voter load if opened).\n"
    "  get_precinct_centroids(precinct_indices) — centroid "
    "coordinates in km of the requested precincts (or all precincts "
    "if `precinct_indices` is omitted).\n"
    "  get_precinct_adjacency() — for each precinct, the indices of "
    "its spatially-adjacent precincts (4-connected on the rasterised "
    "Voronoi map). Useful for connectivity questions.\n"
    "  get_current_assignments(precinct_indices) — for each "
    "precinct, the polling place it is currently assigned to, plus "
    "the assigned travel distance. Omit `precinct_indices` for all.\n"
    "  get_assigned_site_at(x, y) — the polling place currently "
    "serving the precinct that contains coordinate (x, y).\n\n"
    "The map is approximately 0–10 km on each axis, and there are "
    "roughly 80 precincts and 18 opened polling places. When you "
    "have enough information to answer, respond in the format the "
    "user requests."
)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_sites",
            "description": (
                "Return structured info on candidate sites. Each entry "
                "has index, x, y, type, capacity, opened (bool), and "
                "load (voters assigned, only for opened sites)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "opened_only": {
                        "type": "boolean",
                        "description": "If true, return only opened sites.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_precinct_centroids",
            "description": (
                "Return precinct centroid coordinates in km. Each entry "
                "has precinct index, x, y, and voter count. Pass "
                "`precinct_indices` to focus the response, or omit to "
                "return all centroids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional precinct indices to include."),
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
                "Return precinct adjacency (4-connected on the "
                "rasterised Voronoi map). For each precinct, the "
                "neighbours list contains the indices of precincts "
                "sharing a boundary with it. Use with connected-"
                "components on an opened site's assigned-precinct "
                "subgraph to detect non-contiguous service areas."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_assignments",
            "description": (
                "Return current precinct-to-polling-place assignments. "
                "Each entry includes precinct index, centroid, voters, "
                "assigned site index, and assigned travel distance. "
                "Pass `precinct_indices` to focus the response, or "
                "omit to return all assignments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "precinct_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional precinct indices to include."),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_assigned_site_at",
            "description": (
                "Given a coordinate (x, y) in km, return the precinct "
                "at that location and the index of the opened polling "
                "place to which the precinct is assigned. Useful for "
                "spot-checking attribution at a specific point. The "
                "map spans approximately 0 ≤ x ≤ 10 and 0 ≤ y ≤ 10 km."
            ),
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
]


# ---------------------------------------------------------------------------
# Prompt builder — note: image_b64 is intentionally ignored here.
# ---------------------------------------------------------------------------
def build(*, task_name: str, question: str,
          image_b64: str = "",
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build the no-map message list. `image_b64` and `view_info` are
    accepted for runner compatibility but intentionally unused — this
    prompt operates entirely without the rendered map. The runner will
    have set view_info["has_visual"] = False since USES_IMAGE = False
    on this module, so the task module's format_question already
    selected the no-visual question text upstream of build()."""
    _ = (image_b64, view_info)  # explicitly unused
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": [
            {"type": "text", "text": question},
        ]},
    ]


def get_tools() -> List[Dict[str, Any]]:
    """Return the OpenAI-format tool schemas this prompt exposes."""
    return TOOLS


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, args_str: str, *,
                   instance, solution) -> str:
    """Execute one tool call and return JSON text (or a JSON error)."""
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"could not parse args JSON: {e}"})

    if name == "list_sites":
        opened_only = bool(args.get("opened_only", False))
        return json.dumps(_list_sites(instance, solution,
                                       opened_only=opened_only))

    if name == "get_precinct_centroids":
        return json.dumps(_get_precinct_centroids(
            instance, args.get("precinct_indices")))

    if name == "get_precinct_adjacency":
        return json.dumps(_get_precinct_adjacency_data(instance))

    if name == "get_current_assignments":
        return json.dumps(_get_current_assignments(
            instance, solution, args.get("precinct_indices")))

    if name == "get_assigned_site_at":
        try:
            x = float(args["x"])
            y = float(args["y"])
        except (KeyError, TypeError, ValueError) as e:
            return json.dumps({"error":
                                f"x and y are required numbers ({e})."})

        xmin, ymin, xmax, ymax = instance.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return json.dumps({
                "error": (f"({x}, {y}) is outside the map bounds "
                          f"[{xmin}, {xmax}] × [{ymin}, {ymax}]."),
            })
        G = len(instance.grid_xs)
        ix = int(np.clip(round((x - xmin) / (xmax - xmin) * (G - 1)),
                          0, G - 1))
        iy = int(np.clip(round((y - ymin) / (ymax - ymin) * (G - 1)),
                          0, G - 1))
        precinct_index = int(instance.precinct_label_grid[iy, ix])
        assigned_site = int(np.argmax(solution.y[precinct_index]))
        return json.dumps({
            "x": x, "y": y,
            "precinct_index": precinct_index,
            "assigned_site_index": assigned_site,
            "assigned_site_x":
                float(instance.site_locations[assigned_site, 0]),
            "assigned_site_y":
                float(instance.site_locations[assigned_site, 1]),
        })

    return json.dumps({"error": f"unknown tool: {name}"})
