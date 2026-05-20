"""Multimodal-full prompt: rendered map + full structured tool surface.

The cleanest "best of both worlds" cell of the experimental matrix:

    cell                map?    tool surface
    zero_shot           yes     none
    with_attribution    yes     1 tool (get_assigned_site_at)
    tools_only          no      5 tools (list_sites, centroids,
                                          adjacency, assignments,
                                          get_assigned_site_at)
    multimodal_full     yes     5 tools (same as tools_only)

The diagnostic comparison this unlocks: `multimodal_full` vs
`tools_only` differ ONLY in whether the model sees the map. Any F1 lift
of multimodal_full over tools_only is therefore the marginal value of
vision *given perfect structured access*. That isolates the multimodal
contribution from any thin-tools confound the with_attribution_* cells
introduce.

Reuse strategy. Tool schemas and the dispatcher are imported verbatim
from prompts.tools_only — same surface, same semantics, so any future
change to a tool propagates to both prompts automatically. Only the
build() function differs: multimodal_full embeds the image; tools_only
does not.

Pair with models/openai_vlm_tools.py (or any adapter advertising
SUPPORTS_TOOLS=True).
"""
from __future__ import annotations

from typing import Any, Dict, List

# Reuse the tool schemas and dispatcher from tools_only verbatim. The
# point of multimodal_full is to test the marginal value of vision
# given identical structured access — so the tool surface must match
# tools_only exactly.
from prompts.tools_only import (  # noqa: E402
    TOOLS,
    get_tools,
    dispatch_tool,
)


NAME = "multimodal_full"

# Prompt capability — read by eval_perception.py to build view_info.
USES_IMAGE = True


# ---------------------------------------------------------------------------
# Shared tool block (identical to tools_only's, slightly trimmed for
# co-presentation with the visual context)
# ---------------------------------------------------------------------------
_TOOL_BLOCK = (
    "You also have access to the following structured tools:\n"
    "  list_sites(opened_only) — candidate or opened polling places "
    "(index, location in km, capacity, type, opened/closed status, "
    "current voter load if opened).\n"
    "  get_precinct_centroids(precinct_indices) — centroid "
    "coordinates in km of the requested precincts (omit "
    "`precinct_indices` for all precincts).\n"
    "  get_precinct_adjacency() — for each precinct, the indices of "
    "its spatially-adjacent precincts (4-connected on the rasterised "
    "Voronoi map). Useful for connectivity questions.\n"
    "  get_current_assignments(precinct_indices) — for each "
    "precinct, the polling place it is currently assigned to and the "
    "assigned travel distance. Omit `precinct_indices` for all.\n"
    "  get_assigned_site_at(x, y) — the polling place currently "
    "serving the precinct that contains coordinate (x, y).\n\n"
    "Use BOTH channels as appropriate to the question:\n"
    "  - The MAP is best for spatial PATTERNS — clusters, "
    "irregular catchment shapes, large under-served regions, "
    "obviously-disjoint same-color patches. Visual scans surface "
    "anomalies that would be expensive to enumerate via tools.\n"
    "  - The structured TOOLS are best for PRECISE values — exact "
    "distances, specific assigned-site indices, connectivity "
    "verification, candidate enumeration. Tools are exact; the map "
    "is approximate.\n\n"
    "Common workflow: scan the map visually to localise the issue, "
    "then call structured tools to confirm specific entities and "
    "values. The map is roughly 0–10 km on each axis, with about 80 "
    "precincts and 18 opened polling places. When you have enough "
    "information, respond in the format the user requests."
)

_TOOL_BLOCK_2 = (
    "You also have access to the following structured tools:\n"
    "  list_sites(opened_only) — candidate or opened polling places "
    "(index, location in km, capacity, type, opened/closed status, "
    "current voter load if opened).\n"
    "  get_precinct_centroids(precinct_indices) — centroid "
    "coordinates in km of the requested precincts (omit "
    "`precinct_indices` for all precincts).\n"
    "  get_precinct_adjacency() — for each precinct, the indices of "
    "its spatially-adjacent precincts (4-connected on the rasterised "
    "Voronoi map).\n"
    "  get_current_assignments(precinct_indices) — for each "
    "precinct, the polling place it is currently assigned to and the "
    "assigned travel distance. Omit `precinct_indices` for all.\n"
    "  get_assigned_site_at(x, y) — the polling place currently "
    "serving the precinct that contains coordinate (x, y).\n\n"
    "Common workflow: scan the map visually to localise the issue, "
    "then call structured tools to confirm specific entities and "
    "values. The map is roughly 0–10 km on each axis, with about 80 "
    "precincts and 18 opened polling places. When you have enough "
    "information, respond in the format the user requests."
)

_TOOL_BLOCK_3 = (
    "You also have access to the following structured tools:\n"
    "  list_sites(opened_only) — candidate or opened polling places "
    "(index, location in km, capacity, type, opened/closed status, "
    "current voter load if opened).\n"
    "  get_precinct_centroids(precinct_indices) — centroid "
    "coordinates in km of the requested precincts (omit "
    "`precinct_indices` for all precincts).\n"
    "  get_precinct_adjacency() — for each precinct, the indices of "
    "its spatially-adjacent precincts (4-connected on the rasterised "
    "Voronoi map).\n"
    "  get_current_assignments(precinct_indices) — for each "
    "precinct, the polling place it is currently assigned to and the "
    "assigned travel distance. Omit `precinct_indices` for all.\n"
    "  get_assigned_site_at(x, y) — the polling place currently "
    "serving the precinct that contains coordinate (x, y).\n\n"
    "Common workflow: scan the map to identify non-contiguous service areas, then"
    " query a coordinate inside the suspected region with the get_assigned_site_at tool to attribute it to a specific polling place. You "
    "may call the tool as many times as you need. Use the data available from other tools such as the get_precinct_adjacency tool to verify contiguity."
    " The map is roughly 0–10 km on each axis, with about 80 "
    "precincts and 18 opened polling places. When you have enough "
    "information, respond in the format the user requests."
)


SYSTEM_TEXT = (
    "You are a careful analyst examining a map of a polling-place "
    "plan, with full structured-tool access alongside the map.\n\n"
    "On the map: each precinct is filled with the color of its "
    "assigned opened polling place. Opened polling places are drawn "
    "as red circles with white index labels. Closed candidate "
    "polling places (sites that COULD be opened but currently are "
    "not) are drawn as smaller gray circles with their own index "
    "labels.\n\n"
    + _TOOL_BLOCK
)


SYSTEM_TEXT_NO_MARKERS = (
    "You are a careful analyst examining a map of a polling-place "
    "plan, with full structured-tool access alongside the map.\n\n"
    "On the map: each precinct is filled with the color of its "
    "assigned opened polling place, designating a catchment area. "
    "Polling-place locations themselves are NOT drawn on the map — "
    "the only visible signal is the colored fill.\n\n"
    + _TOOL_BLOCK_3
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build(*, task_name: str, question: str,
          image_b64: str,
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build the multimodal_full message list with both image and tools."""
    info = view_info or {}
    has_markers = bool(info.get("has_site_markers", True))
    sys_text = SYSTEM_TEXT if has_markers else SYSTEM_TEXT_NO_MARKERS
    return [
        {"role": "system", "content": sys_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ],
        },
    ]


# Re-export so the runner's `getattr(prompt, "get_tools", None)` and
# `getattr(prompt, "dispatch_tool", None)` lookups succeed identically
# to tools_only.
__all__ = ["NAME", "USES_IMAGE", "SYSTEM_TEXT", "SYSTEM_TEXT_NO_MARKERS",
            "TOOLS", "build", "get_tools", "dispatch_tool"]
