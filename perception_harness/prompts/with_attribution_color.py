"""Tool-using prompt with a color-disambiguated attribution tool.

Sibling of `prompts/with_attribution.py`. The tool signature gains an
optional `color` parameter:

    get_assigned_site_at(x, y, color?)

When the LLM passes a color in addition to a coordinate, the
dispatcher resolves to the polling place whose catchment uses that
color, regardless of whether the coordinate landed on the correct
side of a catchment boundary. This addresses a failure mode observed
in earlier runs: the LLM intended to inspect (e.g.) the magenta
fragment, picked an approximate coordinate, but the coordinate
landed in a neighbouring cyan catchment and the tool returned the
wrong site.

Resolution semantics:
  - color present, valid, and exactly one opened site uses that color
    -> return that site directly.
  - color present, valid, and multiple opened sites share that color
    (palette wrap-around with >18 opened sites; rare with K=18) ->
    return the color-matching site nearest (x, y).
  - color present and the precinct at (x, y) is assigned to a
    DIFFERENT color than the one provided -> still return the
    color-resolved site, but include a warning telling the LLM both
    what its coordinate actually pointed at AND the color-matched
    site, so the LLM can update its understanding.
  - color absent -> behave exactly like the original
    get_assigned_site_at(x, y) for backward compatibility.

Color vocabulary is the canonical 18-name palette used by
renderers/v2.py and renderers/v2_no_markers.py. The tool schema's
`enum` constraint lists the names so well-behaved models pick from
the controlled set; the dispatcher additionally normalises whitespace,
underscore/hyphen, and case for robustness.

Tools and dispatcher are NOT shared with `with_attribution.py` —
that prompt deliberately exposes a simpler 2-arg tool surface for
A/B comparison. Pair this prompt with `models/openai_vlm_tools.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Make instance_generator importable for Instance/Solution introspection.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))
sys.path.insert(0, str(HARNESS_ROOT))

# Import the canonical palette from the renderer so any future palette
# tweak in renderers/v2.py automatically flows through here.
from renderers.v2 import PALETTE_18  # noqa: E402

# Parallel list of human-readable color names for the controlled
# vocabulary the LLM uses. Index k maps to PALETTE_18[k]. If you
# change the palette, update this list to match.
PALETTE_18_NAMES: List[str] = [
    "red", "green", "yellow", "blue", "orange", "purple",
    "cyan", "magenta", "lime", "teal", "brown", "maroon",
    "olive", "navy", "pink", "mint", "gray", "dark_slate",
]
assert len(PALETTE_18) == len(PALETTE_18_NAMES), (
    "PALETTE_18_NAMES must have the same length as PALETTE_18 — they "
    "are parallel arrays. If you change one, change the other."
)


NAME = "with_attribution_color"

# Prompt capability — read by eval_perception.py to build view_info.
USES_IMAGE = True


# ---------------------------------------------------------------------------
# Color-name resolution helpers
# ---------------------------------------------------------------------------
def _normalise_color(name: Optional[str]) -> str:
    """Lowercase + collapse whitespace/hyphens to underscores. Maps
    'Dark Slate', 'dark-slate', and 'DARK_SLATE' to the canonical form."""
    if not name:
        return ""
    s = name.strip().lower()
    for ch in (" ", "-"):
        s = s.replace(ch, "_")
    # Collapse repeated underscores so "dark__slate" -> "dark_slate".
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _site_color_index(opened_idx: np.ndarray, site_index: int) -> int:
    """Return the palette slot for `site_index`, or -1 if not opened.

    Mirrors how renderers/v2.py assigns colors: the k-th opened site
    (in opened_idx-order) gets slot `k % len(PALETTE_18)`."""
    where = np.where(opened_idx == site_index)[0]
    if len(where) == 0:
        return -1
    return int(where[0]) % len(PALETTE_18_NAMES)


def _color_to_site_indices(color_name: Optional[str],
                            opened_idx: np.ndarray) -> Optional[List[int]]:
    """Resolve a color name to the list of opened sites using that color.

    Returns None for an unknown color name (the dispatcher then emits
    an error response listing valid colors). Empty list ([]) means the
    name was valid but no opened site uses that palette slot, which
    can happen if there are fewer opened sites than palette colors.
    """
    norm = _normalise_color(color_name)
    if norm not in PALETTE_18_NAMES:
        return None
    slot = PALETTE_18_NAMES.index(norm)
    n_slots = len(PALETTE_18_NAMES)
    return [int(j) for k, j in enumerate(opened_idx) if (k % n_slots) == slot]


# ---------------------------------------------------------------------------
# System texts
# ---------------------------------------------------------------------------
_VALID_COLORS_LINE = "Valid color names: " + ", ".join(PALETTE_18_NAMES) + "."

_TOOL_BLOCK = (
    "You have one tool available:\n"
    "  get_assigned_site_at(x, y, color) — given a coordinate in km on "
    "the map AND the visual color of the region you intend to inspect, "
    "returns the polling place whose catchment uses that color. The "
    "color is the primary disambiguator; the coordinate is used only "
    "to break ties when multiple polling places share a color (rare). "
    "If `color` is omitted, the tool falls back to a pure coordinate "
    f"lookup — but PROVIDING THE COLOR IS STRONGLY RECOMMENDED.\n\n"
    f"{_VALID_COLORS_LINE}\n\n"
    "WHY COLOR HELPS. Coordinates read from the map are approximate. "
    "A point near a catchment boundary can land on the wrong side of "
    "the boundary, in which case a coordinate-only lookup would return "
    "the WRONG polling place. Providing the color you visually observe "
    "in the region you're inspecting lets the tool resolve to the "
    "correct site even when your coordinate is slightly off.\n\n"
    "If your color choice and coordinate disagree (the precinct at the "
    "coordinate has a different color than what you said), the response "
    "will include a `warning` telling you what color the coordinate "
    "actually pointed at AND the site corresponding to the color you "
    "named. Use this signal to update your understanding."
)

_STRATEGY_BLOCK = (
    "Strategy:\n"
    "  1. Scan the map for candidate split-catchment patterns: a "
    "colored region that appears in two or more separated places.\n"
    "  2. For EACH candidate fragment, call get_assigned_site_at with "
    "a coordinate inside the fragment AND the color you observe there. "
    "Two visually similar fragments belong to the same catchment iff "
    "the tool returns the same `assigned_site_index` for both.\n"
    "  3. Only flag a site as having a split catchment after confirming "
    "via tool calls that two or more fragments belong to the same "
    "site.\n"
    "  4. Use the tool as many times as needed. Prefer fewer "
    "high-confidence sites over many uncertain ones.\n\n"
    "When you have enough information, respond in the format the user "
    "requests."
)


SYSTEM_TEXT = (
    "You are a careful visual analyst examining a map of a polling-"
    "place plan. Each precinct on the map is filled with the color of "
    "the opened polling place it is assigned to. The opened polling "
    "places are drawn as red circles with white index labels.\n\n"
    + _TOOL_BLOCK + "\n\n"
    + _STRATEGY_BLOCK
)


SYSTEM_TEXT_NO_MARKERS = (
    "You are a careful visual analyst examining a map of a polling-"
    "place plan. Each precinct is filled with the color of the opened "
    "polling place it is assigned to, designating a catchment area. "
    "Polling-place locations themselves are NOT drawn — the only "
    "visible signal is the colored fill.\n\n"
    + _TOOL_BLOCK + "\n\n"
    + _STRATEGY_BLOCK
)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_assigned_site_at",
            "description": (
                "Given a coordinate (x, y) in km AND optionally the "
                "color name you observe at that location, return the "
                "polling place whose catchment uses that color. The "
                "color is the primary disambiguator; the coordinate "
                "breaks ties when multiple polling places share a "
                "color (rare). If color is omitted, falls back to a "
                "pure coordinate lookup. The map spans approximately "
                "0 ≤ x ≤ 10 and 0 ≤ y ≤ 10 km. " + _VALID_COLORS_LINE
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number",
                          "description": "X coordinate in km."},
                    "y": {"type": "number",
                          "description": "Y coordinate in km."},
                    "color": {
                        "type": "string",
                        "enum": list(PALETTE_18_NAMES),
                        "description": (
                            "The color name you observe at (x, y). "
                            "Strongly recommended — the tool will "
                            "use color to disambiguate when the "
                            "coordinate is approximate."
                        ),
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
]


def get_tools() -> List[Dict[str, Any]]:
    return TOOLS


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build(*, task_name: str, question: str,
          image_b64: str,
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build the color-aware tool-using multimodal message list."""
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
                    "detail": "original"
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
def dispatch_tool(name: str, args_str: str, *,
                   instance, solution, **_kwargs) -> str:
    """Execute one tool call and return JSON text."""
    if name != "get_assigned_site_at":
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"could not parse args JSON: {e}"})

    try:
        x = float(args["x"])
        y = float(args["y"])
    except (KeyError, TypeError, ValueError) as e:
        return json.dumps({"error":
                            f"x and y are required numbers ({e})."})

    color = args.get("color")
    xmin, ymin, xmax, ymax = instance.bounds
    in_bounds = (xmin <= x <= xmax) and (ymin <= y <= ymax)

    # Resolve the precinct/site at the coordinate (when in bounds), so
    # we can either return it directly (color absent) or cross-check
    # against color (color present).
    coord_precinct: Optional[int] = None
    coord_site: Optional[int] = None
    if in_bounds:
        G = len(instance.grid_xs)
        ix = int(np.clip(round((x - xmin) / (xmax - xmin) * (G - 1)),
                          0, G - 1))
        iy = int(np.clip(round((y - ymin) / (ymax - ymin) * (G - 1)),
                          0, G - 1))
        coord_precinct = int(instance.precinct_label_grid[iy, ix])
        coord_site = int(np.argmax(solution.y[coord_precinct]))

    opened_idx = np.where(solution.x == 1)[0]

    # ---- color absent: backward-compatible coord-only behaviour ----
    if not color:
        if not in_bounds:
            return json.dumps({
                "error": (f"({x}, {y}) is outside the map bounds "
                          f"[{xmin}, {xmax}] × [{ymin}, {ymax}]."),
            })
        return json.dumps({
            "x": x, "y": y,
            "precinct_index": coord_precinct,
            "assigned_site_index": coord_site,
            "assigned_site_x":
                float(instance.site_locations[coord_site, 0]),
            "assigned_site_y":
                float(instance.site_locations[coord_site, 1]),
        })

    # ---- color present: color is the primary disambiguator ----
    matches = _color_to_site_indices(color, opened_idx)
    if matches is None:
        return json.dumps({
            "error": f"unknown color name '{color}'.",
            "valid_colors": list(PALETTE_18_NAMES),
        })
    if not matches:
        return json.dumps({
            "error": (f"no opened polling place uses the color "
                      f"'{color}' in this plan."),
            "valid_colors_in_use": [
                PALETTE_18_NAMES[k % len(PALETTE_18_NAMES)]
                for k in range(len(opened_idx))
            ],
        })

    # Multiple sites share this color (palette wrap with >18 sites) —
    # pick the one nearest the requested coordinate. With K=18 in the
    # standard setup this list almost always has exactly one entry.
    if len(matches) == 1:
        chosen = int(matches[0])
    else:
        site_xy = instance.site_locations[matches]
        target = np.array([x, y], dtype=float)
        dists = np.linalg.norm(site_xy - target, axis=1)
        chosen = int(matches[int(np.argmin(dists))])

    response: Dict[str, Any] = {
        "x": x, "y": y,
        "color_provided": _normalise_color(color),
        "assigned_site_index": chosen,
        "assigned_site_x": float(instance.site_locations[chosen, 0]),
        "assigned_site_y": float(instance.site_locations[chosen, 1]),
    }

    # Cross-check the coordinate's actual catchment against the
    # color-resolved site. When they disagree, surface a warning so the
    # LLM knows its coordinate landed in a neighbouring catchment.
    if in_bounds:
        coord_color_slot = _site_color_index(opened_idx, coord_site)
        coord_color_name = (PALETTE_18_NAMES[coord_color_slot]
                             if coord_color_slot >= 0 else None)
        response["precinct_at_coord_index"] = coord_precinct
        response["precinct_at_coord_assigned_site_index"] = coord_site
        response["precinct_at_coord_color"] = coord_color_name
        if coord_site != chosen:
            response["warning"] = (
                f"The precinct at ({x}, {y}) is assigned to site "
                f"{coord_site} ({coord_color_name}), not "
                f"{response['color_provided']}. The query coordinate "
                f"may have landed in a neighbouring catchment. The "
                f"site returned (assigned_site_index={chosen}) is the "
                f"polling place whose catchment uses color "
                f"'{response['color_provided']}'. If you intended to "
                f"inspect a different region with that color, try "
                f"other coordinates inside it."
            )

    return json.dumps(response)
