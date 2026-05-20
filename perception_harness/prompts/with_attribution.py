"""Tool-using prompt for the contiguity / attribution problem.

Exposes a single tool to the VLM:

    get_assigned_site_at(x, y)
        Returns the precinct at coordinate (x, y) and the index of the
        opened polling place to which that precinct is currently
        assigned, plus that site's coordinates so the VLM can verify
        the lookup against what it sees on the map.

The tool exists because dropping the assignment lines (renderers/v2.py)
removed the line-based attribution channel: the categorical fill tells
the VLM that two regions exist with the same color, but the disjoint
fragment of a split catchment doesn't contain its site's red marker
by definition. A coordinate-level lookup tool fills the gap without
giving away the topology answer (the VLM still has to decide which
coordinates to query).

Contract this prompt module exposes (the runner picks these up):

    build(task_name, question, image_b64) -> list[dict]   (always)
    get_tools() -> list[dict]                              (tool-using)
    dispatch_tool(name, args_str, *, instance, solution) -> str

When the runner sees `get_tools` and `dispatch_tool` on a prompt module
AND the model adapter advertises `SUPPORTS_TOOLS=True`, it pairs them
up via the model's tool-call loop. Otherwise the prompt degrades
gracefully to a single-shot call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Make instance_generator importable so we can introspect Instance/Solution.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "with_attribution"

# Prompt capability — read by eval_perception.py to build view_info.
USES_IMAGE = True


SYSTEM_TEXT = (
    "You are a careful visual analyst examining a map of a polling-place "
    "plan. Each precinct on the map is filled with the color of the "
    "opened polling place it is assigned to. The opened polling places "
    "are drawn as red circles with white index labels.\n\n"
    "You have one tool available:\n"
    "  get_assigned_site_at(x, y) — given a coordinate in km on the "
    "map, returns the precinct at that location and the index of the "
    "polling place to which the precinct is currently assigned. Use "
    "this when you can see a region on the map but cannot tell from "
    "color alone which polling place it belongs to. Common case: a "
    "region that does NOT contain a red marker — its assigned site is "
    "elsewhere on the map. The map is roughly 0–10 km on each axis.\n\n"
    "Strategy: scan the map visually for the pattern the user asks "
    "about; for any region whose assigned site is ambiguous, query a "
    "coordinate inside the region with the tool to attribute it. You "
    "may call the tool as many times as you need before producing your "
    "final answer. When you have enough information, respond in the "
    "format requested by the user."
)

SYSTEM_TEXT_NO_MARKERS = (
    "You are a careful visual analyst examining a map of a polling-place "
    "plan. Each precinct on the map is filled with the color of the "
    "opened polling place it is assigned to, designating a catchment area.\n\n"
    "You have one tool available:\n"
    "  get_assigned_site_at(x, y) — given a coordinate in km on the "
    "map, returns the precinct at that location and the index of the "
    "polling place to which the precinct is currently assigned. Use "
    "this when you can see a region on the map but cannot tell from "
    "color alone which polling place it belongs to. The map is roughly 0–10 km on each axis.\n\n"
    "Strategy: scan the map visually for the pattern the user asks "
    "about; , query a coordinate inside the region with the tool to attribute it. You "
    "may call the tool as many times as you need before producing your "
    "final answer. When you have enough information, respond in the "
    "format requested by the user."
)


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_assigned_site_at",
            "description": (
                "Given a coordinate (x, y) in km on the map, return the "
                "precinct at that location and the index of the opened "
                "polling place to which that precinct is currently "
                "assigned. Use this to attribute a colored region to a "
                "specific polling place when the region does not contain "
                "the red marker of its assigned site, or when colors are "
                "visually ambiguous. The map spans approximately "
                "0 ≤ x ≤ 10 and 0 ≤ y ≤ 10 km."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number",
                          "description": "X coordinate in km."},
                    "y": {"type": "number",
                          "description": "Y coordinate in km."},
                },
                "required": ["x", "y"],
            },
        },
    },
]


def build(*, task_name: str, question: str,
          image_b64: str,
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build the tool-using multimodal message list.

    The system text varies on view_info["has_site_markers"]:
        True  -> SYSTEM_TEXT (mentions red dots + white index labels)
        False -> SYSTEM_TEXT_NO_MARKERS (no marker references)
    Defaults to the marker-aware text when view_info is missing, since
    the canonical baseline render (renderers/v2.py, renderers/base.py)
    has markers visible.
    """
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
                        "detail": "original"
                    },
                },
            ],
        },
    ]


def get_tools() -> List[Dict[str, Any]]:
    """Return the OpenAI-format tool schemas this prompt exposes."""
    return TOOLS


def dispatch_tool(name: str, args_str: str, *,
                   instance, solution) -> str:
    """Execute one tool call and return the textual result.

    The adapter calls this for every tool_call the model emits; the
    return value is appended to the conversation as a `tool` message.
    """
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
        return json.dumps({"error": f"x and y are required numbers ({e})."})

    xmin, ymin, xmax, ymax = instance.bounds
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return json.dumps({
            "error": (f"({x}, {y}) is outside the map bounds "
                      f"[{xmin}, {xmax}] × [{ymin}, {ymax}]."),
        })

    # Locate the precinct at (x, y) via the rasterised label grid —
    # same lookup as agent_tools.get_precinct_at in instance_generator.
    G = len(instance.grid_xs)
    ix = int(np.clip(round((x - xmin) / (xmax - xmin) * (G - 1)), 0, G - 1))
    iy = int(np.clip(round((y - ymin) / (ymax - ymin) * (G - 1)), 0, G - 1))
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
