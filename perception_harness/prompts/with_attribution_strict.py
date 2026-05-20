"""Tool-using prompt with precision-focused system text.

Sibling of `prompts/with_attribution.py`, kept separate so the original
remains the baseline. Same tool surface (`get_assigned_site_at`); only
the system text changes.

Motivation. The first round of contiguity-identify runs against the
original `with_attribution` prompt produced a strong-recall / weak-
precision pattern: micro precision ≈ 0.42, recall ≈ 0.47, with three
false positives for every true positive. The model was flagging more
sites than were actually split, suggesting it was acting on visual
proximity rather than confirmed catchment splits. This variant
explicitly requires tool-based verification before flagging and tells
the model to prefer fewer high-confidence answers over many uncertain
ones.

Reuse. Tool schemas + dispatcher are imported from with_attribution so
both prompts call exactly the same tool surface; only the system
text differs. Pair with `models/openai_vlm_tools.py` (any adapter
advertising `SUPPORTS_TOOLS=True`).

Compare to the original by running both side-by-side:
    --prompts with_attribution with_attribution_strict
"""
from __future__ import annotations

from typing import Any, Dict, List

# Reuse the tool schema and dispatcher from the baseline so any
# improvements to the tool surface flow to both variants automatically.
from prompts.with_attribution import (  # noqa: E402
    TOOLS,
    get_tools,
    dispatch_tool,
)


NAME = "with_attribution_strict"

# Prompt capability — read by eval_perception.py to build view_info.
USES_IMAGE = True


# ---------------------------------------------------------------------------
# Precision-focused system texts
# ---------------------------------------------------------------------------
# Both variants share the same VERIFICATION REQUIREMENT block; only the
# scene description differs depending on whether site markers are visible.

_VERIFICATION_BLOCK = (
    "VERIFICATION REQUIREMENT — IMPORTANT.\n"
    "False positives are heavily penalised. Do NOT flag a site as "
    "having a split catchment based on visual proximity or color "
    "similarity alone. Colors can be hard to distinguish, especially "
    "for pale or adjacent hues, and visually-similar regions in "
    "different parts of the map may belong to entirely different "
    "polling places.\n\n"
    "Strategy:\n"
    "  1. Scan the map for candidate split-catchment patterns: a "
    "colored region that appears to repeat in two or more separated "
    "places.\n"
    "  2. For EACH candidate split, query "
    "get_assigned_site_at(x, y) at a coordinate inside each suspected "
    "fragment. Two visually similar regions are only the same "
    "catchment if the tool returns the SAME assigned_site_index for "
    "coordinates inside both.\n"
    "  3. Only after you have confirmed via tool calls that two or "
    "more fragments belong to the SAME site should you flag that "
    "site as having a split catchment.\n"
    "  4. If you cannot confirm via tool calls that two fragments "
    "belong to the same site, do NOT flag that site. Prefer "
    "reporting fewer high-confidence sites over many uncertain ones. "
    "An empty list is the correct answer when no split has been "
    "verified.\n\n"
    "Use the tool as many times as needed to build confidence. When "
    "you have verified the splits, respond in the format the user "
    "requests."
)

SYSTEM_TEXT = (
    "You are a careful visual analyst examining a map of a polling-"
    "place plan. Each precinct on the map is filled with the color "
    "of the opened polling place it is assigned to. The opened "
    "polling places are drawn as red circles with white index "
    "labels.\n\n"
    "You have one tool available:\n"
    "  get_assigned_site_at(x, y) — given a coordinate in km on the "
    "map, returns the precinct at that location and the index of "
    "the polling place to which the precinct is currently assigned. "
    "The map is roughly 0–10 km on each axis.\n\n"
    + _VERIFICATION_BLOCK
)

SYSTEM_TEXT_NO_MARKERS = (
    "You are a careful visual analyst examining a map of a polling-"
    "place plan. Each precinct on the map is filled with the color "
    "of the opened polling place it is assigned to, designating a "
    "catchment area. Polling-place locations themselves are NOT "
    "drawn — the only visible signal is the colored fill.\n\n"
    "You have one tool available:\n"
    "  get_assigned_site_at(x, y) — given a coordinate in km on the "
    "map, returns the precinct at that location and the index of "
    "the polling place to which the precinct is currently assigned. "
    "The map is roughly 0–10 km on each axis.\n\n"
    + _VERIFICATION_BLOCK
)


def build(*, task_name: str, question: str,
          image_b64: str,
          view_info: Dict[str, Any] | None = None,
          **_kwargs) -> List[Dict[str, Any]]:
    """Build the strict tool-using multimodal message list.

    Picks SYSTEM_TEXT or SYSTEM_TEXT_NO_MARKERS based on
    view_info["has_site_markers"], same convention as the baseline
    with_attribution prompt.
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
                    },
                    "detail": "original"
                },
            ],
        },
    ]


# Re-export for parity with the baseline so the runner's
# `getattr(prompt, "get_tools", None)` and `getattr(prompt,
# "dispatch_tool", None)` lookups succeed identically.
__all__ = ["NAME", "USES_IMAGE", "SYSTEM_TEXT", "SYSTEM_TEXT_NO_MARKERS",
            "TOOLS", "build", "get_tools", "dispatch_tool"]
