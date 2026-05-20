"""Contiguity perception task.

Two sub-tasks:

  identify : "List the indices of all opened polling places whose service
             area is split into two or more disconnected pieces."
             Ground truth = the culprit-site indices from the generator
             metadata. Score = F1 between predicted and true sets.

  describe : "Describe what you see going on with the catchments."
             Ground truth = a list of expected concept keywords.
             Score = fraction of expected concepts mentioned (case-insensitive
             substring match).

Each task module exports three functions with the same signatures across
archetypes:

    format_question(meta, task) -> str
    parse_response(raw_text, task) -> dict
    score(parsed, ground_truth, task) -> float
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------
# The question text varies along two axes that are pulled from view_info:
#   has_visual         : whether the model actually sees a rendered map.
#   has_site_markers   : whether opened sites are drawn on the map as
#                        red circles + white index labels.
#
# Three regimes that matter:
#   visual + markers   : "the map shows... the red dots..."
#   visual + no markers: "the map shows... colored regions..." (no markers)
#   no visual          : "in this polling-place plan..." + structural
#                        definition of non-contiguous (uses adjacency,
#                        not visual disjointness).

_IDENTIFY_TASK_VISUAL = (
    "A polling place's *service area* is the set of precincts assigned "
    "to it — i.e. all precincts sharing its color on the map.\n\n"
    "A service area is **non-contiguous** when its precincts form two "
    "or more visually disjoint patches on the map (the same color "
    "appears in separated pieces, with other colors between them).\n\n"
    "For every opened polling place whose service area is non-contiguous, "
    "report it together with a brief reason explaining the evidence.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '  {"split_sites": [\n'
    '    {"index": <i>, "reason": "<short justification>"},\n'
    '    ...\n'
    "  ]}\n"
    "Each `reason` should briefly state which colored regions you saw, "
    "roughly where on the map, and (if you used the tool) which "
    "coordinates / tool calls confirmed the split. Keep each reason "
    "under 40 words. Do not include any other text outside the JSON."
)

_IDENTIFY_TASK_NO_VISUAL = (
    "A polling place's *service area* is the set of precincts currently "
    "assigned to it.\n\n"
    "A service area is **non-contiguous** when the precincts assigned "
    "to that polling place split into two or more connected components "
    "on the precinct adjacency graph (the get_precinct_adjacency tool "
    "returns this graph).\n\n"
    "For every opened polling place whose service area is non-contiguous, "
    "report it together with a brief reason explaining the evidence.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '  {"split_sites": [\n'
    '    {"index": <i>, "reason": "<short justification>"},\n'
    '    ...\n'
    "  ]}\n"
    "Each `reason` should briefly state which connected components you "
    "found and which precincts populate them. Keep each reason under "
    "40 words. Do not include any other text outside the JSON."
)

_CONTEXT_VISUAL_MARKERS = (
    "The map shows a polling-place plan for a county. Each precinct is "
    "filled with the color of its assigned opened polling place. The "
    "opened polling places are drawn as red circles with white index "
    "labels."
)

_CONTEXT_VISUAL_NO_MARKERS = (
    "The map shows a polling-place plan for a county. Each precinct is "
    "filled with the color of its assigned opened polling place. "
    "Polling-place locations themselves are NOT drawn on the map — the "
    "only visible signal is the colored service-area fill."
)

_CONTEXT_NO_VISUAL = (
    "You are examining a polling-place plan for a county with roughly "
    "80 precincts and 18 opened polling places. No map is provided — "
    "use the structured tools to inspect assignments and adjacency."
)

_DESCRIBE_TASK_VISUAL = (
    "Look at the colored service areas and describe in 2–3 sentences "
    "what you notice about their topology — specifically, whether each "
    "service area looks like one connected region, or whether you see "
    "any visible anomalies."
)

_DESCRIBE_TASK_NO_VISUAL = (
    "Using the structured tools (especially get_precinct_adjacency and "
    "get_current_assignments), describe in 2–3 sentences what you "
    "notice about the topology of the service areas — specifically, "
    "whether each polling place's assigned precincts form one connected "
    "component on the adjacency graph, or whether any are split."
)


def _context(view_info: Dict[str, Any] | None) -> str:
    info = view_info or {}
    if not info.get("has_visual", True):
        return _CONTEXT_NO_VISUAL
    if info.get("has_site_markers", True):
        return _CONTEXT_VISUAL_MARKERS
    return _CONTEXT_VISUAL_NO_MARKERS


def format_question(meta: Dict[str, Any], task: str, *,
                     view_info: Dict[str, Any] | None = None,
                     **_kwargs) -> str:
    info = view_info or {}
    has_visual = info.get("has_visual", True)
    if task == "identify":
        body = _IDENTIFY_TASK_VISUAL if has_visual else _IDENTIFY_TASK_NO_VISUAL
        return _context(view_info) + "\n\n" + body
    if task == "describe":
        body = _DESCRIBE_TASK_VISUAL if has_visual else _DESCRIBE_TASK_NO_VISUAL
        return _context(view_info) + "\n\n" + body
    raise ValueError(f"Unknown contiguity task: {task!r}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
# Models routinely wrap JSON in markdown code fences or include preamble.
# We scan for balanced top-level {...} substrings (string-aware so braces
# inside JSON string values don't fool the balance counter), then return
# the LAST parseable one — models that "think out loud" put the answer
# at the end. A simple `\{[^{}]*\}` regex would fail on the new
# justification-aware response format because the inner {"index": ...,
# "reason": ...} entries introduce nested braces.

def _find_outer_json_objects(text: str) -> List[str]:
    """Return every balanced top-level `{...}` substring of `text`."""
    objects: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start:i + 1])
                start = -1
    return objects


def _extract_last_json_object(raw: str) -> Dict[str, Any]:
    for candidate in reversed(_find_outer_json_objects(raw or "")):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


def _parse_site_entries(items: Any) -> tuple:
    """Parse a list of site entries that may be bare integers, dicts of
    the form {"index": i, "reason": "..."}, or a mix.

    Returns (sites_in_input_order, reasons_dict_keyed_by_index). Works
    for any identify task that follows the {"sites_field": [...]} shape.
    Backward-compatible with the older bare-integer list format.
    """
    sites: list = []
    reasons: Dict[int, str] = {}
    if not isinstance(items, list):
        return sites, reasons
    for v in items:
        idx = None
        reason = None
        if isinstance(v, dict):
            try:
                idx = int(v.get("index"))
            except (TypeError, ValueError):
                continue
            r = v.get("reason")
            if r is not None:
                reason = str(r)
        else:
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
        if idx is None:
            continue
        sites.append(idx)
        if reason is not None:
            # First reason wins on duplicates.
            reasons.setdefault(idx, reason)
    return sites, reasons


def parse_response(raw: str, task: str) -> Dict[str, Any]:
    if task == "identify":
        obj = _extract_last_json_object(raw or "")
        sites, reasons = _parse_site_entries(obj.get("split_sites", []))
        return {
            "split_sites": sorted(set(sites)),
            "reasons": reasons,
        }
    if task == "describe":
        return {"text": raw or ""}
    raise ValueError(f"Unknown contiguity task: {task!r}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _f1(pred: set, truth: set) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    return 2 * precision * recall / (precision + recall)


def score(parsed: Dict[str, Any], ground_truth: Dict[str, Any],
          task: str) -> float:
    if task == "identify":
        pred = set(int(j) for j in parsed.get("split_sites", []))
        truth = set(int(j) for j in ground_truth.get("split_sites", []))
        return _f1(pred, truth)
    if task == "describe":
        text = (parsed.get("text") or "").lower()
        concepts = ground_truth.get("concepts", [])
        if not concepts:
            return 1.0
        hits = sum(1 for c in concepts if c.lower() in text)
        return hits / len(concepts)
    raise ValueError(f"Unknown contiguity task: {task!r}")


# ---------------------------------------------------------------------------
# Secondary scores
# ---------------------------------------------------------------------------
# Identify is a variable-size set-prediction task (unlike shape_niceness'
# fixed top-K), so precision and recall carry distinct information that
# the F1 summary collapses. The runner serialises this dict into the
# per_question.csv `secondary_scores` column. After the failure-mode
# triage on 2026-05-06_170333 showed precision (~0.25 micro) was the
# real bottleneck, surfacing precision and recall natively makes future
# pivots a one-liner instead of a post-hoc re-derivation.
#
# Conventions match the existing _f1: when one side is empty but not
# the other, both precision and recall are 0.0. When both are empty,
# both are 1.0 (correctly reporting "no splits"). Counts are reported
# alongside the rates so pivots can recover the full confusion-matrix
# picture without re-parsing parsed_answer.
def secondary_scores(parsed: Dict[str, Any],
                      ground_truth: Dict[str, Any],
                      task: str) -> Dict[str, float]:
    if task != "identify":
        return {}

    pred = set(int(j) for j in parsed.get("split_sites", []))
    truth = set(int(j) for j in ground_truth.get("split_sites", []))

    out: Dict[str, float] = {
        "pred_size": float(len(pred)),
        "truth_size": float(len(truth)),
        "fp_count": float(len(pred - truth)),
        "fn_count": float(len(truth - pred)),
        "tp_count": float(len(pred & truth)),
    }
    if not pred and not truth:
        out["precision"] = 1.0
        out["recall"] = 1.0
    elif not pred or not truth:
        # Convention matches _f1: an empty side yields 0.0 here so that
        # the F1 derivable from these P/R values agrees with the score
        # column.
        out["precision"] = 0.0
        out["recall"] = 0.0
    else:
        tp = len(pred & truth)
        out["precision"] = tp / len(pred)
        out["recall"] = tp / len(truth)
    return out
