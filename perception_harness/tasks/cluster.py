"""Cluster perception task.

The 'cluster' archetype targets too-tight concentrations of opened
polling places — multiple sites packed into a small region. The task
is fundamentally about *marker positions*: identifying which red dots
on the map sit in close proximity to each other.

Two sub-tasks (mirrors contiguity / shape_niceness):

  identify : "List the polling places that participate in any tight
             cluster." Ground truth = the global set of opened sites
             that appear in any radius-neighbourhood of >= cluster_min_sites
             sites within cluster_radius (matches optimization scoring).
             Score = F1 between predicted and true sets.

  describe : "Describe the spatial distribution of opened polling
             places." Ground truth = a list of expected concept
             keywords (cluster, clustered, dense, packed, concentrated).
             Score = fraction mentioned (case-insensitive substring).

Validity gate: this task REQUIRES that, if a map is shown, it shows the
opened-site markers. Without markers there's no perceptual handle on
*which* polling places form a cluster — the question becomes
ill-defined. The runner consults `is_valid_view(view_info)` and skips
invalid (renderer, prompt) cells with an `invalid_view_for_task` row.

Tools-only mode (no map at all) IS valid for cluster: list_sites
returns coordinates and the LLM can compute the dense-radius-
neighbourhood set analytically. That's a different kind of test than
visual cluster perception, but a meaningful baseline.

Contract:
    format_question(meta, task, *, view_info=None) -> str
    parse_response(raw_text, task) -> dict
    score(parsed, ground_truth, task) -> float
    is_valid_view(view_info) -> bool
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Cluster definition (kept in sync with the optimization metric)
# ---------------------------------------------------------------------------
# A radius-neighbourhood is a "dense cluster" iff it contains at least
# CLUSTER_MIN_SITES opened sites within CLUSTER_RADIUS km of its centre,
# centre included. Both numbers match the README's defaults and queries.py
# (with the 1.35 km scoring cap accounted for upstream by the eval-set
# generator).
CLUSTER_RADIUS = 1.3      # km
CLUSTER_MIN_SITES = 4


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------
_CONTEXT_VISUAL_MARKERS = (
    "The map shows a polling-place plan for a county. The opened "
    "polling places are drawn as red circles with white index labels. "
    "Each precinct is filled with the color of its assigned polling "
    "place — but for this task you should focus on the positions of "
    "the red markers, not the colored fills."
)

_CONTEXT_NO_VISUAL = (
    "You are examining a polling-place plan for a county with roughly "
    "80 precincts and 18 opened polling places. No map is provided — "
    "use list_sites(opened_only=True) to inspect each opened polling "
    "place's coordinates in km."
)


_IDENTIFY_TASK_VISUAL = (
    "Most opened polling places are spread out across the map, but "
    "sometimes a small group of sites sits very close together — a "
    f"*cluster* of {CLUSTER_MIN_SITES} or more opened polling places "
    f"within roughly {CLUSTER_RADIUS:.1f} km of each other.\n\n"
    "Identify every opened polling place that participates in any such "
    "cluster. A site participates in a cluster if there are at least "
    f"{CLUSTER_MIN_SITES} opened polling places (including itself) "
    f"within {CLUSTER_RADIUS:.1f} km of it. Include every participating "
    "site, even if multiple distinct clusters exist. For each site you "
    "report, give a brief reason explaining the evidence (which other "
    "nearby sites form its cluster, where on the map).\n\n"
    "If no tight cluster exists, return an empty list.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    "  {\"clustered_sites\": [\n"
    "    {\"index\": <i>, \"reason\": \"<short justification>\"},\n"
    "    ...\n"
    "  ]}\n"
    "Each `reason` should briefly identify the cluster's other "
    "participants and rough location (e.g. 'cluster with sites 8, 12, "
    "14 around (5.0, 5.0)'). Keep each reason under 40 words. No "
    "other text outside the JSON."
)

_IDENTIFY_TASK_NO_VISUAL = (
    "A *cluster* is a group of opened polling places where each member "
    f"has at least {CLUSTER_MIN_SITES} opened polling places (including "
    f"itself) within {CLUSTER_RADIUS:.1f} km Euclidean distance.\n\n"
    "Use list_sites(opened_only=True) to retrieve the coordinates of "
    "each opened polling place, then compute pairwise distances and "
    "identify every site that participates in any such cluster. "
    "Include every participating site, even if multiple distinct "
    "clusters exist. For each site you report, give a brief reason "
    "naming the other sites in its cluster and rough location.\n\n"
    "If no cluster of the required density exists, return an empty list.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    "  {\"clustered_sites\": [\n"
    "    {\"index\": <i>, \"reason\": \"<short justification>\"},\n"
    "    ...\n"
    "  ]}\n"
    "Each `reason` should briefly identify the cluster's other "
    "participants and rough location (e.g. 'cluster with sites 8, 12, "
    "14 near (5.0, 5.0)'). Keep each reason under 40 words. No other "
    "text outside the JSON."
)


_DESCRIBE_TASK_VISUAL = (
    "Look at the positions of the red opened-polling-place markers on "
    "the map and describe in 2–3 sentences how they are distributed — "
    "are they spread evenly across the map, or do you see clusters / "
    "concentrations of multiple polling places packed close together? "
    "Mention any clusters that stand out as unusually tight."
)

_DESCRIBE_TASK_NO_VISUAL = (
    "Use list_sites(opened_only=True) to inspect the coordinates of "
    "each opened polling place, then describe in 2–3 sentences how the "
    "polling places are spatially distributed — are they spread evenly "
    "or do you see groups of multiple polling places packed close "
    "together? Mention any clusters that stand out as unusually tight."
)


def _context(view_info: Dict[str, Any] | None) -> str:
    """Cluster runs only with markers (visual+markers) or no map at all
    — the visual+no_markers regime is invalid (see is_valid_view), so
    we don't need a no-marker context branch."""
    info = view_info or {}
    if not info.get("has_visual", True):
        return _CONTEXT_NO_VISUAL
    return _CONTEXT_VISUAL_MARKERS


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
    raise ValueError(f"Unknown cluster task: {task!r}")


# ---------------------------------------------------------------------------
# Validity gate
# ---------------------------------------------------------------------------
def is_valid_view(view_info: Dict[str, Any] | None) -> bool:
    """Cluster requires that, if a map is shown, it shows the markers.

    Without markers there is nothing for the model to identify
    clusters OF — the perception task is ill-defined. Tools-only mode
    (no map at all) is fine because the LLM can compute clusters
    analytically from the coordinates it queries.
    """
    info = view_info or {}
    has_visual = bool(info.get("has_visual", True))
    has_markers = bool(info.get("has_site_markers", True))
    return (not has_visual) or has_markers


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
# Brace-balanced + string-aware scanner — handles nested braces in the
# new justification-aware response format where the prior simple regex
# `\{[^{}]*\}` would only match inner {"index": ..., "reason": ...}
# entries instead of the outer {"clustered_sites": [...]} wrapper.

def _find_outer_json_objects(text: str) -> List[str]:
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
    the form {"index": i, "reason": "..."}, or a mix. Returns
    (sites, reasons_dict). Backward compatible with the older
    bare-integer list format."""
    sites: List[int] = []
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
            reasons.setdefault(idx, reason)
    return sites, reasons


def parse_response(raw: str, task: str) -> Dict[str, Any]:
    if task == "identify":
        obj = _extract_last_json_object(raw or "")
        sites, reasons = _parse_site_entries(obj.get("clustered_sites", []))
        return {
            "clustered_sites": sorted(set(sites)),
            "reasons": reasons,
        }
    if task == "describe":
        return {"text": raw or ""}
    raise ValueError(f"Unknown cluster task: {task!r}")


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
        pred = set(int(j) for j in parsed.get("clustered_sites", []))
        truth = set(int(j) for j in ground_truth.get("clustered_sites", []))
        return _f1(pred, truth)
    if task == "describe":
        text = (parsed.get("text") or "").lower()
        concepts = ground_truth.get("concepts", [])
        if not concepts:
            return 1.0
        hits = sum(1 for c in concepts if c.lower() in text)
        return hits / len(concepts)
    raise ValueError(f"Unknown cluster task: {task!r}")
