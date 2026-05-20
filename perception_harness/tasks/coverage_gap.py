"""Coverage-gap perception task.

The 'coverage_gap' archetype targets uncapacitated polling-place plans
where at least one precinct is significantly stranded — its nearest
opened polling place is much farther than typical — and at least one
closed candidate site, if opened, would meaningfully reduce the
maximum travel distance across all precincts.

The perception task is to identify that single closed candidate.

Two sub-tasks:

  identify : "Identify the closed candidate polling place that, if
             opened, would most reduce the maximum travel distance
             across all precincts." Single answer.

             Primary score: fraction-of-optimal-improvement, defined
             as (current_max − new_max_with_agent_pick) divided by
             (current_max − new_max_with_best_pick). Range [0, 1];
             1.0 if the agent picked the analytic best candidate;
             continuous partial credit for picking a candidate that
             achieves some of the achievable improvement.

             Secondary scores: exact_match (binary), in_top3, rank
             (1-indexed; -1 if pick wasn't a closed candidate).

  describe : "Describe what you notice about the coverage of polling
             places." Free-form, scored on concept-keyword overlap
             ("stranded", "underserved", "far", "isolated", "gap",
             "remote", "distant").

Validity gate: identifying a closed candidate by index requires the
candidate markers to be visible (or available via list_sites in
tools-only mode). is_valid_view rejects renderers that hide closed
candidates (v2_no_markers, v2_legend, v2_patch_labels) when a map is
shown. Tools-only mode is always valid.

Contract:
    format_question(meta, task, *, view_info=None) -> str
    parse_response(raw_text, task) -> dict
    score(parsed, ground_truth, task) -> float
    secondary_scores(parsed, ground_truth, task) -> dict
    is_valid_view(view_info) -> bool
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------
_CONTEXT_VISUAL_MARKERS = (
    "The map shows a polling-place plan for a county. Each precinct is "
    "filled with the color of its assigned opened polling place. The "
    "opened polling places are drawn as red circles with white index "
    "labels. Closed candidate polling places (sites that COULD be "
    "opened but currently are not) are drawn as smaller gray circles "
    "with their own index labels."
)

_CONTEXT_NO_VISUAL = (
    "You are examining a polling-place plan for a county with roughly "
    "80 precincts. The plan currently has some opened polling places "
    "and some closed candidate sites. No map is provided — use the "
    "structured tools to inspect site locations, current assignments, "
    "and precinct-to-site distances."
)


_IDENTIFY_TASK_VISUAL = (
    "Some precincts are well-served — their nearest opened polling "
    "place is close. Others may be stranded — their nearest opened "
    "polling place is far away (visible as a colored service area "
    "that extends well beyond its red marker, or as a precinct on "
    "the edge of a large catchment).\n\n"
    "Identify the SINGLE closed candidate polling place (gray "
    "marker) that, if opened, would most reduce the MAXIMUM travel "
    "distance across all precincts. Pick the candidate whose "
    "location best serves the most-stranded precinct(s) without "
    "leaving them far from any open site.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '  {"best_candidate": <site_index>, "reason": "<short justification>"}\n'
    "The `reason` should briefly state which stranded precinct(s) "
    "you identified, where on the map they are, and why this "
    "candidate's location best addresses them. Keep the reason "
    "under 40 words. No other text outside the JSON."
)

_IDENTIFY_TASK_NO_VISUAL = (
    "Use list_sites(opened_only=False) to enumerate all candidate "
    "and opened polling places, get_current_assignments to see "
    "current precinct-to-site assignments and distances, and "
    "get_distance_matrix to read precinct-to-site distances "
    "directly. The current MAXIMUM travel distance across all "
    "precincts is what we want to reduce.\n\n"
    "Identify the SINGLE closed candidate polling place that, if "
    "opened, would most reduce the maximum travel distance across "
    "all precincts. Algorithm: for each closed candidate j, compute "
    "the new maximum travel distance assuming each precinct is "
    "reassigned to its nearest opened site INCLUDING j (i.e., for "
    "each precinct i, new_dist_i = min(current_dist_i, "
    "distance(i, j)); the new max is max over i of new_dist_i). "
    "Pick the candidate that minimises this new maximum.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '  {"best_candidate": <site_index>, "reason": "<short justification>"}\n'
    "The `reason` should briefly identify the most-stranded precinct "
    "(its current distance) and explain why this candidate's "
    "location best serves it. Keep the reason under 40 words. No "
    "other text outside the JSON."
)


_DESCRIBE_TASK_VISUAL = (
    "Look at the map. Do you see any precincts that seem poorly "
    "served — their colored service area extends far from its red "
    "marker, indicating long travel distance? Describe in 2–3 "
    "sentences what you notice about the coverage of polling "
    "places, including any obviously under-served pockets."
)

_DESCRIBE_TASK_NO_VISUAL = (
    "Use the structured tools (especially get_current_assignments) "
    "to inspect each precinct's travel distance to its assigned "
    "polling place. Describe in 2–3 sentences what you notice about "
    "the coverage of polling places — are some precincts much "
    "farther from their nearest opened site than others? Mention "
    "any obviously under-served pockets."
)


def _context(view_info: Dict[str, Any] | None) -> str:
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
    raise ValueError(f"Unknown coverage_gap task: {task!r}")


# ---------------------------------------------------------------------------
# Validity gate
# ---------------------------------------------------------------------------
def is_valid_view(view_info: Dict[str, Any] | None) -> bool:
    """Coverage-gap requires that, if a map is shown, the closed
    candidate sites are visually identifiable (so the agent can
    actually point at one).

    Tools-only mode is always valid: list_sites enumerates closed
    candidates with their indices.
    """
    info = view_info or {}
    has_visual = bool(info.get("has_visual", True))
    has_candidate_markers = bool(info.get("has_candidate_markers", True))
    return (not has_visual) or has_candidate_markers


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
# Brace-balanced + string-aware scanner — same idea as the other tasks,
# robust to nested {"index": ..., "reason": ...} structures and braces
# inside JSON string values.

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


def parse_response(raw: str, task: str) -> Dict[str, Any]:
    if task == "identify":
        obj = _extract_last_json_object(raw or "")
        bc = obj.get("best_candidate")
        # Tolerate string indices and one-element lists.
        if isinstance(bc, list) and bc:
            bc = bc[0]
        try:
            best: Optional[int] = int(bc) if bc is not None else None
        except (TypeError, ValueError):
            best = None
        reason = obj.get("reason")
        out: Dict[str, Any] = {"best_candidate": best, "reasons": {}}
        if best is not None and reason is not None:
            out["reasons"][best] = str(reason)
        return out
    if task == "describe":
        return {"text": raw or ""}
    raise ValueError(f"Unknown coverage_gap task: {task!r}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(parsed: Dict[str, Any], ground_truth: Dict[str, Any],
          task: str) -> float:
    """Primary score for identify is `fraction_of_optimal_improvement`:
    the agent's pick achieves what fraction of the achievable
    improvement to the max travel distance? Range [0, 1]; 1.0 if the
    agent picked the best candidate; 0.0 if the pick achieves no
    improvement (or isn't a closed candidate at all).
    """
    if task == "identify":
        pick = parsed.get("best_candidate")
        if pick is None:
            return 0.0
        per = ground_truth.get("per_candidate_improvement", {})
        best_imp = float(ground_truth.get("best_improvement_km", 0.0))
        if best_imp <= 0:
            # Degenerate pair — no candidate improves things. Treat any
            # non-None pick as 0; only "no candidate exists" returns 0
            # too. Either way, score is 0.
            return 0.0
        pick_data = per.get(str(int(pick)))
        if pick_data is None:
            # Pick isn't among the closed candidates (e.g., agent named
            # an opened site, or a non-existent index). Score 0.
            return 0.0
        agent_imp = float(pick_data.get("improvement_km", 0.0))
        return max(0.0, min(1.0, agent_imp / best_imp))
    if task == "describe":
        text = (parsed.get("text") or "").lower()
        concepts = ground_truth.get("concepts", [])
        if not concepts:
            return 1.0
        hits = sum(1 for c in concepts if c.lower() in text)
        return hits / len(concepts)
    raise ValueError(f"Unknown coverage_gap task: {task!r}")


def secondary_scores(parsed: Dict[str, Any],
                      ground_truth: Dict[str, Any],
                      task: str) -> Dict[str, float]:
    """Return per-task secondary metrics:

      exact_match            : 1.0 if pick == best_candidate, 0.0 else
      in_top3_by_improvement : 1.0 if pick in top-3 by improvement
      rank                   : 1-indexed rank in the full closed-candidate
                                ranking; -1 if pick isn't a closed candidate
      improvement_km         : the agent's pick's improvement in km
                                (raw, not normalised)
    """
    if task != "identify":
        return {}

    pick = parsed.get("best_candidate")
    if pick is None:
        return {
            "exact_match": 0.0,
            "in_top3_by_improvement": 0.0,
            "rank": -1.0,
            "improvement_km": 0.0,
        }

    pick = int(pick)
    best = int(ground_truth.get("best_candidate", -1))
    top3 = set(int(i) for i in ground_truth.get("top3_candidates", []))
    ranking = [int(i) for i in ground_truth.get("ranked_candidates", [])]
    per = ground_truth.get("per_candidate_improvement", {})

    out: Dict[str, float] = {
        "exact_match": 1.0 if pick == best else 0.0,
        "in_top3_by_improvement": 1.0 if pick in top3 else 0.0,
    }
    try:
        out["rank"] = float(ranking.index(pick) + 1)  # 1-indexed
    except ValueError:
        out["rank"] = -1.0
    pick_data = per.get(str(pick))
    out["improvement_km"] = (float(pick_data["improvement_km"])
                              if pick_data else 0.0)
    return out
