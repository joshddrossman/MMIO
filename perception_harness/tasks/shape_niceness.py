"""Shape-niceness perception task.

The 'shape_niceness' archetype targets visibly *ugly* catchments —
elongated, jagged, bowtied service areas with high normalised perimeter
index (NPI = P / (2·√(πA)); 1.0 for a circle, larger for stretched or
irregular shapes). The README flags this archetype as the strongest
case for multimodal access because there's no clean formal handle on
"ugly shape" — vision sidesteps the formalisation problem.

Two sub-tasks (mirrors the contiguity module):

  identify : "List the K polling places whose service areas look most
             elongated/jagged/oddly-shaped." Ground truth = the K sites
             with highest NPI per the generator's metadata. Score = F1
             between predicted and ground-truth set (both size K).

  describe : "Describe the variety of shapes you see." Ground truth =
             a list of expected concept keywords. Score = fraction of
             concepts mentioned (case-insensitive substring).

K is fixed at 3 by default — large enough that random guessing has a
near-zero F1 ceiling against an 18-site map, small enough that the VLM
isn't asked to rank all opened sites. To vary K later, add a sibling
task name ("identify_worst_5", etc.) and dispatch on it inside
format_question / parse_response / score.

Contract every task module satisfies:
    format_question(meta, task) -> str
    parse_response(raw_text, task) -> dict
    score(parsed, ground_truth, task) -> float
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


WORST_K = 3


# ---------------------------------------------------------------------------
# Question formatting
# ---------------------------------------------------------------------------
# The question text varies on two axes pulled from view_info:
#   has_visual         : whether the model actually sees a rendered map.
#   has_site_markers   : whether opened sites are drawn on the map.
#
# Visual modes use perceptual language ("elongated", "bowtied", "look
# most visually..."). The no-visual mode reframes the same task in
# structural terms — the LLM has only centroids + adjacency, so the
# relevant proxy for ugliness is non-compact spatial layout of
# assigned precincts (e.g. high spread or irregular adjacency
# patterns). The LLM has no direct perimeter data, so it cannot
# compute NPI; this is by design.

_IDENTIFY_TASK_VISUAL = (
    "Each opened polling place's *service area* is the region formed "
    "by the precincts assigned to it (one colored region per polling "
    "place).\n\n"
    "Service areas vary in shape. Compact, roughly-circular catchments "
    "are 'nice'. Catchments that look elongated, HIGHLY jagged, non-contiguous, "
    "bowtied, or that have irregular gaps are 'ugly'.\n\n"
    f"Identify the {WORST_K} opened polling places whose service "
    "areas look the MOST irregularlyshaped. Order them worst first, and for each one "
    "give a brief reason explaining what about its catchment shape "
    "makes it irregular.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    "  {\"worst_sites\": [\n"
    "    {\"index\": <i>, \"reason\": \"<short justification>\"},\n"
    "    ...\n"
    "  ]}\n"
    f"Provide exactly {WORST_K} entries, ordered worst first. Each "
    "`reason` should briefly describe the visual shape (e.g. 'long "
    "thin tail extending east', 'bowtied with a narrow neck'). Keep "
    "each reason under 40 words. No other text outside the JSON."
)

_IDENTIFY_TASK_NO_VISUAL = (
    "Each opened polling place's *service area* is the set of "
    "precincts currently assigned to it.\n\n"
    "Service areas vary in spatial layout. A 'nice' service area's "
    "assigned precincts cluster tightly around the polling-place "
    "location and form a compact, roughly-convex group on the "
    "adjacency graph. An 'ugly' service area's precincts are stretched "
    "out in a thin chain, branch into separate fingers, or form "
    "irregular non-compact arrangements.\n\n"
    "Use get_current_assignments to find each polling place's "
    "precincts, get_precinct_centroids to inspect their spatial "
    "layout, and get_precinct_adjacency to see how those precincts "
    "connect to each other. You will not have direct access to "
    "catchment perimeter or area; reason from centroid spread and "
    "adjacency-graph shape.\n\n"
    f"Identify the {WORST_K} opened polling places whose service "
    "areas have the most stretched-out / non-compact spatial "
    "arrangement. Order them worst first, and for each one give a "
    "brief reason explaining what about its layout makes it ugly.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    "  {\"worst_sites\": [\n"
    "    {\"index\": <i>, \"reason\": \"<short justification>\"},\n"
    "    ...\n"
    "  ]}\n"
    f"Provide exactly {WORST_K} entries, ordered worst first. Each "
    "`reason` should briefly describe the spatial layout (e.g. "
    "'precincts strung along a 4 km north-south chain', 'two "
    "branches meeting at a thin neck'). Keep each reason under "
    "40 words. No other text outside the JSON."
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
    "use the structured tools to inspect assignments, centroids, and "
    "adjacency."
)

_DESCRIBE_TASK_VISUAL = (
    "Look at the shapes of the colored service areas and describe in "
    "2–3 sentences what you notice — are they mostly compact and "
    "roughly round, or do you see elongated, stretched, jagged, or "
    "otherwise irregular shapes? Mention any catchments that stand "
    "out as visually unusual."
)

_DESCRIBE_TASK_NO_VISUAL = (
    "Using the structured tools (especially get_current_assignments "
    "and get_precinct_centroids), describe in 2–3 sentences what you "
    "notice about the spatial layout of each polling place's "
    "precincts — are most service areas tightly clustered around "
    "their polling places, or do some span elongated or irregular "
    "regions? Mention any catchments that stand out as having an "
    "unusual layout."
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
    raise ValueError(f"Unknown shape_niceness task: {task!r}")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
# Brace-balanced + string-aware scanner — handles nested braces in the
# new justification-aware response format where the prior simple regex
# `\{[^{}]*\}` would only match inner {"index": ..., "reason": ...}
# entries instead of the outer {"worst_sites": [...]} wrapper.

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
    the form {"index": i, "reason": "..."}, or a mix. Order is preserved
    (worst-first per the prompt). Returns (sites, reasons_dict)."""
    sites: List[int] = []
    reasons: Dict[int, str] = {}
    seen: set = set()
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
        if idx is None or idx in seen:
            continue
        seen.add(idx)
        sites.append(idx)
        if reason is not None:
            reasons.setdefault(idx, reason)
    return sites, reasons


def parse_response(raw: str, task: str) -> Dict[str, Any]:
    if task == "identify":
        obj = _extract_last_json_object(raw or "")
        sites, reasons = _parse_site_entries(obj.get("worst_sites", []))
        return {"worst_sites": sites, "reasons": reasons}
    if task == "describe":
        return {"text": raw or ""}
    raise ValueError(f"Unknown shape_niceness task: {task!r}")


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
        # We grade the top-K *as a set*. Ordering ("worst first") is
        # requested in the prompt to nudge the model toward
        # well-considered picks, but it's not graded here — F1 against
        # the ground-truth top-K set keeps the score interpretable.
        pred = set(int(j) for j in parsed.get("worst_sites", []))
        truth = set(int(j) for j in ground_truth.get("worst_sites", []))
        return _f1(pred, truth)
    if task == "describe":
        text = (parsed.get("text") or "").lower()
        concepts = ground_truth.get("concepts", [])
        if not concepts:
            return 1.0
        hits = sum(1 for c in concepts if c.lower() in text)
        return hits / len(concepts)
    raise ValueError(f"Unknown shape_niceness task: {task!r}")


# ---------------------------------------------------------------------------
# Secondary scores
# ---------------------------------------------------------------------------
# Optional contract: tasks may expose a `secondary_scores` function
# returning a {metric_name: float} dict. The runner serialises it as
# JSON into the per_question.csv `secondary_scores` column. Tasks that
# don't define it leave the column empty.
#
# For shape_niceness identify we report two near-miss metrics — both
# answer "the VLM didn't nail the exact top 3, but were its picks at
# least in the worst-shaped neighbourhood?" along two ranking axes:
#
#   in_top{N}_by_npi_fraction       (against top-N by NPI, primary axis)
#   in_top{N}_by_solidity_fraction  (against top-N by solidity — the
#                                    convex-hull ratio; non-convexity
#                                    rather than elongation/jaggedness)
#
# Comparing the two surfaces VLMs that perceive non-convexity (tails,
# bowties, notches) more reliably than jaggedness — or vice versa.
# An NPI-trained oracle scores 1.0 on the NPI axis but may score lower
# on solidity, quantifying the metric correlation directly.
#
# Range per metric: [0, 1]. Counts are reported alongside fractions so
# pivots can ask either "what fraction" or "how many out of K".
TOP_N_NEIGHBOURHOOD = 6
SECONDARY_METRICS = ("npi", "solidity")


def secondary_scores(parsed: Dict[str, Any],
                      ground_truth: Dict[str, Any],
                      task: str) -> Dict[str, float]:
    """Return per-task secondary metrics, or {} if none apply."""
    if task != "identify":
        return {}

    pred = list(int(j) for j in parsed.get("worst_sites", []))
    out: Dict[str, float] = {}

    for metric in SECONDARY_METRICS:
        gt_key = f"top{TOP_N_NEIGHBOURHOOD}_sites_by_{metric}"
        top_n = set(int(j) for j in ground_truth.get(gt_key, []))
        # If the ground-truth axis is missing (legacy pair predating the
        # field, or scipy-less environment for solidity) silently skip
        # this metric. The runner serialises only the keys we emit.
        if not top_n:
            continue
        if not pred:
            out[f"in_top{TOP_N_NEIGHBOURHOOD}_by_{metric}_fraction"] = 0.0
            out[f"in_top{TOP_N_NEIGHBOURHOOD}_by_{metric}_count"] = 0.0
            continue
        n_in = sum(1 for j in pred if j in top_n)
        out[f"in_top{TOP_N_NEIGHBOURHOOD}_by_{metric}_fraction"] = (
            n_in / len(pred))
        out[f"in_top{TOP_N_NEIGHBOURHOOD}_by_{metric}_count"] = float(n_in)

    return out
