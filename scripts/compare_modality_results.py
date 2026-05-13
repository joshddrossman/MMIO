#!/usr/bin/env python3
"""Compare two run_dataset aggregate.json files (e.g. multimodal vs tools-only).

Usage:
  cd /Users/cat2510/MMIO
  .venv/bin/python scripts/compare_modality_results.py \\
      out/full_dataset/cluster/results_multimodal_vague/aggregate.json \\
      out/full_dataset/cluster/results_tools_only_vague/aggregate.json

Reads aggregate.json only (no API). Stall-nudge counts come from each
pair's ``log_summary.primary_stall_nudges_sent`` when present (current
``run_dataset``). Older ``NN.json`` files omit that field even if the run
nudged; pass ``--trajectory_dir_a`` / ``--trajectory_dir_b`` pointing at the
same dirs used for ``run_dataset.py --trajectory_log_dir`` to backfill from
``{pair_id}_trajectory.json``.

``n_submitted_explicit`` counts only **explicit** ``submit_proposal`` tool
calls; ``0`` means the session ended with an **implicit** submit (assistant
turn with no ``tool_calls`` after at least one resolve/edit).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: Any, width: int = 12) -> str:
    if x is None:
        return "n/a".rjust(width)
    if isinstance(x, float):
        return f"{x:.4f}".rjust(width)
    return str(x).rjust(width)


def _stall_nudges_from_trajectory(path: Path) -> Optional[int]:
    """Return stall-nudge count from a full trajectory JSON, or None."""
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        log = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(log, list):
        return None
    traj = next(
        (e for e in log if e.get("event") == "trajectory_summary"), {})
    if traj.get("primary_stall_nudges_sent") is not None:
        return int(traj["primary_stall_nudges_sent"])
    return sum(
        1 for e in log if e.get("event") == "primary_target_stall_nudge")


def _apply_trajectory_stall_counts(
    agg: Dict[str, Any], trajectory_dir: Optional[Path]
) -> Dict[str, Any]:
    """Overlay primary_stall_nudges_sent from *_trajectory.json when dir set."""
    if trajectory_dir is None:
        return agg
    trajectory_dir = trajectory_dir.expanduser()
    out = dict(agg)
    new_pos: List[Dict[str, Any]] = []
    for row in out.get("pair_outcomes") or []:
        r = dict(row)
        pid = r.get("pair_id")
        if pid is not None:
            tpath = trajectory_dir / f"{pid}_trajectory.json"
            n = _stall_nudges_from_trajectory(tpath)
            if n is not None:
                r["primary_stall_nudges_sent"] = n
        new_pos.append(r)
    out["pair_outcomes"] = new_pos
    return out


def _recompute_stall_and_submit_means(agg: Dict[str, Any]) -> None:
    rows = [
        po for po in (agg.get("pair_outcomes") or [])
        if po.get("status") == "completed"
    ]
    if not rows:
        return
    ns = [int(po.get("primary_stall_nudges_sent", 0)) for po in rows]
    agg["mean_primary_stall_nudges_sent"] = sum(ns) / len(ns)
    agg["total_primary_stall_nudges_sent"] = sum(ns)
    ss = [int(po.get("n_submitted_explicit", 0)) for po in rows]
    agg["mean_n_submitted_explicit"] = sum(ss) / len(ss)


def _enrich_outcomes_from_pair_json(
    agg: Dict[str, Any], results_dir: Path
) -> Dict[str, Any]:
    """Return a shallow copy of agg with pair_outcomes merged from NN.json."""
    out = dict(agg)
    pos = out.get("pair_outcomes") or []
    new_pos: List[Dict[str, Any]] = []
    for po in pos:
        row = dict(po)
        pid = row.get("pair_id")
        if pid is None:
            new_pos.append(row)
            continue
        pj = results_dir / f"{pid}.json"
        if pj.is_file():
            try:
                data = load_json(pj)
            except (OSError, json.JSONDecodeError):
                new_pos.append(row)
                continue
            ls = data.get("log_summary") or {}
            for key in (
                "n_resolves",
                "primary_stall_nudges_sent",
                "n_submitted_explicit",
                "n_view_solution",
            ):
                if key in ls:
                    row[key] = ls[key]
        new_pos.append(row)
    out["pair_outcomes"] = new_pos
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two aggregate.json files from run_dataset.py.",
    )
    parser.add_argument(
        "aggregate_a",
        type=Path,
        help="First aggregate.json (e.g. results_multimodal_vague/aggregate.json)",
    )
    parser.add_argument(
        "aggregate_b",
        type=Path,
        help="Second aggregate.json (e.g. results_tools_only_vague/aggregate.json)",
    )
    parser.add_argument(
        "--label_a", default="A", help="Column label for first file (default: A)")
    parser.add_argument(
        "--label_b", default="B", help="Column label for second file (default: B)")
    parser.add_argument(
        "--enrich_from_pair_json",
        action="store_true",
        help=(
            "Merge n_resolves / stall nudges / submit / view_solution from each "
            "pair's NN.json next to aggregate (same directory as aggregate)."
        ),
    )
    parser.add_argument(
        "--trajectory_dir_a",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory with {pair_id}_trajectory.json for side A (e.g. "
            "verbose_logs/cluster_vague_mm). Fills stall nudges when NN.json "
            "log_summary omits them."
        ),
    )
    parser.add_argument(
        "--trajectory_dir_b",
        type=Path,
        default=None,
        metavar="DIR",
        help="Same as --trajectory_dir_a for side B.",
    )
    args = parser.parse_args()

    a = load_json(args.aggregate_a)
    b = load_json(args.aggregate_b)
    if args.enrich_from_pair_json:
        a = _enrich_outcomes_from_pair_json(a, args.aggregate_a.parent)
        b = _enrich_outcomes_from_pair_json(b, args.aggregate_b.parent)
        for agg in (a, b):
            completed_rows = [
                po for po in (agg.get("pair_outcomes") or [])
                if po.get("status") == "completed"
            ]
            if not completed_rows:
                continue
            if agg.get("mean_n_resolves") is None:
                rs = [int(po.get("n_resolves", 0)) for po in completed_rows]
                agg["mean_n_resolves"] = sum(rs) / len(rs)
                agg["median_n_resolves"] = float(statistics.median(rs))
            if agg.get("mean_primary_stall_nudges_sent") is None:
                ns = [
                    int(po.get("primary_stall_nudges_sent", 0))
                    for po in completed_rows
                ]
                agg["mean_primary_stall_nudges_sent"] = sum(ns) / len(ns)
                agg["total_primary_stall_nudges_sent"] = sum(ns)
            if agg.get("mean_n_view_solution") is None:
                vs = [int(po.get("n_view_solution", 0)) for po in completed_rows]
                agg["mean_n_view_solution"] = sum(vs) / len(vs)
            if agg.get("mean_n_submitted_explicit") is None:
                ss = [
                    int(po.get("n_submitted_explicit", 0))
                    for po in completed_rows
                ]
                agg["mean_n_submitted_explicit"] = sum(ss) / len(ss)
    a = _apply_trajectory_stall_counts(a, args.trajectory_dir_a)
    b = _apply_trajectory_stall_counts(b, args.trajectory_dir_b)
    if args.trajectory_dir_a is not None or args.trajectory_dir_b is not None:
        _recompute_stall_and_submit_means(a)
        _recompute_stall_and_submit_means(b)

    label_a = args.label_a
    label_b = args.label_b

    keys = [
        "n_pairs",
        "n_completed",
        "n_valid_responses",
        "mean_fraction_improved",
        "median_fraction_improved",
        "n_substantial_improvement",
        "n_zero_improvement",
        "n_fully_closed_gap",
        "n_success",
        "n_with_success_criterion",
        "fraction_success",
        "mean_assignment_distance_delta",
        "median_assignment_distance_delta",
        "mean_n_resolves",
        "median_n_resolves",
        "mean_primary_stall_nudges_sent",
        "total_primary_stall_nudges_sent",
        "mean_n_view_solution",
        "mean_n_submitted_explicit",
        "mean_total_tokens",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_cached_tokens",
        "sum_total_tokens",
    ]

    wk = 34
    print("\n" + "=" * 72)
    print("AGGREGATE COMPARISON")
    print("=" * 72)
    print(f"  A ({label_a}): {args.aggregate_a.resolve()}")
    print(f"  B ({label_b}): {args.aggregate_b.resolve()}")
    print()
    header = f"{'metric':<{wk}} {label_a:>12} {label_b:>12}"
    print(header)
    print("-" * len(header))
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if k in ("mean_fraction_improved", "median_fraction_improved",
                 "fraction_success", "mean_n_resolves", "median_n_resolves",
                 "mean_primary_stall_nudges_sent", "mean_assignment_distance_delta",
                 "median_assignment_distance_delta", "mean_n_view_solution",
                 "mean_n_submitted_explicit"):
            sa = fmt(va) if isinstance(va, (int, float)) or va is None else str(va).rjust(12)
            sb = fmt(vb) if isinstance(vb, (int, float)) or vb is None else str(vb).rjust(12)
        else:
            sa = str(va).rjust(12) if va is not None else "n/a".rjust(12)
            sb = str(vb).rjust(12) if vb is not None else "n/a".rjust(12)
        print(f"{k:<{wk}} {sa} {sb}")

    # Per-pair alignment by pair_id
    def by_id(agg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for po in agg.get("pair_outcomes") or []:
            pid = po.get("pair_id")
            if pid is not None:
                out[str(pid)] = po
        return out

    ma, mb = by_id(a), by_id(b)
    all_ids = sorted(set(ma) | set(mb), key=lambda x: (len(x), x))

    print("\n" + "=" * 72)
    print("PER-PAIR TABLE (superscore metrics + log_summary / trajectory)")
    print("=" * 72)
    print(
        "  Legend:\n"
        "    fr  = fraction_improved on benchmark (0–1)\n"
        "    tg  = primary target value on scored response\n"
        "    rs  = number of resolve() applications (feasible + infeasible)\n"
        "    nd  = primary-target stall nudges (only if log_summary or "
        "--trajectory_dir_* has data)\n"
        "    xSub = explicit submit_proposal tool calls "
        "(0 = implicit end: assistant text, no tools)\n"
        "    vw  = view_solution tool calls (0 in tools-only modality)\n"
    )
    cols = (
        "pair",
        "fr_A",
        "fr_B",
        "tg_A",
        "tg_B",
        "rs_A",
        "rs_B",
        "nd_A",
        "nd_B",
        "xSub_A",
        "xSub_B",
        "vw_A",
    )
    row_fmt = (
        "{:<6} {:>6} {:>6} {:>5} {:>5} {:>4} {:>4} {:>4} {:>4} {:>6} {:>6} {:>4}"
    )
    print(row_fmt.format(*cols))
    print("-" * 72)

    def g(po: Optional[Dict[str, Any]], key: str, default: Any = 0) -> Any:
        if not po:
            return default
        return po.get(key, default)

    for pid in all_ids:
        pa, pb = ma.get(pid), mb.get(pid)
        if not pa and not pb:
            continue
        print(row_fmt.format(
            pid,
            f"{float(g(pa, 'fraction_improved', 0)):.2f}" if pa else "-",
            f"{float(g(pb, 'fraction_improved', 0)):.2f}" if pb else "-",
            f"{g(pa, 'target_response', 0):.0f}" if pa else "-",
            f"{g(pb, 'target_response', 0):.0f}" if pb else "-",
            int(g(pa, "n_resolves", 0)),
            int(g(pb, "n_resolves", 0)),
            int(g(pa, "primary_stall_nudges_sent", 0)),
            int(g(pb, "primary_stall_nudges_sent", 0)),
            int(g(pa, "n_submitted_explicit", 0)),
            int(g(pb, "n_submitted_explicit", 0)),
            int(g(pa, "n_view_solution", 0)),
        ))

    missing = [
        k for k in ("mean_n_resolves", "mean_primary_stall_nudges_sent")
        if a.get(k) is None or b.get(k) is None
    ]
    if missing:
        print(
            "\nNote: Some aggregate trajectory means are missing. Use "
            "--enrich_from_pair_json and/or re-run run_dataset.py with the "
            "current tree; for stall nudges on older NN.json, pass "
            f"--trajectory_dir_a / --trajectory_dir_b. Missing: {missing}",
            flush=True,
        )


if __name__ == "__main__":
    main()
