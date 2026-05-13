#!/usr/bin/env python3
"""Option A: total-distance guard sensitivity without re-running the agent.

Reads completed pair JSON from run_dataset (``NN.json``) only — no API, no
``run_dataset`` execution.

1) **final** (default): For the *submitted* ``score``, recomputes whether the
   ``total_weighted_distance`` guard passes at each swept
   ``max_pct_increase`` value. Other guards keep their recorded ``passed``
   flags from the same snapshot; ``feasible`` is unchanged. Reports synthetic
   ``valid`` / ``success`` (cluster: success iff valid and target_response<=0).

2) **explored**: Approximate *counterfactual superscore* if only the
   total-distance cap were perturbed. Re-ranks ``superscore.explored_scores``
   with the same lexicographic key as ``run_dataset._select_best_explored_solution``
   (``test_agent``), but ``valid`` / ``success`` are re-derived from swept TD
   plus a **heuristic** for p90:

   - If a row was **valid** at run time: synthetic valid at pct *p* is
     ``td_passes(p)`` (p90 assumed unchanged when only TD is swept).
   - If **invalid** and ``td_passes(nominal_pct)`` is **False**: treat invalidity
     as (at least) TD-related → synthetic valid at *p* is ``td_passes(p)``.
   - If **invalid** but ``td_passes(nominal_pct)`` is **True**: treat as
     p90/other failure → synthetic valid stays **False** for all *p* in this
     script (TD-only sweep cannot rescue).

   This does **not** re-run ``query.score()``; it cannot model p90 slack vs
   ``fraction_improved`` exactly per explore without stored per-candidate
   guard rows.

   When ``superscore.explored_scores_full`` is present (newer runs), explored
   mode uses each row's stored ``guards`` / ``feasible`` / ``fraction_improved``
   and only re-evaluates the total-distance bound at each swept pct (other
   guards keep their recorded ``passed`` flags).

Usage::

  python scripts/guard_sensitivity_total_distance.py \\
      --results_dir out/full_dataset/cluster/results_vague_gpt54_paired/multimodal \\
      --pct_grid 0.02,0.03,0.04,0.05,0.06,0.08,0.10

  python scripts/guard_sensitivity_total_distance.py \\
      --results_dir_mm .../multimodal --results_dir_to .../tools_only \\
      --scope explored --pct_grid 0.05,0.08,0.12,0.20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GUARD_TD = "total_weighted_distance"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_pct_grid(s: str) -> List[float]:
    out: List[float] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty --pct_grid")
    return sorted(set(out))


def _guard_td(score: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for g in score.get("guards") or []:
        if g.get("name") == GUARD_TD:
            return g
    return None


def _td_passes(response_td: float, baseline_td: float, pct: float) -> bool:
    if baseline_td <= 0:
        return response_td <= 0
    return response_td <= baseline_td * (1.0 + pct)


def _synthetic_valid_final(
    score: Dict[str, Any], pct: float
) -> Tuple[bool, Optional[str]]:
    """Re-evaluate only total_weighted_distance at pct; other guards fixed."""
    td = _guard_td(score)
    if td is None:
        return False, "missing_total_weighted_distance_guard"
    baseline_td = float(td["baseline"])
    response_td = float(td["response"])
    td_ok = _td_passes(response_td, baseline_td, pct)
    other_ok = all(
        bool(g.get("passed"))
        for g in (score.get("guards") or [])
        if g.get("name") != GUARD_TD
    )
    feasible = bool(score.get("feasible", True))
    return feasible and td_ok and other_ok, None


def _cluster_success(valid: bool, target_response: float) -> bool:
    return valid and float(target_response) <= 0.0


def _superscore_key(
    success: bool,
    valid: bool,
    fraction_improved: float,
    raw_improvement: float,
    assignment_distance_delta: float,
    source: Optional[str],
) -> Tuple:
    return (
        success,
        valid,
        float(fraction_improved),
        float(raw_improvement),
        -float(assignment_distance_delta),
        source not in (None, "baseline"),
    )


def _explored_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    sup = data.get("superscore") or {}
    full_rows = sup.get("explored_scores_full")
    if full_rows:
        out: List[Dict[str, Any]] = []
        for r in full_rows:
            sc = r.get("score") or {}
            merged: Dict[str, Any] = {k: v for k, v in r.items() if k != "score"}
            merged.update({
                "fraction_improved": sc.get("fraction_improved"),
                "raw_improvement": sc.get("raw_improvement"),
                "valid": sc.get("valid"),
                "success": sc.get("success"),
                "target_baseline": sc.get("target_baseline"),
                "target_response": sc.get("target_response"),
                "assignment_distance_delta": sc.get("assignment_distance_delta"),
                "guards": sc.get("guards"),
                "feasible": sc.get("feasible"),
                "query_id": sc.get("query_id"),
            })
            out.append(merged)
        return out
    return list(sup.get("explored_scores") or [])


def _best_explored_at_pct(
    data: Dict[str, Any],
    pct: float,
    nominal_pct: float,
) -> Optional[Dict[str, Any]]:
    """Return best explored row after TD-only synthetic validity, or None."""
    score = data.get("score") or {}
    td_final = _guard_td(score)
    if td_final is None:
        return None
    baseline_td_final = float(td_final["baseline"])
    target_baseline = float(score.get("target_baseline", 0.0))
    rows = _explored_rows(data)
    if not rows:
        return None

    scored: List[Dict[str, Any]] = []
    for row in rows:
        td = _guard_td(row)
        if td is None:
            continue
        baseline_td = float(td["baseline"])
        response_td = float(td["response"])
        fi = float(row.get("fraction_improved", 0.0))
        tr = float(row.get("target_response", 0.0))
        raw_imp = float(
            row.get("raw_improvement", target_baseline - tr))
        src = row.get("source")
        feasible = bool(row.get("feasible", True))

        if row.get("guards"):
            td_ok = _td_passes(response_td, baseline_td, pct)
            other_ok = all(
                bool(g.get("passed"))
                for g in (row.get("guards") or [])
                if g.get("name") != GUARD_TD
            )
            syn_valid = feasible and td_ok and other_ok
        else:
            # Legacy thin rows: approximate response TD from final snapshot.
            delta = float(row.get("assignment_distance_delta", 0.0))
            response_td_legacy = baseline_td_final + delta
            orig_valid = bool(row.get("valid"))
            td_nominal = _td_passes(
                response_td_legacy, baseline_td_final, nominal_pct)
            if orig_valid:
                syn_valid = _td_passes(
                    response_td_legacy, baseline_td_final, pct)
            else:
                if td_nominal:
                    syn_valid = False
                else:
                    syn_valid = _td_passes(
                        response_td_legacy, baseline_td_final, pct)

        qid = str(row.get("query_id") or score.get("query_id", ""))
        meta = data.get("metadata") or {}
        if "cluster" in qid:
            syn_success = _cluster_success(syn_valid, tr)
        elif "coverage_gap" in qid:
            thr = float(meta.get("success_threshold_fraction_improved", 0.3))
            syn_success = syn_valid and fi >= thr - 1e-12
        elif "contiguity" in qid:
            thr = float(meta.get("success_threshold_fraction_improved", 0.5))
            syn_success = syn_valid and fi >= thr - 1e-12
        elif "shape_niceness" in qid:
            thr = float(meta.get("success_threshold_fraction_improved", 0.02))
            syn_success = syn_valid and fi >= thr - 1e-12
        else:
            syn_success = syn_valid and fi > 1e-6

        scored.append({
            "row": row,
            "syn_valid": syn_valid,
            "syn_success": syn_success,
            "key": _superscore_key(
                syn_success,
                syn_valid,
                fi,
                raw_imp,
                float(row.get("assignment_distance_delta", 0.0)),
                str(src) if src is not None else None,
            ),
        })

    return max(scored, key=lambda x: x["key"])


def iter_pair_jsons(results_dir: Path) -> Iterable[Path]:
    for p in sorted(results_dir.glob("[0-9][0-9].json")):
        if p.name == "aggregate.json":
            continue
        yield p


def run_final_sensitivity(
    data: Dict[str, Any], pct_grid: List[float]
) -> List[Dict[str, Any]]:
    score = data.get("score") or {}
    qid = str(score.get("query_id", ""))
    fr = float(score.get("fraction_improved", 0.0))
    tr = float(score.get("target_response", 0.0))
    rows_out: List[Dict[str, Any]] = []
    for pct in pct_grid:
        v, err = _synthetic_valid_final(score, pct)
        if "cluster" in qid:
            succ = _cluster_success(v, tr)
        elif "coverage_gap" in qid:
            succ = v and fr >= 0.2
        else:
            succ = v and fr > 1e-6
        rows_out.append({
            "pct_total_dist": pct,
            "synthetic_valid": v,
            "synthetic_success": succ,
            "fraction_improved": fr,
            "target_response": tr,
            "error": err or "",
        })
    return rows_out


def run_explored_sensitivity(
    data: Dict[str, Any],
    pct_grid: List[float],
    nominal_pct: float,
) -> List[Dict[str, Any]]:
    score = data.get("score") or {}
    out: List[Dict[str, Any]] = []
    for pct in pct_grid:
        best = _best_explored_at_pct(data, pct, nominal_pct)
        if best is None:
            out.append({"pct_total_dist": pct, "error": "no_explored_scores"})
            continue
        row = best["row"]
        tr = float(row.get("target_response", 0.0))
        out.append({
            "pct_total_dist": pct,
            "best_explored_index": row.get("explored_index"),
            "best_source": row.get("source"),
            "synthetic_valid": best["syn_valid"],
            "synthetic_success": best["syn_success"],
            "fraction_improved": float(row.get("fraction_improved", 0.0)),
            "target_response": tr,
            "assignment_distance_delta": float(
                row.get("assignment_distance_delta", 0.0)),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option A: sweep total_weighted_distance guard (no API).",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Single results directory (NN.json files).",
    )
    parser.add_argument(
        "--results_dir_mm",
        type=Path,
        default=None,
        help="Multimodal results dir (paired with --results_dir_to).",
    )
    parser.add_argument(
        "--results_dir_to",
        type=Path,
        default=None,
        help="Tools-only results dir.",
    )
    parser.add_argument(
        "--pct_grid",
        type=str,
        default="0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20",
        help="Comma-separated max_pct_increase values for total distance.",
    )
    parser.add_argument(
        "--nominal_pct",
        type=float,
        default=0.05,
        help="Assumed benchmark pct for classifying TD vs p90-only failures "
        "in explored mode (default 0.05 = default_guards).",
    )
    parser.add_argument(
        "--scope",
        choices=("final", "explored"),
        default="final",
        help="final=submitted score only; explored=counterfactual superscore "
        "replay on explored_scores (TD sweep + heuristic; see docstring).",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Write long-form CSV (pair_id, modality, pct, columns...).",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print per-pct aggregates (needs both MM and TO dirs). Metrics that "
        "vary with pct are synthetic_valid / synthetic_success; "
        "fraction_improved on the submitted solution is invariant in --scope final.",
    )
    args = parser.parse_args()
    pct_grid = _parse_pct_grid(args.pct_grid)

    dirs: List[Tuple[str, Path]] = []
    if args.results_dir_mm and args.results_dir_to:
        dirs = [("multimodal", args.results_dir_mm), ("tools_only", args.results_dir_to)]
    elif args.results_dir:
        dirs = [("single", args.results_dir)]
    else:
        print("Provide --results_dir or both --results_dir_mm and --results_dir_to.",
              file=sys.stderr)
        sys.exit(2)

    for _label, rdir in dirs:
        if not rdir.is_dir():
            print(f"ERROR: not a directory or missing: {rdir.resolve()}", file=sys.stderr)
            sys.exit(2)

    csv_rows: List[Dict[str, Any]] = []

    for modality_label, rdir in dirs:
        for jf in iter_pair_jsons(rdir):
            data = load_json(jf)
            if data.get("status") != "completed":
                continue
            pair_id = str(data.get("pair_id") or jf.stem)
            if args.scope == "final":
                sens = run_final_sensitivity(data, pct_grid)
            else:
                sens = run_explored_sensitivity(
                    data, pct_grid, args.nominal_pct)

            for row in sens:
                csv_rows.append({
                    "pair_id": pair_id,
                    "modality": modality_label,
                    "scope": args.scope,
                    **row,
                })

    if not csv_rows:
        print(
            "No completed pair rows written. Check that:\n"
            "  - Paths are real directories (not shell placeholders like .../multimodal).\n"
            "  - Pair JSON files exist and have status == \"completed\".\n"
            "  - For --scope explored, superscore.explored_scores must be present.",
            file=sys.stderr,
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with args.output_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(csv_rows)
            print(f"Wrote {len(csv_rows)} rows to {args.output_csv.resolve()}")
        elif not csv_rows:
            pass  # already warned above

    if args.print_summary and len(dirs) == 2:
        _print_paired_summary(csv_rows, pct_grid, args.scope)


def _print_paired_summary(
    csv_rows: List[Dict[str, Any]], pct_grid: List[float], scope: str
) -> None:
    """MM vs TO: metrics that actually depend on pct vs fi (often constant)."""
    def rows_at(pct: float) -> List[Dict[str, Any]]:
        return [
            r for r in csv_rows
            if abs(float(r["pct_total_dist"]) - pct) < 1e-12 and not r.get("error")
        ]

    print(f"\n=== Guard sensitivity summary  (scope={scope}) ===")
    if scope == "final":
        print(
            "Note: submitted-solution fraction_improved does NOT change with pct.\n"
            "      Below, synthetic_* recomputes only the total-distance guard at "
            "each pct;\n"
            "      other guards keep their recorded pass/fail from the JSON.\n"
        )
    else:
        print(
            "Note: explored mode re-ranks superscore.explored_scores (TD sweep + "
            "p90 heuristic).\n"
            "      fraction_improved can change with pct when the winning explore "
            "switches.\n"
        )

    for pct in pct_grid:
        rp = rows_at(pct)
        mm = {r["pair_id"]: r for r in rp if r["modality"] == "multimodal"}
        to = {r["pair_id"]: r for r in rp if r["modality"] == "tools_only"}
        common = sorted(set(mm) & set(to))
        if not common:
            print(f"  pct={pct:.4f}  n_pairs=0  (no overlapping pair_ids)")
            continue

        def _bget(d: Dict[str, Any], k: str) -> bool:
            return bool(d.get(k))

        n_mm_v = sum(1 for pid in common if _bget(mm[pid], "synthetic_valid"))
        n_to_v = sum(1 for pid in common if _bget(to[pid], "synthetic_valid"))
        n_mm_s = sum(1 for pid in common if _bget(mm[pid], "synthetic_success"))
        n_to_s = sum(1 for pid in common if _bget(to[pid], "synthetic_success"))
        mm_s_to_f = sum(
            1 for pid in common
            if _bget(mm[pid], "synthetic_success") and not _bget(to[pid], "synthetic_success"))
        to_s_mm_f = sum(
            1 for pid in common
            if _bget(to[pid], "synthetic_success") and not _bget(mm[pid], "synthetic_success"))

        fi_mm = sum(1 for pid in common if float(mm[pid].get("fraction_improved", 0))
                    > float(to[pid].get("fraction_improved", 0)))
        fi_to = sum(1 for pid in common if float(to[pid].get("fraction_improved", 0))
                    > float(mm[pid].get("fraction_improved", 0)))
        fi_tie = len(common) - fi_mm - fi_to

        print(f"  pct={pct:.4f}  n_pairs={len(common)}")
        print(f"           synthetic_valid:  MM={n_mm_v:2d}  TO={n_to_v:2d}")
        print(f"           synthetic_success: MM={n_mm_s:2d}  TO={n_to_s:2d}  "
              f"(MM_only={mm_s_to_f}  TO_only={to_s_mm_f})")
        print(f"           fraction_improved: MM_wins={fi_mm}  TO_wins={fi_to}  "
              f"ties={fi_tie}")


if __name__ == "__main__":
    main()
