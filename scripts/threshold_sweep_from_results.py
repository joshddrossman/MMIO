#!/usr/bin/env python3
"""Offline analysis: success rates under alternative *improvement* thresholds.

Context
-------
Each result JSON (from ``run_dataset.py``) stores:

- ``score`` — the **selected** response (superscore winner) with full
  ``queries.ArchetypeQuery.score`` fields.
- ``superscore.explored_scores_full`` — one entry per **feasible**
  explored candidate the agent actually produced, each with the same
  ``score`` shape (``fraction_improved``, ``valid``, guards, etc.).

The **mining** idea:

1. **Threshold sweep** — For archetypes whose *stated* success rule is a
   lower bound on ``fraction_improved`` (coverage_gap, contiguity,
   shape_niceness), recompute how many runs would count as "success" if
   the bar were τ instead of the value baked into ``queries.py`` at run
   time. This separates "how strict is the rule?" from "how good were
   the trajectories?".

2. **Oracle over explored vs. selected** — For each τ, compare:

   - **selected@τ** — does the *chosen* solution satisfy
     ``valid`` and ``fraction_improved >= τ``?
   - **explored_oracle@τ** — does *any* **feasible** explored row satisfy
     ``valid`` and ``fraction_improved >= τ``?

   If oracle@τ >> selected@τ, superscore selection or exploration breadth
   is leaving primary-metric improvement on the table *within the logged
   tree*. If they track closely, the bottleneck is exploration / proposal
   quality, not the selector.

3. **Cluster** — Official success is ``target_response <= 0`` (no dense
   cluster), not ``fraction_improved >= τ``. This script still reports a
   **fraction_improved** threshold curve for cluster so you can compare
   "how much relative progress" across archetypes on one axis; it also
   prints the official cluster rule separately.

Guards: this sweep uses the **same** per-row ``valid`` flag already
computed at run time (feasible ∧ guards). It does **not** re-simulate
guard parameter sweeps; pair ``guard_sensitivity_total_distance.py``
for that.

Usage::

    python scripts/threshold_sweep_from_results.py \\
        --results_root /path/to/MMIO/full_dataset \\
        --query_type vague \\
        --out_json /tmp/threshold_sweep.json

Optional: ``--out_csv_prefix`` writes one CSV per archetype×modality.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_result(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _explored_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    ss = d.get("superscore") or {}
    rows = ss.get("explored_scores_full")
    if not rows:
        return []
    return list(rows)


def _selected_score(d: Dict[str, Any]) -> Dict[str, Any]:
    return d.get("score") or {}


def _best_valid_fraction(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    best: Optional[float] = None
    for row in rows:
        sc = row.get("score") or {}
        if not bool(sc.get("valid")):
            continue
        f = float(sc.get("fraction_improved", 0.0))
        best = f if best is None else max(best, f)
    return best


def _cluster_official_success(sc: Dict[str, Any]) -> bool:
    return bool(sc.get("valid")) and float(sc.get("target_response", 1.0)) <= 0.0


def fraction_threshold_success(sc: Dict[str, Any], tau: float) -> bool:
    if not bool(sc.get("valid")):
        return False
    return float(sc.get("fraction_improved", 0.0)) >= tau - 1e-12


def _tau_key(tau: float) -> str:
    return f"{tau:.6f}"


def sweep_one_group(
    paths: List[Path],
    archetype: str,
    modality: str,
    thresholds: List[float],
) -> Dict[str, Any]:
    """Aggregate counts for one (archetype, modality) bucket."""
    n = len(paths)
    missing_explored = 0
    per_pair: List[Dict[str, Any]] = []

    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for p in paths:
        loaded.append((p, _load_result(p)))

    cluster_official: List[bool] = []

    for p, d in loaded:
        rows = _explored_rows(d)
        if not rows:
            missing_explored += 1
        sel = _selected_score(d)
        sf = float(sel.get("fraction_improved", 0.0))
        of = _best_valid_fraction(rows)
        co = _cluster_official_success(sel)
        cluster_official.append(co)
        per_pair.append(
            {
                "pair_id": d.get("pair_id"),
                "path": str(p),
                "selected_fraction_improved": sf,
                "selected_valid": bool(sel.get("valid")),
                "selected_target_response": sel.get("target_response"),
                "oracle_best_valid_fraction_improved": of,
                "n_feasible_explored": (d.get("superscore") or {}).get(
                    "n_feasible_explored"
                ),
                "cluster_official_success": co,
            }
        )

    curves_selected: Dict[str, int] = {}
    curves_oracle: Dict[str, int] = {}
    for tau in thresholds:
        ks = _tau_key(tau)
        curves_selected[ks] = sum(
            1
            for _, d in loaded
            if fraction_threshold_success(_selected_score(d), tau)
        )
        curves_oracle[ks] = 0
        for _, d in loaded:
            rows = _explored_rows(d)
            ok = any(
                fraction_threshold_success(r.get("score") or {}, tau) for r in rows
            )
            curves_oracle[ks] += int(ok)

    out: Dict[str, Any] = {
        "archetype": archetype,
        "modality": modality,
        "n_result_files": n,
        "n_missing_explored_scores_full": missing_explored,
        "cluster_official_success_count": sum(cluster_official)
        if archetype == "cluster"
        else None,
        "cluster_official_success_rate": (
            sum(cluster_official) / n if archetype == "cluster" and n else None
        ),
        "thresholds": thresholds,
        "n_selected_pass_fraction_ge_tau": curves_selected,
        "n_oracle_explored_pass_fraction_ge_tau": curves_oracle,
        "per_pair_summary": per_pair,
    }
    return out


def _discover_files(
    results_root: Path, query_type: str
) -> Dict[Tuple[str, str], List[Path]]:
    """Map (archetype, modality) -> list of result json paths."""
    buckets: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        if arch_dir.name.startswith(".") or arch_dir.name == "results_paired_vague":
            continue
        for modality in ("multimodal", "tools_only"):
            res_dir = arch_dir / f"results_{modality}_{query_type}"
            if not res_dir.is_dir():
                continue
            for p in sorted(res_dir.glob("*.json")):
                if p.name == "aggregate.json":
                    continue
                buckets[(arch_dir.name, modality)].append(p)
    return buckets


def _write_csv_curve(path: Path, thresholds: List[float], selected: Dict[str, int], oracle: Dict[str, int], n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "rate_selected", "rate_oracle_explored", "n_selected", "n_oracle", "n_pairs"])
        for tau in thresholds:
            ks = _tau_key(tau)
            ns = selected[ks]
            no = oracle[ks]
            w.writerow([tau, ns / n if n else 0.0, no / n if n else 0.0, ns, no, n])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage::")[0])
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("full_dataset"),
        help="Multi-archetype dataset root (contains cluster/, etc.).",
    )
    parser.add_argument("--query_type", default="vague")
    parser.add_argument(
        "--tau_step",
        type=float,
        default=0.05,
        help="Grid step from 0 to 1 inclusive.",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Write full JSON report (includes per-pair rows).",
    )
    parser.add_argument(
        "--out_csv_prefix",
        type=Path,
        default=None,
        help="If set, write <prefix>_<arch>_<mod>.csv per bucket.",
    )
    args = parser.parse_args()

    if not args.results_root.is_dir():
        print(f"ERROR: results_root not a directory: {args.results_root}", file=sys.stderr)
        sys.exit(1)

    thresholds = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        thresholds.append(round(t, 8))
        t += args.tau_step
    if thresholds[-1] < 1.0:
        thresholds.append(1.0)

    buckets = _discover_files(args.results_root, args.query_type)
    report: Dict[str, Any] = {
        "results_root": str(args.results_root.resolve()),
        "query_type": args.query_type,
        "tau_step": args.tau_step,
        "buckets": {},
    }

    for (arch, mod), paths in sorted(buckets.items()):
        if not paths:
            continue
        block = sweep_one_group(paths, arch, mod, thresholds)
        report["buckets"][f"{arch}/{mod}"] = block
        print(f"\n=== {arch}  {mod}  (n={len(paths)}) ===")
        if arch == "cluster" and block.get("cluster_official_success_rate") is not None:
            print(
                f"  cluster official (target_response<=0 & valid): "
                f"{block['cluster_official_success_count']}/{len(paths)} "
                f"= {block['cluster_official_success_rate']:.3f}"
            )
        if block["n_missing_explored_scores_full"]:
            print(
                f"  WARNING: {block['n_missing_explored_scores_full']} files "
                f"without explored_scores_full (oracle curve degraded)"
            )
        # Print a compact table: every 0.1
        print("  tau   selected_rate  oracle_explored_rate  (gap)")
        for tau in thresholds:
            if abs(tau * 10 - round(tau * 10)) > 1e-6 and abs(tau - 1.0) > 1e-6:
                continue
            ks = _tau_key(tau)
            ns = block["n_selected_pass_fraction_ge_tau"][ks]
            no = block["n_oracle_explored_pass_fraction_ge_tau"][ks]
            n = len(paths)
            rs, ro = ns / n, no / n
            print(f"  {tau:4.2f}   {rs:6.3f}         {ro:6.3f}              {ro - rs:+6.3f}")
        if args.out_csv_prefix:
            csv_path = Path(f"{args.out_csv_prefix}_{arch}_{mod}.csv")
            _write_csv_curve(
                csv_path,
                thresholds,
                block["n_selected_pass_fraction_ge_tau"],
                block["n_oracle_explored_pass_fraction_ge_tau"],
                len(paths),
            )
            print(f"  wrote {csv_path}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
