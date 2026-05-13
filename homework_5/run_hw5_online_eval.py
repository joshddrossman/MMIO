#!/usr/bin/env python3
"""Compare two homework benchmark runs (e.g. tools_only vs multimodal).

Reads homework_5/outputs/runs/<exp>/<modality>/aggregate.json for each run
and prints a Markdown table (paste into your PDF / notebook).

  python homework_5/run_hw5_online_eval.py \\
      --a homework_5/outputs/runs/hw5_main/tools_only/aggregate.json \\
      --b homework_5/outputs/runs/hw5_main/multimodal/aggregate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _mean_wall_clock(agg: Dict[str, Any]) -> Optional[float]:
    po = agg.get("pair_outcomes") or []
    times: List[float] = []
    for row in po:
        t = row.get("elapsed_sec")
        if t is not None:
            times.append(float(t))
    return statistics.mean(times) if times else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Path to aggregate.json (config A)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--b", required=True, help="Path to aggregate.json (config B)")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", default=None, help="Optional path to write table.md")
    args = p.parse_args()

    pa, pb = Path(args.a), Path(args.b)
    if not pa.is_file() or not pb.is_file():
        print("ERROR: aggregate files not found.", file=sys.stderr)
        sys.exit(1)

    A, B = _load(pa), _load(pb)

    rows = [
        ("Model", A.get("model"), B.get("model")),
        ("Modality", A.get("modality"), B.get("modality")),
        ("Query type (dataset setting)", A.get("query_type"), B.get("query_type")),
        ("Pairs completed", A.get("n_completed"), B.get("n_completed")),
        ("Fraction success", A.get("fraction_success"), B.get("fraction_success")),
        ("Mean fraction improved", f"{A.get('mean_fraction_improved', 0):.3f}",
         f"{B.get('mean_fraction_improved', 0):.3f}"),
        ("Mean assignment-distance delta",
         f"{A.get('mean_assignment_distance_delta', 0):.1f}" if A.get("mean_assignment_distance_delta") is not None else "n/a",
         f"{B.get('mean_assignment_distance_delta', 0):.1f}" if B.get("mean_assignment_distance_delta") is not None else "n/a"),
        ("Mean wall-clock sec / pair (from JSON)",
         f"{_mean_wall_clock(A):.1f}" if _mean_wall_clock(A) else "n/a",
         f"{_mean_wall_clock(B):.1f}" if _mean_wall_clock(B) else "n/a"),
    ]

    lines = [
        f"| Metric | {args.label_a} | {args.label_b} |",
        "| --- | --- | --- |",
    ]
    for name, va, vb in rows:
        lines.append(f"| {name} | {va} | {vb} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
