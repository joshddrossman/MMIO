#!/usr/bin/env python3
"""Build the headline modality comparison table from run_dataset aggregate.json files.

No API calls. Expects the standard layout::

    <dataset_root>/<archetype>/results_multimodal_vague/aggregate.json
    <dataset_root>/<archetype>/results_tools_only_vague/aggregate.json

Optionally include a flat run tree (e.g. marker-free shape rerun)::

    <flat_root>/multimodal/aggregate.json
    <flat_root>/tools_only/aggregate.json

Usage::

    python scripts/paper_headline_table.py
    python scripts/paper_headline_table.py --dataset-root full_dataset \\
        --flat-run out/runs/shape_niceness_marker_free_vague \\
        --flat-label shape_niceness_marker_free
    python scripts/paper_headline_table.py --out paper/headline_table_vague.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

ARCHETYPE_ORDER = ("cluster", "coverage_gap", "contiguity", "shape_niceness")
MODALITY_ORDER = ("multimodal", "tools_only")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_aggregate(agg: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    n_done = int(agg.get("n_completed", 0))
    n_ok = int(agg.get("n_success", 0))
    frac = float(agg.get("fraction_success", 0.0))
    mfi = float(agg.get("mean_fraction_improved", 0.0))
    mad = float(agg.get("mean_assignment_distance_delta", 0.0))
    pct = 100.0 * frac if n_done else 0.0
    sr = f"**{pct:.1f}%** ({n_ok}/{n_done})" if n_done else "n/a"
    mfi_s = f"{mfi:.3f}"
    mad_s = f"{mad:+.0f}"
    return sr, mfi_s, mad_s, str(n_done), str(n_ok)


def _collect_dataset_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for arch in ARCHETYPE_ORDER:
        for mod in MODALITY_ORDER:
            p = root / arch / f"results_{mod}_vague" / "aggregate.json"
            if not p.is_file():
                continue
            agg = _load(p)
            sr, mfi, mad, _, _ = _row_from_aggregate(agg)
            rows.append({
                "archetype": arch,
                "modality": mod.replace("_", " "),
                "success_rate": sr,
                "mean_frac": mfi,
                "mean_delta_td": mad,
                "source": str(p.relative_to(REPO_ROOT)),
            })
    return rows


def _collect_flat_run_rows(flat: Path, label: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for mod in MODALITY_ORDER:
        p = flat / mod / "aggregate.json"
        if not p.is_file():
            continue
        agg = _load(p)
        sr, mfi, mad, _, _ = _row_from_aggregate(agg)
        rows.append({
            "archetype": label,
            "modality": mod.replace("_", " "),
            "success_rate": sr,
            "mean_frac": mfi,
            "mean_delta_td": mad,
            "source": str(p.relative_to(REPO_ROOT)),
        })
    return rows


def _markdown(rows: List[Dict[str, str]]) -> str:
    lines = [
        "| Archetype | Modality | Success rate | Mean fraction improved | Mean Δ assignment distance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['archetype']} | {r['modality']} | {r['success_rate']} | "
            f"{r['mean_frac']} | {r['mean_delta_td']} |"
        )
    lines.append("")
    lines.append("<!-- Sources: -->")
    for r in rows:
        lines.append(f"<!-- {r['archetype']} {r['modality']}: {r['source']} -->")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "full_dataset",
        help="Root containing per-archetype results_*_vague dirs",
    )
    ap.add_argument(
        "--flat-run",
        type=Path,
        default=None,
        help="Optional flat layout: <path>/multimodal|tools_only/aggregate.json",
    )
    ap.add_argument(
        "--flat-label",
        default="shape_niceness_marker_free",
        help="Archetype column label for --flat-run rows",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown to this path (default: print to stdout only)",
    )
    args = ap.parse_args()

    root = (REPO_ROOT / args.dataset_root).resolve() if not args.dataset_root.is_absolute() else args.dataset_root
    rows = _collect_dataset_rows(root)
    if args.flat_run:
        flat = (REPO_ROOT / args.flat_run).resolve() if not args.flat_run.is_absolute() else args.flat_run
        rows.extend(_collect_flat_run_rows(flat, args.flat_label))

    md = _markdown(rows)
    if args.out:
        out = (REPO_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out.relative_to(REPO_ROOT)}")
    else:
        print(md)


if __name__ == "__main__":
    main()
