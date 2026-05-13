#!/usr/bin/env python3
"""Audit explore-replay pickles and offline-rescore shape_niceness results.

Example::

    python scripts/audit_and_rescore_shape_niceness.py \\
        --dataset-dir full_dataset/shape_niceness \\
        --out-dir full_dataset/shape_niceness/offline_rescore_vague

Writes ``audit.csv``, ``rescore.csv``, and ``rescore.json`` under ``--out-dir``.
No API calls; requires ``*.explore_replay.pkl`` next to each result JSON.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.shape_niceness_replay_metrics import (
    audit_shape_niceness_dataset,
    rescore_shape_niceness_dataset,
    write_audit_csv,
    write_rescore_csv,
    write_rescore_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("full_dataset/shape_niceness"),
        help="Archetype root (contains pairs/ and results_*).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <dataset-dir>/offline_rescore_vague).",
    )
    args = p.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = (args.out_dir or (dataset_dir / "offline_rescore_vague")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = audit_shape_niceness_dataset(dataset_dir)
    write_audit_csv(audit_rows, out_dir / "audit.csv")

    records = rescore_shape_niceness_dataset(dataset_dir)
    write_rescore_csv(records, out_dir / "rescore.csv")
    write_rescore_json(records, out_dir / "rescore.json")

    n_ok = sum(1 for r in records if not r.get("error"))
    print(f"Wrote {out_dir / 'audit.csv'} ({len(audit_rows)} rows)")
    print(f"Wrote {out_dir / 'rescore.csv'} ({len(records)} rows, {n_ok} without error)")


if __name__ == "__main__":
    main()
