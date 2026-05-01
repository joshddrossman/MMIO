"""Top-level orchestrator that generates the full multi-archetype dataset.

Runs the per-archetype dataset generators sequentially:
  cluster        -> out/<root>/cluster/         (N pairs)
  coverage_gap   -> out/<root>/coverage_gap/    (N pairs)
  contiguity     -> out/<root>/contiguity/      (N pairs)
  shape_niceness -> out/<root>/shape_niceness/  (N pairs)

A top-level out/<root>/index.json links to each archetype's index.

Run:
    python generate_full_dataset.py --root out/full_dataset --n_per_archetype 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from dataset_generator import generate_archetype_dataset, ARCHETYPE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out/full_dataset",
                        help="Root output directory.")
    parser.add_argument("--n_per_archetype", type=int, default=25,
                        help="How many pairs to generate per archetype.")
    parser.add_argument("--seed_max", type=int, default=15000,
                        help=("Maximum seed to try per archetype "
                              "(higher = more chances to fill the quota)."))
    parser.add_argument("--no_render", action="store_true",
                        help="Skip per-pair PNG rendering.")
    parser.add_argument("--skip_archetypes", nargs="*", default=[],
                        choices=ARCHETYPE_NAMES,
                        help="Archetypes to skip.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help=("If an archetype fails, continue to the next "
                              "instead of exiting."))
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip_archetypes)
    summary: Dict[str, Any] = {
        "n_per_archetype": args.n_per_archetype,
        "seed_max": args.seed_max,
        "archetypes": {},
    }

    print("=" * 64)
    print(f"FULL DATASET GENERATION  -> {root}")
    print(f"  {args.n_per_archetype} pairs per archetype  |  "
           f"seed_max={args.seed_max}")
    print("=" * 64)

    n_archetypes = len(ARCHETYPE_NAMES)
    for k, archetype in enumerate(ARCHETYPE_NAMES, start=1):
        if archetype in skip:
            continue
        print(f"\n[{k}/{n_archetypes}] {archetype} archetype")
        try:
            idx = generate_archetype_dataset(
                archetype=archetype,
                n_pairs=args.n_per_archetype,
                output_dir=str(root / archetype),
                seed_max=args.seed_max,
                render_baselines=not args.no_render,
                verbose=True,
            )
            summary["archetypes"][archetype] = {
                "n_pairs": idx["n_pairs"],
                "path": str(root / archetype),
            }
        except Exception as e:
            print(f"  {archetype} failed: {e}")
            summary["archetypes"][archetype] = {"error": str(e)[:200]}
            if not args.continue_on_error:
                sys.exit(1)

    with open(root / "index.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"FULL DATASET SUMMARY")
    print("=" * 64)
    total_pairs = 0
    for arch, info in summary["archetypes"].items():
        if "error" in info:
            print(f"  {arch:18s}  FAILED  ({info['error'][:80]})")
        else:
            n = info["n_pairs"]
            total_pairs += n
            print(f"  {arch:18s}  {n} pairs at {info['path']}")
    print(f"\n  total pairs: {total_pairs}")
    print(f"  top-level index: {root / 'index.json'}")
    print(f"\nNext: run the agent across each archetype with run_dataset.py:")
    for arch in ARCHETYPE_NAMES:
        if arch in skip or "error" in summary["archetypes"].get(arch, {}):
            continue
        print(f"  python run_dataset.py --dataset_dir {root / arch}")


if __name__ == "__main__":
    main()
