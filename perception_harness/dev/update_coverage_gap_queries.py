"""In-place update of vague/precise queries in existing pairs.

Use this when you've revised the templates in
instance_generator/dataset_generator.py (e.g. the coverage_gap rewrite
to the max-distance regime) but already have generated pairs you don't
want to re-build from scratch.

For each pair_dir under <full_dataset_dir>/<archetype>/pairs/:
  - Load query_metadata.json
  - Re-run dataset_generator._build_query_texts with a per-pair
    deterministic RNG (seeded by hash of sampling_seed + pair_id, so
    re-runs are idempotent and adding new pairs later doesn't perturb
    existing pairs' template choices)
  - Update vague_text, precise_text, vague_template_idx,
    precise_template_idx in query_metadata.json
  - Update the matching record in the per-archetype index.json so the
    denormalised template indices stay in sync

Run (default updates coverage_gap):
    python dev/update_coverage_gap_queries.py

Other archetypes:
    python dev/update_coverage_gap_queries.py --archetype shape_niceness

Dry run:
    python dev/update_coverage_gap_queries.py --dry_run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))

from instance import Instance  # noqa: E402
from dataset_generator import _build_query_texts  # noqa: E402


def _per_pair_rng(sampling_seed: int, pair_id: str) -> np.random.Generator:
    """Deterministic RNG seeded per-pair via md5.

    Per-pair seeding (rather than a single RNG walked through all pairs
    in order) means the script is idempotent regardless of iteration
    order, and adding new pairs later won't perturb existing pairs'
    template choices.
    """
    h = hashlib.md5(f"{sampling_seed}_{pair_id}".encode()).hexdigest()
    seed = int(h[:8], 16)
    return np.random.default_rng(seed)


def _update_per_archetype_index(archetype_dir: Path,
                                  updates: Dict[str, Dict[str, int]]) -> bool:
    """Sync vague_template_idx + precise_template_idx in index.json's
    pair records. Returns True if any record was modified."""
    idx_path = archetype_dir / "index.json"
    if not idx_path.exists():
        return False
    idx = json.loads(idx_path.read_text())
    changed = False
    for record in idx.get("pairs", []):
        pid = record.get("pair_id")
        if pid in updates:
            new = updates[pid]
            if (record.get("vague_template_idx") != new["vague_template_idx"]
                    or record.get("precise_template_idx")
                        != new["precise_template_idx"]):
                record["vague_template_idx"] = new["vague_template_idx"]
                record["precise_template_idx"] = new["precise_template_idx"]
                changed = True
    if changed:
        idx_path.write_text(json.dumps(idx, indent=2))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetype", default="coverage_gap",
                    help="Which archetype's pairs to update.")
    ap.add_argument(
        "--full_dataset_dir",
        default=str(HARNESS_ROOT / "eval_set" / "full_dataset"),
        help="Path to the full_dataset directory.")
    ap.add_argument("--sampling_seed", type=int, default=42,
                    help=("Seed for the per-pair RNG. Use the same value "
                          "across re-runs to keep template choices stable."))
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would change but don't write.")
    args = ap.parse_args()

    archetype_dir = Path(args.full_dataset_dir) / args.archetype
    pairs_dir = archetype_dir / "pairs"
    if not pairs_dir.is_dir():
        print(f"ERROR: {pairs_dir} not found", file=sys.stderr)
        sys.exit(1)

    pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
    print(f"Found {len(pair_dirs)} pair(s) under {pairs_dir}")

    n_updated = 0
    n_unchanged = 0
    n_skipped = 0
    index_updates: Dict[str, Dict[str, int]] = {}

    for pd in pair_dirs:
        qmeta_path = pd / "query_metadata.json"
        if not qmeta_path.exists():
            print(f"  {pd.name}: no query_metadata.json, skipping")
            n_skipped += 1
            continue
        meta = json.loads(qmeta_path.read_text())
        if meta.get("archetype") != args.archetype:
            print(f"  {pd.name}: archetype mismatch "
                  f"({meta.get('archetype')!r}), skipping")
            n_skipped += 1
            continue

        # Load instance — _build_query_texts takes it as an argument
        # though for the coverage_gap branch it isn't actually consulted.
        inst_path = pd / "instance.pkl"
        try:
            inst = Instance.load(str(inst_path)) if inst_path.exists() else None
        except Exception as e:
            print(f"  {pd.name}: failed to load instance ({e}), "
                   f"continuing without it")
            inst = None

        rng = _per_pair_rng(args.sampling_seed, pd.name)
        try:
            vague, precise, vidx, pidx = _build_query_texts(
                args.archetype, inst, meta, rng)
        except Exception as e:
            print(f"  {pd.name}: query generation failed: {e}")
            n_skipped += 1
            continue

        old_vague = meta.get("vague_text", "")
        old_precise = meta.get("precise_text", "")
        old_vidx = meta.get("vague_template_idx")
        old_pidx = meta.get("precise_template_idx")

        unchanged = (vague == old_vague and precise == old_precise
                     and old_vidx == int(vidx)
                     and old_pidx == int(pidx))
        if unchanged:
            n_unchanged += 1
            continue

        if args.dry_run:
            print(f"  {pd.name}: would update "
                  f"(vidx {old_vidx} -> {vidx}, "
                  f"pidx {old_pidx} -> {pidx})")
            print(f"    OLD vague: {old_vague[:80]}...")
            print(f"    NEW vague: {vague[:80]}...")
            n_updated += 1
            continue

        meta["vague_text"] = vague
        meta["precise_text"] = precise
        meta["vague_template_idx"] = int(vidx)
        meta["precise_template_idx"] = int(pidx)
        qmeta_path.write_text(json.dumps(meta, indent=2))
        index_updates[pd.name] = {
            "vague_template_idx": int(vidx),
            "precise_template_idx": int(pidx),
        }
        print(f"  {pd.name}: updated (vidx={vidx}, pidx={pidx})")
        n_updated += 1

    # Sync the denormalised template indices in per-archetype index.json.
    if not args.dry_run and index_updates:
        if _update_per_archetype_index(archetype_dir, index_updates):
            print(f"\nUpdated {archetype_dir / 'index.json'} "
                  f"({len(index_updates)} pair record(s))")

    print(f"\nDone."
          f" updated={n_updated}, unchanged={n_unchanged}, "
          f"skipped={n_skipped}, total={len(pair_dirs)}"
          f"{' (DRY RUN — no files written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
