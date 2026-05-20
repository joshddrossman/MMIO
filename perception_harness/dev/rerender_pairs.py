"""Re-render existing pair PNGs without regenerating the dataset.

Walks `eval_set/full_dataset/<archetype>/pairs/` and, for each pair,
re-runs the named renderer modules against the existing instance.pkl
and baseline_solution.pkl, overwriting `views/<renderer>.png`. Useful
after a renderer-code change (palette tweak, outline addition, etc.)
when you don't want to re-run the gurobipy generation step.

Idempotent: re-runs produce identical output unless the renderer code
changes.

Run (default re-renders v2_no_markers across all archetypes present):
    python dev/rerender_pairs.py

Filter to specific archetypes:
    python dev/rerender_pairs.py --archetypes coverage_gap shape_niceness

Re-render multiple views at once:
    python dev/rerender_pairs.py --renderers v2 v2_no_markers
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))

from instance import Instance, Solution  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full_dataset_dir",
        default=str(HARNESS_ROOT / "eval_set" / "full_dataset"),
        help="Path to the full_dataset directory.")
    ap.add_argument("--renderers", nargs="+",
                    default=["v2_no_markers"],
                    help=("Renderer module name(s) to re-render. Each "
                          "is written to <pair_dir>/views/<name>.png. "
                          "Default: v2_no_markers."))
    ap.add_argument("--archetypes", nargs="+", default=None,
                    help=("Limit to these archetypes; default: every "
                          "archetype subdirectory found under "
                          "full_dataset/."))
    ap.add_argument("--dry_run", action="store_true",
                    help="List the pairs and renders but don't write.")
    args = ap.parse_args()

    full_root = Path(args.full_dataset_dir)
    if not full_root.is_dir():
        print(f"ERROR: {full_root} not found", file=sys.stderr)
        sys.exit(1)

    # Resolve renderer modules up front so any import error is reported
    # before we touch the filesystem.
    renderer_mods = {}
    for name in args.renderers:
        try:
            renderer_mods[name] = importlib.import_module(f"renderers.{name}")
        except ImportError as e:
            print(f"ERROR: cannot import renderer {name!r}: {e}",
                   file=sys.stderr)
            sys.exit(1)

    # Determine archetypes — explicit list or auto-discover.
    if args.archetypes:
        archetypes = list(args.archetypes)
    else:
        archetypes = sorted(
            d.name for d in full_root.iterdir()
            if d.is_dir() and (d / "pairs").is_dir()
        )

    if not archetypes:
        print(f"ERROR: no archetype subdirs found under {full_root}",
               file=sys.stderr)
        sys.exit(1)

    total_pairs_seen = 0
    total_renders_done = 0
    total_renders_failed = 0

    for archetype in archetypes:
        pairs_dir = full_root / archetype / "pairs"
        if not pairs_dir.is_dir():
            print(f"  {archetype}: no pairs/ subdir, skipping")
            continue

        pair_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())
        print(f"\n=== {archetype}: {len(pair_dirs)} pair(s) ===")

        for pd in pair_dirs:
            inst_path = pd / "instance.pkl"
            sol_path = pd / "baseline_solution.pkl"
            if not (inst_path.exists() and sol_path.exists()):
                print(f"  {pd.name}: missing instance/solution, skipping")
                continue

            try:
                inst = Instance.load(str(inst_path))
                sol = Solution.load(str(sol_path))
            except Exception as e:
                print(f"  {pd.name}: pickle load failed: {e}")
                continue

            views_dir = pd / "views"
            if not args.dry_run:
                views_dir.mkdir(parents=True, exist_ok=True)

            rendered_names = []
            for renderer_name, renderer in renderer_mods.items():
                try:
                    png = renderer.render(inst, sol)
                except Exception as e:
                    print(f"    {renderer_name}: render failed: {e}")
                    total_renders_failed += 1
                    continue
                out_path = views_dir / f"{renderer_name}.png"
                if not args.dry_run:
                    out_path.write_bytes(png)
                rendered_names.append(renderer_name)
                total_renders_done += 1
            total_pairs_seen += 1

            if rendered_names:
                marker = " (dry run)" if args.dry_run else ""
                print(f"  {pd.name}: {', '.join(rendered_names)}{marker}")

    print(f"\nDone. Pairs processed: {total_pairs_seen}, "
           f"renders {'planned' if args.dry_run else 'written'}: "
           f"{total_renders_done}"
           f"{f', failed: {total_renders_failed}' if total_renders_failed else ''}")


if __name__ == "__main__":
    main()
