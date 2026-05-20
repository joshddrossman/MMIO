"""Single-instance orchestrator: generate one archetype-engineered instance,
solve the baseline, render layered views, save artifacts.

For multi-pair dataset generation, use dataset_generator.py or
generate_full_dataset.py instead.

Run:
    python main.py --archetype cluster                # default
    python main.py --archetype coverage_gap --out_dir out/cg
    python main.py --archetype contiguity
    python main.py --archetype shape_niceness
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from instance import Instance, Solution
from rendering import render_view, render_all_pair_views


ARCHETYPE_GENERATORS = {
    "cluster":         "generate_cluster_instance",
    "coverage_gap":    "generate_coverage_gap_instance",
    "contiguity":      "generate_contiguity_instance",
    "shape_niceness":  "generate_shape_niceness_instance",
}


def _save_instance(out_dir, instance, solution, metadata, archetype):
    """Save instance + baseline + JSON-safe metadata."""
    os.makedirs(os.path.join(out_dir, 'views'), exist_ok=True)
    instance.save(os.path.join(out_dir, 'instance.pkl'))
    solution.save(os.path.join(out_dir, 'baseline_solution.pkl'))

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        return obj

    with open(os.path.join(out_dir, 'query_metadata.json'), 'w') as f:
        json.dump(make_json_safe(metadata), f, indent=2)
    print(f'\nSaved {archetype} instance + metadata to {out_dir}/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archetype', default='cluster',
                        choices=list(ARCHETYPE_GENERATORS.keys()),
                        help='Which archetype to engineer.')
    parser.add_argument('--out_dir', default=None,
                        help='Default: out/<archetype>_single.')
    parser.add_argument('--base_seed', type=int, default=1)
    parser.add_argument('--max_attempts', type=int, default=15)
    parser.add_argument('--no_views', action='store_true',
                        help='Skip rendering views.')
    args = parser.parse_args()

    out_dir = args.out_dir or f'out/{args.archetype}_single'
    os.makedirs(out_dir, exist_ok=True)

    # Lazy import: generation requires gurobipy at runtime.
    import generation
    gen_fn = getattr(generation, ARCHETYPE_GENERATORS[args.archetype])

    print(f'Generating {args.archetype} instance...')
    instance, solution, metadata = gen_fn(
        base_seed=args.base_seed,
        max_attempts=args.max_attempts,
        verbose=True,
    )

    _save_instance(out_dir, instance, solution, metadata, args.archetype)

    if args.no_views:
        return
    print('\nRendering views...')
    render_all_pair_views(
        instance, solution,
        os.path.join(out_dir, 'views'),
        title_prefix=f'{args.archetype} — ',
    )
    print(f'\nViews saved to: {os.path.join(out_dir, "views")}/')


if __name__ == '__main__':
    main()
