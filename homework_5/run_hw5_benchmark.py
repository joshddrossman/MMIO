#!/usr/bin/env python3
"""Run the Homework 5 benchmark tasks; write JSON only under homework_5/outputs/.

Examples:
  cd /path/to/MMIO && python homework_5/run_hw5_benchmark.py --modality tools_only
  cd /path/to/MMIO && python homework_5/run_hw5_benchmark.py --modality multimodal --max-iters 20 --no-render

Requires OPENAI_API_KEY, gurobipy license, dependencies from requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Repo imports (run from repo root recommended)
_HW5 = Path(__file__).resolve().parent
_REPO = _HW5.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from run_dataset import aggregate, run_one_pair  # noqa: E402

TASKS_FILE = _HW5 / "benchmark_tasks.json"


def load_tasks():
    with open(TASKS_FILE) as f:
        cfg = json.load(f)
    return cfg["tasks"], cfg.get("model_default", "gpt-5-mini")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modality", choices=("tools_only", "multimodal"), required=True,
    )
    parser.add_argument("--model", default=None, help="default from benchmark_tasks.json")
    parser.add_argument("--max-iters", type=int, default=18)
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip map renders (faster; no PNG writes).",
    )
    parser.add_argument(
        "--experiment", default="hw5_run",
        help="Subfolder name under homework_5/outputs/runs/",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N tasks from benchmark_tasks.json (for A/B tests).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    tasks, default_model = load_tasks()
    if args.limit is not None:
        tasks = tasks[: args.limit]
    model = args.model or default_model
    enable_visual = args.modality == "multimodal"

    out_root = _HW5 / "outputs" / "runs" / args.experiment / args.modality
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = out_root / "per_task_json"
    traj_dir = out_root / "trajectories"
    views_root = out_root / "views"
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_render:
        views_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "modality": args.modality,
        "model": model,
        "max_iters": args.max_iters,
        "n_tasks": len(tasks),
        "started_unix": time.time(),
        "tasks": [],
    }

    all_results: List[Dict[str, Any]] = []

    for spec in tasks:
        arch = spec["archetype"]
        qtype = spec["query_type"]
        task_id = spec["task_id"]
        pair_rel = spec.get("pair_dir") or f"pairs/{spec.get('pair_id', '')}"
        pair_dir = _REPO / "full_dataset" / arch / pair_rel

        vdir = None
        if not args.no_render:
            vdir = views_root / task_id
            vdir.mkdir(parents=True, exist_ok=True)

        tdir = traj_dir / task_id
        tdir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {task_id} {arch}/{pair_rel} {qtype} ({args.modality}) ---", flush=True)
        t0 = time.time()
        result = run_one_pair(
            pair_dir,
            model=model,
            max_iters=args.max_iters,
            render_response=not args.no_render,
            enable_visual=enable_visual,
            query_type=qtype,
            verbose=True,
            temperature=args.temperature,
            seed=args.seed,
            views_dir_override=vdir,
            trajectory_log_dir=tdir,
        )
        elapsed = time.time() - t0
        result = dict(result)
        result["task_id"] = task_id
        result["hw5_category"] = spec.get("category")
        result["archetype"] = arch
        if result.get("elapsed_sec") is None:
            result["elapsed_sec"] = elapsed

        out_path = results_dir / f"{task_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        manifest["tasks"].append({
            "task_id": task_id,
            "result_json": str(out_path.relative_to(_HW5)),
            "wall_clock_sec": elapsed,
        })
        all_results.append(result)

    agg = aggregate(all_results)
    agg["homework"] = {
        "experiment": args.experiment,
        "modality": args.modality,
        "model": model,
    }
    agg_path = out_root / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nWrote aggregate: {agg_path}", flush=True)
    print(f"Fraction success (completed): {agg.get('fraction_success')}", flush=True)


if __name__ == "__main__":
    main()
