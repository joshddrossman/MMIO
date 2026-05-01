"""Run the optimization agent across an archetype dataset and report a
benchmark score.

For each pair in the dataset:
  1. Load instance / baseline / query metadata.
  2. Pick the appropriate text variant (vague or precise) based on the
     --query_type flag.
  3. Construct an ArchetypeQuery from metadata via the dispatch table in
     queries.ARCHETYPE_FACTORIES.
  4. Run the agent (modality controlled by --no_visual).
  5. Score the agent's final solution and save the per-pair result.

Aggregate metrics across all completed pairs are printed at the end and
saved to <results_dir>/aggregate.json.

The 2x2 experimental matrix is:
        modality (multimodal / tools_only)  ×  query_type (vague / precise)

Run:
    export OPENAI_API_KEY=...
    python run_dataset.py --dataset_dir out/full_dataset/contiguity \\
                          --query_type vague --no_visual
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from instance import Instance, Solution
from queries import ARCHETYPE_FACTORIES, ARCHETYPE_NAMES
from agent_tools import Proposal, apply_proposal


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _summarise_log(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    tools_used: Dict[str, int] = {}
    for e in log:
        if e.get("event") == "tool_call":
            n = e.get("name", "?")
            tools_used[n] = tools_used.get(n, 0) + 1
    resolve_events = [e for e in log
                        if e.get("event") == "resolve_applied"]
    local_edit_events = [e for e in log
                          if e.get("event") == "local_edit"]
    submit_events = [e for e in log
                       if e.get("event") == "solution_submitted"]
    return {
        "n_assistant_messages": sum(1 for e in log
                                      if e.get("event") == "assistant"),
        "n_resolves": len(resolve_events),
        "n_resolves_infeasible":
            sum(1 for e in resolve_events if not e.get("feasible")),
        "n_local_edits": len(local_edit_events),
        "local_edit_tool_counts": {
            tool: sum(1 for e in local_edit_events
                       if e.get("tool") == tool)
            for tool in {e.get("tool") for e in local_edit_events
                          if e.get("tool")}
        },
        "n_submitted_explicit": len(submit_events),
        "n_tool_calls": sum(v for v in tools_used.values()),
        "tool_call_counts": tools_used,
    }


def _select_best_explored_solution(
    query,
    instance: Instance,
    baseline: Solution,
    explored_solutions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score all feasible explored solutions and choose the benchmark best."""
    scored: List[Dict[str, Any]] = []
    for idx, entry in enumerate(explored_solutions):
        sol = entry["solution"]
        if not bool(sol.metadata.get("feasible", True)):
            continue
        score = query.score(instance, baseline, sol)
        scored.append({
            "explored_index": idx,
            "source": entry.get("source"),
            "iteration": entry.get("iteration"),
            "resolve_index": entry.get("resolve_index"),
            "n_resolves": entry.get("n_resolves", 0),
            "n_local_edits": entry.get("n_local_edits", 0),
            "proposal": entry.get("proposal"),
            "solution": sol,
            "score": score,
        })

    if not scored:
        score = query.score(instance, baseline, baseline)
        scored.append({
            "explored_index": 0,
            "source": "baseline_fallback",
            "iteration": -1,
            "resolve_index": None,
            "n_resolves": 0,
            "n_local_edits": 0,
            "proposal": None,
            "solution": baseline,
            "score": score,
        })

    def key(item: Dict[str, Any]):
        score = item["score"]
        return (
            bool(score.get("success", False)),
            bool(score.get("valid", False)),
            float(score.get("fraction_improved", 0.0)),
            float(score.get("raw_improvement", 0.0)),
            -float(score.get("assignment_distance_delta", 0.0)),
        )

    best = max(scored, key=key)
    best["n_feasible_explored"] = len(scored)
    best["all_scores"] = [
        {
            "explored_index": item["explored_index"],
            "source": item["source"],
            "iteration": item["iteration"],
            "resolve_index": item["resolve_index"],
            "n_resolves": item["n_resolves"],
            "n_local_edits": item["n_local_edits"],
            "target_response": item["score"]["target_response"],
            "fraction_improved": item["score"]["fraction_improved"],
            "valid": item["score"]["valid"],
            "success": item["score"].get("success"),
            "assignment_distance_delta":
                item["score"].get("assignment_distance_delta"),
        }
        for item in scored
    ]
    return best


def _pick_query_text(meta: Dict[str, Any], query_type: str) -> str:
    """Return the text variant for the requested query_type. Falls back
    to whatever's available if a variant is missing (e.g., legacy data)."""
    if query_type == "vague":
        return meta.get("vague_text") or meta.get("text") or ""
    if query_type == "precise":
        return meta.get("precise_text") or meta.get("text") or ""
    raise ValueError(f"Unknown query_type: {query_type!r}")


def run_one_pair(
    pair_dir: Path,
    model: str = "gpt-4o",
    max_iters: int = 25,
    render_response: bool = True,
    enable_visual: bool = True,
    query_type: str = "vague",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent on one pair and return a result dict.

    Two orthogonal experimental knobs:
      enable_visual : True (multimodal) or False (tools-only).
      query_type    : "vague" (no entity reference) or "precise" (names
                      the offending site / region).
    """
    from test_agent import run_agent
    from rendering import render_view

    pair_id = pair_dir.name
    instance = Instance.load(str(pair_dir / "instance.pkl"))
    baseline = Solution.load(str(pair_dir / "baseline_solution.pkl"))
    with open(pair_dir / "query_metadata.json") as f:
        meta = json.load(f)

    archetype = meta.get("archetype")
    factory = ARCHETYPE_FACTORIES.get(archetype)
    if factory is None:
        raise ValueError(
            f"Unknown archetype '{archetype}' in {pair_dir}/query_metadata.json. "
            f"Known: {list(ARCHETYPE_FACTORIES)}"
        )

    text = _pick_query_text(meta, query_type)
    query = factory(
        query_id=f"{archetype}_{pair_id}_{query_type}",
        text=text,
        metadata_dict=meta,
    )

    target_baseline = float(query.target_metric_fn(instance, baseline))
    modality = "multimodal" if enable_visual else "tools_only"

    if verbose:
        print(f"\n=== Pair {pair_id}  ({archetype}, {modality}, {query_type}) ===")
        print(f"  base_seed={meta.get('base_seed')}")
        if archetype == "cluster":
            print(f"  center={meta.get('cluster_center')}  "
                   f"size={meta.get('cluster_size')}  "
                   f"radius={meta.get('cluster_radius')}")
        elif archetype == "coverage_gap":
            print(f"  center={meta.get('coverage_gap_center')}  "
                   f"severity={meta.get('severity_achieved', float('nan')):.2f}  "
                   f"affected={len(meta.get('affected_precincts') or [])} precincts")
        elif archetype == "contiguity":
            n_split = meta.get('n_split_sites_baseline', 0)
            worst_j = meta.get('worst_culprit_site')
            print(f"  n_split_sites={n_split}  "
                   f"worst_site={worst_j}")
        elif archetype == "shape_niceness":
            print(f"  mean_NPI={meta.get('mean_npi_baseline', float('nan')):.2f}  "
                   f"max_NPI={meta.get('max_npi_baseline', float('nan')):.2f}  "
                   f"worst_site={meta.get('worst_catchment_site')}")
        print(f"  baseline target value: {target_baseline:.3f}")

    t0 = time.time()
    try:
        proposal_dict, final_solution, log, explored_solutions = run_agent(
            instance, baseline, text,
            annotation_polygons=None,  # vague/precise replaces annotation
            model=model,
            max_iters=max_iters,
            save_log_path=None,
            enable_visual=enable_visual,
        )
    except Exception as e:
        traceback.print_exc()
        return {
            "pair_id": pair_id,
            "modality": modality,
            "query_type": query_type,
            "status": "error",
            "error": str(e)[:300],
            "metadata": meta,
            "baseline_target": target_baseline,
            "elapsed_sec": time.time() - t0,
        }
    agent_time = time.time() - t0

    n_local_edits_total = sum(1 for e in log if e.get("event") == "local_edit")
    if proposal_dict is None and n_local_edits_total == 0:
        if verbose:
            print("  agent did not modify the solution.")
        return {
            "pair_id": pair_id,
            "modality": modality,
            "query_type": query_type,
            "status": "no_proposal",
            "metadata": meta,
            "baseline_target": target_baseline,
            "log_summary": _summarise_log(log),
            "elapsed_sec": agent_time,
        }

    # Superscore: score every feasible solution the agent explored and select
    # the best benchmark outcome. submit_proposal only ends the reasoning loop.
    best_explored = _select_best_explored_solution(
        query, instance, baseline, explored_solutions)
    new_solution = best_explored["solution"]
    score = best_explored["score"]

    if verbose:
        n_props = sum(1 for e in log if e.get("event") == "resolve_applied")
        n_infeasible = sum(1 for e in log if e.get("event") == "resolve_applied"
                                              and not e.get("feasible"))
        print(f"  iterations: {n_props} resolve(s) "
               f"({n_infeasible} infeasible), {n_local_edits_total} local edit(s)")
        if proposal_dict is not None:
            print(f"  final proposal: open={proposal_dict.get('force_open') or []}  "
                   f"close={proposal_dict.get('force_close') or []}  "
                   f"assign={proposal_dict.get('force_assign') or []}  "
                   f"weights={proposal_dict.get('precinct_weight_multipliers') or []}")
        print(f"  superscore selected: source={best_explored.get('source')}  "
              f"iteration={best_explored.get('iteration')}  "
              f"feasible explored={best_explored.get('n_feasible_explored')}")
        print(f"  target: {score['target_baseline']:.3f} -> "
               f"{score['target_response']:.3f}  "
               f"({score['fraction_improved']*100:.0f}% improved)  "
               f"valid={score['valid']}")
        print(f"  assignment distance: "
               f"{score['baseline_assignment_distance']:.0f} -> "
               f"{score['final_assignment_distance']:.0f}  "
               f"(delta {score['assignment_distance_delta']:+.0f})")

    if render_response:
        views_dir = pair_dir / "views"
        views_dir.mkdir(exist_ok=True)
        # All after-views match the baseline_text_only style — colored
        # service areas + closed candidates + opened sites + assignment
        # lines. Annotation polygon for archetypes that have a natural
        # anchor (cluster, coverage_gap).
        region_polys = None
        cx = cy = rr = None
        if archetype == "cluster":
            cx, cy = meta["cluster_center"]
            rr = float(meta["cluster_radius"])
        elif archetype == "coverage_gap":
            cx, cy = meta["coverage_gap_center"]
            rr = float(meta["coverage_gap_radius"])
        if rr is not None:
            ang = np.linspace(0, 2 * np.pi, 30, endpoint=False)
            poly = np.column_stack([cx + rr * np.cos(ang),
                                      cy + rr * np.sin(ang)])
            region_polys = [np.vstack([poly, poly[0:1]])]

        after_layers = ["closed_sites", "solution", "assignments"]
        render_view(
            instance, new_solution,
            layers=after_layers,
            region=region_polys,
            title=(f"Pair {pair_id} AFTER ({archetype}, {modality}, {query_type})  |  "
                   f"target {score['target_baseline']:.2f} -> {score['target_response']:.2f}  "
                   f"({score['fraction_improved']*100:.0f}% improved, "
                   f"valid={score['valid']})"),
            save_path=str(views_dir / f"after_{modality}_{query_type}.png"),
        )

    return {
        "pair_id": pair_id,
        "modality": modality,
        "query_type": query_type,
        "status": "completed",
        "metadata": meta,
        "proposal": proposal_dict,
        "score": score,
        "superscore": {
            "selected_explored_index": best_explored["explored_index"],
            "selected_source": best_explored["source"],
            "selected_iteration": best_explored["iteration"],
            "selected_resolve_index": best_explored["resolve_index"],
            "selected_n_resolves": best_explored["n_resolves"],
            "selected_n_local_edits": best_explored["n_local_edits"],
            "n_feasible_explored": best_explored["n_feasible_explored"],
            "explored_scores": best_explored["all_scores"],
        },
        "log_summary": _summarise_log(log),
        "elapsed_sec": agent_time,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-pair results into a benchmark summary."""
    completed = [r for r in results if r.get("status") == "completed"]
    fractions = [r["score"]["fraction_improved"] for r in completed]
    valid_flags = [r["score"]["valid"] for r in completed]

    n_substantial = sum(1 for f in fractions if f > 0.5)
    n_full_close = sum(1 for f in fractions if f >= 0.99)
    n_zero = sum(1 for f in fractions if f <= 1e-6)

    success_flags = [bool(r["score"].get("success"))
                      for r in completed if "success" in r.get("score", {})]
    n_with_success_criterion = len(success_flags)
    n_success = sum(success_flags)

    # Secondary metric: the response solution's voter-weighted total
    # assignment distance. Tracked per-pair AND aggregated across the
    # archetype so two equally-effective fixes can be compared on
    # solution quality.
    final_dists = [r["score"].get("final_assignment_distance")
                    for r in completed
                    if r["score"].get("final_assignment_distance") is not None]
    base_dists = [r["score"].get("baseline_assignment_distance")
                   for r in completed
                   if r["score"].get("baseline_assignment_distance") is not None]
    deltas = [r["score"].get("assignment_distance_delta")
               for r in completed
               if r["score"].get("assignment_distance_delta") is not None]

    pair_outcomes = []
    for r in results:
        po = {"pair_id": r.get("pair_id"), "status": r.get("status")}
        if r.get("status") == "completed":
            po.update({
                "fraction_improved": r["score"]["fraction_improved"],
                "valid": r["score"]["valid"],
                "success": r["score"].get("success"),
                "target_baseline": r["score"]["target_baseline"],
                "target_response": r["score"]["target_response"],
                "baseline_assignment_distance":
                    r["score"].get("baseline_assignment_distance"),
                "final_assignment_distance":
                    r["score"].get("final_assignment_distance"),
                "assignment_distance_delta":
                    r["score"].get("assignment_distance_delta"),
                "n_actions": (
                    (
                        len((r["proposal"] or {}).get("force_open") or [])
                        + len((r["proposal"] or {}).get("force_close") or [])
                        + len((r["proposal"] or {}).get("force_assign") or [])
                        + len((r["proposal"] or {})
                              .get("precinct_weight_multipliers") or [])
                    )
                    if r.get("proposal") is not None else 0
                ),
                "n_local_edits":
                    (r.get("log_summary") or {}).get("n_local_edits", 0),
                "elapsed_sec": r.get("elapsed_sec"),
            })
        pair_outcomes.append(po)

    return {
        "n_pairs": len(results),
        "n_completed": len(completed),
        "n_no_proposal": sum(1 for r in results
                              if r.get("status") == "no_proposal"),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "n_valid_responses": sum(1 for v in valid_flags if v),
        "n_substantial_improvement": n_substantial,
        "n_fully_closed_gap": n_full_close,
        "n_zero_improvement": n_zero,
        "mean_fraction_improved": float(np.mean(fractions)) if fractions else 0.0,
        "median_fraction_improved": float(np.median(fractions)) if fractions else 0.0,
        "fraction_valid": (sum(valid_flags) / len(valid_flags)) if valid_flags else 0.0,
        "n_with_success_criterion": n_with_success_criterion,
        "n_success": n_success,
        "fraction_success": (n_success / n_with_success_criterion
                              if n_with_success_criterion else None),
        # Secondary-metric aggregates: solution-quality picture across the cell.
        "mean_baseline_assignment_distance":
            float(np.mean(base_dists)) if base_dists else None,
        "mean_final_assignment_distance":
            float(np.mean(final_dists)) if final_dists else None,
        "mean_assignment_distance_delta":
            float(np.mean(deltas)) if deltas else None,
        "median_assignment_distance_delta":
            float(np.median(deltas)) if deltas else None,
        "pair_outcomes": pair_outcomes,
    }


def _is_single_archetype_dir(d: Path) -> bool:
    return ((d / "pairs").is_dir() and (d / "index.json").exists())


def _find_archetype_subdirs(root: Path,
                              archetype_filter: Optional[List[str]] = None
                              ) -> List[Path]:
    out = []
    for name in ARCHETYPE_NAMES:
        d = root / name
        if _is_single_archetype_dir(d):
            if archetype_filter is None or name in archetype_filter:
                out.append(d)
    return out


def run_archetype_dataset(
    dataset_dir: Path,
    *,
    model: str,
    max_iters: int,
    enable_visual: bool,
    query_type: str,
    no_render: bool,
    first_n: Optional[int],
    skip_existing: bool,
    results_dir_override: Optional[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the agent across one archetype dataset."""
    with open(dataset_dir / "index.json") as f:
        index = json.load(f)
    pairs = index["pairs"]
    if first_n:
        pairs = pairs[:first_n]

    modality = "tools_only" if not enable_visual else "multimodal"
    default_results_dir = dataset_dir / f"results_{modality}_{query_type}"
    results_dir = Path(results_dir_override or default_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    archetype_label = index.get("archetype", dataset_dir.name)
    if verbose:
        print(f"\n>>> Archetype: {archetype_label}  |  pairs: {len(pairs)}  "
               f"|  results: {results_dir}")

    all_results: List[Dict[str, Any]] = []
    for record in pairs:
        pair_id = record["pair_id"]
        result_path = results_dir / f"{pair_id}.json"
        if skip_existing and result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            if verbose:
                print(f"\n=== Pair {pair_id} (cached) ===")
        else:
            try:
                pair_dir = dataset_dir / record["pair_dir"]
                result = run_one_pair(
                    pair_dir,
                    model=model,
                    max_iters=max_iters,
                    render_response=not no_render,
                    enable_visual=enable_visual,
                    query_type=query_type,
                    verbose=verbose,
                )
            except KeyboardInterrupt:
                print("\nInterrupted; partial results saved.")
                break
            except Exception as e:
                print(f"  FATAL on pair {pair_id}: {e}")
                traceback.print_exc()
                result = {
                    "pair_id": pair_id,
                    "status": "error",
                    "error": str(e)[:300],
                }
            with open(result_path, "w") as f:
                json.dump(_make_json_safe(result), f, indent=2)
        all_results.append(result)

    agg = aggregate(all_results)
    agg["archetype"] = archetype_label
    agg["modality"] = modality
    agg["query_type"] = query_type
    agg["model"] = model
    with open(results_dir / "aggregate.json", "w") as f:
        json.dump(_make_json_safe(agg), f, indent=2)
    return agg


def _print_archetype_summary(agg: Dict[str, Any]):
    print("\n" + "=" * 64)
    print(f"BENCHMARK SUMMARY  archetype={agg.get('archetype', '?')}  "
           f"({agg['n_pairs']} pairs, model={agg.get('model')}, "
           f"modality={agg['modality']}, query_type={agg['query_type']})")
    print("=" * 64)
    print(f"  completed                   : {agg['n_completed']}")
    print(f"  no proposal submitted       : {agg['n_no_proposal']}")
    print(f"  error                       : {agg['n_error']}")
    print(f"  valid responses             : {agg['n_valid_responses']} "
           f"({agg['fraction_valid']*100:.0f}%)")
    print(f"  substantial improvement     : {agg['n_substantial_improvement']} "
           f"(fraction_improved > 0.5)")
    print(f"  fully closed gap            : {agg['n_fully_closed_gap']}")
    print(f"  zero improvement            : {agg['n_zero_improvement']}")
    print(f"  mean fraction_improved      : {agg['mean_fraction_improved']:.3f}")
    print(f"  median fraction_improved    : {agg['median_fraction_improved']:.3f}")
    if agg.get("n_with_success_criterion"):
        fs = agg.get("fraction_success", 0.0) or 0.0
        print(f"  binary success (archetype)  : {agg['n_success']}/"
               f"{agg['n_with_success_criterion']} ({fs * 100:.0f}%)")
    if agg.get("mean_assignment_distance_delta") is not None:
        print(f"  mean assignment-dist delta  : "
               f"{agg['mean_assignment_distance_delta']:+.0f} voter-km "
               f"(vs baseline {agg['mean_baseline_assignment_distance']:.0f})")


def _print_cross_archetype_rollup(per_archetype: Dict[str, Dict[str, Any]],
                                    modality: str, query_type: str, model: str):
    if not per_archetype:
        return
    print("\n" + "#" * 64)
    print(f"CROSS-ARCHETYPE ROLLUP  ({len(per_archetype)} archetypes, "
           f"model={model}, modality={modality}, query_type={query_type})")
    print("#" * 64)
    header = (f"  {'archetype':<18} {'n_pairs':>8} {'n_complete':>11} "
              f"{'mean_frac':>10} {'success':>10}")
    print(header)
    total_pairs = total_completed = total_success = total_with_crit = 0
    fracs = []
    for arch, agg in per_archetype.items():
        np_ = agg.get("n_pairs", 0)
        nc = agg.get("n_completed", 0)
        mf = agg.get("mean_fraction_improved", 0.0)
        ns = agg.get("n_success", 0)
        nwc = agg.get("n_with_success_criterion", 0)
        succ_str = (f"{ns}/{nwc} ({(ns / nwc) * 100:.0f}%)"
                     if nwc else "n/a")
        print(f"  {arch:<18} {np_:>8} {nc:>11} {mf:>10.3f} {succ_str:>10}")
        total_pairs += np_
        total_completed += nc
        total_success += ns
        total_with_crit += nwc
        if nc > 0:
            fracs.extend([mf] * nc)

    overall_mean = (sum(fracs) / len(fracs)) if fracs else 0.0
    overall_succ = ((total_success / total_with_crit)
                     if total_with_crit else 0.0)
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<18} {total_pairs:>8} {total_completed:>11} "
           f"{overall_mean:>10.3f} "
           f"{f'{total_success}/{total_with_crit} ({overall_succ * 100:.0f}%)':>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", default="out/full_dataset/contiguity",
        help=("Path to a single-archetype dataset (with index.json + pairs/) "
               "OR a multi-archetype root containing per-archetype "
               "subdirectories. run_dataset.py auto-detects."),
    )
    parser.add_argument(
        "--archetypes", nargs="*", default=None,
        choices=ARCHETYPE_NAMES,
        help=("When --dataset_dir is a multi-archetype root, restrict to "
               "this subset of archetypes. Default: run all that are "
               "present."),
    )
    parser.add_argument("--results_dir", default=None,
                        help="Override per-archetype results directory.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--max_iters", type=int, default=25)
    parser.add_argument("--first_n", type=int, default=None,
                        help="Only run the first N pairs (per archetype).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip pairs whose result JSON already exists.")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip rendering after-view per pair.")
    parser.add_argument("--no_visual", action="store_true",
                        help=("Disable view_solution and the initial map for "
                              "the agent. The agent operates in tools-only "
                              "mode."))
    parser.add_argument("--query_type", default="vague",
                        choices=["vague", "precise"],
                        help=("'vague': stakeholder-style critique with no "
                              "specific entity reference. 'precise': names "
                              "the offending site / region. The 2x2 matrix "
                              "is modality x query_type."))
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset_dir does not exist: {dataset_dir}",
               file=sys.stderr)
        sys.exit(1)

    enable_visual = not args.no_visual
    modality = "tools_only" if not enable_visual else "multimodal"
    print(f"Modality: {modality}  |  Query type: {args.query_type}  "
           f"|  Model: {args.model}")

    if _is_single_archetype_dir(dataset_dir):
        if args.archetypes:
            print("WARNING: --archetypes is ignored when --dataset_dir is a "
                   "single-archetype directory.")
        agg = run_archetype_dataset(
            dataset_dir,
            model=args.model, max_iters=args.max_iters,
            enable_visual=enable_visual, query_type=args.query_type,
            no_render=args.no_render, first_n=args.first_n,
            skip_existing=args.skip_existing,
            results_dir_override=args.results_dir,
        )
        _print_archetype_summary(agg)
        return

    archetype_dirs = _find_archetype_subdirs(dataset_dir, args.archetypes)
    if not archetype_dirs:
        if args.archetypes:
            print(f"ERROR: none of the requested archetypes "
                   f"({args.archetypes}) were found under {dataset_dir}.",
                   file=sys.stderr)
        else:
            print(f"ERROR: {dataset_dir} contains neither pairs/ "
                   f"(single-archetype) nor any per-archetype "
                   f"subdirectories.", file=sys.stderr)
        sys.exit(1)
    if args.results_dir:
        print("WARNING: --results_dir is ignored when iterating over a "
               "multi-archetype root; each archetype writes to its own "
               "results subdirectory.")

    print(f"Multi-archetype root detected: {dataset_dir}")
    print(f"Archetypes to run: {[d.name for d in archetype_dirs]}")

    per_archetype: Dict[str, Dict[str, Any]] = {}
    for arch_dir in archetype_dirs:
        try:
            agg = run_archetype_dataset(
                arch_dir,
                model=args.model, max_iters=args.max_iters,
                enable_visual=enable_visual, query_type=args.query_type,
                no_render=args.no_render, first_n=args.first_n,
                skip_existing=args.skip_existing,
                results_dir_override=None,
            )
        except KeyboardInterrupt:
            print("\nInterrupted; aborting remaining archetypes.")
            break
        except Exception as e:
            print(f"  FATAL on archetype {arch_dir.name}: {e}")
            traceback.print_exc()
            continue
        _print_archetype_summary(agg)
        per_archetype[arch_dir.name] = agg

    rollup_dir = dataset_dir / f"results_{modality}_{args.query_type}"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    rollup_summary = {
        "modality": modality, "query_type": args.query_type,
        "model": args.model,
        "archetypes": per_archetype,
    }
    with open(rollup_dir / "summary.json", "w") as f:
        json.dump(_make_json_safe(rollup_summary), f, indent=2)
    _print_cross_archetype_rollup(per_archetype, modality, args.query_type,
                                    args.model)
    print(f"\n  cross-archetype summary: {rollup_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
