"""Run the optimization agent across an archetype dataset and report a
benchmark score.

For each pair in the dataset:
  1. Load instance / baseline / query metadata.
  2. Pick the appropriate text variant (vague or precise) based on the
     --query_type flag.
  3. Construct an ArchetypeQuery from metadata via the dispatch table in
     queries.ARCHETYPE_FACTORIES.
  4. Run the agent (default: paired multimodal + tools-only per pair,
     counterbalanced order; use --no_paired_modalities for a single modality
     and --no_visual for tools-only-only).
  5. Score the agent's final solution and save the per-pair result.

Aggregate metrics across all completed pairs are printed at the end and
saved to <results_dir>/aggregate.json.

The 2x2 experimental matrix is:
        modality (multimodal / tools_only)  ×  query_type (vague / precise)

Run:
    export OPENAI_API_KEY=...

    Full per-pair trajectories (tool calls, resolve_applied, stall nudges):
    python run_dataset.py --dataset_dir out/full_dataset/cluster \
        --trajectory_log_dir verbose_logs/cluster_vague_mm --query_type vague

    Compare multimodal vs tools-only aggregates (+ per-pair resolves/nudges):
    .venv/bin/python scripts/compare_modality_results.py \
        out/full_dataset/cluster/results_multimodal_vague/aggregate.json \
        out/full_dataset/cluster/results_tools_only_vague/aggregate.json \
        --label_a multimodal --label_b tools_only --enrich_from_pair_json \
        --trajectory_dir_a verbose_logs/cluster_vague_mm \
        --trajectory_dir_b verbose_logs/cluster_vague_tools_only

    Default paired benchmark (multimodal ↔ tools-only each pair, balanced order):
    python run_dataset.py --dataset_dir out/full_dataset/cluster \
        --query_type vague --model gpt-5.4 \
        --results_parent out/full_dataset/cluster/results_vague_gpt54_paired

    Legacy single-modality run (one invocation = one modality):
    python run_dataset.py --dataset_dir ... --no_paired_modalities --no_visual ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from instance import Instance, Solution
from queries import (
    ARCHETYPE_FACTORIES,
    ARCHETYPE_NAMES,
    guard_specs_to_public_config,
)
from agent_tools import Proposal, apply_proposal

# Per-pair JSON schema; bump when adding required result keys.
RESULT_SCHEMA_VERSION = 1
# Pickle bundle written alongside pair JSON for offline superscore / guard sweeps.
EXPLORE_REPLAY_SCHEMA_VERSION = 1


def _baseline_solution_digest(baseline: Solution) -> str:
    """Stable hash of baseline open/assign pattern and objective (drift check)."""
    h = hashlib.sha256()
    h.update(np.asarray(baseline.x).astype(np.int8).tobytes())
    h.update(np.asarray(baseline.y).astype(np.int8).tobytes())
    h.update(repr(float(baseline.objective)).encode("ascii"))
    return h.hexdigest()


def _merge_cache_result_defaults(result: Dict[str, Any]) -> None:
    """When loading --skip_existing JSON from an older schema, add stub keys."""
    if "result_schema_version" not in result:
        result["result_schema_version"] = 0
    if "termination_reason" not in result:
        result["termination_reason"] = "unknown_legacy"
    if "baseline_digest" not in result:
        result["baseline_digest"] = None
    if "guard_config" not in result:
        result["guard_config"] = None
    ss = result.get("superscore")
    if isinstance(ss, dict):
        ss.setdefault("explored_scores_full", None)
        ss.setdefault("explore_replay_path", None)


def _write_explore_replay_pickle(path: Path, bundle: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)


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

    # View-solution breakdown (multimodal only; empty for tools-only runs).
    view_events = [e for e in log
                   if e.get("event") == "tool_call" and e.get("name") == "view_solution"]
    view_purpose_counts: Dict[str, int] = {}
    for e in view_events:
        p = e.get("view_purpose") or "unspecified"
        view_purpose_counts[p] = view_purpose_counts.get(p, 0) + 1

    # Pull view stats from trajectory_summary if present (avoids re-counting).
    traj = next((e for e in log if e.get("event") == "trajectory_summary"), {})
    stall_from_events = sum(
        1 for e in log if e.get("event") == "primary_target_stall_nudge")
    stall_from_traj = traj.get("primary_stall_nudges_sent")
    primary_stall_nudges_sent = (
        int(stall_from_traj) if stall_from_traj is not None else stall_from_events
    )

    summary: Dict[str, Any] = {
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
        "primary_stall_nudges_sent": primary_stall_nudges_sent,
    }
    usage_events = [e for e in log if e.get("event") == "api_usage"]
    if usage_events:
        summary["n_model_calls"] = len(usage_events)
        summary["usage"] = {
            "prompt_tokens": int(sum(int(e.get("prompt_tokens", 0))
                                      for e in usage_events)),
            "completion_tokens": int(sum(int(e.get("completion_tokens", 0))
                                          for e in usage_events)),
            "total_tokens": int(sum(int(e.get("total_tokens", 0))
                                     for e in usage_events)),
            "cached_tokens": int(sum(int(e.get("cached_tokens", 0))
                                      for e in usage_events)),
            "reasoning_tokens": int(sum(int(e.get("reasoning_tokens", 0))
                                         for e in usage_events)),
        }
    if view_events:
        summary["n_view_solution"] = len(view_events)
        summary["view_purpose_counts"] = view_purpose_counts
        summary["had_baseline_view"] = traj.get("had_baseline_view", False)
        summary["pending_post_action_view_at_end"] = traj.get(
            "pending_post_action_view_at_end", False)
    return summary


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
            item["source"] != "baseline",  # prefer agent action over baseline on tie
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
    # Full production score dict per feasible explore (offline guard sweeps).
    best["all_scores_full"] = [
        _make_json_safe(
            {
                "explored_index": item["explored_index"],
                "source": item["source"],
                "iteration": item["iteration"],
                "resolve_index": item["resolve_index"],
                "n_resolves": item["n_resolves"],
                "n_local_edits": item["n_local_edits"],
                "proposal": item.get("proposal"),
                "score": item["score"],
            }
        )
        for item in scored
    ]
    return best, scored


def _summarise_superscore_diagnostics(
    best_explored: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive human-readable superscore diagnostics from explored scores."""
    explored = list(best_explored.get("all_scores") or [])
    n_explored_feasible = int(best_explored.get("n_feasible_explored", len(explored)))
    n_explored_valid = sum(1 for e in explored if bool(e.get("valid")))
    n_explored_primary_improved = sum(
        1 for e in explored if float(e.get("fraction_improved", 0.0)) > 1e-9
    )
    n_explored_primary_improved_but_invalid = sum(
        1
        for e in explored
        if float(e.get("fraction_improved", 0.0)) > 1e-9 and not bool(e.get("valid"))
    )

    selected_source = best_explored.get("source")
    selected_score = best_explored.get("score") or {}
    selected_valid = bool(selected_score.get("valid", False))
    selected_fraction_improved = float(selected_score.get("fraction_improved", 0.0))

    reason = "Selected by superscore ordering."
    if selected_source == "baseline":
        if (n_explored_primary_improved > 0
                and n_explored_primary_improved_but_invalid
                == n_explored_primary_improved):
            reason = ("Baseline selected: all explored primary-improving "
                      "candidates were invalid.")
        elif n_explored_primary_improved == 0:
            reason = "Baseline selected: no explored candidate improved the primary target."
        else:
            reason = ("Baseline selected: explored candidates did not beat baseline "
                      "under superscore tie-breaks.")
    elif not selected_valid:
        reason = ("Selected explored candidate is invalid because no valid explored "
                  "candidate outranked it.")
    elif selected_fraction_improved <= 1e-9:
        reason = ("Selected explored candidate is valid but did not improve the "
                  "primary target.")
    else:
        reason = ("Selected explored candidate is valid and improves the "
                  "primary target.")

    return {
        "selected_source": selected_source,
        "selected_valid": selected_valid,
        "n_explored_feasible": n_explored_feasible,
        "n_explored_valid": n_explored_valid,
        "n_explored_primary_improved": n_explored_primary_improved,
        "n_explored_primary_improved_but_invalid":
            n_explored_primary_improved_but_invalid,
        "selection_reason": reason,
    }


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
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    views_dir_override: Optional[Path] = None,
    trajectory_log_dir: Optional[Path] = None,
    system_prompt_suffix: Optional[str] = None,
    replay_bundle_path: Optional[Path] = None,
    marker_free_maps: bool = False,
) -> Dict[str, Any]:
    """Run the agent on one pair and return a result dict.

    Two orthogonal experimental knobs:
      enable_visual : True (multimodal) or False (tools-only).
      query_type    : "vague" (no entity reference) or "precise" (names
                      the offending site / region).
      marker_free_maps : when True with multimodal (enable_visual), every
                      map uses rendering_v2_no_markers only; ignored when
                      tools-only.
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

    if marker_free_maps and not enable_visual:
        raise ValueError(
            "marker_free_maps=True requires enable_visual=True (multimodal)."
        )
    mf = bool(enable_visual and marker_free_maps)

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
            print(f"  meta mean_NPI={meta.get('mean_npi_baseline', float('nan')):.2f}  "
                   f"max_NPI={meta.get('max_npi_baseline', float('nan')):.2f}  "
                   f"worst_site={meta.get('worst_catchment_site')}  "
                   f"(live primary = worst-k mean NPI; see query metadata)")
        if mf:
            print("  marker_free_maps: True (v2_no_markers only)")
        print(f"  baseline target value: {target_baseline:.3f}")

    log_path: Optional[str] = None
    if trajectory_log_dir is not None:
        trajectory_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(trajectory_log_dir / f"{pair_id}_trajectory.json")

    t0 = time.time()
    try:
        proposal_dict, final_solution, log, explored_solutions = run_agent(
            instance, baseline, text,
            query=query,
            annotation_polygons=None,  # vague/precise replaces annotation
            model=model,
            max_iters=max_iters,
            save_log_path=log_path,
            enable_visual=enable_visual,
            temperature=temperature,
            seed=seed,
            system_prompt_suffix=system_prompt_suffix,
            marker_free_maps=mf,
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
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "termination_reason": "error",
            "baseline_digest": _baseline_solution_digest(baseline),
            "guard_config": guard_specs_to_public_config(query.guards),
            "marker_free_maps": mf,
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
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "termination_reason": next(
                (e.get("termination_reason", "unknown")
                 for e in reversed(log)
                 if e.get("event") == "trajectory_summary"),
                "unknown",
            ),
            "baseline_digest": _baseline_solution_digest(baseline),
            "guard_config": guard_specs_to_public_config(query.guards),
            "marker_free_maps": mf,
        }

    # Superscore: score every feasible solution the agent explored and select
    # the best benchmark outcome. submit_proposal only ends the reasoning loop.
    best_explored, scored_rows = _select_best_explored_solution(
        query, instance, baseline, explored_solutions)
    new_solution = best_explored["solution"]
    score = best_explored["score"]
    superscore_diag = _summarise_superscore_diagnostics(best_explored)

    explore_replay_filename: Optional[str] = None
    if replay_bundle_path is not None:
        bundle: Dict[str, Any] = {
            "replay_schema_version": EXPLORE_REPLAY_SCHEMA_VERSION,
            "pair_id": pair_id,
            "query_id": query.query_id,
            "archetype": query.archetype,
            "query_type": query_type,
            "modality": modality,
            "baseline_digest": _baseline_solution_digest(baseline),
            "guard_config": guard_specs_to_public_config(query.guards),
            "rows": [
                {
                    "explored_index": r["explored_index"],
                    "source": r.get("source"),
                    "iteration": r.get("iteration"),
                    "resolve_index": r.get("resolve_index"),
                    "n_resolves": r.get("n_resolves", 0),
                    "n_local_edits": r.get("n_local_edits", 0),
                    "proposal": r.get("proposal"),
                    "solution": r["solution"],
                    "score": r["score"],
                }
                for r in scored_rows
            ],
        }
        _write_explore_replay_pickle(replay_bundle_path, bundle)
        explore_replay_filename = replay_bundle_path.name

    termination_reason = next(
        (e.get("termination_reason", "unknown")
         for e in reversed(log)
         if e.get("event") == "trajectory_summary"),
        "unknown",
    )

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
        print(f"  superscore diagnostics: selected_valid="
              f"{superscore_diag['selected_valid']}  "
              f"feasible/valid/improved/improved_invalid="
              f"{superscore_diag['n_explored_feasible']}/"
              f"{superscore_diag['n_explored_valid']}/"
              f"{superscore_diag['n_explored_primary_improved']}/"
              f"{superscore_diag['n_explored_primary_improved_but_invalid']}")
        print(f"  superscore reason: {superscore_diag['selection_reason']}")
        print(f"  target: {score['target_baseline']:.3f} -> "
               f"{score['target_response']:.3f}  "
               f"({score['fraction_improved']*100:.0f}% improved)  "
               f"valid={score['valid']}")
        print(f"  assignment distance: "
               f"{score['baseline_assignment_distance']:.0f} -> "
               f"{score['final_assignment_distance']:.0f}  "
               f"(delta {score['assignment_distance_delta']:+.0f})")

    if render_response:
        views_dir = Path(views_dir_override) if views_dir_override else (
            pair_dir / "views")
        views_dir.mkdir(parents=True, exist_ok=True)
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
        "run_config": {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "enable_visual": bool(enable_visual),
            "marker_free_maps": mf,
            "query_type": query_type,
            "max_iters": max_iters,
            "guard_config": guard_specs_to_public_config(query.guards),
        },
        "status": "completed",
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "termination_reason": termination_reason,
        "baseline_digest": _baseline_solution_digest(baseline),
        "guard_config": guard_specs_to_public_config(query.guards),
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
            "selected_valid": superscore_diag["selected_valid"],
            "n_explored_valid": superscore_diag["n_explored_valid"],
            "n_explored_primary_improved":
                superscore_diag["n_explored_primary_improved"],
            "n_explored_primary_improved_but_invalid":
                superscore_diag["n_explored_primary_improved_but_invalid"],
            "selection_reason": superscore_diag["selection_reason"],
            "explored_scores": best_explored["all_scores"],
            "explored_scores_full": best_explored["all_scores_full"],
            "explore_replay_path": explore_replay_filename,
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
    usage_logs = [
        (r.get("log_summary") or {}).get("usage")
        for r in completed
        if (r.get("log_summary") or {}).get("usage")
    ]

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
                "n_resolves": (r.get("log_summary") or {}).get("n_resolves", 0),
                "primary_stall_nudges_sent": (
                    (r.get("log_summary") or {}).get(
                        "primary_stall_nudges_sent", 0)),
                "n_submitted_explicit": (
                    (r.get("log_summary") or {}).get("n_submitted_explicit", 0)),
                "n_view_solution": (r.get("log_summary") or {}).get(
                    "n_view_solution", 0),
                "selected_source": (r.get("superscore") or {}).get(
                    "selected_source"),
                "n_explored_feasible": (r.get("superscore") or {}).get(
                    "n_feasible_explored"),
                "n_explored_valid": (r.get("superscore") or {}).get(
                    "n_explored_valid"),
                "n_explored_primary_improved": (r.get("superscore") or {}).get(
                    "n_explored_primary_improved"),
                "n_explored_primary_improved_but_invalid":
                    (r.get("superscore") or {}).get(
                        "n_explored_primary_improved_but_invalid"),
                "selection_reason": (r.get("superscore") or {}).get(
                    "selection_reason"),
                "total_tokens": (((r.get("log_summary") or {}).get("usage") or {}).get(
                    "total_tokens")),
                "prompt_tokens": (((r.get("log_summary") or {}).get("usage") or {}).get(
                    "prompt_tokens")),
                "completion_tokens":
                    (((r.get("log_summary") or {}).get("usage") or {}).get(
                        "completion_tokens")),
                "cached_tokens": (((r.get("log_summary") or {}).get("usage") or {}).get(
                    "cached_tokens")),
                "reasoning_tokens":
                    (((r.get("log_summary") or {}).get("usage") or {}).get(
                        "reasoning_tokens")),
                "elapsed_sec": r.get("elapsed_sec"),
            })
        pair_outcomes.append(po)

    out: Dict[str, Any] = {
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

    # Trajectory-style aggregates (from per-pair log_summary).
    ls_list = [
        (r.get("log_summary") or {}) for r in completed
        if r.get("log_summary")
    ]
    if ls_list:
        out["mean_n_resolves"] = float(
            np.mean([int(x.get("n_resolves", 0)) for x in ls_list]))
        out["median_n_resolves"] = float(
            np.median([int(x.get("n_resolves", 0)) for x in ls_list]))
        out["mean_primary_stall_nudges_sent"] = float(
            np.mean([int(x.get("primary_stall_nudges_sent", 0))
                     for x in ls_list]))
        out["total_primary_stall_nudges_sent"] = int(
            sum(int(x.get("primary_stall_nudges_sent", 0)) for x in ls_list))
        out["mean_n_view_solution"] = float(
            np.mean([int(x.get("n_view_solution", 0)) for x in ls_list]))
        out["mean_n_submitted_explicit"] = float(
            np.mean([int(x.get("n_submitted_explicit", 0)) for x in ls_list]))

    # Superscore selection diagnostics across completed pairs.
    selected_sources: Dict[str, int] = {}
    improved_invalid_total = 0
    for r in completed:
        ss = r.get("superscore") or {}
        src = str(ss.get("selected_source") or "unknown")
        selected_sources[src] = selected_sources.get(src, 0) + 1
        improved_invalid_total += int(
            ss.get("n_explored_primary_improved_but_invalid", 0))
    if completed:
        out["selected_source_counts"] = selected_sources
        out["total_explored_primary_improved_but_invalid"] = improved_invalid_total
    if usage_logs:
        out["mean_total_tokens"] = float(
            np.mean([u.get("total_tokens", 0) for u in usage_logs]))
        out["mean_prompt_tokens"] = float(
            np.mean([u.get("prompt_tokens", 0) for u in usage_logs]))
        out["mean_completion_tokens"] = float(
            np.mean([u.get("completion_tokens", 0) for u in usage_logs]))
        out["mean_cached_tokens"] = float(
            np.mean([u.get("cached_tokens", 0) for u in usage_logs]))
        out["mean_reasoning_tokens"] = float(
            np.mean([u.get("reasoning_tokens", 0) for u in usage_logs]))
        out["sum_total_tokens"] = int(
            sum(int(u.get("total_tokens", 0)) for u in usage_logs))
        out["sum_prompt_tokens"] = int(
            sum(int(u.get("prompt_tokens", 0)) for u in usage_logs))
        out["sum_completion_tokens"] = int(
            sum(int(u.get("completion_tokens", 0)) for u in usage_logs))
        out["sum_cached_tokens"] = int(
            sum(int(u.get("cached_tokens", 0)) for u in usage_logs))
        out["sum_reasoning_tokens"] = int(
            sum(int(u.get("reasoning_tokens", 0)) for u in usage_logs))

    return out


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
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    views_dir_override: Optional[Path] = None,
    trajectory_log_dir: Optional[Path] = None,
    system_prompt_suffix: Optional[str] = None,
    marker_free_maps: bool = False,
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
        replay_bundle_path = result_path.with_name(result_path.stem + ".explore_replay.pkl")
        if skip_existing and result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            _merge_cache_result_defaults(result)
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
                    temperature=temperature,
                    seed=seed,
                    views_dir_override=views_dir_override,
                    trajectory_log_dir=trajectory_log_dir,
                    system_prompt_suffix=system_prompt_suffix,
                    replay_bundle_path=replay_bundle_path,
                    marker_free_maps=marker_free_maps,
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
    agg["marker_free_maps"] = bool(enable_visual and marker_free_maps)
    with open(results_dir / "aggregate.json", "w") as f:
        json.dump(_make_json_safe(agg), f, indent=2)
    return agg


def _annotate_paired_modality_meta(
    result: Dict[str, Any],
    *,
    pair_index: int,
    modality_key: str,
    run_order_tokens: List[str],
) -> None:
    """Attach reproducibility metadata for default paired-modality runs."""
    if result.get("status") not in ("completed", "no_proposal", "error"):
        return
    result["paired_modality_meta"] = {
        "counterbalance_scheme": "pair_index_parity",
        "pair_index_in_subset": pair_index,
        "modality_for_this_result": modality_key,
        "run_order_this_pair": list(run_order_tokens),
    }


def run_archetype_dataset_paired(
    dataset_dir: Path,
    *,
    model: str,
    max_iters: int,
    query_type: str,
    no_render: bool,
    first_n: Optional[int],
    skip_existing: bool,
    results_dir_multimodal: Path,
    results_dir_tools_only: Path,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    views_dir_override: Optional[Path] = None,
    trajectory_log_dir: Optional[Path] = None,
    system_prompt_suffix: Optional[str] = None,
    marker_free_maps: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run multimodal and tools-only for each pair, counterbalancing order.

    Even pair index (0-based in the selected subset): multimodal first,
    then tools-only. Odd index: tools-only first, then multimodal.
    Mitigates simple run-order confounds (latency drift, cache warmup).
    """
    with open(dataset_dir / "index.json") as f:
        index = json.load(f)
    pairs = index["pairs"]
    if first_n:
        pairs = pairs[:first_n]

    results_dir_multimodal.mkdir(parents=True, exist_ok=True)
    results_dir_tools_only.mkdir(parents=True, exist_ok=True)
    traj_mm = (trajectory_log_dir / "multimodal"
               if trajectory_log_dir is not None else None)
    traj_to = (trajectory_log_dir / "tools_only"
               if trajectory_log_dir is not None else None)

    archetype_label = index.get("archetype", dataset_dir.name)
    if verbose:
        print(f"\n>>> PAIRED MODALITIES  archetype={archetype_label}  "
               f"|  pairs: {len(pairs)}")
        print(f"    multimodal results: {results_dir_multimodal}")
        print(f"    tools_only results: {results_dir_tools_only}")
        if trajectory_log_dir is not None:
            print(f"    trajectory logs:    {trajectory_log_dir}/"
                   f"multimodal | tools_only")

    all_mm: List[Dict[str, Any]] = []
    all_to: List[Dict[str, Any]] = []

    for pair_index, record in enumerate(pairs):
        pair_id = record["pair_id"]
        if pair_index % 2 == 0:
            sequence: List[Tuple[str, bool, Optional[Path]]] = [
                ("multimodal", True, traj_mm),
                ("tools_only", False, traj_to),
            ]
            order_tokens = ["multimodal", "tools_only"]
        else:
            sequence = [
                ("tools_only", False, traj_to),
                ("multimodal", True, traj_mm),
            ]
            order_tokens = ["tools_only", "multimodal"]

        if verbose:
            print(f"\n--- Pair {pair_id}  (counterbalance index {pair_index}: "
                   f"order {' → '.join(order_tokens)}) ---")

        pair_mm: Optional[Dict[str, Any]] = None
        pair_to: Optional[Dict[str, Any]] = None

        for modality_key, enable_vis, traj in sequence:
            pair_dir = dataset_dir / record["pair_dir"]
            if modality_key == "multimodal":
                result_path = results_dir_multimodal / f"{pair_id}.json"
                dest_list = "mm"
            else:
                result_path = results_dir_tools_only / f"{pair_id}.json"
                dest_list = "to"

            if traj is not None:
                traj.mkdir(parents=True, exist_ok=True)

            cached = bool(skip_existing and result_path.exists())
            if cached:
                with open(result_path) as f:
                    result = json.load(f)
                _merge_cache_result_defaults(result)
                if verbose:
                    print(f"    {modality_key}: (cached)")
            else:
                try:
                    replay_bundle_path = result_path.with_name(
                        result_path.stem + ".explore_replay.pkl")
                    result = run_one_pair(
                        pair_dir,
                        model=model,
                        max_iters=max_iters,
                        render_response=not no_render,
                        enable_visual=enable_vis,
                        query_type=query_type,
                        verbose=verbose,
                        temperature=temperature,
                        seed=seed,
                        views_dir_override=views_dir_override,
                        trajectory_log_dir=traj,
                        system_prompt_suffix=system_prompt_suffix,
                        replay_bundle_path=replay_bundle_path,
                        marker_free_maps=marker_free_maps if enable_vis else False,
                    )
                except KeyboardInterrupt:
                    print("\nInterrupted; partial paired results saved.")
                    raise
                except Exception as e:
                    print(f"  FATAL pair {pair_id} {modality_key}: {e}")
                    traceback.print_exc()
                    result = {
                        "pair_id": pair_id,
                        "modality": modality_key,
                        "status": "error",
                        "error": str(e)[:300],
                    }
                _annotate_paired_modality_meta(
                    result,
                    pair_index=pair_index,
                    modality_key=modality_key,
                    run_order_tokens=order_tokens,
                )
                with open(result_path, "w") as f:
                    json.dump(_make_json_safe(result), f, indent=2)

            if cached:
                _annotate_paired_modality_meta(
                    result,
                    pair_index=pair_index,
                    modality_key=modality_key,
                    run_order_tokens=order_tokens,
                )

            if dest_list == "mm":
                pair_mm = result
            else:
                pair_to = result

        all_mm.append(pair_mm if pair_mm is not None else {"pair_id": pair_id,
                                                           "status": "error"})
        all_to.append(pair_to if pair_to is not None else {"pair_id": pair_id,
                                                            "status": "error"})

    agg_mm = aggregate(all_mm)
    agg_mm["archetype"] = archetype_label
    agg_mm["modality"] = "multimodal"
    agg_mm["query_type"] = query_type
    agg_mm["model"] = model
    agg_mm["paired_modality_benchmark"] = True
    agg_mm["counterbalance_scheme"] = "pair_index_parity"
    agg_mm["marker_free_maps"] = bool(marker_free_maps)
    with open(results_dir_multimodal / "aggregate.json", "w") as f:
        json.dump(_make_json_safe(agg_mm), f, indent=2)

    agg_to = aggregate(all_to)
    agg_to["archetype"] = archetype_label
    agg_to["modality"] = "tools_only"
    agg_to["query_type"] = query_type
    agg_to["model"] = model
    agg_to["paired_modality_benchmark"] = True
    agg_to["counterbalance_scheme"] = "pair_index_parity"
    agg_to["marker_free_maps"] = False
    with open(results_dir_tools_only / "aggregate.json", "w") as f:
        json.dump(_make_json_safe(agg_to), f, indent=2)

    return agg_mm, agg_to


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
    if agg.get("selected_source_counts"):
        print(f"  superscore selected source  : {agg['selected_source_counts']}")
    if agg.get("total_explored_primary_improved_but_invalid") is not None:
        print("  explored improved but invalid: "
              f"{agg['total_explored_primary_improved_but_invalid']}")
    if agg.get("sum_total_tokens") is not None:
        print("  token usage (sum prompt/completion/cached): "
              f"{agg['sum_prompt_tokens']}/"
              f"{agg['sum_completion_tokens']}/"
              f"{agg['sum_cached_tokens']}  "
              f"(total {agg['sum_total_tokens']})")


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
                        help=("Override results directory. When using the "
                              "default paired benchmark, interpreted as a parent: "
                              "<dir>/multimodal and <dir>/tools_only (unless "
                              "overridden by --results_dir_multimodal / "
                              "--results_dir_tools_only). When using "
                              "--no_paired_modalities, this is the single "
                              "results directory for that run."))
    parser.add_argument("--model", default="gpt-5.4")
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


    parser.add_argument("--temperature", type=float, default=None,
                        help="LLM sampling temperature (e.g. 0.0). "
                             "Omit to use the model's API default.")
    parser.add_argument("--seed", type=int, default=None,
                        help="LLM seed for reproducibility (e.g. 42). "
                             "Omit to use the model's API default.")
    parser.set_defaults(use_paired_modality_benchmark=True)
    paired_group = parser.add_mutually_exclusive_group()
    paired_group.add_argument(
        "--no_paired_modalities",
        action="store_false",
        dest="use_paired_modality_benchmark",
        help=(
            "Legacy: run only one modality per invocation (--no_visual for "
            "tools-only). Default is paired multimodal + tools_only per pair "
            "with counterbalanced run order "
            "(even pair index → multimodal first)."
        ),
    )
    paired_group.add_argument(
        "--paired_modalities",
        action="store_true",
        dest="use_paired_modality_benchmark",
        help=(
            "Compatibility: paired modality benchmark is already the default. "
            "Use --no_paired_modalities to disable pairing."
        ),
    )
    parser.add_argument(
        "--results_parent",
        default=None,
        metavar="DIR",
        help=(
            "Default paired runs: write to DIR/multimodal and DIR/tools_only. "
            "Lower priority than --results_dir_multimodal / "
            "--results_dir_tools_only."
        ),
    )
    parser.add_argument(
        "--results_dir_multimodal",
        default=None,
        metavar="DIR",
        help=(
            "Default paired runs only: explicit multimodal results directory."
        ),
    )
    parser.add_argument(
        "--results_dir_tools_only",
        default=None,
        metavar="DIR",
        help=(
            "Default paired runs only: explicit tools_only results directory."
        ),
    )
    parser.add_argument(
        "--trajectory_log_dir",
        default=None,
        metavar="DIR",
        help=(
            "If set, write full agent trajectories (same schema as "
            "test_agent.py --save_log) to DIR/{pair_id}_trajectory.json "
            "per pair. When --dataset_dir is a multi-archetype root, logs "
            "go under DIR/<archetype_name>/ to avoid pair id collisions."
        ),
    )
    parser.add_argument(
        "--marker_free_maps",
        action="store_true",
        help=(
            "Multimodal-only: use rendering_v2_no_markers for the initial map, "
            "every view_solution call, and all auto-attached post-action maps. "
            "The view_solution tool accepts only layers=['v2_no_markers']. "
            "Use with --results_parent or explicit results dirs so you do not "
            "overwrite standard multimodal benchmarks."
        ),
    )
    args = parser.parse_args()

    paired = bool(args.use_paired_modality_benchmark)
    if paired and (
            bool(args.results_dir_multimodal) ^ bool(args.results_dir_tools_only)):
        print(
            "ERROR: default paired benchmark requires BOTH "
            "--results_dir_multimodal and --results_dir_tools_only if you "
            "set either one (or omit both and use --results_parent / "
            "--results_dir / dataset defaults).",
            file=sys.stderr,
        )
        sys.exit(1)

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: dataset_dir does not exist: {dataset_dir}",
               file=sys.stderr)
        sys.exit(1)

    if args.marker_free_maps and not paired and args.no_visual:
        print(
            "ERROR: --marker_free_maps requires multimodal runs "
            "(omit --no_visual with --no_paired_modalities).",
            file=sys.stderr,
        )
        sys.exit(1)

    if paired and args.no_visual:
        print(
            "WARNING: default paired benchmark ignores --no_visual (both "
            "modalities run per pair). Use --no_paired_modalities for "
            "tools-only-only.",
            flush=True,
        )

    enable_visual = not args.no_visual if not paired else True
    modality = "tools_only" if not enable_visual else "multimodal"
    mode_line = ("paired multimodal↔tools_only  |  Query type: "
                   f"{args.query_type}  |  Model: {args.model}"
                   if paired else
                   f"Modality: {modality}  |  Query type: {args.query_type}  "
                   f"|  Model: {args.model}")
    print(mode_line)
    if args.marker_free_maps:
        print("  marker_free_maps: True (multimodal branch uses v2_no_markers only)")
    traj_root: Optional[Path] = None
    if args.trajectory_log_dir:
        traj_root = Path(args.trajectory_log_dir).expanduser()
        print(f"Trajectory logs: {traj_root.resolve()}")

    if _is_single_archetype_dir(dataset_dir):
        if args.archetypes:
            print("WARNING: --archetypes is ignored when --dataset_dir is a "
                   "single-archetype directory.")

        def _paired_result_paths(ds: Path) -> Tuple[Path, Path]:
            if args.results_dir_multimodal and args.results_dir_tools_only:
                return (
                    Path(args.results_dir_multimodal),
                    Path(args.results_dir_tools_only),
                )
            if args.results_parent:
                rp = Path(args.results_parent).expanduser()
                return rp / "multimodal", rp / "tools_only"
            if args.results_dir:
                rd = Path(args.results_dir).expanduser()
                return rd / "multimodal", rd / "tools_only"
            mm = ds / f"results_multimodal_{args.query_type}"
            to_ = ds / f"results_tools_only_{args.query_type}"
            return mm, to_

        if paired:
            mm_dir, to_dir = _paired_result_paths(dataset_dir)
            agg_mm, agg_to = run_archetype_dataset_paired(
                dataset_dir,
                model=args.model, max_iters=args.max_iters,
                query_type=args.query_type,
                no_render=args.no_render, first_n=args.first_n,
                skip_existing=args.skip_existing,
                results_dir_multimodal=mm_dir,
                results_dir_tools_only=to_dir,
                temperature=args.temperature, seed=args.seed,
                trajectory_log_dir=traj_root,
                marker_free_maps=args.marker_free_maps,
            )
            _print_archetype_summary(agg_mm)
            _print_archetype_summary(agg_to)
            return

        agg = run_archetype_dataset(
            dataset_dir,
            model=args.model, max_iters=args.max_iters,
            enable_visual=enable_visual, query_type=args.query_type,
            no_render=args.no_render, first_n=args.first_n,
            skip_existing=args.skip_existing,
            results_dir_override=args.results_dir,
            temperature=args.temperature, seed=args.seed,
            trajectory_log_dir=traj_root,
            marker_free_maps=args.marker_free_maps,
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
    if args.results_dir or args.results_parent or args.results_dir_multimodal:
        print("WARNING: --results_dir / --results_parent / modality-specific "
               "result overrides are ignored for multi-archetype roots unless "
               "you use single-archetype --dataset_dir or run per archetype.")

    print(f"Multi-archetype root detected: {dataset_dir}")
    print(f"Archetypes to run: {[d.name for d in archetype_dirs]}")

    per_archetype: Dict[str, Dict[str, Any]] = {}
    for arch_dir in archetype_dirs:
        try:
            traj_dir = traj_root / arch_dir.name if traj_root else None
            if paired:
                mm_dir = arch_dir / f"results_multimodal_{args.query_type}"
                to_dir = arch_dir / f"results_tools_only_{args.query_type}"
                agg_mm, agg_to = run_archetype_dataset_paired(
                    arch_dir,
                    model=args.model, max_iters=args.max_iters,
                    query_type=args.query_type,
                    no_render=args.no_render, first_n=args.first_n,
                    skip_existing=args.skip_existing,
                    results_dir_multimodal=mm_dir,
                    results_dir_tools_only=to_dir,
                    temperature=args.temperature, seed=args.seed,
                    trajectory_log_dir=traj_dir,
                    marker_free_maps=args.marker_free_maps,
                )
                _print_archetype_summary(agg_mm)
                _print_archetype_summary(agg_to)
                per_archetype[arch_dir.name] = {
                    "multimodal": agg_mm,
                    "tools_only": agg_to,
                }
            else:
                agg = run_archetype_dataset(
                    arch_dir,
                    model=args.model, max_iters=args.max_iters,
                    enable_visual=enable_visual, query_type=args.query_type,
                    no_render=args.no_render, first_n=args.first_n,
                    skip_existing=args.skip_existing,
                    results_dir_override=None,
                    temperature=args.temperature, seed=args.seed,
                    trajectory_log_dir=traj_dir,
                    marker_free_maps=args.marker_free_maps,
                )
                _print_archetype_summary(agg)
                per_archetype[arch_dir.name] = agg
        except KeyboardInterrupt:
            print("\nInterrupted; aborting remaining archetypes.")
            break
        except Exception as e:
            print(f"  FATAL on archetype {arch_dir.name}: {e}")
            traceback.print_exc()
            continue

    rollup_modality = "paired" if paired else modality
    rollup_dir = dataset_dir / f"results_{rollup_modality}_{args.query_type}"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    rollup_summary: Dict[str, Any] = {
        "query_type": args.query_type,
        "model": args.model,
        "archetypes": per_archetype,
    }
    if args.marker_free_maps:
        rollup_summary["marker_free_maps"] = True
    if paired:
        rollup_summary["paired_modalities"] = True
    else:
        rollup_summary["modality"] = modality
    with open(rollup_dir / "summary.json", "w") as f:
        json.dump(_make_json_safe(rollup_summary), f, indent=2)
    if paired:
        print(
            "\n  (paired modalities: see per-archetype summaries above; "
            "cross-archetype rollup table skipped — nested mm/to per arch "
            "in summary.json)",
            flush=True,
        )
    else:
        _print_cross_archetype_rollup(per_archetype, modality, args.query_type,
                                       args.model)
    print(f"\n  multi-archetype summary: {rollup_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
