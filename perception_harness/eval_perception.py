"""Run the perception evaluation cross-product.

Iterates renderer × prompt × model × ground_truth_row, calls the model on
each rendered map, parses + scores the response, and writes a flat per-row
CSV plus a small aggregate summary.

Run (oracle baseline — no API key needed, sanity check only):
    python eval_perception.py --models tool_oracle --first_n 5

Run (real VLM):
    python eval_perception.py \\
        --renderers base \\
        --prompts zero_shot \\
        --models openai_vlm:gpt-4o \\
        --archetypes contiguity \\
        --difficulties easy med hard \\
        --first_n 5

Outputs land under results/<run_name>/.
"""
from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import importlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make instance_generator importable so Instance/Solution pickles load.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))
sys.path.insert(0, str(HERE))

from instance import Instance, Solution  # noqa: E402


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def _load_pair(source_dir: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load (instance, solution, meta) from a pair directory.

    Reads `query_metadata.json` (the unified-dataset filename, also
    what partner's run_dataset.py uses) when present, falling back
    to `meta.json` for backward compatibility with eval sets generated
    before the unified-layout refactor.
    """
    inst = Instance.load(str(source_dir / "instance.pkl"))
    sol = Solution.load(str(source_dir / "baseline_solution.pkl"))
    qmeta = source_dir / "query_metadata.json"
    legacy_meta = source_dir / "meta.json"
    meta_path = qmeta if qmeta.exists() else legacy_meta
    with open(meta_path) as f:
        meta = json.load(f)
    return inst, sol, meta


def _parse_model_spec(spec: str) -> Tuple[str, str]:
    """'openai_vlm:gpt-4o' -> ('openai_vlm', 'gpt-4o')."""
    mod, _, model_id = spec.partition(":")
    return mod, (model_id or "default")


def _import(kind: str, name: str):
    """Import a renderer / prompt / task / model module by name."""
    return importlib.import_module(f"{kind}.{name}")


def _render_or_die(renderer, instance, solution) -> Optional[bytes]:
    try:
        return renderer.render(instance, solution)
    except Exception as e:
        print(f"  RENDER FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", default="eval_set/ground_truth.csv",
                    help="Path to the eval-set ground truth CSV.")
    ap.add_argument("--renderers", nargs="+", default=["v2"])
    ap.add_argument("--prompts",   nargs="+", default=["zero_shot"])
    ap.add_argument("--models",    nargs="+", default=["tool_oracle"],
                    help=("Provider:model pairs, e.g. 'openai_vlm:gpt-4o'. "
                          "Use 'tool_oracle' for the analytic baseline."))
    ap.add_argument("--archetypes", nargs="+", default=None,
                    help="Filter rows by archetype (default: all).")
    ap.add_argument("--difficulties", nargs="+", default=None,
                    choices=["easy", "med", "hard"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Filter rows by task name (default: all).")
    ap.add_argument("--first_n", type=int, default=None,
                    help="Limit to the first N filtered rows.")
    ap.add_argument("--out_dir", default=None,
                    help="Default: results/<UTC-timestamp>/.")
    args = ap.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = HERE / gt_path
    if not gt_path.exists():
        print(f"ERROR: ground truth CSV not found at {gt_path}.", file=sys.stderr)
        print("       Run: python eval_set/build_eval_set.py", file=sys.stderr)
        sys.exit(1)

    with open(gt_path) as f:
        rows = list(csv.DictReader(f))

    def _keep(r: Dict[str, str]) -> bool:
        if args.archetypes and r["archetype"] not in args.archetypes:
            return False
        if args.difficulties and r["difficulty"] not in args.difficulties:
            return False
        if args.tasks and r["task"] not in args.tasks:
            return False
        return True

    rows = [r for r in rows if _keep(r)]
    if args.first_n:
        rows = rows[:args.first_n]

    if not rows:
        print("No ground-truth rows match the filters.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "results" /
        _dt.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the run config for traceability.
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "renderers": args.renderers,
            "prompts": args.prompts,
            "models": args.models,
            "archetypes": args.archetypes,
            "difficulties": args.difficulties,
            "tasks": args.tasks,
            "first_n": args.first_n,
            "n_rows": len(rows),
            "ground_truth": str(gt_path),
        }, f, indent=2)

    # Resolve modules once.
    renderers = {n: _import("renderers", n) for n in args.renderers}
    prompts = {n: _import("prompts", n) for n in args.prompts}
    models: Dict[str, Tuple[Any, str]] = {}
    for spec in args.models:
        mod_name, model_id = _parse_model_spec(spec)
        models[spec] = (_import("models", mod_name), model_id)

    out_csv = out_dir / "per_question.csv"
    out_jsonl = out_dir / "per_question.jsonl"
    fieldnames = [
        "row_id",
        "pair_id", "archetype", "difficulty", "task",
        "renderer", "prompt", "model",
        "raw_response", "parsed_answer",
        "score", "secondary_scores",
        "tool_calls", "tool_call_log",
        "elapsed_sec", "error",
    ]

    # Per-cell aggregate counters.
    agg: Dict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)

    n_total = (len(rows) * len(renderers)
                * len(prompts) * len(models))
    n_done = 0
    print(f"\nRunning {n_total} evaluations "
           f"({len(rows)} rows × {len(renderers)} renderer(s) × "
           f"{len(prompts)} prompt(s) × {len(models)} model(s))")
    print(f"Writing to: {out_csv}")
    print(f"        + : {out_jsonl}  (full untruncated records)")
    print()

    # Two parallel result files:
    #   per_question.csv   — pandas-friendly headline summary; long
    #                         fields are TRUNCATED to ~1500 chars for
    #                         spreadsheet legibility, so this file is
    #                         best for quick pivots, not for full
    #                         post-analysis of long model responses.
    #   per_question.jsonl — same row set, but with raw_response,
    #                         parsed_answer, secondary_scores, and
    #                         tool_call_log stored UNTRUNCATED and as
    #                         native JSON (dicts/lists), so post-
    #                         analysis has the complete model output.
    # `row_id` (sequential int starting at 0) appears in both files and
    # links them: row N in CSV ↔ JSON object on line N (0-indexed) in
    # JSONL.
    row_id = 0

    with open(out_csv, "w", newline="") as csv_f, \
         open(out_jsonl, "w", encoding="utf-8") as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        # Cache renders per (pair_id, renderer_name) — re-rendering the same
        # map for describe + identify (or for multiple prompts) is wasteful.
        render_cache: Dict[Tuple[str, str], Optional[bytes]] = {}

        for r in rows:
            archetype = r["archetype"]
            task_name = r["task"]
            try:
                task_mod = _import("tasks", archetype)
            except ModuleNotFoundError:
                print(f"  SKIP {r['pair_id']}/{task_name} — no task module "
                       f"for archetype '{archetype}'.")
                continue

            source_dir = HERE / r["source_dir"]
            try:
                instance, solution, meta = _load_pair(source_dir)
            except Exception as e:
                print(f"  LOAD FAILED for {source_dir}: {e}", file=sys.stderr)
                continue

            ground_truth = json.loads(r["answer_json"])

            for renderer_name, renderer in renderers.items():
                cache_key = (r["pair_id"], renderer_name)
                if cache_key not in render_cache:
                    png_bytes = _render_or_die(
                        renderer, instance, solution)
                    # Persist the rendered map to disk so it can be
                    # inspected after the run (debugging mismatches between
                    # what the VLM reported and what was actually visible).
                    # Saved next to the source pickles, mirroring the
                    # views/ layout instance_generator/ already uses.
                    if png_bytes is not None:
                        save_path = source_dir / "views" / f"{renderer_name}.png"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_bytes(png_bytes)
                    render_cache[cache_key] = png_bytes
                png_bytes = render_cache[cache_key]
                image_b64 = (base64.b64encode(png_bytes).decode("ascii")
                             if png_bytes else None)

                # Renderer capability flags — defaults are conservative
                # (assume markers visible, lines absent). Read once per
                # renderer; combined with the prompt's USES_IMAGE below
                # to form per-cell view_info.
                renderer_has_markers = bool(getattr(
                    renderer, "HAS_SITE_MARKERS", True))
                renderer_has_candidates = bool(getattr(
                    renderer, "HAS_CANDIDATE_MARKERS", True))
                renderer_has_lines = bool(getattr(
                    renderer, "HAS_ASSIGNMENT_LINES", False))

                for prompt_name, prompt in prompts.items():
                    # Prompt-side capability — defaults to True so legacy
                    # prompts continue to receive the image.
                    uses_image = bool(getattr(prompt, "USES_IMAGE", True))

                    # view_info is the runtime fact every task / prompt
                    # consults to pick text variants. has_visual gates
                    # the marker / lines flags so a tools-only prompt
                    # paired with a marker-rich renderer still reports
                    # has_site_markers=False (the model never sees them).
                    view_info = {
                        "has_visual": uses_image,
                        "has_site_markers":
                            uses_image and renderer_has_markers,
                        "has_candidate_markers":
                            uses_image and renderer_has_candidates,
                        "has_assignment_lines":
                            uses_image and renderer_has_lines,
                        "renderer_name": renderer_name,
                        "prompt_name": prompt_name,
                    }

                    # Per-task validity gate. Some tasks reject certain
                    # (renderer, prompt) pairs — e.g. cluster requires
                    # markers when a map is shown. Tasks that don't
                    # define is_valid_view default to "always valid."
                    is_valid_fn = getattr(task_mod, "is_valid_view", None)
                    if is_valid_fn is not None and not is_valid_fn(view_info):
                        for spec in models:
                            n_done += 1
                            base = {
                                "row_id": row_id,
                                "pair_id": r["pair_id"],
                                "archetype": archetype,
                                "difficulty": r["difficulty"],
                                "task": task_name,
                                "renderer": renderer_name,
                                "prompt": prompt_name,
                                "model": spec,
                                "raw_response": "",
                                "tool_calls": 0,
                                "elapsed_sec": 0.0,
                                "error": "invalid_view_for_task",
                            }
                            writer.writerow({
                                **base,
                                "parsed_answer": "",
                                "score": "",
                                "secondary_scores": "",
                                "tool_call_log": "",
                            })
                            jsonl_f.write(json.dumps({
                                **base,
                                "parsed_answer": None,
                                "score": None,
                                "secondary_scores": {},
                                "tool_call_log": [],
                            }) + "\n")
                            row_id += 1
                        continue

                    # Task question depends on view_info — must compute
                    # per (renderer, prompt) cell, not once per row.
                    try:
                        question = task_mod.format_question(
                            meta, task_name, view_info=view_info)
                    except TypeError:
                        # Legacy task that doesn't accept view_info —
                        # fall back to the old signature.
                        question = task_mod.format_question(meta, task_name)

                    if uses_image and image_b64 is None:
                        # Render failed AND the prompt would have used
                        # the image. Write a row marking the failure.
                        for spec in models:
                            n_done += 1
                            csv_row = {
                                "row_id": row_id,
                                "pair_id": r["pair_id"],
                                "archetype": archetype,
                                "difficulty": r["difficulty"],
                                "task": task_name,
                                "renderer": renderer_name,
                                "prompt": prompt_name,
                                "model": spec,
                                "raw_response": "",
                                "parsed_answer": "",
                                "score": "",
                                "secondary_scores": "",
                                "tool_calls": 0,
                                "tool_call_log": "",
                                "elapsed_sec": 0.0,
                                "error": "render_failed",
                            }
                            writer.writerow(csv_row)
                            jsonl_record = {
                                "row_id": row_id,
                                "pair_id": r["pair_id"],
                                "archetype": archetype,
                                "difficulty": r["difficulty"],
                                "task": task_name,
                                "renderer": renderer_name,
                                "prompt": prompt_name,
                                "model": spec,
                                "raw_response": "",
                                "parsed_answer": None,
                                "score": None,
                                "secondary_scores": {},
                                "tool_calls": 0,
                                "tool_call_log": [],
                                "elapsed_sec": 0.0,
                                "error": "render_failed",
                            }
                            jsonl_f.write(json.dumps(jsonl_record) + "\n")
                            row_id += 1
                        continue

                    # Pass view_info to the prompt builder; legacy
                    # prompts without the kwarg fall back gracefully.
                    try:
                        messages = prompt.build(
                            task_name=task_name,
                            question=question,
                            image_b64=image_b64 or "",
                            view_info=view_info,
                        )
                    except TypeError:
                        messages = prompt.build(
                            task_name=task_name,
                            question=question,
                            image_b64=image_b64 or "",
                        )

                    # If this prompt module advertises tools, capture
                    # the schema list and the dispatcher once per prompt.
                    # Adapters that opt in via SUPPORTS_TOOLS=True will
                    # run the tool-call loop; others ignore them.
                    prompt_tools = (prompt.get_tools()
                                    if hasattr(prompt, "get_tools")
                                    else None)
                    prompt_dispatcher = getattr(
                        prompt, "dispatch_tool", None)

                    for spec, (model, model_id) in models.items():
                        n_done += 1
                        is_oracle = hasattr(model, "compute_answer")
                        supports_tools = getattr(
                            model, "SUPPORTS_TOOLS", False)
                        # Per-call tool log; the tool-using adapter
                        # appends one entry per tool_call it dispatches.
                        tool_log: List[Dict[str, Any]] = []
                        t0 = time.time()
                        err = ""
                        try:
                            if is_oracle:
                                raw_result = model.compute_answer(
                                    instance=instance, solution=solution,
                                    archetype=archetype, task=task_name)
                            elif (supports_tools and prompt_tools
                                    and prompt_dispatcher is not None):
                                raw_result = model.query(
                                    messages,
                                    model_id=model_id,
                                    tools=prompt_tools,
                                    tool_dispatcher=prompt_dispatcher,
                                    dispatcher_kwargs={
                                        "instance": instance,
                                        "solution": solution,
                                        "_tool_log": tool_log,
                                    },
                                )
                            else:
                                raw_result = model.query(messages,
                                                         model_id=model_id)
                        except Exception as e:
                            raw_result = ""
                            err = f"{type(e).__name__}: {e}"
                            traceback.print_exc()
                        dt = time.time() - t0

                        # Adapters may return either a plain str (legacy
                        # adapters, the analytic oracle) or a dict with
                        # "text" + API metadata (openai_vlm[_tools]).
                        # Normalise so the rest of the loop is uniform.
                        if isinstance(raw_result, dict):
                            raw = raw_result.get("text", "") or ""
                            api_meta = {k: v for k, v in raw_result.items()
                                         if k != "text"}
                        else:
                            raw = raw_result or ""
                            api_meta = {}

                        try:
                            parsed = task_mod.parse_response(raw, task_name)
                            score_val = float(task_mod.score(
                                parsed, ground_truth, task_name))
                        except Exception as e:
                            parsed = {}
                            score_val = float("nan")
                            err = err or f"score_error: {e}"

                        # Secondary scores: optional per-task contract.
                        # Tasks that don't define `secondary_scores` leave
                        # the column empty.
                        secondary: Dict[str, Any] = {}
                        if hasattr(task_mod, "secondary_scores"):
                            try:
                                secondary = (task_mod.secondary_scores(
                                    parsed, ground_truth, task_name) or {})
                            except Exception as e:
                                secondary = {"_error":
                                              f"{type(e).__name__}: {e}"}

                        # CSV row — long fields truncated for spreadsheet
                        # legibility (raw_response and tool_call_log can be
                        # tens of KB, which Excel/Sheets handle poorly).
                        writer.writerow({
                            "row_id": row_id,
                            "pair_id": r["pair_id"],
                            "archetype": archetype,
                            "difficulty": r["difficulty"],
                            "task": task_name,
                            "renderer": renderer_name,
                            "prompt": prompt_name,
                            "model": spec,
                            "raw_response": (raw or "")[:1500],
                            "parsed_answer": json.dumps(parsed),
                            "score": f"{score_val:.4f}"
                                if score_val == score_val else "",
                            "secondary_scores":
                                json.dumps(secondary) if secondary else "",
                            "tool_calls": len(tool_log),
                            "tool_call_log":
                                json.dumps(tool_log)[:1500]
                                if tool_log else "",
                            "elapsed_sec": f"{dt:.3f}",
                            "error": err,
                        })

                        # JSONL record — full untruncated payload, native
                        # types (parsed_answer, secondary_scores as dicts;
                        # tool_call_log as list). Use this for any post-
                        # analysis that touches model output text or tool
                        # traces; cross-link to CSV via row_id.
                        #
                        # api_metadata captures everything the adapter
                        # extracted from the provider response: token
                        # usage (incl. cached + reasoning), finish_reason,
                        # resolved model snapshot, system_fingerprint,
                        # request_id, n_turns, per_turn_finish_reasons,
                        # and (for tool-using runs) the full conversation
                        # trace with image data redacted. Adapters that
                        # return a plain string (oracle, legacy) leave
                        # this empty.
                        jsonl_f.write(json.dumps({
                            "row_id": row_id,
                            "pair_id": r["pair_id"],
                            "archetype": archetype,
                            "difficulty": r["difficulty"],
                            "task": task_name,
                            "renderer": renderer_name,
                            "prompt": prompt_name,
                            "model": spec,
                            "raw_response": raw or "",
                            "parsed_answer": parsed,
                            "score": (score_val
                                      if score_val == score_val else None),
                            "secondary_scores": secondary,
                            "tool_calls": len(tool_log),
                            "tool_call_log": tool_log,
                            "elapsed_sec": dt,
                            "error": err,
                            "api_metadata": api_meta,
                        }, default=str) + "\n")
                        row_id += 1

                        if score_val == score_val:
                            agg[(archetype, r["difficulty"], task_name,
                                  renderer_name, prompt_name)].append(
                                score_val)

                        marker = "PASS" if score_val == 1.0 else \
                                  ("ZERO" if score_val == 0.0 else " ok ")
                        print(f"  [{n_done:>3}/{n_total}] "
                               f"{r['pair_id']:<26} {task_name:<8} "
                               f"r={renderer_name:<8} p={prompt_name:<8} "
                               f"m={spec:<28} score={score_val:.3f} "
                               f"[{marker}] ({dt:.1f}s)")

    # Aggregate (mean score) per cell × per model.
    summary: Dict[str, Any] = {"cells": []}
    # Re-compute per (archetype, difficulty, task, renderer, prompt, model).
    cell_scores: Dict[Tuple[str, str, str, str, str, str],
                       List[float]] = defaultdict(list)
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            try:
                s = float(row["score"])
            except (TypeError, ValueError):
                continue
            cell_scores[(
                row["archetype"], row["difficulty"], row["task"],
                row["renderer"], row["prompt"], row["model"],
            )].append(s)
    for key, scores in sorted(cell_scores.items()):
        summary["cells"].append({
            "archetype": key[0],
            "difficulty": key[1],
            "task": key[2],
            "renderer": key[3],
            "prompt": key[4],
            "model": key[5],
            "n": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        })
    with open(out_dir / "aggregate.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {n_done} evaluations.")
    print(f"  per-question rows  : {out_csv}")
    print(f"  per-question JSONL : {out_jsonl}  (full untruncated)")
    print(f"  aggregate summary  : {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
