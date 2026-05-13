"""Audit explore-replay pickles and offline shape metrics for shape_niceness.

Uses pickled ``Solution`` objects from ``*.explore_replay.pkl`` (no API).
Computes per-catchment NPI (via ``generation._per_catchment_npi``) and
raster convex-hull solidity (A / A_hull, capped at 1), then aggregates
mean / worst-k / deltas for superscore-selected vs baseline.
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from generation import _per_catchment_npi


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew monotone chain). points: (n,2)."""
    if len(points) == 0:
        return points
    pts = np.unique(np.round(points, 12), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def build_half(seq: np.ndarray) -> List[np.ndarray]:
        hull: List[np.ndarray] = []
        for p in seq:
            while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(np.asarray(p, dtype=float))
        return hull

    lower = build_half(pts)
    upper = build_half(pts[::-1])
    if not lower or not upper:
        return pts
    hull = lower[:-1] + upper[:-1]
    return np.stack(hull, axis=0)


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _per_catchment_solidity(
    instance: Any, solution: Any
) -> Dict[int, Dict[str, float]]:
    """Convex-hull solidity A / A_hull (capped at 1) per opened catchment.

    Cell centers use ``grid_xs[ix], grid_ys[iy]`` consistent with
    ``_per_catchment_npi`` mask indexing.
    """
    label_grid = instance.precinct_label_grid
    xs = instance.grid_xs
    ys = instance.grid_ys
    cell_w = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    cell_h = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    cell_area = cell_w * cell_h

    assigned = solution.y.argmax(axis=1)
    cell_site = assigned[label_grid]

    out: Dict[int, Dict[str, float]] = {}
    for j in np.where(solution.x == 1)[0]:
        mask = (cell_site == j)
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        A = float(n_cells) * cell_area
        iy, ix = np.where(mask)
        pts = np.column_stack((xs[ix.astype(int)], ys[iy.astype(int)]))
        hull = _convex_hull_vertices(pts)
        a_hull = _polygon_area(hull)
        if a_hull <= 0.0 or n_cells < 3:
            solidity = 1.0
        else:
            solidity = float(min(1.0, A / a_hull))
        out[int(j)] = {"A": A, "A_hull": float(a_hull), "solidity": solidity,
                       "n_cells": float(n_cells)}
    return out


def _mean_of_k_largest(values: Sequence[float], k: int) -> float:
    if not values:
        return float("nan")
    kk = min(k, len(values))
    arr = np.sort(np.asarray(values, dtype=float))
    return float(arr[-kk:].mean())


def _mean_of_k_smallest(values: Sequence[float], k: int) -> float:
    if not values:
        return float("nan")
    kk = min(k, len(values))
    arr = np.sort(np.asarray(values, dtype=float))
    return float(arr[:kk].mean())


def replay_row_superscore_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Lexicographic key aligned with ``run_dataset._select_best_explored_solution``.

    Replay bundle rows carry a ``score`` dict and ``source`` like explored entries.
    """
    sc = row.get("score") or {}
    return (
        bool(sc.get("success", False)),
        bool(sc.get("valid", False)),
        float(sc.get("fraction_improved", 0.0)),
        float(sc.get("raw_improvement", 0.0)),
        -float(sc.get("assignment_distance_delta", 0.0)),
        (row.get("source") or "") != "baseline",
    )


def fraction_improved_directed(
    baseline_val: float, response_val: float, direction: str
) -> float:
    """Same normalization as ``ArchetypeQuery.score`` in ``queries.py``."""
    if not (np.isfinite(baseline_val) and np.isfinite(response_val)):
        return 0.0
    if direction == "minimize":
        improvement = baseline_val - response_val
    elif direction == "maximize":
        improvement = response_val - baseline_val
    else:
        raise ValueError(f"direction must be 'minimize' or 'maximize', got {direction!r}")
    denom = abs(baseline_val) if abs(baseline_val) > 1e-9 else 1.0
    return max(0.0, float(improvement / denom))


def offline_metric_fractions_for_result_path(
    result_json: Path,
    *,
    metric_key: str,
    direction: str,
    preloaded: Optional[Dict[str, Any]] = None,
    oracle_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Replay-based ``fraction_improved`` using any scalar from ``shape_metric_bundle``.

    Returns selected / oracle (valid) / oracle (feasible) primary fractions vs
    the **baseline** geometry for ``metric_key``, plus JSON ``selected_valid``.

    If ``oracle_top_k`` is a positive int, oracle maxima are taken only over the
    top-``oracle_top_k`` replay rows by ``replay_row_superscore_key`` (same order
    as ``run_dataset._select_best_explored_solution``). ``None`` means all rows
    with a loaded ``solution``.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "selected_frac": 0.0,
        "oracle_valid_frac": 0.0,
        "oracle_feasible_frac": 0.0,
        "selected_valid": False,
        "n_explores": 0,
        "error": "",
    }
    data = preloaded
    if data is None:
        with open(result_json, encoding="utf-8") as f:
            data = json.load(f)
    if str(data.get("status", "")) != "completed":
        out["error"] = "not_completed"
        return out
    pair_id = str(data.get("pair_id", result_json.stem))
    dataset_dir = result_json.parent.parent
    pair_dir = dataset_dir / "pairs" / pair_id
    ss = data.get("superscore")
    if not isinstance(ss, dict):
        out["error"] = "no_superscore"
        return out
    rel = ss.get("explore_replay_path")
    if not rel:
        out["error"] = "no_replay_path"
        return out
    replay_path = (result_json.parent / str(rel)).resolve()
    if not replay_path.is_file():
        out["error"] = "replay_missing"
        return out

    from instance import Instance, Solution

    instance = Instance.load(str(pair_dir / "instance.pkl"))
    baseline = Solution.load(str(pair_dir / "baseline_solution.pkl"))
    bundle = load_explore_replay(replay_path)
    rows = list(bundle.get("rows") or [])
    out["n_explores"] = len(rows)
    if not rows:
        out["error"] = "empty_bundle"
        return out

    base_m = shape_metric_bundle(instance, baseline)
    if metric_key not in base_m or isinstance(base_m[metric_key], dict):
        raise KeyError(
            f"Unknown scalar metric_key {metric_key!r}; use a key from shape_metric_bundle."
        )
    baseline_val = float(base_m[metric_key])

    def frac_for_solution(sol: Any) -> float:
        m = shape_metric_bundle(instance, sol)
        return fraction_improved_directed(baseline_val, float(m[metric_key]), direction)

    rows_with_sol = [r for r in rows if r.get("solution") is not None]
    if oracle_top_k is not None and oracle_top_k > 0:
        oracle_rows = sorted(
            rows_with_sol,
            key=replay_row_superscore_key,
            reverse=True,
        )[: int(oracle_top_k)]
    else:
        oracle_rows = rows_with_sol

    best_valid = 0.0
    best_feas = 0.0
    for row in oracle_rows:
        sc = row.get("score") or {}
        sol = row.get("solution")
        if sol is None:
            continue
        f = frac_for_solution(sol)
        if bool(sc.get("feasible", True)):
            best_feas = max(best_feas, f)
        if bool(sc.get("valid")):
            best_valid = max(best_valid, f)

    sel_idx = int(ss.get("selected_explored_index", 0))
    sel_row = pick_selected_row(bundle, sel_idx)
    if sel_row is None:
        out["error"] = "selected_row_missing"
        return out
    sel_sc = sel_row.get("score") or {}
    out["selected_valid"] = bool(sel_sc.get("valid"))
    out["selected_frac"] = frac_for_solution(sel_row["solution"])
    out["oracle_valid_frac"] = float(best_valid)
    out["oracle_feasible_frac"] = float(best_feas)
    out["ok"] = True
    return out


def sweep_shape_offline_tau_for_paths(
    paths: Sequence[Path],
    taus: np.ndarray,
    *,
    metric_key: str,
    direction: str,
    oracle_top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """τ sweep parallel to ``sweep_selected_and_oracle`` but using replay metrics.

    For each τ, returns the fraction of paths where:

    1. **json_shipped** — JSON ``valid`` ∧ JSON ``fraction_improved`` ≥ τ
       (original primary in the saved result).
    2. **offline_selected** — JSON ``valid`` ∧ offline(selected) ≥ τ.
    3. **offline_oracle_valid** — ∃ explored row (optionally restricted to the
       top ``oracle_top_k`` by ``replay_row_superscore_key``) with JSON
       ``valid`` ∧ offline(row) ≥ τ.
    4. **offline_oracle_feasible** — same pool, ``feasible`` ∧ offline(row) ≥ τ.

    Paths with load errors contribute no hits (same as missing data).
    """
    n = len(paths)
    z = np.zeros_like(taus, dtype=np.float64)
    if n == 0:
        return z, z, z, z
    json_hits = np.zeros_like(taus, dtype=np.int32)
    off_sel_hits = np.zeros_like(taus, dtype=np.int32)
    off_ora_v_hits = np.zeros_like(taus, dtype=np.int32)
    off_ora_f_hits = np.zeros_like(taus, dtype=np.int32)

    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            fr = offline_metric_fractions_for_result_path(
                p,
                metric_key=metric_key,
                direction=direction,
                preloaded=data,
                oracle_top_k=oracle_top_k,
            )
        except Exception:
            continue
        if not fr.get("ok"):
            continue
        sel_score = data.get("score") or {}

        def json_passes(tau: float) -> bool:
            if not bool(sel_score.get("valid")):
                return False
            return float(sel_score.get("fraction_improved", 0.0)) >= float(tau) - 1e-12

        for i, tau in enumerate(taus):
            if json_passes(float(tau)):
                json_hits[i] += 1
            if fr["selected_valid"] and fr["selected_frac"] >= float(tau) - 1e-12:
                off_sel_hits[i] += 1
            if fr["oracle_valid_frac"] >= float(tau) - 1e-12:
                off_ora_v_hits[i] += 1
            if fr["oracle_feasible_frac"] >= float(tau) - 1e-12:
                off_ora_f_hits[i] += 1

    return (
        json_hits.astype(np.float64) / n,
        off_sel_hits.astype(np.float64) / n,
        off_ora_v_hits.astype(np.float64) / n,
        off_ora_f_hits.astype(np.float64) / n,
    )


def shape_metric_bundle(instance: Any, solution: Any) -> Dict[str, Any]:
    """Scalar summaries for one solution (opened catchments only)."""
    per_npi = _per_catchment_npi(instance, solution)
    per_sol = _per_catchment_solidity(instance, solution)
    npis = [d["NPI"] for d in per_npi.values()]
    sols = [per_sol[j]["solidity"] for j in per_npi.keys() if j in per_sol]
    return {
        "n_catchments": len(npis),
        "npi_mean": float(np.mean(npis)) if npis else float("nan"),
        "npi_max": float(np.max(npis)) if npis else float("nan"),
        "npi_p90": float(np.percentile(npis, 90)) if npis else float("nan"),
        "npi_mean_worst3": _mean_of_k_largest(npis, 3),
        "npi_mean_worst5": _mean_of_k_largest(npis, 5),
        "npi_mean_worst6": _mean_of_k_largest(npis, 6),
        "solidity_mean": float(np.mean(sols)) if sols else float("nan"),
        "solidity_min": float(np.min(sols)) if sols else float("nan"),
        "solidity_p10": float(np.percentile(sols, 10)) if sols else float("nan"),
        "solidity_mean_worst3": _mean_of_k_smallest(sols, 3),
        "solidity_mean_worst5": _mean_of_k_smallest(sols, 5),
        "solidity_mean_worst6": _mean_of_k_smallest(sols, 6),
        "solidity_mean_best6": _mean_of_k_largest(sols, 6),
        "per_npi": per_npi,
        "per_solidity": per_sol,
    }


def _mean_delta_on_sites(
    base_map: Mapping[int, Dict[str, float]],
    resp_map: Mapping[int, Dict[str, float]],
    site_ids: Sequence[int],
    key: str,
) -> Tuple[float, int]:
    """Mean (response - baseline) for ``key`` over sites open in both maps."""
    deltas: List[float] = []
    for j in site_ids:
        if j not in base_map or j not in resp_map:
            continue
        deltas.append(float(resp_map[j][key]) - float(base_map[j][key]))
    if not deltas:
        return float("nan"), 0
    return float(np.mean(deltas)), len(deltas)


def _worst_k_site_ids_by_npi(per_npi: Mapping[int, Dict[str, float]], k: int
                             ) -> List[int]:
    items = sorted(per_npi.items(), key=lambda kv: kv[1]["NPI"], reverse=True)
    return [int(j) for j, _ in items[:k]]


def _worst_k_site_ids_by_solidity(
    per_sol: Mapping[int, Dict[str, float]], k: int
) -> List[int]:
    items = sorted(per_sol.items(), key=lambda kv: kv[1]["solidity"])
    return [int(j) for j, _ in items[:k]]


def enrich_with_topk_deltas(
    base_bundle: Dict[str, Any], resp_bundle: Dict[str, Any], k: int = 6
) -> Dict[str, float]:
    """Mean deltas on baseline worst-k sites (still open in response)."""
    per_b = base_bundle["per_npi"]
    per_r = resp_bundle["per_npi"]
    sol_b = base_bundle["per_solidity"]
    sol_r = resp_bundle["per_solidity"]
    sites_npi = _worst_k_site_ids_by_npi(per_b, k)
    sites_sol = _worst_k_site_ids_by_solidity(sol_b, k)
    dn, nn = _mean_delta_on_sites(per_b, per_r, sites_npi, "NPI")
    ds, ns = _mean_delta_on_sites(sol_b, sol_r, sites_sol, "solidity")
    return {
        f"delta_npi_mean_on_baseline_worst{k}_sites": dn,
        f"n_sites_used_baseline_worst{k}_npi": float(nn),
        f"delta_solidity_mean_on_baseline_worst{k}_sites": ds,
        f"n_sites_used_baseline_worst{k}_solidity": float(ns),
    }


def load_explore_replay(path: Path) -> Dict[str, Any]:
    """Load a bundle written by ``run_dataset._write_explore_replay_pickle``.

    Pickles embed NumPy arrays; if they were saved with NumPy 2.x, loading
    on NumPy 1.x can raise ``ModuleNotFoundError: numpy._core`` — use the
    same NumPy major version as the run that produced the replay file.
    """
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if "numpy" in str(e).lower():
                raise ModuleNotFoundError(
                    f"{e}\nExplore-replay pickles require a compatible NumPy "
                    "version (e.g. NumPy 2.x if the bundle was saved under "
                    "NumPy 2). Upgrade NumPy in this environment or re-run the "
                    "dataset in an env matching the writer."
                ) from e
            raise


def pick_selected_row(
    bundle: Mapping[str, Any], selected_explored_index: int
) -> Optional[Dict[str, Any]]:
    for row in bundle.get("rows") or []:
        if int(row.get("explored_index", -1)) == int(selected_explored_index):
            return row
    return None


@dataclass
class AuditRow:
    pair_id: str
    modality: str
    result_json: Path
    status: str
    explore_replay_path_field: Optional[str]
    replay_path_resolved: Optional[Path]
    replay_exists: bool


def audit_results_dir(
    results_dir: Path,
    modality_label: str,
) -> List[AuditRow]:
    rows: List[AuditRow] = []
    for json_path in sorted(results_dir.glob("*.json")):
        if json_path.name == "aggregate.json":
            continue
        with open(json_path) as f:
            data = json.load(f)
        pair_id = str(data.get("pair_id", json_path.stem))
        status = str(data.get("status", ""))
        rel = None
        ss = data.get("superscore")
        if isinstance(ss, dict):
            rel = ss.get("explore_replay_path")
        replay_resolved = None
        exists = False
        if rel:
            replay_resolved = (results_dir / rel).resolve()
            exists = replay_resolved.is_file()
        rows.append(
            AuditRow(
                pair_id=pair_id,
                modality=modality_label,
                result_json=json_path,
                status=status,
                explore_replay_path_field=rel,
                replay_path_resolved=replay_resolved,
                replay_exists=exists,
            )
        )
    return rows


def audit_shape_niceness_dataset(
    dataset_dir: Path,
    query_type: str = "vague",
    modalities: Sequence[Tuple[str, Path]] = (
        ("multimodal", Path("results_multimodal_vague")),
        ("tools_only", Path("results_tools_only_vague")),
    ),
) -> List[AuditRow]:
    out: List[AuditRow] = []
    for label, sub in modalities:
        rd = dataset_dir / sub
        if not rd.is_dir():
            continue
        out.extend(audit_results_dir(rd, label))
    return out


def rescore_one_result_json(
    dataset_dir: Path,
    result_json: Path,
    *,
    pair_dir_from_index: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load result + replay + instance; return flat dict for CSV / notebook."""
    with open(result_json) as f:
        result = json.load(f)
    pair_id = str(result["pair_id"])
    modality = str(result.get("modality", ""))
    status = str(result.get("status", ""))

    row_out: Dict[str, Any] = {
        "pair_id": pair_id,
        "modality": modality,
        "status": status,
        "result_json": str(result_json),
    }
    ss = result.get("superscore")
    if not isinstance(ss, dict):
        row_out["error"] = "no_superscore"
        return row_out
    rel = ss.get("explore_replay_path")
    if not rel:
        row_out["error"] = "no_explore_replay_path"
        return row_out
    replay_path = (result_json.parent / rel).resolve()
    if not replay_path.is_file():
        row_out["error"] = "replay_missing"
        row_out["replay_expected"] = str(replay_path)
        return row_out

    if pair_dir_from_index is None:
        pair_dir = dataset_dir / "pairs" / pair_id
    else:
        pair_dir = pair_dir_from_index
    inst_path = pair_dir / "instance.pkl"
    if not inst_path.is_file():
        row_out["error"] = "instance_missing"
        return row_out

    from instance import Instance, Solution

    instance = Instance.load(str(inst_path))
    baseline = Solution.load(str(pair_dir / "baseline_solution.pkl"))
    bundle = load_explore_replay(replay_path)
    sel_idx = int(ss.get("selected_explored_index", 0))
    sel_row = pick_selected_row(bundle, sel_idx)
    if sel_row is None:
        row_out["error"] = "selected_row_not_in_bundle"
        row_out["selected_explored_index"] = sel_idx
        return row_out

    sol_sel = sel_row["solution"]
    base_metrics = shape_metric_bundle(instance, baseline)
    resp_metrics = shape_metric_bundle(instance, sol_sel)

    row_out["replay_path"] = str(replay_path)
    row_out["selected_explored_index"] = sel_idx
    row_out["selected_source"] = ss.get("selected_source")

    for prefix, bundle_m in (("baseline", base_metrics), ("selected", resp_metrics)):
        for k, v in bundle_m.items():
            if k in ("per_npi", "per_solidity"):
                continue
            row_out[f"{prefix}_{k}"] = v

    for k, v in enrich_with_topk_deltas(base_metrics, resp_metrics, k=6).items():
        row_out[k] = v
    for k, v in enrich_with_topk_deltas(base_metrics, resp_metrics, k=5).items():
        row_out[k] = v

    # Deltas (selected - baseline) on global summaries
    for key in (
        "npi_mean",
        "npi_max",
        "npi_p90",
        "npi_mean_worst3",
        "npi_mean_worst5",
        "npi_mean_worst6",
        "solidity_mean",
        "solidity_min",
        "solidity_p10",
        "solidity_mean_worst3",
        "solidity_mean_worst5",
        "solidity_mean_worst6",
        "solidity_mean_best6",
    ):
        row_out[f"delta_{key}"] = float(resp_metrics[key] - base_metrics[key])

    row_out["error"] = ""
    return row_out


def rescore_shape_niceness_dataset(
    dataset_dir: Path,
    query_type: str = "vague",
    modalities: Sequence[Tuple[str, Path]] = (
        ("multimodal", Path("results_multimodal_vague")),
        ("tools_only", Path("results_tools_only_vague")),
    ),
) -> List[Dict[str, Any]]:
    """Rescore every ``*.json`` result (except aggregate) under modality dirs."""
    records: List[Dict[str, Any]] = []
    for _label, sub in modalities:
        rd = dataset_dir / sub
        if not rd.is_dir():
            continue
        for json_path in sorted(rd.glob("*.json")):
            if json_path.name == "aggregate.json":
                continue
            records.append(rescore_one_result_json(dataset_dir, json_path))
    return records


def write_audit_csv(rows: Sequence[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_id",
                "modality",
                "status",
                "explore_replay_path_field",
                "replay_path_resolved",
                "replay_exists",
                "result_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.pair_id,
                    r.modality,
                    r.status,
                    r.explore_replay_path_field or "",
                    str(r.replay_path_resolved) if r.replay_path_resolved else "",
                    r.replay_exists,
                    str(r.result_json),
                ]
            )


def write_rescore_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    keys = sorted({k for rec in records for k in rec.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in keys})


def write_rescore_json(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(records), f, indent=2)
