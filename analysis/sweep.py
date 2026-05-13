"""Load ``run_dataset`` JSON artifacts and build threshold / ECDF summaries.

Designed for notebooks and scripts; depends only on stdlib + numpy.

Each ``PairRecord`` includes:

- ``oracle_best_valid_fraction_improved`` — max primary fraction among **valid**
  logged explores (guards respected).
- ``oracle_max_feasible_fraction_improved`` — max among **feasible** logged
  explores **ignoring** ``valid`` (guard-agnostic ceiling on the visited set).
- ``selected_total_guard_violation`` / ``violation_sum_at_max_feasible_fraction`` —
  sums of per-guard ``violation`` for guard-margin analyses.

Global optimum / MIP regret is **not** in JSON; see ``analysis/optimum_regret.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Mirrors ``queries.make_*_query_from_metadata`` defaults at run time.
OFFICIAL_FRACTION_THRESHOLDS: Dict[str, float] = {
    "coverage_gap": 0.3,
    "contiguity": 0.5,
    # Mirrors ``queries.make_shape_niceness_query_from_metadata`` default
    # (worst-k mean NPI primary; 0.2 on legacy mean-NPI was unattainable).
    "shape_niceness": 0.02,
}

ARCHETYPE_ORDER: Tuple[str, ...] = (
    "cluster",
    "coverage_gap",
    "contiguity",
    "shape_niceness",
)
MODALITY_ORDER: Tuple[str, ...] = ("multimodal", "tools_only")


@dataclass(frozen=True)
class PairRecord:
    """One completed pair-run (one JSON file)."""

    archetype: str
    modality: str
    pair_id: str
    path: Path
    selected_fraction_improved: float
    oracle_best_valid_fraction_improved: float
    #: Max ``fraction_improved`` over logged *feasible* explores (ignore ``valid`` / guards).
    oracle_max_feasible_fraction_improved: float
    selected_valid: bool
    official_success: bool
    target_baseline: float
    target_response: float
    n_feasible_explored: int
    has_explored_scores_full: bool
    selected_total_guard_violation: float
    violation_sum_at_max_feasible_fraction: float


def _explored_rows(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    ss = d.get("superscore") or {}
    rows = ss.get("explored_scores_full")
    if not rows:
        return []
    return list(rows)


def _oracle_best_valid_fraction(
    rows: Sequence[Dict[str, Any]], selected_score: Dict[str, Any]
) -> float:
    best: Optional[float] = None
    for row in rows:
        sc = row.get("score") or {}
        if not bool(sc.get("valid")):
            continue
        f = float(sc.get("fraction_improved", 0.0))
        best = f if best is None else max(best, f)
    if best is None:
        return float(selected_score.get("fraction_improved", 0.0))
    return float(best)


def _guard_violation_sum(score: Dict[str, Any]) -> float:
    return float(
        sum(float(g.get("violation", 0.0) or 0.0) for g in (score.get("guards") or []))
    )


def _oracle_max_feasible_fraction_and_violation(
    rows: Sequence[Dict[str, Any]], selected_score: Dict[str, Any]
) -> Tuple[float, float]:
    """Max primary ``fraction_improved`` over feasible explores, ignoring ``valid``.

    Tie-break on equal fraction: lower total guard ``violation`` sum.

    ``explored_scores_full`` rows are feasible-only in current ``run_dataset``;
    we still honor ``feasible`` when present.
    """
    best_f: Optional[float] = None
    best_v = 0.0
    for row in rows:
        sc = row.get("score") or {}
        if not bool(sc.get("feasible", True)):
            continue
        f = float(sc.get("fraction_improved", 0.0))
        vsum = _guard_violation_sum(sc)
        if best_f is None or f > best_f + 1e-15:
            best_f = f
            best_v = vsum
        elif abs(f - best_f) <= 1e-15 and vsum < best_v:
            best_v = vsum
    if best_f is None:
        f0 = float(selected_score.get("fraction_improved", 0.0))
        return f0, _guard_violation_sum(selected_score)
    return float(best_f), float(best_v)


def load_pair_record(path: Path) -> PairRecord:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = d.get("metadata") or {}
    arch = str(meta.get("archetype") or path.parent.parent.name)
    modality = str(d.get("modality") or "")
    sel = d.get("score") or {}
    rows = _explored_rows(d)
    oracle = _oracle_best_valid_fraction(rows, sel)
    feas_max, viol_at_max = _oracle_max_feasible_fraction_and_violation(rows, sel)
    ss = d.get("superscore") or {}
    return PairRecord(
        archetype=arch,
        modality=modality,
        pair_id=str(d.get("pair_id", path.stem)),
        path=path,
        selected_fraction_improved=float(sel.get("fraction_improved", 0.0)),
        oracle_best_valid_fraction_improved=oracle,
        oracle_max_feasible_fraction_improved=feas_max,
        selected_valid=bool(sel.get("valid")),
        official_success=bool(sel.get("success", False)),
        target_baseline=float(sel.get("target_baseline", 0.0)),
        target_response=float(sel.get("target_response", 0.0)),
        n_feasible_explored=int(ss.get("n_feasible_explored", 0)),
        has_explored_scores_full=bool(rows),
        selected_total_guard_violation=_guard_violation_sum(sel),
        violation_sum_at_max_feasible_fraction=viol_at_max,
    )


def discover_result_paths(
    results_root: Path, query_type: str
) -> List[Tuple[str, str, Path]]:
    """Return list of (archetype, modality, json_path) for pair results."""
    out: List[Tuple[str, str, Path]] = []
    for arch_dir in sorted(results_root.iterdir()):
        if not arch_dir.is_dir():
            continue
        name = arch_dir.name
        if name.startswith(".") or name == "results_paired_vague":
            continue
        for mod in MODALITY_ORDER:
            rdir = arch_dir / f"results_{mod}_{query_type}"
            if not rdir.is_dir():
                continue
            for p in sorted(rdir.glob("*.json")):
                if p.name == "aggregate.json":
                    continue
                out.append((name, mod, p))
    return out


def load_all_records(
    results_root: Path, query_type: str = "vague"
) -> List[PairRecord]:
    return [load_pair_record(p) for _, _, p in discover_result_paths(results_root, query_type)]


def tau_grid(step: float = 0.02) -> np.ndarray:
    t = np.arange(0.0, 1.0 + 1e-9, step, dtype=np.float64)
    if t[-1] < 1.0 - 1e-12:
        t = np.append(t, 1.0)
    return t


def passes_fraction_tau(score: Dict[str, Any], tau: float) -> bool:
    if not bool(score.get("valid")):
        return False
    return float(score.get("fraction_improved", 0.0)) >= tau - 1e-12


def success_at_tau_for_offline_selection(
    archetype: str, score: Dict[str, Any], tau: float
) -> bool:
    """``success`` slot in the offline superscore key (mirrors ``run_dataset`` intent).

    - **cluster**: ``feasible`` ∧ ``valid`` ∧ ``target_response`` ≤ 0 (independent of τ).
    - **Others**: ``passes_fraction_tau`` i.e. ``valid`` ∧ ``fraction_improved`` ≥ τ.
    """
    if not bool(score.get("feasible", True)):
        return False
    if archetype == "cluster":
        return bool(score.get("valid", False)) and float(
            score.get("target_response", 1.0)
        ) <= 0.0
    return passes_fraction_tau(score, float(tau))


def offline_superscore_key_at_tau(
    item: Dict[str, Any], tau: float, archetype: str
) -> Tuple[Any, ...]:
    """Lexicographic key aligned with ``run_dataset._select_best_explored_solution`` but with τ-success."""
    sc = item.get("score") or {}
    s_tau = success_at_tau_for_offline_selection(archetype, sc, tau)
    return (
        s_tau,
        bool(sc.get("valid", False)),
        float(sc.get("fraction_improved", 0.0)),
        float(sc.get("raw_improvement", 0.0)),
        -float(sc.get("assignment_distance_delta", 0.0)),
        (item.get("source") or "") != "baseline",
    )


def offline_best_explored_row(
    rows: Sequence[Dict[str, Any]], tau: float, archetype: str
) -> Optional[Dict[str, Any]]:
    """``argmax`` over ``explored_scores_full`` rows under ``offline_superscore_key_at_tau``."""
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: offline_superscore_key_at_tau(r, float(tau), archetype),
    )


def sweep_offline_reselect_pass_same_tau(
    paths: Sequence[Path], taus: np.ndarray, archetype: str
) -> np.ndarray:
    """Pass-rate vs τ: offline re-superscore **at** τ, then test ``valid ∧ frac ≥ τ`` on the winner.

    For each abscissa value τ, the winner is ``max(explored_scores_full, key=offline_key_τ)``.
    The plotted value is the fraction of runs where that winner passes ``passes_fraction_tau(·, τ)``.
    """
    n = len(paths)
    if n == 0:
        return np.zeros_like(taus, dtype=np.float64)
    out = np.zeros(len(taus), dtype=np.float64)
    for j, tau in enumerate(taus):
        t = float(tau)
        hits = 0
        for p in paths:
            d = _load_json(p)
            rows = _explored_rows(d)
            best = offline_best_explored_row(rows, t, archetype)
            if best is None:
                sc = d.get("score") or {}
            else:
                sc = best.get("score") or {}
            if passes_fraction_tau(sc, t):
                hits += 1
        out[j] = hits / n
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise TypeError(
            f"Expected a JSON object (pair-run result) from {path}, "
            f"got {type(d).__name__}. Wrong file (e.g. *_trajectory.json)?"
        )
    return d


def sweep_selected_valid_and_feasible_oracle(
    paths: Sequence[Path], taus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pass-rate τ: selected uses ``valid`` ∧ frac≥τ; oracle uses max feasible frac (any logged explore)."""
    n = len(paths)
    if n == 0:
        z = np.zeros_like(taus, dtype=np.float64)
        return z, z
    sel_hits = np.zeros_like(taus, dtype=np.int32)
    ora_hits = np.zeros_like(taus, dtype=np.int32)
    for p in paths:
        d = _load_json(p)
        sel = d.get("score") or {}
        rows = _explored_rows(d)
        max_feas = 0.0
        for row in rows:
            sc = row.get("score") or {}
            if not bool(sc.get("feasible", True)):
                continue
            max_feas = max(max_feas, float(sc.get("fraction_improved", 0.0)))
        if not rows:
            max_feas = float(sel.get("fraction_improved", 0.0))
        for i, tau in enumerate(taus):
            t = float(tau)
            if passes_fraction_tau(sel, t):
                sel_hits[i] += 1
            if max_feas >= t - 1e-12:
                ora_hits[i] += 1
    return sel_hits.astype(np.float64) / n, ora_hits.astype(np.float64) / n


def sweep_selected_and_oracle(
    paths: Sequence[Path], taus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of paths where selected (resp. any explored valid) passes τ."""
    n = len(paths)
    if n == 0:
        z = np.zeros_like(taus, dtype=np.float64)
        return z, z
    sel_hits = np.zeros_like(taus, dtype=np.int32)
    ora_hits = np.zeros_like(taus, dtype=np.int32)
    for p in paths:
        d = _load_json(p)
        sel = d.get("score") or {}
        for i, tau in enumerate(taus):
            if passes_fraction_tau(sel, float(tau)):
                sel_hits[i] += 1
        rows = _explored_rows(d)
        for i, tau in enumerate(taus):
            ok = any(passes_fraction_tau(r.get("score") or {}, float(tau)) for r in rows)
            if ok:
                ora_hits[i] += 1
    return sel_hits.astype(np.float64) / n, ora_hits.astype(np.float64) / n


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted x and F(x) step function nodes (for plotting)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    y = np.arange(1, n + 1, dtype=np.float64) / n
    return v, y


def records_by_bucket(
    records: Sequence[PairRecord],
) -> Dict[Tuple[str, str], List[PairRecord]]:
    m: Dict[Tuple[str, str], List[PairRecord]] = {}
    for r in records:
        m.setdefault((r.archetype, r.modality), []).append(r)
    for k in m:
        m[k].sort(key=lambda x: x.pair_id)
    return m


def group_paths(
    records: Sequence[PairRecord],
) -> Dict[Tuple[str, str], List[Path]]:
    m: Dict[Tuple[str, str], List[Path]] = {}
    for r in records:
        m.setdefault((r.archetype, r.modality), []).append(r.path)
    return m


def official_fraction_threshold(
    archetype: str,
    *,
    overrides: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Return the τ used for official fraction bars (and optional plot-time overrides)."""
    if overrides is not None and archetype in overrides:
        return float(overrides[archetype])
    return OFFICIAL_FRACTION_THRESHOLDS.get(archetype)


def record_success_for_threshold_plot(
    record: PairRecord,
    *,
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> bool:
    """Success rate for bar charts: logged ``official_success``, or counterfactual τ.

    When ``official_threshold_overrides`` contains ``record.archetype`` (and the
    archetype is not ``cluster``), success is **recomputed** from the shipped
    score only, as ``valid`` ∧ ``fraction_improved`` ≥ τ — the same test as the
    solid "selected" curve in the τ sweep figures. This matches a lower official
    bar **without** re-running the API (cluster stays logged-only).
    """
    if official_threshold_overrides is None:
        return record.official_success
    if record.archetype == "cluster":
        return record.official_success
    if record.archetype not in official_threshold_overrides:
        return record.official_success
    tau = float(official_threshold_overrides[record.archetype])
    return bool(record.selected_valid) and record.selected_fraction_improved >= tau - 1e-12


def selection_gap_summary(records: Sequence[PairRecord]) -> Dict[str, Any]:
    """How often oracle fraction exceeds selected (same pair)."""
    gaps = []
    gaps_feas = []
    for r in records:
        gaps.append(r.oracle_best_valid_fraction_improved - r.selected_fraction_improved)
        gaps_feas.append(
            r.oracle_max_feasible_fraction_improved - r.selected_fraction_improved
        )
    arr = np.array(gaps, dtype=np.float64)
    arr_f = np.array(gaps_feas, dtype=np.float64)
    return {
        "mean_oracle_minus_selected": float(arr.mean()) if arr.size else 0.0,
        "frac_strictly_positive_gap": float((arr > 1e-9).mean()) if arr.size else 0.0,
        "max_gap": float(arr.max()) if arr.size else 0.0,
        "mean_feasible_oracle_minus_selected": float(arr_f.mean()) if arr_f.size else 0.0,
        "frac_feasible_oracle_strictly_above_selected": float((arr_f > 1e-9).mean())
        if arr_f.size
        else 0.0,
        "max_feasible_oracle_gap": float(arr_f.max()) if arr_f.size else 0.0,
    }
