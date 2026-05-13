"""Matplotlib figure builders for post-run analysis (spacious layouts)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .shape_niceness_replay_metrics import sweep_shape_offline_tau_for_paths
from .sweep import (
    ARCHETYPE_ORDER,
    MODALITY_ORDER,
    PairRecord,
    ecdf,
    official_fraction_threshold,
    record_success_for_threshold_plot,
    sweep_offline_reselect_pass_same_tau,
    sweep_selected_and_oracle,
    sweep_selected_valid_and_feasible_oracle,
    tau_grid,
)

# Distinct, colour-blind-friendly (Okabe–Ito inspired)
COLOR_MM = "#0072B2"
COLOR_TO = "#D55E00"
COLOR_ORACLE = "#009E73"
COLOR_SELECTED = "#CC79A7"


def _thr(arch: str, official_threshold_overrides: Optional[Dict[str, float]]) -> Optional[float]:
    return official_fraction_threshold(arch, overrides=official_threshold_overrides)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _xlim_right_for_threshold_curves(
    arch: str,
    taus: np.ndarray,
    rates: np.ndarray,
    tau_step: float,
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> float:
    """Right edge for τ axis: shape_niceness stays tight; others data-driven."""
    # rates: shape (4, n_taus) — MM sel, MM ora, TO sel, TO ora
    if arch == "shape_niceness":
        return 0.10

    eps = 0.02
    active = np.any(rates > eps, axis=0)
    if not np.any(active):
        return min(1.0, float(0.25 + 3 * tau_step))
    last_i = int(np.where(active)[0][-1])
    right = float(taus[last_i]) + 3 * tau_step + 0.04
    thr = _thr(arch, official_threshold_overrides)
    if thr is not None:
        right = max(right, thr + 0.05)
    return min(1.0, max(0.18, right))


def fig_threshold_curves(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    tau_step: float = 0.02,
    figsize: Tuple[float, float] = (9.5, 3.25),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """One row per archetype: MM and TO overlaid (selected solid, oracle dashed)."""
    taus_full = tau_grid(tau_step)
    n_arch = len(ARCHETYPE_ORDER)
    fig, axes = plt.subplots(
        n_arch,
        1,
        figsize=(figsize[0], figsize[1] * n_arch),
        sharex=False,
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for row, arch in enumerate(ARCHETYPE_ORDER):
        ax = axes_flat[row]
        curves: List[Tuple[np.ndarray, np.ndarray, str, str, str]] = []
        rate_rows: List[np.ndarray] = []
        for mod, color in [
            ("Multimodal", COLOR_MM),
            ("Tools-only", COLOR_TO),
        ]:
            paths = paths_by_arch_mod.get((arch, mod), [])
            sel, ora = sweep_selected_and_oracle(paths, taus_full)
            rate_rows.append(np.asarray(sel))
            rate_rows.append(np.asarray(ora))
            mod_short = "MM" if mod == "multimodal" else "TO"
            curves.append((sel, ora, color, mod_short, mod))

        rates = np.vstack(rate_rows) if rate_rows else np.zeros((1, len(taus_full)))
        x_right = _xlim_right_for_threshold_curves(
            arch, taus_full, rates, tau_step, official_threshold_overrides
        )
        mask = taus_full <= x_right + 1e-12
        tx = taus_full[mask]

        for sel, ora, color, mod_short, _mod in curves:
            ax.plot(
                tx,
                np.asarray(sel)[mask],
                color=color,
                lw=2.1,
                ls="-",
                label=f"{mod_short} selected",
            )
            ax.plot(
                tx,
                np.asarray(ora)[mask],
                color=color,
                lw=2.0,
                ls="--",
                alpha=0.92,
                label=f"{mod_short} oracle",
            )

        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            if thr <= x_right + 1e-9:
                ax.axvline(
                    thr,
                    color="0.35",
                    ls=":",
                    lw=1.6,
                    label=f"Official bar ({thr:g})",
                )
            else:
                ax.annotate(
                    f"Official τ={thr:g} (right of axis)",
                    xy=(0.97, 0.12),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=9,
                    color="0.35",
                )
        if arch == "cluster":
            ax.annotate(
                "Official cluster success uses target_response ≤ 0 "
                "(not this curve).",
                xy=(0.98, 0.04),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.45),
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.0, x_right)
        ax.set_ylabel("Fraction of pairs")
        ax.set_title(arch.replace("_", " ").title(), loc="left", fontsize=12, pad=10)
        if row == n_arch - 1:
            ax.set_xlabel(r"Threshold $\tau$ on primary fraction_improved")
        ax.legend(loc="upper right", framealpha=0.94, ncol=2, fontsize=9)

    fig.suptitle(
        "Success rate vs improvement over baseline at threshold τ",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.07, hspace=0.48)
    return fig


def fig_threshold_curves_guard_agnostic_oracle(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    tau_step: float = 0.02,
    figsize: Tuple[float, float] = (9.5, 3.25),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """τ sweep: **selected** still requires ``valid`` ∧ frac≥τ; **oracle** = max frac over logged *feasible* explores (guards ignored)."""
    taus_full = tau_grid(tau_step)
    n_arch = len(ARCHETYPE_ORDER)
    fig, axes = plt.subplots(
        n_arch,
        1,
        figsize=(figsize[0], figsize[1] * n_arch),
        sharex=False,
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for row, arch in enumerate(ARCHETYPE_ORDER):
        ax = axes_flat[row]
        rate_rows: List[np.ndarray] = []
        curves: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
        for mod, color in [
            ("multimodal", COLOR_MM),
            ("tools_only", COLOR_TO),
        ]:
            paths = paths_by_arch_mod.get((arch, mod), [])
            sel, ora = sweep_selected_valid_and_feasible_oracle(paths, taus_full)
            rate_rows.append(np.asarray(sel))
            rate_rows.append(np.asarray(ora))
            mod_short = "MM" if mod == "multimodal" else "TO"
            curves.append((sel, ora, color, mod_short))
        rates = np.vstack(rate_rows) if rate_rows else np.zeros((1, len(taus_full)))
        x_right = _xlim_right_for_threshold_curves(
            arch, taus_full, rates, tau_step, official_threshold_overrides
        )
        mask = taus_full <= x_right + 1e-12
        tx = taus_full[mask]
        for sel, ora, color, mod_short in curves:
            ax.plot(
                tx,
                np.asarray(sel)[mask],
                color=color,
                lw=2.1,
                ls="-",
                label=f"{mod_short} selected (valid)",
            )
            ax.plot(
                tx,
                np.asarray(ora)[mask],
                color=color,
                lw=2.0,
                ls="--",
                alpha=0.92,
                label=f"{mod_short} oracle (feasible max)",
            )
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            if thr <= x_right + 1e-9:
                ax.axvline(
                    thr,
                    color="0.35",
                    ls=":",
                    lw=1.6,
                    label=f"Official bar ({thr:g})",
                )
            else:
                ax.annotate(
                    f"Official τ={thr:g} (right of axis)",
                    xy=(0.97, 0.12),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=9,
                    color="0.35",
                )
        if arch == "cluster":
            ax.annotate(
                "Official cluster success uses target_response ≤ 0 "
                "(not this curve).",
                xy=(0.98, 0.04),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.45),
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.0, x_right)
        ax.set_ylabel("Fraction of pairs")
        ax.set_title(arch.replace("_", " ").title(), loc="left", fontsize=12, pad=10)
        if row == n_arch - 1:
            ax.set_xlabel(r"Threshold $\tau$ on primary fraction_improved")
        ax.legend(loc="upper right", framealpha=0.94, ncol=2, fontsize=8)
    fig.suptitle(
        "τ sweep: selected (valid) vs guard-agnostic feasible oracle\n"
        "(oracle = max fraction_improved among logged feasible explores)",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.07, hspace=0.48)
    return fig


def fig_threshold_offline_tau_reselection(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    tau_step: float = 0.02,
    figsize: Tuple[float, float] = (9.5, 3.25),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """τ sweep: **shipped** vs **offline τ-reselection** vs **valid oracle**.

    For each abscissa τ, the offline curve recomputes ``argmax`` over
    ``explored_scores_full`` using the same lex order as ``run_dataset`` but
    replacing ``success`` with ``success_at_τ`` (fraction threshold archetypes)
    or cluster ``target_response`` ≤ 0, then tests ``valid ∧ fraction ≥ τ`` on
    that winner (self-consistent pass bar).
    """
    taus_full = tau_grid(tau_step)
    n_arch = len(ARCHETYPE_ORDER)
    fig, axes = plt.subplots(
        n_arch,
        1,
        figsize=(figsize[0], figsize[1] * n_arch),
        sharex=False,
        sharey=True,
        constrained_layout=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for row, arch in enumerate(ARCHETYPE_ORDER):
        ax = axes_flat[row]
        rate_rows: List[np.ndarray] = []
        bundles: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
        for mod, color in [
            ("multimodal", COLOR_MM),
            ("tools_only", COLOR_TO),
        ]:
            paths = paths_by_arch_mod.get((arch, mod), [])
            sel, ora = sweep_selected_and_oracle(paths, taus_full)
            off = sweep_offline_reselect_pass_same_tau(paths, taus_full, arch)
            bundles.append((mod, color, sel, ora, off))
            rate_rows.extend(
                [np.asarray(sel), np.asarray(off), np.asarray(ora)]
            )

        rates = np.vstack(rate_rows) if rate_rows else np.zeros((1, len(taus_full)))
        x_right = _xlim_right_for_threshold_curves(
            arch, taus_full, rates, tau_step, official_threshold_overrides
        )
        mask = taus_full <= x_right + 1e-12
        tx = taus_full[mask]

        for mod, color, sel, ora, off in bundles:
            mod_short = "MM" if mod == "multimodal" else "TO"
            ax.plot(
                tx,
                np.asarray(sel)[mask],
                color=color,
                lw=2.0,
                ls="-",
                label=f"{mod_short} shipped",
            )
            ax.plot(
                tx,
                np.asarray(off)[mask],
                color=color,
                lw=2.0,
                ls="-.",
                label=f"{mod_short} offline@τ",
            )
            ax.plot(
                tx,
                np.asarray(ora)[mask],
                color=color,
                lw=1.8,
                ls="--",
                alpha=0.88,
                label=f"{mod_short} valid oracle",
            )

        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            if thr <= x_right + 1e-9:
                ax.axvline(
                    thr,
                    color="0.35",
                    ls=":",
                    lw=1.4,
                    label=f"Official bar ({thr:g})",
                )
            else:
                ax.annotate(
                    f"Official τ={thr:g} (right of axis)",
                    xy=(0.97, 0.1),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=9,
                    color="0.35",
                )
        if arch == "cluster":
            ax.annotate(
                "Offline first key = cluster target_response ≤ 0 (τ on x-axis "
                "does not change that tier).",
                xy=(0.98, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
            )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0.0, x_right)
        ax.set_ylabel("Fraction of pairs")
        ax.set_title(arch.replace("_", " ").title(), loc="left", fontsize=12, pad=10)
        if row == n_arch - 1:
            ax.set_xlabel(r"Threshold $\tau$ (selection bar = evaluation bar)")
        ax.legend(loc="upper right", framealpha=0.94, ncol=2, fontsize=7)

    fig.suptitle(
        "Offline τ-reselection on logged explores vs shipped superscore\n"
        "(per τ: argmax with τ-defined success, then pass valid ∧ frac ≥ τ on winner)",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.07, hspace=0.5)
    return fig


def _shape_rescore_delta_stats_by_modality(
    rescore_csv: Path,
    delta_col: str,
) -> Dict[str, Dict[str, float]]:
    """Per-modality median / quartiles of one ``delta_*`` column (boxplot summary)."""
    buckets: Dict[str, List[float]] = {"multimodal": [], "tools_only": []}
    with open(rescore_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            err = (row.get("error") or "").strip()
            if err:
                continue
            mod = row.get("modality") or ""
            if mod not in buckets:
                continue
            raw = row.get(delta_col)
            if raw is None or raw == "":
                continue
            try:
                x = float(raw)
            except ValueError:
                continue
            if np.isfinite(x):
                buckets[mod].append(x)
    stats: Dict[str, Dict[str, float]] = {}
    for mod, vals in buckets.items():
        if not vals:
            stats[mod] = {
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }
            continue
        a = np.asarray(vals, dtype=np.float64)
        stats[mod] = {
            "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
        }
    return stats


def fig_shape_niceness_offline_tau(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    tau_step: float = 0.02,
    metric_key: str = "npi_mean_worst6",
    direction: str = "minimize",
    figsize: Tuple[float, float] = (9.5, 3.6),
    show_json_shipped: bool = False,
    oracle_top_k: Optional[int] = 6,
    rescore_csv: Optional[Path] = None,
    show_rescore_delta_bands: bool = True,
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """τ sweep for **shape_niceness** only: replay offline metric vs optional JSON primary.

    For each τ (x-axis), the left y-axis is the fraction of pair-runs where the
    condition holds.

    **Curves per modality** (MM / TO), by default:

    - **Offline selected** — ``valid`` ∧ offline fraction (``metric_key``) ≥ τ
      for the superscore-selected replay row.
    - **Offline valid oracle** — best offline fraction among **valid** explores,
      optionally restricted to the **top ``oracle_top_k``** rows by the same
      lexicographic superscore tuple as ``run_dataset._select_best_explored_solution``.

    If ``show_json_shipped=True``, also plot **shipped (JSON)** —
    ``valid`` ∧ logged ``fraction_improved`` ≥ τ.

    When ``rescore_csv`` points to ``rescore.csv`` from
    ``scripts/audit_and_rescore_shape_niceness.py`` and ``show_rescore_delta_bands``,
    a **right-hand axis** shows the **median** and **IQR** of ``delta_{metric_key}``
    (selected − baseline), matching the boxplot summary for the same scalar.

    Offline fractions use ``fraction_improved_directed`` (same normalization as
    ``queries.ArchetypeQuery.score``). Set ``oracle_top_k=None`` for an oracle
    over **all** explores with solutions (legacy behaviour).
    """
    taus_full = tau_grid(tau_step)
    fig, ax = plt.subplots(figsize=figsize)
    rate_rows: List[np.ndarray] = []
    bundles: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for mod, color in [
        ("multimodal", COLOR_MM),
        ("tools_only", COLOR_TO),
    ]:
        paths = paths_by_arch_mod.get(("shape_niceness", mod), [])
        js, osel, orav, _oraf = sweep_shape_offline_tau_for_paths(
            paths,
            taus_full,
            metric_key=metric_key,
            direction=direction,
            oracle_top_k=oracle_top_k,
        )
        bundles.append((mod, color, js, osel, orav))
        if show_json_shipped:
            rate_rows.append(np.asarray(js))
        rate_rows.extend([np.asarray(osel), np.asarray(orav)])

    rates = np.vstack(rate_rows) if rate_rows else np.zeros((1, len(taus_full)))
    eps = 0.02
    active = np.any(rates > eps, axis=0)
    if np.any(active):
        last_i = int(np.where(active)[0][-1])
        x_right = min(1.0, float(taus_full[last_i]) + 3 * tau_step + 0.06)
    else:
        x_right = min(1.0, 0.25 + 3 * tau_step)
    thr = _thr("shape_niceness", official_threshold_overrides)
    if thr is not None:
        x_right = max(x_right, thr + 0.08)
    mask = taus_full <= x_right + 1e-12
    tx = taus_full[mask]

    ora_label_suffix = (
        f"top-{oracle_top_k} valid oracle"
        if oracle_top_k is not None and oracle_top_k > 0
        else "valid oracle (full tree)"
    )

    for mod, color, js, osel, orav in bundles:
        mod_short = "MM" if mod == "multimodal" else "TO"
        if show_json_shipped:
            ax.plot(
                tx,
                np.asarray(js)[mask],
                color=color,
                lw=2.0,
                ls="-",
                label=f"{mod_short} shipped (JSON primary)",
                zorder=3,
            )
        ax.plot(
            tx,
            np.asarray(osel)[mask],
            color=color,
            lw=2.0,
            ls="-.",
            label=f"{mod_short} offline selected ({metric_key})",
            zorder=3,
        )
        ax.plot(
            tx,
            np.asarray(orav)[mask],
            color=color,
            lw=1.75,
            ls="--",
            alpha=0.88,
            label=f"{mod_short} {ora_label_suffix}",
            zorder=3,
        )

    if thr is not None and thr <= x_right + 1e-9:
        ax.axvline(
            thr,
            color="0.35",
            ls=":",
            lw=1.4,
            label=f"Official bar ({thr:g})",
            zorder=2,
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.0, x_right)
    if show_json_shipped:
        ax.set_xlabel(
            r"Threshold $\tau$ on JSON primary fraction or replay offline fraction"
        )
    else:
        ax.set_xlabel(r"Threshold $\tau$ on replay offline fraction")
    ax.set_ylabel("Fraction of pairs")

    delta_col = f"delta_{metric_key}"
    if (
        rescore_csv is not None
        and rescore_csv.is_file()
        and show_rescore_delta_bands
    ):
        stats = _shape_rescore_delta_stats_by_modality(rescore_csv, delta_col)
        ax2 = ax.twinx()
        pooled: List[float] = []
        for mod, color in [
            ("multimodal", COLOR_MM),
            ("tools_only", COLOR_TO),
        ]:
            st = stats.get(mod, {})
            q25, med, q75 = st.get("q25"), st.get("median"), st.get("q75")
            for v in (q25, med, q75):
                if v is not None and np.isfinite(v):
                    pooled.append(float(v))
            if med is not None and np.isfinite(med):
                if (
                    q25 is not None
                    and q75 is not None
                    and np.isfinite(q25)
                    and np.isfinite(q75)
                ):
                    ax2.axhspan(
                        float(q25),
                        float(q75),
                        alpha=0.14,
                        color=color,
                        zorder=0,
                    )
                ax2.axhline(
                    float(med),
                    color=color,
                    ls=":",
                    lw=2.1,
                    alpha=0.9,
                    zorder=1,
                )
        if pooled:
            lo, hi = min(pooled), max(pooled)
            span = hi - lo
            pad = 0.06 * (span if span > 1e-12 else max(abs(hi), 1.0))
            ax2.set_ylim(lo - pad, hi + pad)
        ax2.set_ylabel(
            f"Δ {metric_key}\n(selected − baseline)\nmedian (dots) & IQR (band)",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.11, right=0.88, top=0.88, bottom=0.14)
    else:
        fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.14)

    k_note = (
        f", oracle ∈ top-{oracle_top_k} by superscore"
        if oracle_top_k is not None and oracle_top_k > 0
        else ", oracle over all explores"
    )
    ax.set_title(
        f"Shape niceness — τ sweep ({metric_key}, {direction}{k_note})",
        loc="left",
        fontsize=12,
        pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.94, ncol=2, fontsize=7)
    return fig


def fig_ecdf_oracle(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (12, 8),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """2×2 ECDF of oracle-best valid fraction_improved."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, arch in zip(axes_flat, ARCHETYPE_ORDER):
        for mod, color, ls in [
            ("multimodal", COLOR_MM, "-"),
            ("tools_only", COLOR_TO, "--"),
        ]:
            vals = np.array(
                [
                    r.oracle_best_valid_fraction_improved
                    for r in records
                    if r.archetype == arch and r.modality == mod
                ],
                dtype=np.float64,
            )
            xs, ys = ecdf(vals)
            ax.step(xs, ys, where="post", color=color, ls=ls, lw=2.0, label=mod)
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axvline(thr, color="0.25", ls=":", lw=1.4)
            ax.text(
                thr + 0.01,
                0.08,
                f"official τ={thr:g}",
                rotation=90,
                va="bottom",
                fontsize=9,
                color="0.25",
            )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(arch.replace("_", " ").title())
        ax.set_xlabel("Oracle best valid fraction_improved")
        ax.set_ylabel("ECDF")
        ax.legend(loc="lower right")
    fig.suptitle(
        "How much primary improvement was achievable among feasible explores?\n"
        "(ECDF of max valid fraction_improved within each run)",
        fontsize=13,
    )
    return fig


def fig_ecdf_feasible_oracle_vs_selected(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (12, 8),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """ECDF: **selected** fraction vs **feasible-max** fraction (ignore guard validity)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    for ax, arch in zip(axes.ravel(), ARCHETYPE_ORDER):
        for mod, color, ls in [
            ("multimodal", COLOR_MM, "-"),
            ("tools_only", COLOR_TO, "--"),
        ]:
            sel_vals = np.array(
                [
                    r.selected_fraction_improved
                    for r in records
                    if r.archetype == arch and r.modality == mod
                ],
                dtype=np.float64,
            )
            ora_vals = np.array(
                [
                    r.oracle_max_feasible_fraction_improved
                    for r in records
                    if r.archetype == arch and r.modality == mod
                ],
                dtype=np.float64,
            )
            xs, ys = ecdf(sel_vals)
            ax.step(
                xs,
                ys,
                where="post",
                color=color,
                ls=ls,
                lw=2.0,
                label=f"{mod} selected",
            )
            xs2, ys2 = ecdf(ora_vals)
            ax.step(
                xs2,
                ys2,
                where="post",
                color=color,
                ls=ls,
                lw=1.6,
                alpha=0.55,
                label=f"{mod} max feasible",
            )
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axvline(thr, color="0.25", ls=":", lw=1.2, alpha=0.7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(arch.replace("_", " ").title())
        ax.set_xlabel("fraction_improved")
        ax.set_ylabel("ECDF")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "ECDF: selected (opaque) vs max feasible fraction among logged explores (fainter)\n"
        "Same colour = modality; feasible oracle ignores guard validity.",
        fontsize=12,
    )
    return fig


def fig_scatter_feasible_oracle_vs_selected(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """x = max feasible fraction (ignore guards); y = selected fraction (what shipped)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    lim = (-0.02, 1.02)
    for ax, arch in zip(axes.ravel(), ARCHETYPE_ORDER):
        for mod, marker, color in [
            ("multimodal", "o", COLOR_MM),
            ("tools_only", "^", COLOR_TO),
        ]:
            xs = [
                r.oracle_max_feasible_fraction_improved
                for r in records
                if r.archetype == arch and r.modality == mod
            ]
            ys = [
                r.selected_fraction_improved
                for r in records
                if r.archetype == arch and r.modality == mod
            ]
            ax.scatter(
                xs,
                ys,
                s=36,
                alpha=0.75,
                marker=marker,
                edgecolors="0.15",
                linewidths=0.35,
                facecolors=color,
                label=mod,
            )
        ax.plot(lim, lim, "k--", lw=1.0, alpha=0.4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(arch.replace("_", " ").title())
        ax.set_xlabel("Max feasible fraction (logged explores, guards ignored)")
        ax.set_ylabel("Selected fraction_improved")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Below diagonal: max feasible primary fraction (logged) exceeds what was shipped "
        "(often guard-invalidating). On diagonal: same value on this axis.",
        fontsize=12,
    )
    return fig


def fig_hist_oracle_vs_selected(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (14, 8),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """For each archetype: overlaid histograms of oracle vs selected (pooled modalities)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    bins = np.linspace(0, 1, 23)
    for ax, arch in zip(axes.ravel(), ARCHETYPE_ORDER):
        ora = np.array(
            [r.oracle_best_valid_fraction_improved for r in records if r.archetype == arch],
            dtype=np.float64,
        )
        sel = np.array(
            [r.selected_fraction_improved for r in records if r.archetype == arch],
            dtype=np.float64,
        )
        ax.hist(
            ora,
            bins=bins,
            alpha=0.55,
            label="Oracle (best valid explore)",
            color=COLOR_ORACLE,
            density=True,
        )
        ax.hist(
            sel,
            bins=bins,
            alpha=0.45,
            label="Selected",
            color=COLOR_SELECTED,
            density=True,
        )
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axvline(thr, color="0.2", ls=":", lw=1.5, label=f"Official τ={thr:g}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction_improved")
        ax.set_ylabel("Density")
        ax.set_title(arch.replace("_", " ").title())
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Distribution of primary improvement: selected vs best explored\n"
        "(both modalities pooled per archetype)",
        fontsize=13,
    )
    return fig


def fig_scatter_selected_vs_oracle(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (12, 10),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """One subplot per archetype; points = pair-runs; diagonal = perfect selection."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    lim = (-0.02, 1.02)
    for ax, arch in zip(axes.ravel(), ARCHETYPE_ORDER):
        for mod, marker, color in [
            ("multimodal", "o", COLOR_MM),
            ("tools_only", "^", COLOR_TO),
        ]:
            xs = [
                r.oracle_best_valid_fraction_improved
                for r in records
                if r.archetype == arch and r.modality == mod
            ]
            ys = [
                r.selected_fraction_improved
                for r in records
                if r.archetype == arch and r.modality == mod
            ]
            ax.scatter(
                xs,
                ys,
                s=38,
                alpha=0.75,
                marker=marker,
                edgecolors="0.15",
                linewidths=0.4,
                facecolors=color,
                label=mod,
            )
        ax.plot(lim, lim, "k--", lw=1.0, alpha=0.45, label="y = x (oracle = selected)")
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axvline(thr, color="0.35", ls=":", lw=1)
            ax.axhline(thr, color="0.35", ls=":", lw=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(arch.replace("_", " ").title())
        ax.set_xlabel("Oracle best valid fraction_improved")
        ax.set_ylabel("Selected fraction_improved")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Selection vs exploration ceiling (per pair-run)\n"
        "Points below the diagonal: superscore chose less primary improvement than some valid explore",
        fontsize=13,
    )
    return fig


def fig_oracle_marginal_sensitivity(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    tau_step: float = 0.05,
    figsize: Tuple[float, float] = (12, 9),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Finite-difference slope of oracle pass-rate vs τ (two modalities per archetype)."""
    taus = tau_grid(tau_step)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    for ax, arch in zip(axes.ravel(), ARCHETYPE_ORDER):
        for mod, color, ls in [
            ("multimodal", COLOR_MM, "-"),
            ("tools_only", COLOR_TO, "--"),
        ]:
            paths = paths_by_arch_mod.get((arch, mod), [])
            _, ora = sweep_selected_and_oracle(paths, taus)
            d_tau = np.diff(taus)
            d_rate = np.diff(ora)
            marginal = d_rate / np.maximum(d_tau, 1e-12)
            centers = 0.5 * (taus[:-1] + taus[1:])
            ax.plot(
                centers,
                marginal,
                drawstyle="steps-mid",
                color=color,
                ls=ls,
                lw=2.0,
                label=mod,
            )
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axvline(thr, color="0.35", ls=":", lw=1.2, alpha=0.8)
        ax.axhline(0, color="0.45", lw=0.7)
        ax.set_title(arch.replace("_", " ").title())
        ax.set_xlabel(r"$\tau$ (bin centre)")
        ax.set_ylabel(r"$d\,\mathrm{pass}^{oracle}/d\tau$  (finite diff.)")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Sensitivity of oracle success rate to the improvement bar\n"
        "Positive spikes: many pairs barely clear τ in that band; "
        "drops to ~0 after the mass of pairs is excluded",
        fontsize=13,
    )
    return fig


def fig_violin_oracle_by_modality(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (14, 4.2),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Violin plots: oracle best fraction, split MM vs TO per archetype (readable spacing)."""
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True, constrained_layout=True)
    for ax, arch in zip(axes, ARCHETYPE_ORDER):
        data = []
        labels = []
        for mod in MODALITY_ORDER:
            vals = [
                r.oracle_best_valid_fraction_improved
                for r in records
                if r.archetype == arch and r.modality == mod
            ]
            if vals:
                data.append(vals)
                labels.append(mod.replace("_", "\n"))
        if data:
            parts = ax.violinplot(
                data,
                positions=range(1, len(data) + 1),
                showmeans=True,
                showmedians=False,
                widths=0.55,
            )
            for b in parts["bodies"]:
                b.set_alpha(0.55)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(arch.replace("_", " ").title(), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.axhline(thr, color="0.3", ls=":", lw=1.2)
        ax.set_ylabel("Oracle best valid fraction" if arch == "cluster" else "")
    fig.suptitle(
        "Spread of exploration ceilings (oracle) — multimodal vs tools-only",
        fontsize=13,
    )
    return fig


def fig_heatmap_oracle_pass(
    paths_by_arch_mod: Dict[Tuple[str, str], List[Path]],
    *,
    modality: str = "multimodal",
    tau_step: float = 0.05,
    figsize: Tuple[float, float] = (11, 4.5),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Heatmap: archetype × τ → oracle pass rate (one modality for clarity)."""
    taus = tau_grid(tau_step)
    mat = []
    for arch in ARCHETYPE_ORDER:
        paths = paths_by_arch_mod.get((arch, modality), [])
        _, ora = sweep_selected_and_oracle(paths, taus)
        mat.append(ora)
    Z = np.array(mat, dtype=np.float64)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax.set_yticks(np.arange(Z.shape[0]))
    ax.set_yticklabels([a.replace("_", " ") for a in ARCHETYPE_ORDER])
    ax.set_xticks(np.arange(Z.shape[1]))
    ax.set_xticklabels([f"{t:g}" for t in taus], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(r"Threshold $\tau$ (column = pass if valid ∧ fraction_improved ≥ τ)")
    ax.set_title(f"Oracle pass rate by archetype and τ — {modality}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Fraction of pairs")
    for arch in ARCHETYPE_ORDER:
        thr = _thr(arch, official_threshold_overrides)
        if thr is None:
            continue
        j = int(np.argmin(np.abs(taus - thr)))
        ax.axvline(j, color="white", ls="-", lw=1.8, alpha=0.95)
    return fig


def fig_modality_comparison_bars(
    records: Sequence[PairRecord],
    *,
    figsize: Tuple[float, float] = (10, 5),
    official_threshold_overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Side-by-side success rate: MM vs TO per archetype.

    By default uses logged ``official_success``. Pass ``official_threshold_overrides``
    (e.g. ``{"shape_niceness": 0.02}``) to **recompute** success for those
    archetypes as ``valid`` ∧ ``fraction_improved`` ≥ τ from the shipped score
    only — no API re-run (cluster stays logged-only).
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    x = np.arange(len(ARCHETYPE_ORDER))
    w = 0.36
    mm_rates = []
    to_rates = []
    for arch in ARCHETYPE_ORDER:
        mm = [r for r in records if r.archetype == arch and r.modality == "multimodal"]
        to = [r for r in records if r.archetype == arch and r.modality == "tools_only"]
        mm_rates.append(
            np.mean(
                [
                    record_success_for_threshold_plot(
                        r, official_threshold_overrides=official_threshold_overrides
                    )
                    for r in mm
                ]
            )
            if mm
            else 0.0
        )
        to_rates.append(
            np.mean(
                [
                    record_success_for_threshold_plot(
                        r, official_threshold_overrides=official_threshold_overrides
                    )
                    for r in to
                ]
            )
            if to
            else 0.0
        )
    ax.bar(x - w / 2, mm_rates, width=w, label="multimodal", color=COLOR_MM)
    ax.bar(x + w / 2, to_rates, width=w, label="tools_only", color=COLOR_TO)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in ARCHETYPE_ORDER])
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    if official_threshold_overrides:
        ax.set_title("Benchmark success (counterfactual τ where overridden; else logged)")
    else:
        ax.set_title("Benchmark success (as logged at run time)")
    ax.legend()
    for i, arch in enumerate(ARCHETYPE_ORDER):
        thr = _thr(arch, official_threshold_overrides)
        if thr is not None:
            ax.text(
                i,
                1.02,
                f"frac bar τ={thr:g}",
                ha="center",
                fontsize=8,
                color="0.35",
            )
    return fig
