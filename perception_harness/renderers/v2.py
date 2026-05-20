"""Renderer v2 — canonical view (partner's rendering style).

Adapted from rendering_vCat.py (partner's renderer used in the
optimization side) into the renderer-module API the perception
harness expects: a single `render(instance, solution, **kwargs)
-> bytes` entry point plus capability flags.

Visual design (preserved from vCat):
  - Saturated categorical fill on each precinct (tab10 / tab20 /
    tab20+tab20b depending on opened-site count) at high alpha (0.90).
  - Strong precinct boundaries in dark grey.
  - Closed candidate sites: light fill (#f5f5f5) with dark border,
    medium size, index labels above marker.
  - Opened polling places: red, larger, on top, with white-on-stroke
    bold index labels in the marker.
  - Assignment lines from each precinct centroid to its assigned site.

This is the single canonical visual the optimization agent and the
perception harness both see — so any improvements to it propagate
across both pipelines.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import ListedColormap, Normalize

# Make instance_generator importable for Instance/Solution pickle classes.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "v2"

# Renderer capabilities — read by eval_perception.py to build view_info,
# which the prompt + task modules consult to pick text variants.
HAS_SITE_MARKERS = True        # opened-site identity is visible
HAS_CANDIDATE_MARKERS = True   # closed candidates drawn as gray dots
HAS_ASSIGNMENT_LINES = True    # precinct -> assigned-site segments


def _assignment_colormap(n_open: int) -> Tuple[ListedColormap, Normalize]:
    """Distinct categorical colors for opened-site catchments, sized to n_open."""
    if n_open <= 1:
        return ListedColormap(["#d62728"]), Normalize(vmin=0, vmax=1)
    if n_open <= 10:
        cmap = plt.cm.get_cmap("tab10", n_open)
        return cmap, Normalize(vmin=0, vmax=max(n_open - 1, 1))
    if n_open <= 20:
        cmap = plt.cm.get_cmap("tab20", n_open)
        return cmap, Normalize(vmin=0, vmax=n_open - 1)
    # tab20 + tab20b => up to 40 clearly separated hues.
    t20 = plt.cm.tab20(np.linspace(0, 1, 20, endpoint=False))
    t20b = plt.cm.tab20b(np.linspace(0, 1, 20, endpoint=False))
    merged = np.vstack([t20, t20b])
    if n_open <= len(merged):
        cmap = ListedColormap(merged[:n_open])
    else:
        cmap = ListedColormap(
            np.vstack([merged] * ((n_open + 39) // 40))[:n_open])
    return cmap, Normalize(vmin=0, vmax=n_open - 1)


def render(instance, solution, *,
           dpi: int = 120,
           fill_alpha: float = 0.90,
           figsize: Tuple[float, float] = (9.5, 9.5),
           show_precinct_labels: bool = False,
           min_precinct_label_area: float = 0.20) -> bytes:
    """Render the canonical view as PNG bytes."""
    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin, xmax, ymax = instance.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    label_grid = instance.precinct_label_grid
    xs = instance.grid_xs
    ys = instance.grid_ys

    opened_idx = np.where(solution.x == 1)[0] if solution is not None \
        else np.array([], dtype=int)
    n_open = len(opened_idx)

    # Categorical service-area fill. Each precinct gets the colour of
    # its assigned opened polling place.
    if n_open > 0:
        site_to_color = -np.ones(instance.n_sites, dtype=int)
        for k, j in enumerate(opened_idx):
            site_to_color[j] = k
        assigned = solution.y.argmax(axis=1)
        precinct_color_idx = site_to_color[assigned]
        color_grid = precinct_color_idx[label_grid]
        cmap, norm = _assignment_colormap(n_open)
        ax.pcolormesh(xs, ys, color_grid, cmap=cmap, norm=norm,
                      shading="auto", alpha=fill_alpha, rasterized=True)
    else:
        # No solution provided — fall back to a plain saturated fill
        # so precincts are still visible.
        n = instance.n_precincts
        plain_colors = ["#1f77b4", "#ff7f0e", "#2ca02c",
                        "#d62728", "#9467bd"]
        cmap = ListedColormap(plain_colors)
        color_field = (np.arange(n) % len(plain_colors)).astype(int)
        value_grid = color_field[label_grid]
        ax.pcolormesh(xs, ys, value_grid, cmap=cmap,
                      norm=Normalize(vmin=0, vmax=len(plain_colors) - 1),
                      shading="auto", alpha=0.78, rasterized=True)

    # Precinct boundaries — strong dark lines so structure reads at
    # any layer combination.
    ax.contour(xs, ys, label_grid.astype(float),
               levels=np.arange(instance.n_precincts + 1) - 0.5,
               colors="#1a1a1a", linewidths=0.55, alpha=0.72)

    # Closed candidate sites — light fill, dark border, medium size.
    if solution is not None:
        opened_mask = (solution.x == 1)
        closed_mask = ~opened_mask
        if closed_mask.any():
            ax.scatter(instance.site_locations[closed_mask, 0],
                       instance.site_locations[closed_mask, 1],
                       marker="o", c="#f5f5f5", s=72,
                       edgecolor="#0d0d0d", linewidths=1.25,
                       alpha=0.95, zorder=3, label="Candidate site")

        # Opened sites — red, large, on top.
        if opened_mask.any():
            ax.scatter(instance.site_locations[opened_mask, 0],
                       instance.site_locations[opened_mask, 1],
                       marker="o", c="red", s=260,
                       edgecolor="black", linewidths=1.6,
                       zorder=6, label="Opened polling place")

        # Site index labels — bigger fonts and a black stroke around
        # opened-site labels so they read clearly under VLM downsampling.
        opened_stroke = [
            patheffects.Stroke(linewidth=2.0, foreground="black"),
            patheffects.Normal(),
        ]
        for j in range(instance.n_sites):
            x_j, y_j = instance.site_locations[j]
            if opened_mask[j]:
                t = ax.text(x_j, y_j, str(j), color="white",
                            fontsize=10, fontweight="bold",
                            ha="center", va="center", zorder=7)
                t.set_path_effects(opened_stroke)
            else:
                ax.text(x_j, y_j + 0.22, str(j), color="#333333",
                        fontsize=8, ha="center", va="bottom",
                        zorder=4, alpha=0.95)

        # Assignment lines (precinct centroid -> assigned site).
        for i in range(instance.n_precincts):
            j_assigned = np.where(solution.y[i] > 0.5)[0]
            if len(j_assigned) == 0:
                continue
            j = int(j_assigned[0])
            p = instance.precinct_centroids[i]
            s = instance.site_locations[j]
            ax.plot([p[0], s[0]], [p[1], s[1]],
                    color="#1a1a1a", linewidth=1.0, alpha=0.7,
                    zorder=3, solid_capstyle="round")

    # Optional precinct-index labels (off by default).
    if show_precinct_labels:
        for i in range(instance.n_precincts):
            if instance.precinct_areas[i] < min_precinct_label_area:
                continue
            cx, cy = instance.precinct_centroids[i]
            ax.text(cx, cy, str(i), color="black", fontsize=6,
                    ha="center", va="center", zorder=4, alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.12",
                              facecolor="white", edgecolor="none",
                              alpha=0.65))

    ax.set_xlabel("km")
    ax.set_ylabel("km")
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_, loc="lower right",
                  fontsize=8, framealpha=0.85)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
