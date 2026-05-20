"""Renderer v2_legend — v2 with a color→site legend on the right.

Same visual as v2_no_markers (no red circles, no closed-site dots, no
gray precinct boundaries, dark per-catchment outlines) PLUS a legend
panel on the right of the map associating each color with its opened
polling-place index. Lighter-touch attribution help than
v2_patch_labels: the LLM still has to color-match between the map and
the legend, but it isn't left guessing which palette slot owns which
site.

Layout: the figure is widened (12.5 in) to fit the legend without
distorting the square map area. Legend is a 2-column block of colored
patches anchored just outside the right edge of the axes.

Pair with any prompt; `view_info["has_site_markers"]` reports True so
prompts and tasks (including cluster) that key on it accept this
renderer.
"""
from __future__ import annotations

import colorsys
import io
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize, to_rgb

# Make instance_generator importable so Instance/Solution pickles load.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "v2_legend"

# Renderer capabilities — read by eval_perception.py to build view_info.
# Opened-site identity IS available via the legend, so HAS_SITE_MARKERS
# is True. The legend covers OPENED sites only — closed candidates are
# not shown on the map and not enumerated in the legend, so
# HAS_CANDIDATE_MARKERS is False (coverage_gap perception, which needs
# to point at a closed candidate, will reject this renderer).
HAS_SITE_MARKERS = True
HAS_CANDIDATE_MARKERS = False
HAS_ASSIGNMENT_LINES = False


PALETTE_18: List[str] = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000",
    "#808000", "#000075", "#fabed4", "#aaffc3", "#808080", "#2f4f4f",
]


def _darken(hex_color: str, factor: float = 0.55) -> tuple:
    r, g, b = to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, l * factor)
    return colorsys.hls_to_rgb(h, l, s)


def render(instance, solution, *,
           dpi: int = 120,
           fill_alpha: float = 0.75,
           outline_width: float = 1.6,
           outline_darken: float = 0.55,
           figsize: tuple = (12.5, 9.5),
           legend_fontsize: int = 10,
           legend_ncol: int = 2) -> bytes:
    """Render the v2 view + a right-side color-to-site legend."""
    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin, xmax, ymax = instance.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    label_grid = instance.precinct_label_grid
    xs = instance.grid_xs
    ys = instance.grid_ys

    opened_idx = np.where(solution.x == 1)[0]
    n_open = len(opened_idx)

    site_to_slot = -np.ones(instance.n_sites, dtype=int)
    for k, j in enumerate(opened_idx):
        site_to_slot[j] = k % len(PALETTE_18)

    # Categorical service-area fill.
    if n_open > 0:
        assigned = solution.y.argmax(axis=1)
        precinct_color_idx = site_to_slot[assigned]
        color_grid = precinct_color_idx[label_grid]
        n_slots = min(n_open, len(PALETTE_18))
        cmap = ListedColormap(PALETTE_18[:max(n_slots, 1)])
        norm = Normalize(vmin=0, vmax=max(n_slots - 1, 1))
        ax.pcolormesh(xs, ys, color_grid, cmap=cmap, norm=norm,
                      shading="auto", alpha=fill_alpha, rasterized=True)

    # Per-catchment dark outlines (same as v2_no_markers).
    if n_open > 0:
        cell_site = assigned[label_grid]
        for k, j in enumerate(opened_idx):
            mask = (cell_site == j).astype(float)
            if mask.sum() == 0:
                continue
            slot_color = PALETTE_18[k % len(PALETTE_18)]
            outline_color = _darken(slot_color, factor=outline_darken)
            ax.contour(xs, ys, mask, levels=[0.5],
                       colors=[outline_color],
                       linewidths=outline_width, alpha=0.95)

    # Color legend on the right. One entry per opened site, sorted by
    # site index so the legend reads in numeric order regardless of
    # the order opened_idx happens to be in.
    if n_open > 0:
        order = np.argsort(opened_idx)
        handles = []
        for pos in order:
            site_idx = int(opened_idx[pos])
            slot = int(pos) % len(PALETTE_18)
            color = PALETTE_18[slot]
            # alpha matches the fill so the legend swatch matches what's
            # actually drawn on the map (post-alpha-blend).
            handles.append(mpatches.Patch(
                color=color, label=f"Site {site_idx}",
                alpha=fill_alpha,
            ))
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=legend_fontsize,
            title="Polling place colors",
            title_fontsize=legend_fontsize + 1,
            ncol=legend_ncol,
            handlelength=1.5,
            handleheight=1.0,
            borderpad=0.6,
            labelspacing=0.4,
        )

    ax.set_xlabel("km")
    ax.set_ylabel("km")
    fig.tight_layout()

    buf = io.BytesIO()
    # bbox_inches="tight" preserves the legend that sits outside the axes.
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
