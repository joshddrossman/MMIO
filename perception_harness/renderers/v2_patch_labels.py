"""Renderer v2_patch_labels — labels overlaid on each colored patch.

Same visual as v2_no_markers (no red circles, no closed-site dots, no
gray precinct boundaries, dark per-catchment outlines on the same
18-color palette) PLUS a site-index label drawn inside every connected
component of every opened catchment.

  - A contiguous catchment gets ONE label at the cell-weighted
    centroid of its only component.
  - A non-contiguous catchment gets ONE label per disjoint component.

This makes attribution near-trivial: the LLM reads the index off the
patch instead of having to color-match against a separate marker.
That's an aggressive intervention — for the contiguity perception
task it gives away the topology directly (you can see one or two
labels of the same number) — but it cleanly isolates "can the VLM
attribute correctly" from "can the VLM perceive the split topology."
For shape and cluster the labels just make attribution easier without
revealing the answer.

Pair freely with any prompt; the resulting `view_info["has_site_markers"]`
will report True so cluster (which requires markers when a map is
shown) accepts this renderer.
"""
from __future__ import annotations

import colorsys
import io
import sys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import ListedColormap, Normalize, to_rgb

# Make instance_generator importable so we can reuse its connected-
# components / adjacency helpers.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "v2_patch_labels"

# Renderer capabilities — read by eval_perception.py to build view_info.
# Opened-site identity IS visible (patch labels placed inside each
# catchment piece), so HAS_SITE_MARKERS is True. Closed candidates are
# NOT drawn at all in this renderer, so HAS_CANDIDATE_MARKERS is False
# (coverage_gap perception, which points at a closed candidate, will
# reject this renderer via is_valid_view).
HAS_SITE_MARKERS = True
HAS_CANDIDATE_MARKERS = False
HAS_ASSIGNMENT_LINES = False


# Same 18-color palette as v2 / v2_no_markers — keep all three in sync.
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
           figsize: tuple = (9.5, 9.5),
           label_fontsize: int = 12,
           label_stroke_width: float = 2.5) -> bytes:
    """Render the v2 view + per-component patch labels and return PNG bytes."""
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

    # Per-catchment dark outline in a darker shade of the fill color.
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

    # Per-component labels. For each opened site, find the connected
    # components of its assigned-precinct subgraph; draw one label per
    # component at the cell-weighted centroid of that component.
    if n_open > 0:
        from generation import _precinct_adjacency, _connected_components
        adj = _precinct_adjacency(instance.precinct_label_grid)
        XX, YY = np.meshgrid(xs, ys)
        stroke = [
            patheffects.Stroke(linewidth=label_stroke_width,
                               foreground="black"),
            patheffects.Normal(),
        ]
        for j in opened_idx:
            members = np.where(assigned == j)[0]
            if len(members) == 0:
                continue
            comps = _connected_components(members, adj)
            for comp in comps:
                # Cell-weighted centroid of this component.
                comp_set = set(int(p) for p in comp)
                comp_mask = np.isin(label_grid, list(comp_set))
                if not comp_mask.any():
                    continue
                cx = float(XX[comp_mask].mean())
                cy = float(YY[comp_mask].mean())
                t = ax.text(
                    cx, cy, str(int(j)),
                    color="white", fontsize=label_fontsize,
                    fontweight="bold", ha="center", va="center",
                    zorder=7,
                )
                t.set_path_effects(stroke)

    ax.set_xlabel("km")
    ax.set_ylabel("km")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
