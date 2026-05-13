"""Renderer v2_no_markers — v2 minus markers/lines, plus catchment outlines.

Identical visual design to v2 (saturated tab10/tab20 categorical fill,
high alpha) except:
  - NO opened-site red circles
  - NO closed-site gray dots
  - NO site index labels (opened or closed)
  - NO assignment lines
  - NO precinct boundaries
  - ADDED: a black outline around each opened catchment (the union of
    precincts assigned to that polling place)

The map shows the categorical service-area fill plus a clear black
border around each catchment. Subdivisions WITHIN a catchment
(individual precinct boundaries) are absent — only catchment-level
borders remain. The agent must rely entirely on color-region
perception, but the catchment boundary signal is unambiguous.

Pair with prompts/with_attribution_color.py (or any tool-using prompt)
to test whether structured tools can substitute for the visual handles
v2 normally provides while still preserving catchment-shape clarity.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

# Make instance_generator importable for Instance/Solution pickle classes.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "v2_no_markers"

# Renderer capabilities — the whole point of this renderer is to deny
# the model both site identities AND closed-candidate identities. The
# perception harness's coverage_gap and cluster tasks (which require
# candidate / site markers) reject this renderer via is_valid_view.
HAS_SITE_MARKERS = False
HAS_CANDIDATE_MARKERS = False
HAS_ASSIGNMENT_LINES = False


def _assignment_colormap(n_open: int) -> Tuple[ListedColormap, Normalize]:
    """Same palette as v2 — keep both renderers visually consistent so
    the only difference between them is the marker layer."""
    if n_open <= 1:
        return ListedColormap(["#d62728"]), Normalize(vmin=0, vmax=1)
    if n_open <= 10:
        cmap = plt.cm.get_cmap("tab10", n_open)
        return cmap, Normalize(vmin=0, vmax=max(n_open - 1, 1))
    if n_open <= 20:
        cmap = plt.cm.get_cmap("tab20", n_open)
        return cmap, Normalize(vmin=0, vmax=n_open - 1)
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
           outline_color: str = "black",
           outline_width: float = 1.8,
           figsize: Tuple[float, float] = (9.5, 9.5)) -> bytes:
    """Render the marker-free view (with black catchment outlines) as PNG bytes."""
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

        # Black catchment outlines — one per opened polling place. Each
        # catchment's mask is the set of cells whose precinct is
        # assigned to that site; ax.contour at level=0.5 traces the
        # boundary of that mask. Precinct-internal boundaries don't
        # appear (we never call contour on the precinct label_grid),
        # so the only black lines on the map are catchment-to-catchment
        # / catchment-to-edge borders.
        cell_site = assigned[label_grid]
        for j in opened_idx:
            mask = (cell_site == j).astype(float)
            if mask.sum() == 0:
                continue
            ax.contour(xs, ys, mask, levels=[0.5],
                       colors=[outline_color],
                       linewidths=outline_width, alpha=0.95)
    else:
        # No solution — fall back to a plain saturated fill so the map
        # still has visible content for debugging.
        n = instance.n_precincts
        plain_colors = ["#1f77b4", "#ff7f0e", "#2ca02c",
                        "#d62728", "#9467bd"]
        cmap = ListedColormap(plain_colors)
        color_field = (np.arange(n) % len(plain_colors)).astype(int)
        value_grid = color_field[label_grid]
        ax.pcolormesh(xs, ys, value_grid, cmap=cmap,
                      norm=Normalize(vmin=0, vmax=len(plain_colors) - 1),
                      shading="auto", alpha=0.78, rasterized=True)

    # NO precinct boundaries, NO site markers, NO labels, NO assignment
    # lines. The black catchment outlines above are the only line signal.

    ax.set_xlabel("km")
    ax.set_ylabel("km")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
