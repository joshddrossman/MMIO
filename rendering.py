"""Layered renderer: produces the visual the agent's view tool returns.

Supported layers (combine freely):
    'precincts'            faint precinct boundaries + light fill
    'population_density'   voters per km^2 heatmap
    'closed_sites'         all candidate sites (light gray dots)
    'solution'             opened sites (red) + closed sites (gray)
    'assignments'          precinct centroids -> assigned site lines, and
                           a categorical fill colouring each precinct by
                           its assigned opened site (so non-contiguous
                           catchments and oddly-shaped catchments are
                           visible at a glance)

A single annotated region (or list of regions) can be overlaid via the
`region` parameter. Regions are arrays of (x, y) polygon vertices in km.
"""
from typing import List, Optional, Tuple, Union, Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon

from instance import Instance, Solution


KNOWN_LAYERS = {
    'precincts',
    'population_density',
    'closed_sites',
    'solution',
    'assignments',
}


def _heatmap_spec(instance: Instance, layer: str):
    """Return (per-precinct values, cmap name, colorbar label) for the
    population_density heatmap layer, or None."""
    if layer == 'population_density':
        density = np.where(
            instance.precinct_areas > 0,
            instance.precinct_voters / np.maximum(instance.precinct_areas, 1e-6),
            0.0,
        )
        return density, 'YlOrRd', 'Voters per km²'
    return None


def render_all_pair_views(
    instance,
    solution,
    views_dir,
    annotation_polygon=None,
    title_prefix: str = "",
    file_prefix: str = "",
):
    """Render the canonical view set for one dataset pair.

      plain.png                    : precincts + candidate sites (no solution)
      baseline_text_only.png       : solution + assignments (categorical
                                      service-area fill by assigned site)
      baseline_text_annotated.png  : same + annotation polygon (only if
                                      annotation_polygon provided)
      population_density.png       : solution overlaid on voter density
    """
    import os
    from pathlib import Path
    views_dir = Path(views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)

    base_layers = ['closed_sites', 'solution', 'assignments']

    def _save(layers, region, title, fname):
        render_view(
            instance, solution if 'solution' in layers else None,
            layers=layers, region=region,
            title=(title_prefix + title) if title_prefix else title,
            save_path=str(views_dir / f"{file_prefix}{fname}"),
        )

    # (a) Plain map — no solution overlay.
    _save(['closed_sites'], None,
          "Plain map: precincts + candidate sites",
          "plain.png")

    # (b) Solution + assignments (the canonical baseline view).
    _save(base_layers, None,
          "Baseline solution (text-only)",
          "baseline_text_only.png")

    # (c) Solution + annotation polygon.
    if annotation_polygon is not None:
        region = (annotation_polygon if isinstance(annotation_polygon, list)
                   else [np.asarray(annotation_polygon)])
        _save(base_layers, region,
              "Baseline solution + annotation",
              "baseline_text_annotated.png")

    # (d) Density layer (still useful as visual context).
    _save(['population_density', 'solution'], None,
          "Solution + Population density", "population_density.png")


def render_view(
    instance: Instance,
    solution: Optional[Solution] = None,
    layers: Optional[List[str]] = None,
    region: Optional[Union[np.ndarray, Iterable[np.ndarray]]] = None,
    figsize: Tuple[float, float] = (9.5, 9.5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    show_site_labels: bool = True,
    show_precinct_labels: bool = False,
    min_precinct_label_area: float = 0.20,
):
    """Render a layered view. Returns the matplotlib Figure."""
    if layers is None:
        layers = ['precincts']
    unknown = [l for l in layers if l not in KNOWN_LAYERS]
    if unknown:
        raise ValueError(f"Unknown layers: {unknown}. Valid: {sorted(KNOWN_LAYERS)}")

    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin, xmax, ymax = instance.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    label_grid = instance.precinct_label_grid
    xs = instance.grid_xs
    ys = instance.grid_ys

    # Pick a single heatmap-like background layer.
    heat_layer = None
    for l in layers:
        if l == 'population_density':
            heat_layer = l
            break

    if heat_layer is not None:
        values, cmap, cbar_label = _heatmap_spec(instance, heat_layer)
        value_grid = values[label_grid]
        vmin, vmax = float(values.min()), float(values.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(xs, ys, value_grid, cmap=cmap, norm=norm,
                           shading='auto', alpha=0.85, rasterized=True)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(cbar_label)
    elif 'assignments' in layers and solution is not None:
        # CATEGORICAL SERVICE-AREA FILL: each precinct is colored by the
        # opened site it is assigned to. Makes non-contiguous catchments
        # visually obvious (the same color appears in disjoint patches)
        # AND makes catchment shape readable for the shape archetype.
        opened_idx = np.where(solution.x == 1)[0]
        if len(opened_idx) > 0:
            site_to_color = -np.ones(instance.n_sites, dtype=int)
            for k, j in enumerate(opened_idx):
                site_to_color[j] = k
            assigned = solution.y.argmax(axis=1)
            precinct_color_idx = site_to_color[assigned]
            n_open = len(opened_idx)
            cmap_name = 'tab20' if n_open <= 20 else 'gist_ncar'
            color_grid = precinct_color_idx[label_grid]
            color_norm = Normalize(vmin=0, vmax=max(n_open - 1, 1))
            ax.pcolormesh(xs, ys, color_grid, cmap=cmap_name, norm=color_norm,
                          shading='auto', alpha=0.50, rasterized=True)
        else:
            n = instance.n_precincts
            color_field = (np.arange(n) % 5).astype(float) / 5.0
            value_grid = color_field[label_grid]
            ax.pcolormesh(xs, ys, value_grid, cmap='Pastel1', shading='auto',
                          alpha=0.30, rasterized=True)
    elif 'precincts' in layers:
        # Plain precinct fill (variegated pastel) so boundaries are visible.
        n = instance.n_precincts
        color_field = (np.arange(n) % 5).astype(float) / 5.0
        value_grid = color_field[label_grid]
        ax.pcolormesh(xs, ys, value_grid, cmap='Pastel1', shading='auto',
                      alpha=0.30, rasterized=True)

    # Precinct boundaries: ALWAYS drawn so the agent can see precinct
    # structure under any layer combination.
    ax.contour(xs, ys, label_grid.astype(float),
               levels=np.arange(instance.n_precincts + 1) - 0.5,
               colors='gray', linewidths=0.4, alpha=0.6)

    # Closed (candidate) sites
    if 'closed_sites' in layers or 'solution' in layers:
        opened_mask = (solution.x == 1) if solution is not None else np.zeros(instance.n_sites, dtype=bool)
        closed_mask = ~opened_mask
        if closed_mask.any():
            ax.scatter(instance.site_locations[closed_mask, 0],
                       instance.site_locations[closed_mask, 1],
                       marker='o', c='lightgray', s=60,
                       edgecolor='gray', linewidths=1.0,
                       alpha=0.85, zorder=3, label='Candidate site')

    # Solution: opened sites (drawn on top, larger so the index label is legible)
    if 'solution' in layers and solution is not None:
        opened_mask = (solution.x == 1)
        if opened_mask.any():
            ax.scatter(instance.site_locations[opened_mask, 0],
                       instance.site_locations[opened_mask, 1],
                       marker='o', c='red', s=260,
                       edgecolor='black', linewidths=1.6,
                       zorder=6, label='Opened polling place')

    # Site index labels — bigger fonts and a black stroke around opened-site
    # labels so they read clearly when the image is downsampled by a VLM.
    if show_site_labels and ('closed_sites' in layers or 'solution' in layers):
        opened_mask = (solution.x == 1) if solution is not None else np.zeros(instance.n_sites, dtype=bool)
        opened_stroke = [
            patheffects.Stroke(linewidth=2.0, foreground='black'),
            patheffects.Normal(),
        ]
        for j in range(instance.n_sites):
            x_j, y_j = instance.site_locations[j]
            if opened_mask[j]:
                t = ax.text(x_j, y_j, str(j), color='white', fontsize=10,
                            fontweight='bold', ha='center', va='center', zorder=7)
                t.set_path_effects(opened_stroke)
            else:
                ax.text(x_j, y_j + 0.22, str(j), color='#333333', fontsize=8,
                        ha='center', va='bottom', zorder=4, alpha=0.95)

    # Assignment lines (precinct centroid -> assigned site)
    if 'assignments' in layers and solution is not None:
        for i in range(instance.n_precincts):
            j_assigned = np.where(solution.y[i] > 0.5)[0]
            if len(j_assigned) == 0:
                continue
            j = int(j_assigned[0])
            p = instance.precinct_centroids[i]
            s = instance.site_locations[j]
            ax.plot([p[0], s[0]], [p[1], s[1]],
                    color='#1a1a1a', linewidth=1.0, alpha=0.7, zorder=3,
                    solid_capstyle='round')

    # Precinct labels (off by default).
    if show_precinct_labels:
        for i in range(instance.n_precincts):
            if instance.precinct_areas[i] < min_precinct_label_area:
                continue
            cx, cy = instance.precinct_centroids[i]
            ax.text(cx, cy, str(i), color='black', fontsize=6,
                    ha='center', va='center', zorder=4,
                    alpha=0.85,
                    bbox=dict(boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor='none', alpha=0.65))

    # Annotated region(s)
    if region is not None:
        regions = list(region) if (isinstance(region, list) or
                                    (isinstance(region, np.ndarray) and region.ndim == 3)) else [region]
        for r in regions:
            arr = np.asarray(r)
            poly = MplPolygon(arr, closed=True,
                              facecolor='yellow', alpha=0.30,
                              edgecolor='orange', linewidth=2.5, zorder=4)
            ax.add_patch(poly)

    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel('km')
    ax.set_ylabel('km')

    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_, loc='lower right', fontsize=8, framealpha=0.85)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')

    if show:
        plt.show()

    return fig
