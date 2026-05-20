"""Baseline renderer — exactly the view the optimization agent currently sees.

This is the canonical 'baseline_text_only' layer combination from
instance_generator/rendering.py: closed_sites + solution + assignments,
with categorical service-area fill colored by assigned opened site.

Keep this file thin. The contract every renderer must satisfy:

    render(instance, solution, **kwargs) -> bytes  (PNG)

Variants (high_contrast, thick_outlines, npi_overlay, ...) live in sibling
modules and may take additional kwargs but must keep the (instance,
solution) signature compatible so the runner can swap them by name.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Make instance_generator importable.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))

import matplotlib.pyplot as plt
from rendering import render_view


NAME = "base"

# Renderer capabilities — read by eval_perception.py to build view_info,
# which the prompt + task modules consult to pick text variants. Defaults
# (when a renderer omits these flags) are conservative: assume markers are
# visible, assume assignment lines are NOT.
HAS_SITE_MARKERS = True        # opened-site identity is visible somewhere
HAS_CANDIDATE_MARKERS = True   # closed candidate sites visible w/ index labels
HAS_ASSIGNMENT_LINES = True    # precinct-centroid -> site segments

LAYERS = ["closed_sites", "solution", "assignments"]


def render(instance, solution, *,
           show_precinct_labels: bool = False,
           dpi: int = 120) -> bytes:
    """Render the canonical baseline view as PNG bytes."""
    fig = render_view(
        instance, solution,
        layers=LAYERS,
        show_precinct_labels=show_precinct_labels,
        title=None,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
