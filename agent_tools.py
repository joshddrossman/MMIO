"""Tools the optimization agent uses to identify entities and act on them.

The agent's typical workflow:

    1. View the current solution with one or more layers (e.g. demographic_black)
       to perceive the spatial pattern that motivates the user's critique.
    2. Identify which precincts and sites are relevant to acting on. There are
       two complementary identification paths:
         (a) Visual: the rendered map carries site index labels (and optional
             precinct index labels) so the agent can read indices directly.
         (b) Structured lookup: list_precincts_in_region(polygon),
             get_precinct_at(x, y), list_sites(opened_only=True),
             get_site_at(x, y) return precinct/site indices and metadata
             given a region or coordinate.
       For multi-entity actions (a critique that affects a cluster), the
       structured path is preferred for reliability; the visual labels exist
       for single-entity references and verification.
    3. Build a Proposal: which sites to force open / close, which precincts
       to force-assign to which site.
    4. Call apply_proposal(instance, proposal) to re-solve under the fixings
       and obtain a new Solution.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib.path import Path

from instance import Instance, Solution
from solver import solve_baseline, FixingConstraints
from rendering import render_view


# ---------------------------------------------------------------------------
# Identification: visual -> action indices
# ---------------------------------------------------------------------------
def list_sites(
    instance: Instance,
    solution: Optional[Solution] = None,
    opened_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return structured info on candidate sites.

    Each entry has: index, x, y, type, capacity, opened (bool, if solution given),
    and load (voters assigned, if solution given).
    """
    out: List[Dict[str, Any]] = []
    site_loads = None
    if solution is not None:
        voters = instance.precinct_voters
        site_loads = (voters[:, None] * solution.y).sum(axis=0)
    for j in range(instance.n_sites):
        opened = bool(solution.x[j]) if solution is not None else False
        if opened_only and not opened:
            continue
        info: Dict[str, Any] = {
            'index': j,
            'x': float(instance.site_locations[j, 0]),
            'y': float(instance.site_locations[j, 1]),
            'type': instance.site_types[j],
            'capacity': int(instance.site_capacity[j]),
            'opened': opened,
        }
        if site_loads is not None:
            info['load'] = int(site_loads[j])
        out.append(info)
    return out


def get_site_at(
    instance: Instance,
    x: float,
    y: float,
    max_distance: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """Return the site nearest to (x, y) within max_distance, or None.

    Useful when the agent has identified a site by approximate coordinate
    (e.g. "close the northernmost opened site") and needs its index.
    """
    target = np.array([x, y], dtype=float)
    dists = np.linalg.norm(instance.site_locations - target, axis=1)
    j = int(np.argmin(dists))
    if dists[j] > max_distance:
        return None
    return {
        'index': j,
        'x': float(instance.site_locations[j, 0]),
        'y': float(instance.site_locations[j, 1]),
        'type': instance.site_types[j],
        'capacity': int(instance.site_capacity[j]),
        'distance': float(dists[j]),
    }


def list_precincts_in_region(
    instance: Instance,
    region: np.ndarray,
    centroid_only: bool = True,
) -> List[Dict[str, Any]]:
    """Return precincts whose centroid lies inside the polygon `region`.

    region : (P, 2) polygon vertices in km.
    centroid_only : if True (default), use centroid containment. If False,
                    include any precinct whose label-grid raster has any cell
                    inside the polygon (more inclusive).

    Returned per precinct: index, x, y, voters.
    """
    region_arr = np.asarray(region)
    path = Path(region_arr)

    if centroid_only:
        inside = path.contains_points(instance.precinct_centroids)
        idx = np.where(inside)[0]
    else:
        xs = instance.grid_xs
        ys = instance.grid_ys
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
        in_mask = path.contains_points(pts).reshape(XX.shape)
        labels_in = np.unique(instance.precinct_label_grid[in_mask])
        idx = labels_in

    out: List[Dict[str, Any]] = []
    for i in idx:
        i = int(i)
        out.append({
            'index': i,
            'x': float(instance.precinct_centroids[i, 0]),
            'y': float(instance.precinct_centroids[i, 1]),
            'voters': int(instance.precinct_voters[i]),
        })
    return out


def get_precinct_at(
    instance: Instance,
    x: float,
    y: float,
) -> Optional[Dict[str, Any]]:
    """Return the precinct containing the point (x, y), or None if out of bounds."""
    xmin, ymin, xmax, ymax = instance.bounds
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return None
    G = len(instance.grid_xs)
    ix = int(np.clip(round((x - xmin) / (xmax - xmin) * (G - 1)), 0, G - 1))
    iy = int(np.clip(round((y - ymin) / (ymax - ymin) * (G - 1)), 0, G - 1))
    i = int(instance.precinct_label_grid[iy, ix])
    return {
        'index': i,
        'x': float(instance.precinct_centroids[i, 0]),
        'y': float(instance.precinct_centroids[i, 1]),
        'voters': int(instance.precinct_voters[i]),
    }


def get_precinct_centroids(
    instance: Instance,
    precinct_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Return precinct centroid coordinates, optionally for a subset."""
    if precinct_indices is None:
        idx = list(range(instance.n_precincts))
    else:
        idx = [int(i) for i in precinct_indices]

    centroids: List[Dict[str, Any]] = []
    for i in idx:
        if not (0 <= i < instance.n_precincts):
            continue
        centroids.append({
            "precinct": int(i),
            "x": float(instance.precinct_centroids[i, 0]),
            "y": float(instance.precinct_centroids[i, 1]),
            "voters": int(instance.precinct_voters[i]),
        })

    return {
        "n_precincts": int(instance.n_precincts),
        "n_returned": len(centroids),
        "centroids": centroids,
        "note": "x and y are precinct centroid coordinates in km.",
    }


def get_current_assignments(
    instance: Instance,
    solution: Solution,
    precinct_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Return current precinct -> polling-place assignments.

    Each assignment includes the precinct centroid, voter count, assigned site,
    and assigned travel distance. The optional precinct filter keeps outputs
    manageable when an agent is investigating a specific region.
    """
    if precinct_indices is None:
        idx = list(range(instance.n_precincts))
    else:
        idx = [int(i) for i in precinct_indices]

    assignments: List[Dict[str, Any]] = []
    for i in idx:
        if not (0 <= i < instance.n_precincts):
            continue
        site_arr = np.where(solution.y[i] > 0.5)[0]
        site_index: Optional[int] = int(site_arr[0]) if len(site_arr) else None
        record: Dict[str, Any] = {
            "precinct": int(i),
            "x": float(instance.precinct_centroids[i, 0]),
            "y": float(instance.precinct_centroids[i, 1]),
            "voters": int(instance.precinct_voters[i]),
            "assigned_site": site_index,
        }
        if site_index is not None:
            distance = float(instance.distance_matrix[i, site_index])
            record.update({
                "assigned_distance": distance,
                "weighted_distance": distance * int(instance.precinct_voters[i]),
                "site_x": float(instance.site_locations[site_index, 0]),
                "site_y": float(instance.site_locations[site_index, 1]),
                "site_type": instance.site_types[site_index],
            })
        assignments.append(record)

    return {
        "n_precincts": int(instance.n_precincts),
        "n_returned": len(assignments),
        "assignments": assignments,
        "note": (
            "assignments[k].assigned_site is the polling-place site currently "
            "serving that precinct in the current solution."
        ),
    }


def get_distance_matrix_data(
    instance: Instance,
    solution: Optional[Solution] = None,
    precinct_indices: Optional[Iterable[int]] = None,
    site_indices: Optional[Iterable[int]] = None,
    opened_only: bool = False,
) -> Dict[str, Any]:
    """Return a JSON-friendly precinct -> site distance matrix slice."""
    if precinct_indices is None:
        precinct_idx = list(range(instance.n_precincts))
    else:
        precinct_idx = [
            int(i) for i in precinct_indices
            if 0 <= int(i) < instance.n_precincts
        ]

    if site_indices is None:
        if opened_only and solution is not None:
            site_idx = [int(j) for j in np.where(solution.x == 1)[0]]
        else:
            site_idx = list(range(instance.n_sites))
    else:
        site_idx = [
            int(j) for j in site_indices
            if 0 <= int(j) < instance.n_sites
        ]

    matrix = instance.distance_matrix[np.ix_(precinct_idx, site_idx)]
    return {
        "units": "km",
        "precinct_indices": precinct_idx,
        "site_indices": site_idx,
        "distances": matrix.tolist(),
        "note": (
            "distances[r][c] is the travel distance from "
            "precinct_indices[r] to site_indices[c]."
        ),
    }


# ---------------------------------------------------------------------------
# Action: force_open / force_close / force_assign + apply
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    """A set of agent actions to apply via re-solving the MILP.

    Hard actions (variable fixings):
        force_open: site indices required to be opened.
        force_close: site indices required to be closed.
        force_assign: (precinct_idx, site_idx) pairs pinning an assignment.

    Soft action (objective modification):
        precinct_weight_multipliers: precinct_idx -> multiplier (default 1.0).
            Multiplies that precinct's contribution to the objective by the
            given factor. Use values > 1.0 to prioritize a precinct's travel
            distance; useful for equity-style critiques without forcing a
            specific configuration. The optimizer will then favor solutions
            that reduce the boosted precincts' travel distances, all else
            equal. Multipliers are clamped to [0.0, 100.0] in the solver.
    """
    force_open: List[int] = field(default_factory=list)
    force_close: List[int] = field(default_factory=list)
    force_assign: List[Tuple[int, int]] = field(default_factory=list)
    precinct_weight_multipliers: Dict[int, float] = field(default_factory=dict)

    def to_constraints(self) -> FixingConstraints:
        return FixingConstraints(
            force_open=list(self.force_open),
            force_close=list(self.force_close),
            force_assign=list(self.force_assign),
            precinct_weight_multipliers=dict(self.precinct_weight_multipliers),
        )


def apply_proposal(
    instance: Instance,
    proposal: Proposal,
    time_limit: float = 60.0,
    mip_gap: float = 0.005,
) -> Solution:
    """Re-solve the optimization problem under the proposal's fixings."""
    return solve_baseline(
        instance,
        constraints=proposal.to_constraints(),
        time_limit=time_limit,
        mip_gap=mip_gap,
    )


# ---------------------------------------------------------------------------
# Local-edit helpers (modify a Solution in place WITHOUT re-solving the MILP)
# ---------------------------------------------------------------------------
# Used by the agent's local-search-style tools (force_assign with freeze_rest,
# swap_assignments). Each returns (new_solution, summary_text). The new
# Solution carries an updated y matrix, a fresh objective, and a feasible
# flag based on a capacity check at every opened site. The opened-site
# vector x is unchanged.
#
# Capacity is the only constraint that local edits can violate (each
# precinct still has exactly one assignment by construction, and assigning
# to an opened site preserves the y[i,j] <= x[j] link). Budget on opened
# sites is also unchanged because x doesn't move.

def _capacity_status(instance: 'Instance', y_new: np.ndarray, x: np.ndarray
                       ) -> Tuple[bool, List[Tuple[int, int, int]]]:
    """Compute (is_feasible, [(site, load, capacity), ...]) for overloaded sites."""
    voters = instance.precinct_voters
    site_loads = (voters[:, None] * y_new).sum(axis=0)
    overloaded: List[Tuple[int, int, int]] = []
    for j in np.where(x == 1)[0]:
        load = int(site_loads[j])
        cap = int(instance.site_capacity[j])
        if load > cap:
            overloaded.append((int(j), load, cap))
    return len(overloaded) == 0, overloaded


def apply_local_assignment(
    instance: 'Instance',
    solution: Solution,
    precinct_index: int,
    site_index: int,
) -> Tuple[Solution, str]:
    """Force-assign precinct -> site WITHOUT re-solving the MILP.

    Constraints:
      - target site must already be opened (else error, no change).
      - precinct_index / site_index in range (else error, no change).

    The new Solution has:
      - x unchanged (no opening/closing without an MILP solve).
      - y row updated for the affected precinct.
      - objective recomputed.
      - metadata['feasible'] reflects the capacity check after the move.

    Capacity violations are NOT rejected — the new infeasible solution is
    returned so the agent can see the consequence and recover with another
    local edit. Use this for local-search-style refinement; for changes
    that need to open or close sites, use submit_proposal.
    """
    if not (0 <= precinct_index < instance.n_precincts):
        return solution, (f"ERROR: precinct_index {precinct_index} out of range "
                            f"[0, {instance.n_precincts}).")
    if not (0 <= site_index < instance.n_sites):
        return solution, (f"ERROR: site_index {site_index} out of range "
                            f"[0, {instance.n_sites}).")
    if int(solution.x[site_index]) != 1:
        return solution, (
            f"ERROR: site {site_index} is not currently opened. "
            f"Local force_assign requires the target site to be open. "
            f"Use submit_proposal with force_open + force_assign to open "
            f"a closed site as part of an MILP-based change."
        )

    prev_site_arr = np.where(solution.y[precinct_index] > 0.5)[0]
    if len(prev_site_arr) == 0:
        prev_site = -1
    else:
        prev_site = int(prev_site_arr[0])
    if prev_site == site_index:
        return solution, (
            f"No change: precinct {precinct_index} is already assigned "
            f"to site {site_index}."
        )

    new_y = solution.y.copy()
    if prev_site >= 0:
        new_y[precinct_index, prev_site] = 0
    new_y[precinct_index, site_index] = 1

    voters = instance.precinct_voters
    D = instance.distance_matrix
    new_obj = float((voters[:, None] * D * new_y).sum())

    feasible, overloaded = _capacity_status(instance, new_y, solution.x)

    new_metadata = dict(solution.metadata)
    new_metadata["feasible"] = bool(feasible)
    new_metadata["solver_status"] = "local_force_assign"
    new_metadata["last_action"] = (
        f"force_assign(precinct={precinct_index}, site={site_index}, "
        f"freeze_rest=True)"
    )
    new_sol = Solution(
        x=solution.x.copy(),
        y=new_y,
        objective=new_obj,
        solver_status="local_force_assign",
        metadata=new_metadata,
    )

    moved_voters = int(voters[precinct_index])
    delta = new_obj - float(solution.objective)
    parts = [
        f"force_assign (frozen): precinct {precinct_index} "
        f"({moved_voters} voters) reassigned from site {prev_site} "
        f"to site {site_index}."
    ]
    parts.append(f"Objective {solution.objective:.0f} -> {new_obj:.0f} "
                 f"(delta {delta:+.0f}).")
    if feasible:
        parts.append("Capacity OK at all opened sites.")
    else:
        overload_str = "; ".join(
            f"site {j} (load {load}/{cap})" for j, load, cap in overloaded
        )
        parts.append(
            f"INFEASIBLE — capacity now violated at {overload_str}. "
            f"Reverse this move or reassign other precincts to bring "
            f"loads under capacity before finalizing."
        )
    return new_sol, " ".join(parts)


def apply_local_swap(
    instance: 'Instance',
    solution: Solution,
    precinct_a_index: int,
    precinct_b_index: int,
) -> Tuple[Solution, str]:
    """Swap the assigned sites of two precincts WITHOUT re-solving the MILP.

    Both precincts' currently-assigned sites are necessarily opened (since
    they had assignments). Net capacity change at the two sites is
    +/- (voters_a - voters_b); other sites are unaffected. Returns the
    new Solution and a summary; capacity violations are reported but not
    rejected.
    """
    if precinct_a_index == precinct_b_index:
        return solution, "No change: cannot swap a precinct with itself."
    for i in (precinct_a_index, precinct_b_index):
        if not (0 <= i < instance.n_precincts):
            return solution, (
                f"ERROR: precinct index {i} out of range "
                f"[0, {instance.n_precincts}).")

    site_a_arr = np.where(solution.y[precinct_a_index] > 0.5)[0]
    site_b_arr = np.where(solution.y[precinct_b_index] > 0.5)[0]
    if len(site_a_arr) == 0 or len(site_b_arr) == 0:
        return solution, ("ERROR: one of the precincts has no current "
                            "assignment; cannot swap.")
    site_a = int(site_a_arr[0])
    site_b = int(site_b_arr[0])
    if site_a == site_b:
        return solution, (
            f"No change: both precincts ({precinct_a_index}, "
            f"{precinct_b_index}) are already assigned to the same site "
            f"({site_a})."
        )

    new_y = solution.y.copy()
    new_y[precinct_a_index, site_a] = 0
    new_y[precinct_a_index, site_b] = 1
    new_y[precinct_b_index, site_b] = 0
    new_y[precinct_b_index, site_a] = 1

    voters = instance.precinct_voters
    D = instance.distance_matrix
    new_obj = float((voters[:, None] * D * new_y).sum())

    feasible, overloaded = _capacity_status(instance, new_y, solution.x)

    new_metadata = dict(solution.metadata)
    new_metadata["feasible"] = bool(feasible)
    new_metadata["solver_status"] = "local_swap_assignments"
    new_metadata["last_action"] = (
        f"swap_assignments(precincts={precinct_a_index}/"
        f"{precinct_b_index}, freeze_rest=True)"
    )
    new_sol = Solution(
        x=solution.x.copy(),
        y=new_y,
        objective=new_obj,
        solver_status="local_swap_assignments",
        metadata=new_metadata,
    )

    delta = new_obj - float(solution.objective)
    parts = [
        f"swap_assignments (frozen): precinct {precinct_a_index} "
        f"({int(voters[precinct_a_index])} voters) <-> precinct "
        f"{precinct_b_index} ({int(voters[precinct_b_index])} voters). "
        f"Precinct {precinct_a_index} now -> site {site_b} (was site "
        f"{site_a}); precinct {precinct_b_index} now -> site {site_a} "
        f"(was site {site_b})."
    ]
    parts.append(f"Objective {solution.objective:.0f} -> {new_obj:.0f} "
                 f"(delta {delta:+.0f}).")
    if feasible:
        parts.append("Capacity OK at all opened sites.")
    else:
        overload_str = "; ".join(
            f"site {j} (load {load}/{cap})" for j, load, cap in overloaded
        )
        parts.append(
            f"INFEASIBLE — capacity now violated at {overload_str}. "
            f"Reverse the swap or do further moves before finalizing."
        )
    return new_sol, " ".join(parts)


def get_precinct_adjacency_data(instance: 'Instance') -> Dict[str, Any]:
    """Return precinct adjacency in a JSON-friendly form. Each precinct's
    neighbours come from a 4-connected pass over the precinct label
    raster — i.e. two precincts are neighbours iff their precincts share
    any cell-edge in the rasterized Voronoi map.

    Useful for tools-only agents who want to verify whether a site's
    catchment is contiguous: collect the precincts assigned to that
    site, then run a connected-components search using the adjacency
    list. Two or more components ⇒ a non-contiguous service area.
    """
    from generation import _precinct_adjacency
    adj = _precinct_adjacency(instance.precinct_label_grid)
    return {
        "n_precincts": int(instance.n_precincts),
        "neighbors": [
            sorted(int(v) for v in adj.get(i, []))
            for i in range(instance.n_precincts)
        ],
        "note": (
            "neighbors[i] is the list of precinct indices that share a "
            "boundary with precinct i (4-connected on the rasterized "
            "Voronoi map). Use this to compute connected components on "
            "an opened site's assigned-precinct subgraph: more than one "
            "component means a non-contiguous service area."
        ),
    }


# ---------------------------------------------------------------------------
# View tool: render to PNG bytes for VLM consumption
# ---------------------------------------------------------------------------
def view_solution_png(
    instance: Instance,
    solution: Optional[Solution] = None,
    layers: Optional[List[str]] = None,
    region: Optional[Union[np.ndarray, Iterable[np.ndarray]]] = None,
    show_site_labels: bool = True,
    show_precinct_labels: bool = False,
    title: Optional[str] = None,
) -> bytes:
    """Render and return PNG bytes — the natural payload for a multimodal LLM call."""
    import matplotlib.pyplot as plt
    fig = render_view(
        instance, solution, layers=layers, region=region,
        show_site_labels=show_site_labels,
        show_precinct_labels=show_precinct_labels,
        title=title,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
