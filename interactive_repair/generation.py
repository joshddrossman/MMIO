"""Procedural generation of polling place location instances.

Pipeline:
  1. Generate county geometry (bounding box, river polyline, bridges).
  2. Sample N precinct seeds with non-uniform spatial density.
  3. Compute Voronoi tessellation as a label raster (no scipy needed).
  4. Compute precinct centroids and areas from the raster.
  5. Compute spatially-correlated demographic fields per precinct.
  6. Compute voter counts based on regional density.
  7. Sample M candidate sites with capacities.
  8. Compute precinct->site distance matrix with river-barrier penalty.

Numpy-only; no scipy/shapely dependencies.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

from instance import Instance


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
def default_params() -> Dict[str, Any]:
    """Return a default parameter dict for instance generation."""
    return {
        # Geometry
        'bounds': (0.0, 0.0, 10.0, 10.0),
        'grid_resolution': 200,
        # Counts
        'n_precincts': 80,
        'n_sites': 40,
        'K': 18,
        'target_total_voters': 50_000,
        # Internal y-line that splits the county into a denser south
        # (urban / suburban clusters) and sparser north / rural fringes.
        # This is just a sampling parameter; nothing geographic is rendered
        # at this y-coordinate.
        'y_split': 5.0,
        # Precinct seed distribution
        'south_neighborhood_centers': [(2.5, 3.0), (5.0, 2.2), (7.5, 3.3)],
        'south_neighborhood_sigma': 0.7,
        'frac_south_urban': 0.42,
        'frac_north_suburban': 0.40,
        # remainder is rural
        # Voter density (per km^2)
        'urban_density': 800,
        'suburban_density': 400,
        'rural_density': 80,
        'reg_rate_range': (0.6, 0.8),
        # Sites: undersupplied on the south side relative to its voter
        # share, so the baseline opens fewer sites than would be ideal
        # there — useful for the cluster / coverage_gap archetypes.
        'frac_sites_south': 0.22,
        'frac_sites_north': 0.55,
        # remainder rural
        'capacity_map': {
            'school': 5000,
            'library': 3000,
            'community_center': 4000,
            'church': 3500,
        },
    }


# ---------------------------------------------------------------------------
# Precinct seeds and Voronoi raster
# ---------------------------------------------------------------------------
def sample_precinct_seeds(rng: np.random.Generator, params: Dict[str, Any]) -> np.ndarray:
    """Sample N precinct centroid seeds with non-uniform spatial density.

    Distribution:
      - frac_south_urban: clustered around three south-side neighborhood centers.
      - frac_north_suburban: roughly uniform over the north half.
      - remainder: scattered along the east/west and far-south rural fringes.
    """
    xmin, ymin, xmax, ymax = params['bounds']
    river_y = params['y_split']
    N = params['n_precincts']
    fS = params['frac_south_urban']
    fN = params['frac_north_suburban']
    n_south = int(round(N * fS))
    n_north = int(round(N * fN))
    n_rural = N - n_south - n_north

    south_centers = np.array(params['south_neighborhood_centers'])
    sigma_s = params['south_neighborhood_sigma']
    south_seeds_list = []
    per_cluster = n_south // len(south_centers)
    extras = n_south - per_cluster * len(south_centers)
    for i, c in enumerate(south_centers):
        n_this = per_cluster + (1 if i < extras else 0)
        s = rng.normal(c, sigma_s, (n_this, 2))
        south_seeds_list.append(s)
    south_seeds = np.vstack(south_seeds_list) if south_seeds_list else np.zeros((0, 2))

    north_seeds = rng.uniform(
        [xmin + 0.5, river_y + 0.5],
        [xmax - 0.5, ymax - 0.4],
        (n_north, 2),
    )

    rural_seeds: List[List[float]] = []
    for _ in range(n_rural):
        side = rng.choice(['east', 'west', 'far_south_west', 'far_south_east'])
        if side == 'east':
            rural_seeds.append([rng.uniform(8.7, 9.8), rng.uniform(0.4, 9.6)])
        elif side == 'west':
            rural_seeds.append([rng.uniform(0.3, 1.4), rng.uniform(0.4, 9.6)])
        elif side == 'far_south_west':
            rural_seeds.append([rng.uniform(0.3, 4.0), rng.uniform(0.3, 1.2)])
        else:
            rural_seeds.append([rng.uniform(6.0, 9.7), rng.uniform(0.3, 1.2)])
    rural_seeds_arr = np.array(rural_seeds) if rural_seeds else np.zeros((0, 2))

    seeds = np.vstack([south_seeds, north_seeds, rural_seeds_arr])
    seeds[:, 0] = np.clip(seeds[:, 0], xmin + 0.1, xmax - 0.1)
    seeds[:, 1] = np.clip(seeds[:, 1], ymin + 0.1, ymax - 0.1)
    return seeds


def compute_voronoi_raster(
    seeds: np.ndarray, params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raster Voronoi: assign each grid cell to its nearest seed.

    Returns (label_grid, xs, ys), where label_grid has shape (G_y, G_x) and
    label_grid[iy, ix] is the seed index of the precinct at (xs[ix], ys[iy]).
    """
    xmin, ymin, xmax, ymax = params['bounds']
    G = params['grid_resolution']
    xs = np.linspace(xmin, xmax, G)
    ys = np.linspace(ymin, ymax, G)
    XX, YY = np.meshgrid(xs, ys)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (G*G, 2)
    # Memory-efficient: chunk over grid rows to keep dist matrix manageable
    labels_flat = np.empty(points.shape[0], dtype=np.int32)
    chunk = 4000
    for s in range(0, points.shape[0], chunk):
        e = s + chunk
        d2 = ((points[s:e, None, :] - seeds[None, :, :]) ** 2).sum(-1)  # (chunk, N)
        labels_flat[s:e] = np.argmin(d2, axis=1)
    labels = labels_flat.reshape(G, G)
    return labels, xs, ys


def compute_precinct_attributes(
    label_grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, seeds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centroid and area of each precinct from the raster."""
    N = len(seeds)
    cell_area = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    XX, YY = np.meshgrid(xs, ys)

    centroids = np.zeros((N, 2))
    areas = np.zeros(N)

    flat_labels = label_grid.ravel()
    flat_x = XX.ravel()
    flat_y = YY.ravel()

    counts = np.bincount(flat_labels, minlength=N)
    sum_x = np.bincount(flat_labels, weights=flat_x, minlength=N)
    sum_y = np.bincount(flat_labels, weights=flat_y, minlength=N)

    nonzero = counts > 0
    centroids[nonzero, 0] = sum_x[nonzero] / counts[nonzero]
    centroids[nonzero, 1] = sum_y[nonzero] / counts[nonzero]
    centroids[~nonzero] = seeds[~nonzero]
    areas[:] = counts * cell_area

    return centroids, areas


# ---------------------------------------------------------------------------
# Voters
# ---------------------------------------------------------------------------
def compute_voters(
    centroids: np.ndarray,
    areas: np.ndarray,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Voters per precinct = area * regional density * registration rate, normalised to a target total."""
    N = len(centroids)
    river_y = params['y_split']
    xmin, _, xmax, _ = params['bounds']
    raw = np.zeros(N)
    for i in range(N):
        x, y = centroids[i]
        if y < river_y:
            d = params['urban_density']
        elif y < river_y + 2.5:
            d = params['suburban_density']
        else:
            d = params['suburban_density'] * 0.6
        # Reduce on far edges (rural)
        edge_dist = min(x - xmin, xmax - x)
        if edge_dist < 1.5:
            d *= max(0.18, 0.18 + 0.6 * (edge_dist / 1.5))
        raw[i] = areas[i] * d

    rr_lo, rr_hi = params['reg_rate_range']
    raw *= rng.uniform(rr_lo, rr_hi, N)

    target = params['target_total_voters']
    scale = target / max(raw.sum(), 1e-9)
    voters = (raw * scale).round().astype(int)
    voters = np.maximum(voters, 5)
    return voters


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------
def generate_sites(
    rng: np.random.Generator, params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Sample candidate site locations, types, and capacities."""
    xmin, ymin, xmax, ymax = params['bounds']
    river_y = params['y_split']
    M = params['n_sites']
    fS = params['frac_sites_south']
    fN = params['frac_sites_north']
    n_south = int(round(M * fS))
    n_north = int(round(M * fN))
    n_rural = M - n_south - n_north

    south_centers = np.array(params['south_neighborhood_centers'])
    sites: List[np.ndarray] = []
    for _ in range(n_south):
        c = south_centers[rng.integers(len(south_centers))]
        loc = rng.normal(c, 1.2, 2)
        loc[1] = np.clip(loc[1], 0.4, river_y - 0.4)
        loc[0] = np.clip(loc[0], xmin + 0.3, xmax - 0.3)
        sites.append(loc)

    for _ in range(n_north):
        loc = rng.uniform([xmin + 0.4, river_y + 0.4], [xmax - 0.4, ymax - 0.4], 2)
        sites.append(loc)

    for _ in range(n_rural):
        side = rng.choice(['east', 'west', 'far_south'])
        if side == 'east':
            loc = np.array([rng.uniform(8.7, 9.6), rng.uniform(0.5, 9.5)])
        elif side == 'west':
            loc = np.array([rng.uniform(0.4, 1.3), rng.uniform(0.5, 9.5)])
        else:
            loc = np.array([rng.uniform(2.0, 8.0), rng.uniform(0.4, 1.0)])
        sites.append(loc)

    site_arr = np.vstack(sites)

    type_choices = ['school', 'library', 'community_center', 'church']
    weights = [0.30, 0.25, 0.25, 0.20]
    site_types = list(rng.choice(type_choices, M, p=weights))

    cap_map = params['capacity_map']
    capacities = np.array([cap_map[t] for t in site_types], dtype=int)
    return site_arr, capacities, site_types


# ---------------------------------------------------------------------------
# Distance matrix (pure Euclidean)
# ---------------------------------------------------------------------------
def compute_distance_matrix(
    precinct_centroids: np.ndarray,
    site_locations: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Precinct -> site Euclidean distance matrix (N, M)."""
    diff = precinct_centroids[:, None, :] - site_locations[None, :, :]
    return np.linalg.norm(diff, axis=2)


# ---------------------------------------------------------------------------
# Archetype-specific instance generators (instance-from-query design)
# ---------------------------------------------------------------------------
#
# These generators produce instances *engineered* to exhibit a specific
# archetype's target property. Each returns (instance, baseline_solution,
# metadata) where metadata identifies the affected entities (e.g. the
# precincts that constitute a coverage_gap hole). Queries are then templated on
# the metadata, which guarantees the query is meaningful for the instance.
#
# Each generator must guarantee not just that the property is present, but
# also that AT LEAST ONE feasible action sequence improves the target
# metric without violating guards (a "fix path"). Without this, the agent
# has nothing to do; the experiment is uninformative.
# ---------------------------------------------------------------------------

def _rebuild_instance_with_site_subset(
    base: 'Instance', keep_mask: np.ndarray,
) -> 'Instance':
    """Return a copy of `base` with only the sites where keep_mask is True."""
    from instance import Instance
    return Instance(
        bounds=base.bounds,
        precinct_label_grid=base.precinct_label_grid,
        grid_xs=base.grid_xs,
        grid_ys=base.grid_ys,
        precinct_centroids=base.precinct_centroids,
        precinct_areas=base.precinct_areas,
        precinct_voters=base.precinct_voters,
        site_locations=base.site_locations[keep_mask],
        site_capacity=base.site_capacity[keep_mask],
        site_types=[t for t, k in zip(base.site_types, keep_mask) if k],
        distance_matrix=base.distance_matrix[:, keep_mask],
        K=base.K,
        seed=base.seed,
        params=base.params,
    )


# ---------------------------------------------------------------------------
# Topology helpers (contiguity / shape archetypes)
# ---------------------------------------------------------------------------
def _precinct_adjacency(label_grid: np.ndarray) -> Dict[int, set]:
    """4-connected precinct adjacency derived from the precinct label raster."""
    G = np.asarray(label_grid)
    adj: Dict[int, set] = {}

    def _add(u: int, v: int):
        if u == v:
            return
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))

    # Horizontal neighbours
    a = G[:, :-1]
    b = G[:, 1:]
    diff = a != b
    if diff.any():
        for u, v in zip(a[diff], b[diff]):
            _add(int(u), int(v))
    # Vertical neighbours
    a = G[:-1, :]
    b = G[1:, :]
    diff = a != b
    if diff.any():
        for u, v in zip(a[diff], b[diff]):
            _add(int(u), int(v))
    return adj


def _connected_components(members: Iterable[int],
                            adj: Dict[int, set]) -> List[List[int]]:
    """Connected components of the subgraph induced by `members`, given the
    global adjacency dict."""
    members_set = set(int(m) for m in members)
    seen: set = set()
    comps: List[List[int]] = []
    for start in members_set:
        if start in seen:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            for v in adj.get(u, ()):
                if v in members_set and v not in seen:
                    stack.append(v)
        comps.append(comp)
    return comps


def verify_coverage_gap(
    instance, solution,
    coverage_gap_center: Tuple[float, float],
    coverage_gap_radius: float,
    surrounding_factor: float = 2.5,
    local_anomaly_radius: float = 2.0,
    top_k_anomalies: int = 8,
    max_closed_candidate_center_distance: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Inspect (instance, solution) for the coverage-coverage_gap property.

    Returns a dict with severity / affected / fix-path / uniqueness metadata,
    or None if the property cannot be evaluated (no precincts inside or no
    surrounding ring).

    Severity (mean in-hole nearest-site distance / mean surrounding-ring
    nearest) captures how bad the gap looks locally.

    Uniqueness captures whether the coverage_gap is the *dominant* visible gap
    on the map. For each precinct we compute a "local anomaly" — its
    nearest-site distance minus the mean nearest-site distance of its
    spatial neighbors (within `local_anomaly_radius` km). Coverage-gap-style
    holes have high positive local anomaly: worse than their neighbors.
    We take the top-K precincts globally by positive local anomaly and ask
    what fraction of them sit inside the target coverage_gap. High fraction =
    the coverage_gap is unambiguously the worst gap; low fraction = multiple
    comparably-bad regions exist (and the agent's textual critique is
    ambiguous about which one to address).

    This metric is more robust than a global-distance threshold because it
    cancels out edge effects (rural-fringe precincts that have high raw
    distance but neighbours that are also far from sites).
    """
    opened_idx = np.where(solution.x == 1)[0]
    if len(opened_idx) == 0:
        return None
    nearest = instance.distance_matrix[:, opened_idx].min(axis=1)

    centroids = instance.precinct_centroids
    cx, cy = float(coverage_gap_center[0]), float(coverage_gap_center[1])
    d_from_center = np.linalg.norm(centroids - np.array([cx, cy]), axis=1)

    in_hole = d_from_center < coverage_gap_radius
    surrounding = (
        (d_from_center >= coverage_gap_radius)
        & (d_from_center < coverage_gap_radius * surrounding_factor)
    )
    if not in_hole.any() or not surrounding.any():
        return None

    in_hole_mean = float(nearest[in_hole].mean())
    in_hole_max = float(nearest[in_hole].max())
    surrounding_mean = float(nearest[surrounding].mean())
    severity = in_hole_mean / max(surrounding_mean, 1e-3)

    affected = [int(i) for i in np.where(in_hole)[0]]

    # Fix path: closed candidates near the centre of the coverage_gap.
    # The agent's natural fix is to force-open such a candidate; the coverage_gap
    # must contain at least one for the pair to be solvable. Candidates near the
    # edge of the annotated region are less visually obvious and may not fill
    # the central hole.
    site_d = np.linalg.norm(
        instance.site_locations - np.array([cx, cy]), axis=1
    )
    closed_in_coverage_gap_idx = np.where(
        (solution.x == 0) & (site_d <= coverage_gap_radius)
    )[0]
    if max_closed_candidate_center_distance is None:
        max_closed_candidate_center_distance = coverage_gap_radius
    closed_near_center_idx = np.where(
        (solution.x == 0) & (site_d <= max_closed_candidate_center_distance)
    )[0]

    # Uniqueness via local anomaly: each precinct's nearest-site distance
    # minus the mean of its spatial neighbours (within local_anomaly_radius).
    # Top-K positive anomalies = the most visible "holes" on the map. The
    # target coverage_gap is unambiguous when most of those top-K sit inside it.
    n_pre = len(centroids)
    local_anomaly = np.zeros(n_pre, dtype=float)
    for i in range(n_pre):
        d = np.linalg.norm(centroids - centroids[i], axis=1)
        nb = (d < local_anomaly_radius) & (d > 0)
        if nb.sum() < 3:
            continue
        local_anomaly[i] = float(nearest[i] - nearest[nb].mean())

    # Top-K precincts by local anomaly, keeping only positive values.
    order = np.argsort(-local_anomaly)
    top_pos = [int(i) for i in order[:top_k_anomalies]
                if local_anomaly[i] > 0]
    n_top = len(top_pos)
    n_top_in_coverage_gap = int(sum(1 for i in top_pos if in_hole[i]))
    fraction_in_coverage_gap = (n_top_in_coverage_gap / n_top) if n_top > 0 else 1.0
    outside_top_anomalies = [int(i) for i in top_pos if not in_hole[i]]

    return {
        "affected_precincts": affected,
        "in_hole_mean_distance": in_hole_mean,
        "in_hole_max_distance": in_hole_max,
        "surrounding_mean_distance": surrounding_mean,
        "severity": float(severity),
        "closed_candidates_in_coverage_gap": [int(j) for j in closed_in_coverage_gap_idx],
        "closed_candidates_near_gap_center": [int(j) for j in closed_near_center_idx],
        "max_closed_candidate_center_distance":
            float(max_closed_candidate_center_distance),
        "fix_path_available": bool(len(closed_near_center_idx) > 0),
        # Uniqueness diagnostics (local-anomaly based)
        "uniqueness_local_anomaly_radius": float(local_anomaly_radius),
        "uniqueness_top_k": int(top_k_anomalies),
        "uniqueness_n_top_anomalies": n_top,
        "uniqueness_n_top_in_coverage_gap": n_top_in_coverage_gap,
        "uniqueness_fraction_in_coverage_gap": float(fraction_in_coverage_gap),
        "uniqueness_top_anomaly_indices": top_pos,
        "uniqueness_outside_top_anomalies": outside_top_anomalies,
    }


def generate_coverage_gap_instance(
    base_seed: int = 1,
    severity_target: float = 1.4,
    coverage_gap_center: Tuple[float, float] = (2.5, 4.7),
    coverage_gap_radius: float = 1.2,
    max_closed_candidate_center_distance: Optional[float] = None,
    keep_n_candidates: int = 1,
    max_attempts: int = 15,
    params: Optional[Dict[str, Any]] = None,
    require_unique: bool = False,
    uniqueness_min_fraction: float = 0.5,
    local_anomaly_radius: float = 2.0,
    top_k_anomalies: int = 8,
    verbose: bool = True,
):
    """Generate an instance + baseline with an *engineered* coverage coverage_gap.

    Algorithm:
      1. Generate a base instance with the requested seed.
      2. Find candidate sites whose location lies inside the coverage_gap polygon.
      3. Reduce candidate density inside: keep only the `keep_n_candidates`
         nearest the coverage_gap centre, remove the rest. (Reduction, not
         elimination — leaving at least one closed candidate gives the agent
         a viable fix path: force-open a closed candidate and close some
         less-needed site elsewhere within the same budget K.)
      4. Solve the baseline MILP on the reduced instance.
      5. Verify the coverage_gap property: in-hole nearest-site distance is at
         least `severity_target`× the surrounding mean.
      6. Verify a fix path: at least one closed candidate exists near the
         centre of the coverage_gap.
      7. Record uniqueness diagnostics via local anomaly. If
         `require_unique`, verify uniqueness: at
         least `uniqueness_min_fraction` of the top-K precincts globally
         (ranked by local anomaly — nearest-site distance vs the mean of
         their spatial neighbours within `local_anomaly_radius`) must lie
         inside the target coverage_gap. This rejects instances where multiple
         comparably-anomalous gaps exist on the map and the agent can't
         tell from the description which one to address.
      8. If all pass, return (instance, solution, metadata). Else retry
         with the next seed (base_seed + 17*attempt).

    Parameters specific to uniqueness:
        require_unique : if True, enforce step 7. Default False.
        uniqueness_min_fraction : minimum fraction of the top-K most
            locally-anomalous precincts that must lie inside the coverage_gap.
            Higher = stricter. Default 0.5.
        local_anomaly_radius : neighborhood size (km) for the local-mean
            comparison that defines a precinct's local anomaly. Default 2.0.
        top_k_anomalies : how many top-anomaly precincts to use for the
            uniqueness ratio. Default 8 (~10% of an 80-precinct map).
        max_closed_candidate_center_distance : maximum distance in km from
            the coverage_gap centre for a closed candidate to count as the
            fix path. Default is half the coverage_gap radius.

    Returns
    -------
    (instance, baseline_solution, metadata) where metadata contains:
        archetype, coverage_gap_center, coverage_gap_radius, severity_achieved,
        affected_precincts, in_hole_mean_distance, surrounding_mean_distance,
        closed_candidates_in_coverage_gap, base_seed, sites_removed, attempt.

    Raises
    ------
    RuntimeError if no acceptable instance is found within max_attempts.
    ImportError if gurobipy isn't installed (since solving is required).
    """
    from solver import solve_baseline   # lazy: avoid hard gurobipy dep at import
    if params is None:
        params = default_params()
    if max_closed_candidate_center_distance is None:
        max_closed_candidate_center_distance = 0.5 * coverage_gap_radius

    last_metadata = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        base = generate_instance(seed, params)

        site_d = np.linalg.norm(
            base.site_locations
            - np.array([coverage_gap_center[0], coverage_gap_center[1]]),
            axis=1,
        )
        in_coverage_gap_idx = np.where(site_d <= coverage_gap_radius)[0]

        if len(in_coverage_gap_idx) <= keep_n_candidates:
            new_inst = base
            sites_removed = 0
        else:
            order = in_coverage_gap_idx[np.argsort(site_d[in_coverage_gap_idx])]
            keep = set(order[:keep_n_candidates].tolist())
            keep_mask = np.ones(len(base.site_locations), dtype=bool)
            for j in in_coverage_gap_idx:
                if int(j) not in keep:
                    keep_mask[j] = False
            new_inst = _rebuild_instance_with_site_subset(base, keep_mask)
            sites_removed = int((~keep_mask).sum())

        sol = solve_baseline(new_inst, verbose=False)
        if not sol.metadata.get("feasible", True):
            if verbose:
                print(f"[coverage_gap] attempt {attempt} (seed {seed}): "
                       f"infeasible, skipping")
            continue

        verify = verify_coverage_gap(
            new_inst, sol, coverage_gap_center, coverage_gap_radius,
            local_anomaly_radius=local_anomaly_radius,
            top_k_anomalies=top_k_anomalies,
            max_closed_candidate_center_distance=(
                max_closed_candidate_center_distance),
        )
        if verify is None:
            if verbose:
                print(f"[coverage_gap] attempt {attempt}: no precincts/surrounding")
            continue

        if verbose:
            print(
                f"[coverage_gap] attempt {attempt} (seed {seed}): "
                f"severity={verify['severity']:.2f}, "
                f"in_hole_max={verify['in_hole_max_distance']:.2f}, "
                f"affected={len(verify['affected_precincts'])} precincts, "
                f"closed_in_coverage_gap="
                f"{len(verify['closed_candidates_in_coverage_gap'])}, "
                f"uniq={verify['uniqueness_n_top_in_coverage_gap']}/"
                f"{verify['uniqueness_n_top_anomalies']} "
                f"({verify['uniqueness_fraction_in_coverage_gap']:.2f}), "
                f"removed={sites_removed} candidates"
            )

        last_metadata = verify
        unique_ok = (
            (not require_unique)
            or (verify["uniqueness_fraction_in_coverage_gap"]
                >= uniqueness_min_fraction)
        )
        if (verify["severity"] >= severity_target
                and verify["fix_path_available"]
                and unique_ok):
            # Calibrate the global "stranded" threshold per-pair, so the
            # baseline metric is on a consistent scale across the dataset.
            # Pick threshold so ~5 precincts are above it in the baseline
            # (the engineered affected precincts plus a few naturally-bad
            # ones). The agent's task is then to reduce that count
            # without inflating it elsewhere.
            opened = np.where(sol.x == 1)[0]
            baseline_nearest = (
                new_inst.distance_matrix[:, opened].min(axis=1))
            sorted_nearest = np.sort(baseline_nearest)[::-1]
            target_baseline_strands = 5
            k = min(target_baseline_strands, len(sorted_nearest) - 1)
            # Threshold: midway between the kth and (k+1)th worst, so
            # exactly k precincts are above.
            distance_threshold = float(
                (sorted_nearest[k - 1] + sorted_nearest[k]) / 2.0
            ) if k >= 1 else float(sorted_nearest[0])

            metadata = {
                "archetype": "coverage_gap",
                "coverage_gap_center": [float(coverage_gap_center[0]),
                                      float(coverage_gap_center[1])],
                "coverage_gap_radius": float(coverage_gap_radius),
                "severity_target": float(severity_target),
                "severity_achieved": verify["severity"],
                "affected_precincts": verify["affected_precincts"],
                "in_hole_mean_distance": verify["in_hole_mean_distance"],
                "in_hole_max_distance": verify["in_hole_max_distance"],
                "surrounding_mean_distance": verify["surrounding_mean_distance"],
                "closed_candidates_in_coverage_gap":
                    verify["closed_candidates_in_coverage_gap"],
                "closed_candidates_near_gap_center":
                    verify["closed_candidates_near_gap_center"],
                "max_closed_candidate_center_distance":
                    verify["max_closed_candidate_center_distance"],
                # Calibrated threshold for the global "stranded" metric.
                "coverage_gap_distance_threshold": distance_threshold,
                "coverage_gap_baseline_strand_count":
                    int(target_baseline_strands),
                "uniqueness_local_anomaly_radius":
                    verify["uniqueness_local_anomaly_radius"],
                "uniqueness_top_k": verify["uniqueness_top_k"],
                "uniqueness_n_top_anomalies":
                    verify["uniqueness_n_top_anomalies"],
                "uniqueness_n_top_in_coverage_gap":
                    verify["uniqueness_n_top_in_coverage_gap"],
                "uniqueness_fraction_in_coverage_gap":
                    verify["uniqueness_fraction_in_coverage_gap"],
                "uniqueness_top_anomaly_indices":
                    verify["uniqueness_top_anomaly_indices"],
                "uniqueness_outside_top_anomalies":
                    verify["uniqueness_outside_top_anomalies"],
                "base_seed": seed,
                "attempt": attempt,
                "sites_removed": sites_removed,
            }
            if verbose:
                print(f"[coverage_gap] -> ACCEPTED on attempt {attempt} "
                       f"(threshold={distance_threshold:.2f} km)")
            return new_inst, sol, metadata

    raise RuntimeError(
        f"Could not generate coverage_gap instance with severity >= "
        f"{severity_target} and a fix path within {max_attempts} attempts. "
        f"Last seen: {last_metadata}. Try lowering severity_target or "
        f"adjusting coverage_gap_center / coverage_gap_radius / keep_n_candidates."
    )


# ===========================================================================
# Other archetype-specific instance generators
# ===========================================================================
# All four follow the same shape as `generate_coverage_gap_instance`: try seeds
# until you find one whose baseline solution exhibits the target property at
# sufficient strength, then return (instance, baseline_solution, metadata).
# Metadata identifies the affected entities — used downstream by the
# matching query factory to score the agent's response.

def generate_cluster_instance(
    base_seed: int = 1,
    cluster_radius: float = 1.3,
    cluster_min_sites: int = 4,
    cluster_density_factor: float = 2.0,
    max_attempts: int = 30,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """Find an instance whose baseline has a *dense* cluster of opened sites
    (archetype 1, referential disambiguation).

    Algorithm:
      1. Generate base instance + solve.
      2. For each opened site, count opened sites within `cluster_radius`.
      3. The densest such circle is the candidate cluster.
      4. Accept if the cluster contains ≥ `cluster_min_sites` (default 4 —
         a "cluster" is by definition at least 4 closely-packed sites) AND
         its site density exceeds `cluster_density_factor` × the average
         opened-site density across the map (default 2.0× = "relatively
         densely packed"). The default radius is intentionally tight so
         broad nearby-site constellations do not qualify.

    The agent's task: close enough clustered sites so no radius-
    neighbourhood of threshold size remains. Success criterion (used by
    make_cluster_query_from_metadata): target_response == 0.
    """
    from solver import solve_baseline
    if params is None:
        params = default_params()

    bounds = params['bounds']
    K = params['K']
    map_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    avg_density = K / map_area  # opened sites per km²
    cluster_area = float(np.pi * cluster_radius ** 2)

    last_meta = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        inst = generate_instance(seed, params)
        sol = solve_baseline(inst, verbose=False)
        if not sol.metadata.get("feasible", True):
            continue

        opened_idx = np.where(sol.x == 1)[0]
        if len(opened_idx) == 0:
            continue
        opened_locs = inst.site_locations[opened_idx]

        # Densest opened-site cluster: try each opened site as a candidate centre.
        best_count, best_center, best_members = 0, None, None
        for j in opened_idx:
            c = inst.site_locations[j]
            d = np.linalg.norm(opened_locs - c, axis=1)
            members = opened_idx[d <= cluster_radius]
            if len(members) > best_count:
                best_count = len(members)
                best_center = c
                best_members = members

        if best_count < cluster_min_sites:
            if verbose:
                print(f"[cluster] seed={seed}: best cluster size {best_count} < min {cluster_min_sites}")
            continue

        density_ratio = (best_count / cluster_area) / max(avg_density, 1e-9)
        if density_ratio < cluster_density_factor:
            if verbose:
                print(f"[cluster] seed={seed}: density ratio {density_ratio:.2f} < {cluster_density_factor}")
            continue

        last_meta = {"density_ratio": float(density_ratio), "best_count": int(best_count)}
        meta = {
            "archetype": "cluster",
            "cluster_center": [float(best_center[0]), float(best_center[1])],
            "cluster_radius": float(cluster_radius),
            "affected_sites": [int(j) for j in best_members],
            "cluster_size": int(best_count),
            "cluster_density_ratio": float(density_ratio),
            "cluster_min_sites": int(cluster_min_sites),
            "cluster_density_factor": float(cluster_density_factor),
            "base_seed": seed,
            "attempt": attempt,
        }
        if verbose:
            print(f"[cluster] attempt {attempt}: ACCEPTED, cluster of {best_count} sites at "
                   f"({best_center[0]:.1f}, {best_center[1]:.1f}), density={density_ratio:.2f}×")
        return inst, sol, meta

    raise RuntimeError(f"Could not generate cluster instance in {max_attempts} attempts. "
                        f"Last: {last_meta}")


# ===========================================================================
# Contiguity (non-contiguous service area) archetype
# ===========================================================================
def _scan_all_contiguitys(
    instance: 'Instance', solution,
    adj: Dict[int, set],
) -> List[Dict[str, Any]]:
    """Return one dict per opened site whose assigned-precinct subgraph has
    >= 2 connected components. The dict contains the structural facts we
    care about for both metric scoring and metadata bookkeeping:

        site                         : opened site index
        components                   : list of precinct-index lists, sorted
                                        by voter count DESCENDING (largest
                                        first; the rest are "disjoint
                                        excess").
        component_voters             : list of int, parallel to components.
        n_components                 : len(components).
        largest_component_voters     : int.
        disjoint_excess_voters       : sum of voter counts of all
                                        non-largest components — the metric
                                        we minimise.
        worst_pair_separation_km     : centroid distance between the two
                                        most-voter-heavy components.

    Sites with contiguous catchments (1 component) are NOT included.
    The list is empty when every catchment is contiguous.
    """
    out: List[Dict[str, Any]] = []
    if not solution.metadata.get("feasible", True):
        return out
    assigned = solution.y.argmax(axis=1)
    opened_idx = np.where(solution.x == 1)[0]
    centroids = instance.precinct_centroids
    voters = instance.precinct_voters
    for j in opened_idx:
        members = np.where(assigned == j)[0]
        if len(members) < 2:
            continue
        comps = _connected_components(members, adj)
        if len(comps) < 2:
            continue
        # Voter count per component, then sort descending so [0] is "main".
        comp_voters = [int(voters[c].sum()) for c in comps]
        order = np.argsort(-np.array(comp_voters))
        comps_sorted = [list(map(int, comps[k])) for k in order]
        comp_voters_sorted = [comp_voters[k] for k in order]
        excess = int(sum(comp_voters_sorted[1:]))
        # Separation between the two largest components.
        a = centroids[comps_sorted[0]].mean(axis=0)
        b = centroids[comps_sorted[1]].mean(axis=0)
        sep = float(np.linalg.norm(a - b))
        out.append({
            "site": int(j),
            "components": comps_sorted,
            "component_voters": comp_voters_sorted,
            "n_components": len(comps_sorted),
            "largest_component_voters": comp_voters_sorted[0],
            "disjoint_excess_voters": excess,
            "worst_pair_separation_km": sep,
        })
    return out


def _passes_acceptance(
    culprits: List[Dict[str, Any]],
    min_split_voters: int,
    min_separation: float,
) -> bool:
    """Acceptance: at least one culprit has disjoint_excess_voters >=
    min_split_voters AND worst_pair_separation_km >= min_separation."""
    return any(
        c["disjoint_excess_voters"] >= min_split_voters
        and c["worst_pair_separation_km"] >= min_separation
        for c in culprits
    )


def _engineer_capacity_overflow(base: 'Instance',
                                  rng_seed: int,
                                  shrink_fraction: float = 0.45,
                                  n_to_shrink: int = 4) -> 'Instance':
    """Return a copy of `base` with a few sites' capacities reduced.

    Tightening capacity on a handful of sites forces the MILP to spill over
    into more distant alternatives, which occasionally produces
    non-contiguous service areas. We don't try to be surgical here — the
    rejection-sampling loop is what guarantees a usable instance comes out.
    """
    rng = np.random.default_rng(int(rng_seed))
    new_caps = base.site_capacity.copy()
    M = base.n_sites
    pick = rng.choice(M, size=min(n_to_shrink, M), replace=False)
    for j in pick:
        new_caps[j] = max(int(new_caps[j] * shrink_fraction), 600)
    new_inst = Instance(
        bounds=base.bounds,
        precinct_label_grid=base.precinct_label_grid,
        grid_xs=base.grid_xs, grid_ys=base.grid_ys,
        precinct_centroids=base.precinct_centroids,
        precinct_areas=base.precinct_areas,
        precinct_voters=base.precinct_voters,
        site_locations=base.site_locations,
        site_capacity=new_caps,
        site_types=list(base.site_types),
        distance_matrix=base.distance_matrix,
        K=base.K, seed=base.seed, params=base.params,
    )
    return new_inst


def generate_contiguity_instance(
    base_seed: int = 1,
    min_split_voters: int = 1500,
    min_separation: float = 1.5,
    max_attempts: int = 60,
    capacity_perturbations_per_seed: int = 3,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """Engineer an instance where >= 1 opened site has a non-contiguous
    service area (visible as same-colored disjoint patches on the colored
    assignment view).

    Acceptance: at least ONE opened site's catchment splits with disjoint
    excess voters >= min_split_voters AND component separation >=
    min_separation. The metadata records every culprit found in the
    accepted baseline (not just the first), so the metric's view of the
    instance and the metadata's view of the instance agree.

    Strategy: try the natural baseline first; if no split passes, try a
    few capacity-shrink perturbations (which force overflow). If that
    still fails, advance to the next seed.
    """
    from solver import solve_baseline
    if params is None:
        params = default_params()
    last_reason = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        base = generate_instance(seed, params)
        adj = _precinct_adjacency(base.precinct_label_grid)

        # Candidates: (instance, solution, culprits-list).
        candidates: List[Tuple['Instance', Any, List[Dict[str, Any]]]] = []
        sol = solve_baseline(base, verbose=False)
        culprits = _scan_all_contiguitys(base, sol, adj)
        candidates.append((base, sol, culprits))

        if not _passes_acceptance(culprits, min_split_voters, min_separation):
            # Try a few capacity-shrink perturbations of this seed.
            for k in range(capacity_perturbations_per_seed):
                inst_p = _engineer_capacity_overflow(
                    base, rng_seed=seed * 41 + k * 11,
                    shrink_fraction=0.45,
                    n_to_shrink=4)
                sol_p = solve_baseline(inst_p, verbose=False)
                culprits_p = _scan_all_contiguitys(inst_p, sol_p, adj)
                candidates.append((inst_p, sol_p, culprits_p))
                if _passes_acceptance(culprits_p,
                                       min_split_voters, min_separation):
                    break

        # Pick the first candidate that passes.
        winner = next((c for c in candidates
                        if _passes_acceptance(c[2], min_split_voters,
                                                min_separation)),
                       None)
        if winner is None:
            last_reason = "no split-catchment with sufficient voter mass / separation"
            if verbose:
                if candidates and candidates[-1][2]:
                    summary = ", ".join(
                        f"site {c['site']}: {c['disjoint_excess_voters']}v / "
                        f"{c['worst_pair_separation_km']:.2f}km"
                        for c in candidates[-1][2]
                    )
                    print(f"[contiguity] seed={seed}: "
                           f"saw splits but below thresholds — {summary}")
                else:
                    print(f"[contiguity] seed={seed}: {last_reason}")
            continue
        inst, sol, culprits = winner

        # Metadata: every culprit, plus aggregates the metric will use.
        culprit_records: List[Dict[str, Any]] = []
        total_disjoint = 0
        total_split_sites = len(culprits)
        worst_excess = 0
        worst_site = None
        for c in culprits:
            culprit_records.append({
                "site": int(c["site"]),
                "n_components": int(c["n_components"]),
                "components": [list(map(int, comp))
                                for comp in c["components"]],
                "component_voters":
                    [int(v) for v in c["component_voters"]],
                "largest_component_voters":
                    int(c["largest_component_voters"]),
                "disjoint_excess_voters":
                    int(c["disjoint_excess_voters"]),
                "worst_pair_separation_km":
                    float(c["worst_pair_separation_km"]),
            })
            total_disjoint += int(c["disjoint_excess_voters"])
            if c["disjoint_excess_voters"] > worst_excess:
                worst_excess = int(c["disjoint_excess_voters"])
                worst_site = int(c["site"])

        # Anchor centroid (used by dataset_generator for {region}/{river_side}
        # placeholders) — the worst culprit's smallest component centroid.
        worst = next(c for c in culprits if c["site"] == worst_site)
        # The components inside `culprits` are sorted voter-DESCENDING; the
        # smallest component is the LAST one.
        smallest_comp = worst["components"][-1]
        smallest_centroid = inst.precinct_centroids[smallest_comp].mean(axis=0)

        meta = {
            "archetype": "contiguity",
            "culprits": culprit_records,
            "n_split_sites_baseline": total_split_sites,
            "total_disjoint_excess_voters_baseline": int(total_disjoint),
            "worst_culprit_site": worst_site,
            "worst_culprit_disjoint_voters": int(worst_excess),
            "smallest_component_centroid":
                [float(smallest_centroid[0]), float(smallest_centroid[1])],
            "min_split_voters_threshold": int(min_split_voters),
            "min_separation_threshold_km": float(min_separation),
            "base_seed": seed,
            "attempt": attempt,
        }
        if verbose:
            print(f"[contiguity] attempt {attempt}: ACCEPTED — "
                   f"{total_split_sites} split site(s), "
                   f"total disjoint excess = {total_disjoint} voters; "
                   f"worst: site {worst_site} ({worst_excess} v)")
        return inst, sol, meta
    raise RuntimeError(
        f"Could not generate contiguity instance in {max_attempts} "
        f"attempts. Last: {last_reason}"
    )


# ===========================================================================
# Shape niceness archetype
# ===========================================================================
def _per_catchment_npi(
    instance: 'Instance', solution,
) -> Dict[int, Dict[str, float]]:
    """For each opened site, compute the catchment's normalized perimeter
    index NPI = P / (2 * sqrt(pi * A)). NPI = 1 for a perfect circle and
    grows with elongation / jaggedness / "weirdness". Computed on the
    rasterised precinct label grid:

      A = (cell area) * (number of cells whose precinct is in the catchment)
      P = (cell edge length) * (number of cell-edges where one side is in
          the catchment and the other is not, including grid borders).

    Returns a dict: site_index -> {'A', 'P', 'NPI', 'n_cells'}. Sites with
    empty catchments are omitted.
    """
    label_grid = instance.precinct_label_grid          # (G, G) int
    xs = instance.grid_xs
    ys = instance.grid_ys
    cell_w = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    cell_h = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    cell_area = cell_w * cell_h

    # precinct -> assigned site (argmax over y rows).
    assigned = solution.y.argmax(axis=1)  # (n_precincts,)
    # Map every grid cell to its precinct's assigned site.
    cell_site = assigned[label_grid]  # (G, G)

    out: Dict[int, Dict[str, float]] = {}
    for j in np.where(solution.x == 1)[0]:
        mask = (cell_site == j)
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        A = float(n_cells) * cell_area

        # Perimeter: for each cell in the catchment, count edges to OUTSIDE.
        # Right neighbour
        right = np.zeros_like(mask)
        right[:, :-1] = mask[:, :-1] & ~mask[:, 1:]
        # Left neighbour (cells with a left neighbour outside)
        left = np.zeros_like(mask)
        left[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
        # Down neighbour
        down = np.zeros_like(mask)
        down[:-1, :] = mask[:-1, :] & ~mask[1:, :]
        # Up neighbour
        up = np.zeros_like(mask)
        up[1:, :] = mask[1:, :] & ~mask[:-1, :]
        # Grid borders count too: any cell of the catchment on the outer
        # boundary contributes its outer-edge.
        border_n = int(mask[0, :].sum()) + int(mask[-1, :].sum())
        border_v = int(mask[:, 0].sum()) + int(mask[:, -1].sum())

        n_h_edges = int(left.sum() + right.sum() + border_v)
        n_v_edges = int(up.sum() + down.sum() + border_n)
        # Each horizontal-edge has length cell_h; each vertical-edge has
        # length cell_w. (A "horizontal" cell-edge separates two cells
        # along the x-axis; its length is cell_h.)
        P = n_h_edges * cell_h + n_v_edges * cell_w
        NPI = P / (2.0 * np.sqrt(np.pi * A)) if A > 0 else float('inf')
        out[int(j)] = {
            "A": float(A),
            "P": float(P),
            "NPI": float(NPI),
            "n_cells": int(n_cells),
        }
    return out


def aggregate_npi(per_catchment: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-catchment NPI into solution-level summary stats."""
    if not per_catchment:
        return {"mean_npi": float('nan'),
                 "max_npi": float('nan'),
                 "p90_npi": float('nan'),
                 "n_catchments": 0}
    vals = np.array([d["NPI"] for d in per_catchment.values()])
    return {
        "mean_npi": float(vals.mean()),
        "max_npi": float(vals.max()),
        "p90_npi": float(np.percentile(vals, 90)),
        "n_catchments": int(len(vals)),
    }


def generate_shape_niceness_instance(
    base_seed: int = 1,
    min_mean_npi: float = 1.5,
    min_max_npi: float = 2.0,
    capacity_perturbations_per_seed: int = 3,
    max_attempts: int = 60,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    """Engineer an instance whose baseline catchments are visibly ugly —
    elongated, jagged, or otherwise high-NPI shapes. Acceptance:

      mean_NPI >= min_mean_npi      (the average catchment is at least
                                      mildly elongated)
      max_NPI  >= min_max_npi       (at least one catchment is strikingly bad)

    Strategy: try the natural baseline first; if NPI thresholds aren't met,
    apply a few capacity-shrink perturbations (mirroring the contiguity
    generator) which often produce stretched catchments. Advance the seed
    if nothing works.

    Returns (instance, baseline_solution, metadata) where metadata contains
    per-catchment NPIs, the worst culprit's site index, and the aggregate
    baseline summary (mean / max / p90 of NPI).
    """
    from solver import solve_baseline
    if params is None:
        params = default_params()
    last_summary = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        base = generate_instance(seed, params)

        # Try the natural baseline first.
        candidates: List[Tuple['Instance', Any]] = []
        sol = solve_baseline(base, verbose=False)
        candidates.append((base, sol))

        per_cmap = _per_catchment_npi(base, sol) if sol.metadata.get(
            "feasible", True) else {}
        agg = aggregate_npi(per_cmap)
        if (sol.metadata.get("feasible", True)
                and agg["mean_npi"] >= min_mean_npi
                and agg["max_npi"] >= min_max_npi):
            chosen_inst, chosen_sol = base, sol
        else:
            # Try capacity-shrink perturbations to provoke ugly catchments.
            chosen_inst = chosen_sol = None
            for k in range(capacity_perturbations_per_seed):
                inst_p = _engineer_capacity_overflow(
                    base, rng_seed=seed * 53 + k * 17,
                    shrink_fraction=0.45,
                    n_to_shrink=4,
                )
                sol_p = solve_baseline(inst_p, verbose=False)
                if not sol_p.metadata.get("feasible", True):
                    continue
                per_p = _per_catchment_npi(inst_p, sol_p)
                agg_p = aggregate_npi(per_p)
                last_summary = agg_p
                if (agg_p["mean_npi"] >= min_mean_npi
                        and agg_p["max_npi"] >= min_max_npi):
                    chosen_inst, chosen_sol = inst_p, sol_p
                    per_cmap = per_p
                    agg = agg_p
                    break

        last_summary = agg if chosen_inst is None else last_summary
        if chosen_inst is None:
            if verbose:
                print(f"[shape_niceness] seed={seed}: agg={agg} (below "
                       f"thresholds mean>={min_mean_npi}, max>={min_max_npi})")
            continue

        # Identify the worst-shaped catchment for the precise-query variant.
        worst_site = max(per_cmap.items(), key=lambda kv: kv[1]["NPI"])[0]
        worst_npi = per_cmap[worst_site]["NPI"]

        meta = {
            "archetype": "shape_niceness",
            "per_catchment_npi": {str(k): v
                                    for k, v in per_cmap.items()},
            "mean_npi_baseline": agg["mean_npi"],
            "max_npi_baseline": agg["max_npi"],
            "p90_npi_baseline": agg["p90_npi"],
            "n_catchments_baseline": agg["n_catchments"],
            "worst_catchment_site": int(worst_site),
            "worst_catchment_npi": float(worst_npi),
            "min_mean_npi_threshold": float(min_mean_npi),
            "min_max_npi_threshold": float(min_max_npi),
            "base_seed": seed,
            "attempt": attempt,
        }
        if verbose:
            print(f"[shape_niceness] attempt {attempt}: ACCEPTED — "
                   f"mean_NPI={agg['mean_npi']:.2f}, "
                   f"max_NPI={agg['max_npi']:.2f}, "
                   f"worst site {worst_site} (NPI={worst_npi:.2f})")
        return chosen_inst, chosen_sol, meta
    raise RuntimeError(
        f"Could not generate shape_niceness instance in {max_attempts} "
        f"attempts. Last summary: {last_summary}"
    )


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------
def generate_instance(seed: int, params: Dict[str, Any] = None) -> Instance:
    """Generate one full polling place instance."""
    if params is None:
        params = default_params()
    rng = np.random.default_rng(seed)

    seeds = sample_precinct_seeds(rng, params)
    label_grid, xs, ys = compute_voronoi_raster(seeds, params)
    centroids, areas = compute_precinct_attributes(label_grid, xs, ys, seeds)

    voters = compute_voters(centroids, areas, params, rng)

    sites, capacities, site_types = generate_sites(rng, params)
    D = compute_distance_matrix(centroids, sites, params)

    return Instance(
        bounds=params['bounds'],
        precinct_label_grid=label_grid,
        grid_xs=xs,
        grid_ys=ys,
        precinct_centroids=centroids,
        precinct_areas=areas,
        precinct_voters=voters,
        site_locations=sites,
        site_capacity=capacities,
        site_types=site_types,
        distance_matrix=D,
        K=params['K'],
        seed=seed,
        params=params,
    )
