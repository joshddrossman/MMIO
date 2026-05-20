"""Solution evaluation metrics.

Reference metrics computed on a Solution. The archetype-specific scoring
lives in queries.py — these helpers are for diagnostic / display use.
"""
from typing import Dict
import numpy as np

from instance import Instance, Solution


def per_precinct_distance(instance: Instance, solution: Solution) -> np.ndarray:
    """Distance from each precinct to its assigned site."""
    return (instance.distance_matrix * solution.y).sum(axis=1)


def voter_weighted_mean(values: np.ndarray, voters: np.ndarray,
                          mask: np.ndarray = None) -> float:
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    if not mask.any() or voters[mask].sum() == 0:
        return float('nan')
    return float((values[mask] * voters[mask]).sum() / voters[mask].sum())


def compute_metrics(instance: Instance, solution: Solution) -> Dict[str, float]:
    """Reference metrics on a solution: total weighted distance, p90 / p95
    voter travel distance, mean voter travel distance, site loads, count of
    opened sites, feasibility."""
    voters = instance.precinct_voters
    pdist = per_precinct_distance(instance, solution)

    total_weighted_distance = float((voters * pdist).sum())
    mean_dist_overall = voter_weighted_mean(pdist, voters)

    # Voter-weighted percentiles
    voter_dists_expanded = np.repeat(pdist, voters)
    p90 = float(np.percentile(voter_dists_expanded, 90))
    p95 = float(np.percentile(voter_dists_expanded, 95))

    # Site loads
    site_loads = (voters[:, None] * solution.y).sum(axis=0)
    opened_loads = site_loads[solution.x == 1]

    return {
        'total_weighted_distance': total_weighted_distance,
        'mean_dist_overall': mean_dist_overall,
        'p90_distance': p90,
        'p95_distance': p95,
        'mean_load_opened':
            float(opened_loads.mean()) if len(opened_loads) else float('nan'),
        'max_load':
            float(opened_loads.max()) if len(opened_loads) else float('nan'),
        'sites_opened': int(solution.x.sum()),
        'feasible': bool(solution.metadata.get('feasible', True)),
    }
