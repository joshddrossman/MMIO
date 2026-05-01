"""Archetype query definitions and scoring framework.

A *query* bundles a stakeholder critique with archetype-specific scoring:

    text                  : the critique
    target_metric_fn      : metric the agent should improve
    target_direction      : "minimize" or "maximize"
    guards                : list of GuardSpec (degradation budget)
    success_criterion_fn  : optional binary success check on top of metric

The 4 archetypes are emergent solution properties:

    cluster        : facility density (count of opened sites in a region).
    coverage_gap   : a small interior pocket where voters are stranded
                     relative to the surrounding ring (renamed from doughnut).
    contiguity     : opened sites whose assigned-precinct subgraph is
                     non-contiguous (renamed from split_catchment).
    shape_niceness : opened sites whose catchments have ugly normalised
                     perimeter index (NPI = P / (2*sqrt(pi*A)); 1 = circle,
                     larger = elongated/jagged).

Every score dict includes `final_assignment_distance` — the voter-weighted
total travel distance under the response solution — as a secondary metric
tracked alongside the archetype's primary target.

Each archetype has *vague* and *precise* query variants (different text
templates, same scoring target). Both are stored in pair metadata; the
runner picks one based on a CLI knob and calls the appropriate factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from matplotlib.path import Path

from instance import Instance, Solution


# Cluster scoring should only flag very tight facility concentrations. Older
# datasets may carry a broader historical cluster_radius in metadata; cap the
# scoring radius so those broad constellations no longer count as clusters.
MAX_CLUSTER_SCORING_RADIUS = 1.3


# =========================================================================
# Common metric helpers
# =========================================================================
def total_weighted_distance(instance: Instance, solution: Solution) -> float:
    """Total voter-weighted assignment distance.

    Used (a) as a guard metric, (b) as the secondary `final_assignment_distance`
    metric tracked in every score dict so two equally-effective fixes can be
    compared on solution quality.
    """
    voters = instance.precinct_voters
    return float((voters[:, None] * instance.distance_matrix * solution.y).sum())


def p90_voter_distance(instance: Instance, solution: Solution) -> float:
    voters = instance.precinct_voters
    pdist = (instance.distance_matrix * solution.y).sum(axis=1)
    voter_dists = np.repeat(pdist, voters)
    return float(np.percentile(voter_dists, 90))


def n_sites_in_dense_cluster(
    cluster_radius: float, cluster_min_sites: int,
) -> Callable[[Instance, Solution], float]:
    """Count opened sites that participate in any dense cluster.

    A radius-neighbourhood is a dense cluster if it contains at least
    `cluster_min_sites` opened sites, including its centre. The target
    metric counts the union of all opened sites that appear inside any
    qualifying radius-neighbourhood.

    This is the cluster archetype's target metric. It is GLOBAL — it
    sees clusters anywhere on the map, not just in the engineered region.
    So an agent who breaks up the engineered cluster but inadvertently
    creates a new dense cluster elsewhere is penalised.

    Direction: minimize. Success: target_response == 0 (no radius-
    neighbourhood contains cluster_min_sites or more opened sites).
    """
    R = float(cluster_radius)
    K = int(cluster_min_sites)

    def fn(instance: Instance, solution: Solution) -> float:
        opened_idx = np.where(solution.x == 1)[0]
        if len(opened_idx) < K:
            return 0.0
        opened_xy = instance.site_locations[opened_idx]
        diff = opened_xy[:, None, :] - opened_xy[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        # Find every qualifying radius-neighbourhood, then count the union
        # of opened sites that appear inside at least one such neighbourhood.
        within_count = (d <= R).sum(axis=1)
        qualifying_centres = within_count >= K
        if not np.any(qualifying_centres):
            return 0.0
        in_any_dense_cluster = (d[qualifying_centres] <= R).any(axis=0)
        return float(in_any_dense_cluster.sum())
    return fn


def n_precincts_in_coverage_gap(
    distance_threshold: float = 2.0,
) -> Callable[[Instance, Solution], float]:
    """Count precincts whose nearest-open-site distance exceeds
    `distance_threshold` km. A simple absolute threshold for "stranded
    voters" — voters who have to travel substantially farther than the
    typical precinct.

    This is the coverage_gap archetype's target metric. GLOBAL — it
    flags every stranded precinct anywhere on the map. An agent who
    fixes the engineered gap but inadvertently creates a new one
    elsewhere (e.g., by closing a site that was serving its
    neighbourhood) is penalised: the newly-stranded precincts cross
    the threshold.

    The metric is MONOTONIC under site additions: opening a new site
    can only reduce or preserve a precinct's nearest distance, hence
    can only reduce or preserve the count. (Closing sites can
    increase the count, which is the desired penalty.)

    Direction: minimize. Success: fraction_improved >= 0.5 (combined
    with feasibility + guards).
    """
    T = float(distance_threshold)

    def fn(instance: Instance, solution: Solution) -> float:
        opened_idx = np.where(solution.x == 1)[0]
        if len(opened_idx) == 0:
            return float(instance.n_precincts)
        nearest = instance.distance_matrix[:, opened_idx].min(axis=1)
        return float((nearest > T).sum())
    return fn


def n_discontiguous_catchments() -> Callable[[Instance, Solution], float]:
    """Count of opened sites whose assigned-precinct subgraph has > 1
    connected component on the precinct adjacency graph. The contiguity
    archetype's primary target (direction: minimize)."""
    def fn(instance: Instance, solution: Solution) -> float:
        from generation import _precinct_adjacency, _connected_components
        adj = _precinct_adjacency(instance.precinct_label_grid)
        assigned = solution.y.argmax(axis=1)
        n_split = 0
        for j in np.where(solution.x == 1)[0]:
            members = np.where(assigned == j)[0]
            if len(members) < 2:
                continue
            comps = _connected_components(members, adj)
            if len(comps) > 1:
                n_split += 1
        return float(n_split)
    return fn


def mean_npi() -> Callable[[Instance, Solution], float]:
    """Mean normalised perimeter index across opened catchments. The
    shape_niceness archetype's primary target (direction: minimize).
    NPI = P / (2 * sqrt(pi * A)); 1.0 for a circle, larger for elongated
    or jagged shapes."""
    def fn(instance: Instance, solution: Solution) -> float:
        from generation import _per_catchment_npi, aggregate_npi
        per = _per_catchment_npi(instance, solution)
        agg = aggregate_npi(per)
        v = agg["mean_npi"]
        return float('inf') if (isinstance(v, float) and np.isnan(v)) else float(v)
    return fn


# =========================================================================
# Dataclasses
# =========================================================================
@dataclass
class GuardSpec:
    """A constraint on how much the response can degrade a given metric.

    Pick exactly one of max_pct_increase / max_abs_increase / max_abs_value.
    """
    name: str
    metric_fn: Callable[[Instance, Solution], float]
    max_pct_increase: Optional[float] = None
    max_abs_increase: Optional[float] = None
    max_abs_value: Optional[float] = None

    def evaluate(self, instance: Instance,
                  baseline: Solution, response: Solution) -> Dict[str, Any]:
        baseline_val = self.metric_fn(instance, baseline)
        response_val = self.metric_fn(instance, response)
        if self.max_pct_increase is not None:
            bound = baseline_val * (1.0 + self.max_pct_increase)
        elif self.max_abs_increase is not None:
            bound = baseline_val + self.max_abs_increase
        elif self.max_abs_value is not None:
            bound = self.max_abs_value
        else:
            bound = baseline_val
        violation = max(0.0, response_val - bound)
        return {
            "name": self.name,
            "baseline": float(baseline_val),
            "response": float(response_val),
            "bound": float(bound),
            "violation": float(violation),
            "passed": bool(violation == 0.0),
        }


@dataclass
class ArchetypeQuery:
    """A single stakeholder query with its archetype-specific scoring."""
    query_id: str
    archetype: str
    text: str
    target_metric_fn: Callable[[Instance, Solution], float] = None
    target_direction: str = "minimize"
    guards: List[GuardSpec] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_criterion_fn: Optional[
        Callable[[Dict[str, Any], "Instance", "Solution", "Solution"], bool]
    ] = None

    def score(self, instance: Instance,
              baseline: Solution, response: Solution) -> Dict[str, Any]:
        target_baseline = float(self.target_metric_fn(instance, baseline))
        target_response = float(self.target_metric_fn(instance, response))

        if self.target_direction == "minimize":
            improvement = target_baseline - target_response
        else:
            improvement = target_response - target_baseline
        denom = abs(target_baseline) if abs(target_baseline) > 1e-9 else 1.0
        fraction_improved = max(0.0, improvement / denom)

        guard_results = [g.evaluate(instance, baseline, response) for g in self.guards]
        all_guards_passed = all(g["passed"] for g in guard_results)
        feasible = bool(response.metadata.get("feasible", True))

        # Secondary metric tracked uniformly across archetypes: the response
        # solution's total voter-weighted assignment distance, plus the
        # baseline's value as a reference. Two agents that fix the property
        # equally well can still differ in solution quality — this metric
        # exposes that difference.
        baseline_total_dist = float(
            total_weighted_distance(instance, baseline))
        response_total_dist = float(
            total_weighted_distance(instance, response))

        score_dict = {
            "query_id": self.query_id,
            "archetype": self.archetype,
            "target_direction": self.target_direction,
            "target_baseline": target_baseline,
            "target_response": target_response,
            "raw_improvement": float(improvement),
            "fraction_improved": float(fraction_improved),
            "guards": guard_results,
            "all_guards_passed": all_guards_passed,
            "feasible": feasible,
            "valid": all_guards_passed and feasible,
            "baseline_assignment_distance": baseline_total_dist,
            "final_assignment_distance": response_total_dist,
            "assignment_distance_delta": response_total_dist - baseline_total_dist,
        }
        if self.success_criterion_fn is not None:
            try:
                metric_passes = bool(self.success_criterion_fn(
                    score_dict, instance, baseline, response,
                ))
            except Exception:
                metric_passes = False
            score_dict["success"] = bool(
                metric_passes and feasible and all_guards_passed
            )
        return score_dict


# =========================================================================
# Default guards (reused across archetypes; easy to override per query)
# =========================================================================
def default_guards(
    pct_total_dist_increase: float = 0.05,
    p90_abs_increase: float = 0.0,
) -> List[GuardSpec]:
    return [
        GuardSpec("total_weighted_distance", total_weighted_distance,
                  max_pct_increase=pct_total_dist_increase),
        GuardSpec("p90_voter_distance", p90_voter_distance,
                  max_abs_increase=p90_abs_increase),
    ]


# =========================================================================
# Archetype factories
# =========================================================================
def make_cluster_query_from_metadata(
    query_id: str, text: str, metadata_dict: Dict[str, Any],
    description: str = "",
) -> ArchetypeQuery:
    """Cluster archetype.

    Target metric: GLOBAL count of opened sites that appear inside any
    dense radius-neighbourhood (>= cluster_min_sites opened sites within
    cluster_radius). Direction: minimize.

    The metric is global — an agent who breaks up the engineered cluster
    but inadvertently creates a new tight cluster elsewhere is
    penalised.

    Success: target_response == 0 (no dense radius-neighbourhood of the
    threshold size exists).
    """
    metadata_radius = float(metadata_dict["cluster_radius"])
    cluster_radius = min(metadata_radius, MAX_CLUSTER_SCORING_RADIUS)
    cluster_min_sites = int(metadata_dict.get("cluster_min_sites", 4))

    def _success(score_dict, instance, baseline, response):
        return score_dict.get("target_response", 1.0) <= 0.0

    return ArchetypeQuery(
        query_id=query_id,
        archetype="cluster",
        text=text,
        target_metric_fn=n_sites_in_dense_cluster(
            cluster_radius, cluster_min_sites),
        target_direction="minimize",
        guards=default_guards(pct_total_dist_increase=0.05),
        description=description,
        metadata={**metadata_dict,
                   "cluster_scoring_radius": cluster_radius,
                   "cluster_min_sites": cluster_min_sites,
                   "success_threshold_n_sites_in_cluster": 0},
        success_criterion_fn=_success,
    )


def make_coverage_gap_query_from_metadata(
    query_id: str, text: str, metadata_dict: Dict[str, Any],
    description: str = "",
    success_fraction: float = 0.5,
    distance_threshold: Optional[float] = None,
) -> ArchetypeQuery:
    """Coverage-gap archetype.

    Target metric: GLOBAL count of precincts whose nearest-open-site
    distance exceeds `distance_threshold` km — i.e., voters who are
    "stranded" anywhere on the map. Direction: minimize.

    The metric is global — an agent who fixes the engineered gap but
    inadvertently creates a new stranded pocket elsewhere (by closing
    a site that was serving it) is penalised: the newly-stranded
    precincts cross the threshold.

    Success: fraction_improved >= `success_fraction` (default 0.5 — at
    least half the stranded precincts have been brought within
    threshold without creating new ones). Combined with feasibility +
    guards.
    """
    # If the caller didn't pass an explicit threshold, use the per-pair
    # value the generator stored in metadata (calibrated so the baseline
    # has ~5 stranded precincts). Fallback to 2.0 km for legacy pairs.
    if distance_threshold is None:
        distance_threshold = float(
            metadata_dict.get("coverage_gap_distance_threshold", 2.0))

    def _success(score_dict, instance, baseline, response):
        return score_dict.get("fraction_improved", 0.0) >= success_fraction

    return ArchetypeQuery(
        query_id=query_id,
        archetype="coverage_gap",
        text=text,
        target_metric_fn=n_precincts_in_coverage_gap(
            distance_threshold=distance_threshold,
        ),
        target_direction="minimize",
        guards=default_guards(pct_total_dist_increase=0.05),
        description=description,
        metadata={**metadata_dict,
                   "coverage_gap_distance_threshold":
                       float(distance_threshold),
                   "success_threshold_fraction_improved":
                       float(success_fraction)},
        success_criterion_fn=_success,
    )


def make_contiguity_query_from_metadata(
    query_id: str, text: str, metadata_dict: Dict[str, Any],
    description: str = "",
    success_fraction: float = 0.5,
) -> ArchetypeQuery:
    """Contiguity (formerly split_catchment) archetype.

    Target metric: COUNT of opened sites with non-contiguous catchments.
    Direction: minimize. The metric is interpretable ("how many sites
    are still split?") and supports a clean fractional success criterion.

    Success: fraction_improved >= `success_fraction` (default 0.5 — at
    least half of the split sites have been repaired). Combined with
    feasibility + guards.
    """
    def _success(score_dict, instance, baseline, response):
        return score_dict.get("fraction_improved", 0.0) >= success_fraction

    return ArchetypeQuery(
        query_id=query_id,
        archetype="contiguity",
        text=text,
        target_metric_fn=n_discontiguous_catchments(),
        target_direction="minimize",
        guards=default_guards(pct_total_dist_increase=0.05),
        description=description,
        metadata={**metadata_dict,
                   "success_threshold_fraction_improved":
                       float(success_fraction)},
        success_criterion_fn=_success,
    )


def make_shape_niceness_query_from_metadata(
    query_id: str, text: str, metadata_dict: Dict[str, Any],
    description: str = "",
    success_fraction: float = 0.2,
) -> ArchetypeQuery:
    """Shape-niceness archetype.

    Target metric: mean NPI across opened catchments. Direction: minimize
    (lower NPI means rounder / more compact shapes; 1.0 = perfect circle).

    Success: fraction_improved >= `success_fraction` (default 0.2 — a 20%
    reduction in mean NPI; shapes are visibly nicer overall). Combined
    with feasibility + guards.
    """
    def _success(score_dict, instance, baseline, response):
        return score_dict.get("fraction_improved", 0.0) >= success_fraction

    return ArchetypeQuery(
        query_id=query_id,
        archetype="shape_niceness",
        text=text,
        target_metric_fn=mean_npi(),
        target_direction="minimize",
        guards=default_guards(pct_total_dist_increase=0.05),
        description=description,
        metadata={**metadata_dict,
                   "success_threshold_fraction_improved":
                       float(success_fraction)},
        success_criterion_fn=_success,
    )


# =========================================================================
# Public dispatch table — used by run_dataset.py and friends
# =========================================================================
ARCHETYPE_FACTORIES = {
    "cluster": make_cluster_query_from_metadata,
    "coverage_gap": make_coverage_gap_query_from_metadata,
    "contiguity": make_contiguity_query_from_metadata,
    "shape_niceness": make_shape_niceness_query_from_metadata,
}

ARCHETYPE_NAMES = list(ARCHETYPE_FACTORIES.keys())
