"""Generate a graded eval set for the perception harness.

Re-uses the archetype generators from ../instance_generator. Each archetype
gets three difficulty tiers (easy / med / hard). For now only contiguity is
wired up — the other archetypes follow the same pattern and are stubbed at
the bottom.

Output structure (relative to perception_harness/):

    eval_set/pairs/<archetype>_<difficulty>_<NN>/
        instance.pkl
        baseline_solution.pkl
        meta.json                        # generator metadata + difficulty tag

    eval_set/ground_truth.csv            # flat, one row per (pair, task)

Run:
    cd perception_harness
    python eval_set/build_eval_set.py --archetypes contiguity --n_per_tier 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


# Make instance_generator importable so we can reuse generation.py and the
# Instance/Solution dataclasses (the pickles depend on these modules).
# Also put the harness root on sys.path so we can `import renderers.<name>`
# below to pre-render maps at generation time.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
INSTANCE_GEN = PROJECT_ROOT / "instance_generator"
sys.path.insert(0, str(INSTANCE_GEN))
sys.path.insert(0, str(HARNESS_ROOT))


# ---------------------------------------------------------------------------
# Difficulty tiers per archetype
# ---------------------------------------------------------------------------
# Each tier is a kwargs dict passed to the archetype's generator function.
# Easier tiers raise the acceptance thresholds so accepted pairs exhibit the
# property more strikingly; harder tiers lower them so the property is more
# subtle (and presumably harder for a VLM to pick up from the rendered map).

CONTIGUITY_TIERS: Dict[str, Dict[str, Any]] = {
    "easy": {"min_split_voters": 1, "min_separation": 3.0},
    "med":  {"min_split_voters": 1, "min_separation": 1.8},
    "hard": {"min_split_voters": 1, "min_separation": 1.0},
}

# Shape-niceness tiers — UNCAPACITATED regime.
#
# Generated via _generate_shape_niceness_uncapacitated below, which
# overrides every site's capacity to a non-binding value before solving
# the baseline. Result: catchments are Voronoi-like and almost always
# contiguous on the precinct adjacency graph, isolating "ugly shape"
# from the contiguity artefact that capacity-overflow perturbations
# would otherwise introduce.
#
# NPI in this regime is necessarily smaller than in the capacity-
# perturbed regime, so the acceptance bar is correspondingly lower.
# These values are starting points — tune to your own observed NPI
# distribution after a small generation run.
SHAPE_NICENESS_TIERS: Dict[str, Dict[str, Any]] = {
    "easy": {"min_mean_npi": 1.30, "min_max_npi": 1.65},
    "med":  {"min_mean_npi": 1.20, "min_max_npi": 1.45},
    "hard": {"min_mean_npi": 1.15, "min_max_npi": 1.30},
}

# Cluster tiers — easier tier = denser engineered cluster (higher
# acceptance bar), so accepted pairs exhibit a more obvious cluster of
# polling places packed close together. The optimization metric caps
# `cluster_radius` at 1.35 km regardless, so we hold radius fixed and
# vary `cluster_min_sites` and `cluster_density_factor` instead.
CLUSTER_TIERS: Dict[str, Dict[str, Any]] = {
    "easy": {"cluster_radius": 1.3, "cluster_min_sites": 5,
             "cluster_density_factor": 2.5},
    "med":  {"cluster_radius": 1.3, "cluster_min_sites": 4,
             "cluster_density_factor": 2.0},
    "hard": {"cluster_radius": 1.3, "cluster_min_sites": 4,
             "cluster_density_factor": 1.5},
}

# Coverage-gap tiers — UNCAPACITATED regime.
#
# `min_max_distance` (km) sets how far the worst-served precinct must
# be from its nearest opened polling place; larger = more obviously
# stranded. `min_improvement` (km) sets how much the best closed
# candidate must reduce that maximum if opened; larger = more
# unambiguously the right candidate. Easier tiers raise both bars
# (more striking gap, more obvious fix). The map is 10×10 km; typical
# precinct→nearest-open distances are 0.5–2 km, so a 3+ km worst is
# already noticeably stranded. Tune empirically after a small run.
COVERAGE_GAP_TIERS: Dict[str, Dict[str, Any]] = {
    "easy": {"min_max_distance": 3, "min_improvement": 1.4},
    "med":  {"min_max_distance": 2.6, "min_improvement": 1.0},
    "hard": {"min_max_distance": 2.2, "min_improvement": 0.6},
}
COVERAGE_GAP_SEED_ATTEMPTS = 20_000


# ---------------------------------------------------------------------------
# Per-archetype task definitions
# ---------------------------------------------------------------------------
# Each task definition knows how to derive the ground truth from the
# generator's metadata. Tasks are kept narrow on purpose: one perception
# question per task with a programmatic answer.

def _contiguity_ground_truth(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {task_name: ground_truth_dict} for a contiguity pair."""
    split_sites = sorted(int(c["site"]) for c in meta.get("culprits", []))
    return {
        "identify": {"split_sites": split_sites},
        "describe": {
            # Concepts the VLM ought to mention if it's actually seeing the
            # disjoint same-color patches. Scored as fraction-overlap.
            "concepts": ["disjoint", "split", "disconnected",
                         "non-contiguous", "separated"],
        },
    }


# Top-K size for shape_niceness identify. Mirrors tasks/shape_niceness.WORST_K
# — kept in sync because the prompt asks the VLM for exactly this many sites.
_SHAPE_NICENESS_K = 3
# Secondary-score neighbourhood size. Mirrors
# tasks/shape_niceness.TOP_N_NEIGHBOURHOOD; we record the top-N indices in
# the ground truth so secondary_scores() can compute "are the VLM's picks
# at least in the N worst offenders?" without needing access to the meta.
_SHAPE_NICENESS_TOP_N = 6


# ---------------------------------------------------------------------------
# Uncapacitated shape_niceness generator
# ---------------------------------------------------------------------------
# Mirrors generation.generate_shape_niceness_instance, but:
#   1. Overrides every site's capacity to a non-binding value (total voters)
#      BEFORE solving the baseline. With no binding capacity constraint
#      the MILP is effectively a K-median assignment — each precinct goes
#      to its nearest opened site — so the resulting catchments are
#      Voronoi-like and almost always contiguous on the precinct
#      adjacency graph. This isolates "ugly shape" from the contiguity
#      artefact that capacity-overflow perturbations otherwise introduce.
#   2. Skips the capacity-shrink perturbation fallback. The acceptance
#      gate is purely the natural-Voronoi NPI thresholds.
#
# NPI thresholds in this regime are necessarily lower than the
# capacity-perturbed regime, since pure Voronoi shapes are more compact.
# SHAPE_NICENESS_TIERS below is calibrated for that.

def _generate_shape_niceness_uncapacitated(
    base_seed: int,
    *,
    min_mean_npi: float,
    min_max_npi: float,
    max_attempts: int = 1,
    verbose: bool = False,
    **_unused: Any,
):
    """Generate one uncapacitated shape_niceness pair.

    Same return shape as generation.generate_shape_niceness_instance:
    (instance, baseline_solution, metadata). Raises RuntimeError if no
    seed in [base_seed, base_seed + 17 * (max_attempts - 1)] meets the
    NPI thresholds. The caller (eval_set._generate_one_tier) passes
    max_attempts=1 and iterates seeds itself.
    """
    import generation                 # gurobi-dependent
    from instance import Instance
    from solver import solve_baseline

    last_summary = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        base = generation.generate_instance(seed, None)

        # Override capacities so no constraint binds — true K-median
        # assignment. Setting each site's capacity to total_voters is
        # safely non-binding (any single site could absorb every voter).
        total_voters = int(base.precinct_voters.sum())
        new_caps = np.full(base.n_sites, total_voters, dtype=int)
        uncapped = Instance(
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

        sol = solve_baseline(uncapped, verbose=False)
        if not sol.metadata.get("feasible", True):
            continue

        per_cmap = generation._per_catchment_npi(uncapped, sol)
        agg = generation.aggregate_npi(per_cmap)
        last_summary = agg

        if (agg["mean_npi"] >= min_mean_npi
                and agg["max_npi"] >= min_max_npi):
            worst_site = max(per_cmap.items(),
                              key=lambda kv: kv[1]["NPI"])[0]
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
                "uncapacitated": True,
            }
            if verbose:
                print(f"[shape_niceness/uncapped] attempt {attempt} "
                      f"(seed {seed}): ACCEPTED — "
                      f"mean_NPI={agg['mean_npi']:.2f}, "
                      f"max_NPI={agg['max_npi']:.2f}, "
                      f"worst site {worst_site} "
                      f"(NPI={worst_npi:.2f})")
            return uncapped, sol, meta

    raise RuntimeError(
        f"Could not generate uncapacitated shape_niceness instance "
        f"in {max_attempts} attempt(s). Last summary: {last_summary}"
    )


# ---------------------------------------------------------------------------
# Coverage-gap helpers + uncapacitated generator + ground truth
# ---------------------------------------------------------------------------
# A coverage-gap pair has at least one stranded precinct (its nearest
# opened polling place is significantly farther than typical) AND at
# least one closed candidate site whose opening would meaningfully
# reduce the maximum travel distance across all precincts. The
# perception task: identify that closed candidate.
#
# Like shape_niceness, this is generated UNCAPACITATED (each site's
# capacity overridden to a non-binding value) so that contiguity does
# not arise. With pure K-median assignment, the only thing the model
# is being tested on is whether it can spot the under-served region
# and the candidate that best fixes it.

def _rank_closed_candidates_by_improvement(instance, solution):
    """For each closed candidate site, compute the new maximum travel
    distance across all precincts if that candidate were opened (each
    precinct reassigned to its nearest opened site, including the new
    one). Returns a list of (site_index, new_max_km, improvement_km)
    sorted by improvement descending; ties broken by site_index ascending.

    Used by the coverage_gap generator's acceptance gate, by the
    coverage_gap ground-truth function, and by the tool_oracle's
    coverage_gap branch — single source of truth.
    """
    opened_idx = np.where(solution.x == 1)[0]
    closed_idx = np.where(solution.x == 0)[0]
    if len(opened_idx) == 0 or len(closed_idx) == 0:
        return []
    D = instance.distance_matrix
    current_dists = D[:, opened_idx].min(axis=1)
    current_max = float(current_dists.max())
    out = []
    for j in closed_idx:
        new_dists = np.minimum(current_dists, D[:, j])
        new_max = float(new_dists.max())
        out.append((int(j), new_max, current_max - new_max))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


def _generate_coverage_gap_uncapacitated(
    base_seed: int,
    *,
    min_max_distance: float,
    min_improvement: float,
    max_attempts: int = 1,
    verbose: bool = False,
    **_unused: Any,
):
    """Generate one uncapacitated coverage_gap pair.

    Acceptance criteria:
      - The current solution's maximum travel distance is at least
        `min_max_distance` km (i.e., a meaningful stranded precinct
        exists).
      - At least one closed candidate, if opened, would reduce the
        maximum travel distance by at least `min_improvement` km
        (i.e., a viable fix exists).

    Returns (instance, baseline_solution, metadata) in the same shape
    as the other generators. The metadata records the analytic best
    candidate plus the full ranking so the ground-truth function can
    surface top-K secondary scores cheaply.
    """
    import generation
    from instance import Instance
    from solver import solve_baseline

    last_summary = None
    for attempt in range(max_attempts):
        seed = base_seed + 17 * attempt
        base = generation.generate_instance(seed, None)

        # Override capacities so no constraint binds — true K-median.
        total_voters = int(base.precinct_voters.sum())
        new_caps = np.full(base.n_sites, total_voters, dtype=int)
        uncapped = Instance(
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

        sol = solve_baseline(uncapped, verbose=False)
        if not sol.metadata.get("feasible", True):
            continue

        opened_idx = np.where(sol.x == 1)[0]
        if len(opened_idx) == 0:
            continue
        D = uncapped.distance_matrix
        current_dists = D[:, opened_idx].min(axis=1)
        current_max = float(current_dists.max())
        most_stranded = int(current_dists.argmax())

        ranking = _rank_closed_candidates_by_improvement(uncapped, sol)
        if not ranking:
            continue
        best_idx, best_new_max, best_imp = ranking[0]

        last_summary = {
            "current_max_km": current_max,
            "best_improvement_km": best_imp,
            "best_candidate_idx": best_idx,
        }

        if (current_max >= min_max_distance
                and best_imp >= min_improvement):
            per_candidate = {
                str(idx): {
                    "improvement_km": float(imp),
                    "new_max_km": float(nmax),
                }
                for idx, nmax, imp in ranking
            }

            # Optimization-side metadata that partner's query factories
            # and dataset_generator._build_query_texts expect:
            #   distance_threshold : a per-pair calibrated cutoff so the
            #                        baseline has ~5 "affected" precincts.
            #                        Used for the stranded-count secondary
            #                        diagnostic and as a discriminator for
            #                        which precincts are referenced in
            #                        precise-query text.
            #   affected_precincts : list of precincts with distance >
            #                        threshold (those whose names appear
            #                        in the precise-query template).
            #   coverage_gap_center: centroid of affected precincts (or
            #                        the most-stranded precinct's
            #                        centroid if affected is empty).
            #                        Used by _build_query_texts to derive
            #                        the {region} compass label.
            #   coverage_gap_radius: max distance from center to any
            #                        affected precinct's centroid.
            sorted_dists = np.sort(current_dists)[::-1]
            target_strands = 5
            k = min(target_strands, len(sorted_dists) - 1)
            if k >= 1:
                distance_threshold = float(
                    (sorted_dists[k - 1] + sorted_dists[k]) / 2.0)
            else:
                distance_threshold = float(sorted_dists[0])
            affected_mask = current_dists > distance_threshold
            affected_indices = [int(i) for i
                                 in np.where(affected_mask)[0]]
            if affected_indices:
                aff_centroids = uncapped.precinct_centroids[affected_indices]
                cx = float(aff_centroids[:, 0].mean())
                cy = float(aff_centroids[:, 1].mean())
                cg_radius = float(np.max(np.linalg.norm(
                    aff_centroids - np.array([cx, cy]), axis=1)))
            else:
                # Fallback: use the most-stranded precinct alone.
                cx = float(uncapped.precinct_centroids[most_stranded, 0])
                cy = float(uncapped.precinct_centroids[most_stranded, 1])
                cg_radius = 0.0
                affected_indices = [most_stranded]

            meta = {
                "archetype": "coverage_gap",

                # Perception-side fields (for harness's identify task).
                "current_max_distance_km": current_max,
                "most_stranded_precinct": most_stranded,
                "most_stranded_distance_km": current_max,
                "best_candidate_idx": int(best_idx),
                "best_candidate_new_max_km": float(best_new_max),
                "best_improvement_km": float(best_imp),
                "per_candidate_improvement": per_candidate,
                "candidates_ranked_by_improvement":
                    [int(idx) for idx, _, _ in ranking],

                # Optimization-side fields (for partner's run_dataset.py
                # and queries.py). Partner's templates use
                # coverage_gap_center to derive a {region} label and
                # affected_precincts to list precinct indices in the
                # precise-query text.
                "coverage_gap_center": [cx, cy],
                "coverage_gap_radius": cg_radius,
                "affected_precincts": affected_indices,
                "coverage_gap_distance_threshold": distance_threshold,
                "coverage_gap_baseline_strand_count":
                    int(len(affected_indices)),

                # Generation diagnostics.
                "n_opened": int(len(opened_idx)),
                "n_closed_candidates": int(len(ranking)),
                "min_max_distance_threshold_km": float(min_max_distance),
                "min_improvement_threshold_km": float(min_improvement),
                "base_seed": seed,
                "attempt": attempt,
                "uncapacitated": True,
            }
            if verbose:
                print(f"[coverage_gap/uncapped] attempt {attempt} "
                      f"(seed {seed}): ACCEPTED — "
                      f"max={current_max:.2f}km, "
                      f"best_cand={best_idx} "
                      f"(reduces max to {best_new_max:.2f}km, "
                      f"Δ={best_imp:.2f}km)")
            return uncapped, sol, meta

    raise RuntimeError(
        f"Could not generate uncapacitated coverage_gap instance "
        f"in {max_attempts} attempt(s). Last summary: {last_summary}"
    )


def _coverage_gap_ground_truth(meta: Dict[str, Any]
                                ) -> Dict[str, Dict[str, Any]]:
    """Return {task_name: ground_truth_dict} for a coverage_gap pair.

    The identify ground truth carries:
      - best_candidate          : the analytic best closed candidate
      - top3_candidates         : top-3 by improvement (secondary)
      - ranked_candidates       : full ranking (for rank-based metrics)
      - per_candidate_improvement : {str(idx): {improvement_km, new_max_km}}
      - best_improvement_km     : the achievable improvement (denominator
                                   for fraction_of_optimal_improvement)
    The describe ground truth is the concept-keyword set.
    """
    best = int(meta.get("best_candidate_idx", -1))
    ranking = meta.get("candidates_ranked_by_improvement", [])
    per = meta.get("per_candidate_improvement", {})
    return {
        "identify": {
            "best_candidate": best,
            "top3_candidates": [int(i) for i in ranking[:3]],
            "ranked_candidates": [int(i) for i in ranking],
            "per_candidate_improvement": per,
            "best_improvement_km":
                float(meta.get("best_improvement_km", 0.0)),
        },
        "describe": {
            "concepts": ["stranded", "underserved", "far",
                         "isolated", "gap", "remote", "distant"],
        },
    }


# ---------------------------------------------------------------------------
# Cluster ground truth
# ---------------------------------------------------------------------------
# Mirrors queries.n_sites_in_dense_cluster: a site is "clustered" iff it
# appears in any radius-neighbourhood with >= cluster_min_sites opened
# sites within cluster_radius. The set returned here is the same set
# the optimization metric counts, so the perception ground truth and
# the optimization scoring agree on what "in a cluster" means.

# Scoring cap from queries.MAX_CLUSTER_SCORING_RADIUS — older metadata
# may carry a wider engineered radius, but the perceptual ground truth
# uses the same capped radius the optimization scorer uses.
_CLUSTER_RADIUS_CAP = 1.35


def _globally_clustered_sites(instance, solution, *,
                                radius: float,
                                min_sites: int) -> List[int]:
    """Return the sorted list of opened-site indices that participate in
    any dense radius-neighbourhood (>= `min_sites` opened sites within
    `radius`, centre included)."""
    opened_idx = np.where(solution.x == 1)[0]
    if len(opened_idx) < min_sites:
        return []
    opened_xy = instance.site_locations[opened_idx]
    diff = opened_xy[:, None, :] - opened_xy[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    within_count = (d <= radius).sum(axis=1)
    qualifying_centres = within_count >= min_sites
    if not np.any(qualifying_centres):
        return []
    in_any = (d[qualifying_centres] <= radius).any(axis=0)
    return sorted(int(j) for j in opened_idx[in_any])


def _cluster_ground_truth(meta: Dict[str, Any]
                            ) -> Dict[str, Dict[str, Any]]:
    """Return {task_name: ground_truth_dict} for a cluster pair.

    Uses `globally_clustered_sites` injected into meta at generation
    time. Falls back to the engineered `affected_sites` if the global
    field is absent (legacy meta).
    """
    sites = meta.get("globally_clustered_sites")
    if sites is None:
        sites = meta.get("affected_sites", [])
    return {
        "identify": {
            "clustered_sites": sorted(int(j) for j in sites),
        },
        "describe": {
            "concepts": ["cluster", "clustered", "dense",
                         "packed", "concentrated"],
        },
    }


def _shape_niceness_ground_truth(meta: Dict[str, Any]
                                  ) -> Dict[str, Dict[str, Any]]:
    """Return {task_name: ground_truth_dict} for a shape_niceness pair.

    Two ranking axes are surfaced — primary by NPI (matches the
    optimization system's scoring + the generator's acceptance
    criteria), secondary by solidity (convex-hull ratio; catches
    non-convexity / tails / bowties more directly than NPI).

    NPI ranks descending: high NPI = bad shape (elongated or jagged).
    Solidity ranks ascending: low solidity = bad shape (concave /
    notched / spiral).

    Identify ground-truth fields:
      worst_sites              : top K=3 by NPI (primary).
      top{N}_sites_by_npi      : top N=10 by NPI (secondary, near-miss).
      top{N}_sites_by_solidity : top N=10 by solidity (cross-metric
                                  near-miss; absent on legacy pairs
                                  generated before solidity was added).
    Ties are broken by site index ascending so ranks are deterministic.
    """
    # NPI ranking — primary axis.
    per_npi = meta.get("per_catchment_npi", {})
    items_npi = []
    for k, v in per_npi.items():
        try:
            items_npi.append((int(k), float(v["NPI"])))
        except (KeyError, TypeError, ValueError):
            continue
    # Descending NPI (worst first); ties broken by site index ascending.
    items_npi.sort(key=lambda kv: (-kv[1], kv[0]))
    worst = [k for k, _ in items_npi[:_SHAPE_NICENESS_K]]
    top_n_npi = [k for k, _ in items_npi[:_SHAPE_NICENESS_TOP_N]]

    identify_gt: Dict[str, Any] = {
        "worst_sites": worst,
        f"top{_SHAPE_NICENESS_TOP_N}_sites_by_npi": top_n_npi,
    }

    # Solidity ranking — secondary axis. Skip when the field is absent
    # (legacy pairs / scipy-less environments).
    per_sol = meta.get("per_catchment_solidity", {})
    items_sol = []
    for k, v in per_sol.items():
        try:
            items_sol.append((int(k), float(v["solidity"])))
        except (KeyError, TypeError, ValueError):
            continue
    if items_sol:
        # Ascending solidity (worst first = least convex);
        # ties broken by site index ascending.
        items_sol.sort(key=lambda kv: (kv[1], kv[0]))
        top_n_sol = [k for k, _ in items_sol[:_SHAPE_NICENESS_TOP_N]]
        identify_gt[f"top{_SHAPE_NICENESS_TOP_N}_sites_by_solidity"] = top_n_sol

    return {
        "identify": identify_gt,
        "describe": {
            "concepts": ["elongated", "stretched", "jagged",
                         "irregular", "thin", "tail", "bowtie", "odd"],
        },
    }


ARCHETYPE_CONFIG: Dict[str, Dict[str, Any]] = {
    "contiguity": {
        "tiers": CONTIGUITY_TIERS,
        "ground_truth_fn": _contiguity_ground_truth,
        "tasks": ["identify", "describe"],
        "generator": "generate_contiguity_instance",
    },
    "shape_niceness": {
        "tiers": SHAPE_NICENESS_TIERS,
        "ground_truth_fn": _shape_niceness_ground_truth,
        "tasks": ["identify", "describe"],
        # Callable instead of a string name: the harness uses a local
        # uncapacitated wrapper so contiguity issues don't bleed into
        # shape_niceness pairs. _generate_one_tier accepts either form.
        "generator": _generate_shape_niceness_uncapacitated,
    },
    "cluster": {
        "tiers": CLUSTER_TIERS,
        "ground_truth_fn": _cluster_ground_truth,
        "tasks": ["identify", "describe"],
        "generator": "generate_cluster_instance",
    },
    "coverage_gap": {
        "tiers": COVERAGE_GAP_TIERS,
        "ground_truth_fn": _coverage_gap_ground_truth,
        "tasks": ["identify", "describe"],
        "seed_attempts": COVERAGE_GAP_SEED_ATTEMPTS,
        # Callable wrapper — same pattern as shape_niceness. Uncapacitated
        # K-median assignment; identifies the best closed candidate to
        # open for max-distance reduction.
        "generator": _generate_coverage_gap_uncapacitated,
    },
}


# ---------------------------------------------------------------------------
# Per-catchment solidity (convex-hull ratio)
# ---------------------------------------------------------------------------
# Computed at generation time and written into meta.json next to
# per_catchment_npi. Solidity = catchment_area / convex_hull_area; 1.0 for
# convex catchments, lower for shapes with concave indentations / notches /
# bowtie waists / disjoint pieces. Complements NPI: NPI catches elongation +
# jaggedness, solidity catches non-convexity. Used by the harness's
# shape_niceness ground-truth as a *secondary* ranking axis to surface cases
# where the two metrics disagree about which catchments look worst.

_SCIPY_WARNED = False


def _per_catchment_solidity(instance, solution) -> Dict[int, Dict[str, float]]:
    """For each opened site, return {site_idx: {A, A_hull, solidity}}.

    Computed on cell *centres* of the rasterised catchment mask via
    scipy.spatial.ConvexHull. Catchments smaller than 3 cells are skipped
    (ConvexHull requires at least 3 distinct points). If scipy isn't
    installed the function returns an empty dict and prints one warning;
    downstream ground-truth code degrades gracefully (no top10_by_solidity
    entry in the GT, so the secondary score for solidity is silently
    omitted from per_question.csv).
    """
    global _SCIPY_WARNED
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.qhull import QhullError
    except ImportError:
        if not _SCIPY_WARNED:
            print("  WARNING: scipy not installed — solidity will not be "
                  "computed and the in_top10_by_solidity secondary score "
                  "will be unavailable. `pip install scipy` to enable.")
            _SCIPY_WARNED = True
        return {}

    label_grid = instance.precinct_label_grid
    xs = instance.grid_xs
    ys = instance.grid_ys
    cell_w = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    cell_h = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    cell_area = cell_w * cell_h

    assigned = solution.y.argmax(axis=1)
    cell_site = assigned[label_grid]      # (G, G), each cell -> assigned site
    XX, YY = np.meshgrid(xs, ys)

    out: Dict[int, Dict[str, float]] = {}
    for j in np.where(solution.x == 1)[0]:
        mask = (cell_site == j)
        n_cells = int(mask.sum())
        if n_cells < 3:
            continue
        A = float(n_cells) * cell_area
        pts = np.column_stack([XX[mask], YY[mask]])
        # If all points are colinear, ConvexHull raises QhullError.
        # That's not a meaningful catchment — skip.
        try:
            hull = ConvexHull(pts)
            A_hull = float(hull.volume)   # 2D: .volume is the polygon area
        except Exception:
            continue
        if A_hull <= 0:
            continue
        # Numerical clip: rasterisation can push the ratio fractionally
        # above 1 for nearly-square catchments.
        solidity = max(0.0, min(1.0, A / A_hull))
        out[int(j)] = {
            "A": A,
            "A_hull": A_hull,
            "solidity": float(solidity),
        }
    return out


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------
# Resolves a renderer module by name (matching the runner's --renderers
# convention) and writes <pair_dir>/views/<renderer_name>.png. Same path
# the runner uses, so a downstream `eval_perception.py` run will overwrite
# with bit-identical output unless the renderer code has changed.

def _render_to_disk(instance, solution, pair_dir: Path,
                    renderer_name: str) -> None:
    import importlib
    renderer = importlib.import_module(f"renderers.{renderer_name}")
    png = renderer.render(instance, solution)
    save_path = pair_dir / "views" / f"{renderer_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(png)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------
def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _generate_one_tier(
    archetype: str,
    difficulty: str,
    tier_kwargs: Dict[str, Any],
    n_pairs: int,
    seed_start: int,
    seed_max: int,
    archetype_out_dir: Path,
    renderer_names: List[str],
    template_rng: np.random.Generator,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate `n_pairs` pairs for one (archetype, difficulty) cell.

    Writes pair files into `archetype_out_dir/pairs/<pair_id>/`.

    Returns a tuple (gt_rows, pair_records):
      gt_rows       — for the harness's ground_truth.csv (one per task per pair)
      pair_records  — for the per-archetype index.json (one per pair),
                       matching the schema of
                       instance_generator/dataset_generator.py.
    """
    import generation
    from dataset_generator import _build_query_texts
    cfg = ARCHETYPE_CONFIG[archetype]
    gen_value = cfg["generator"]
    if callable(gen_value):
        gen_fn = gen_value
    else:
        gen_fn = getattr(generation, gen_value)
    gt_fn = cfg["ground_truth_fn"]
    tasks = cfg["tasks"]

    gt_rows: List[Dict[str, Any]] = []
    pair_records: List[Dict[str, Any]] = []
    pair_idx = 0
    seed = seed_start
    attempts = 0

    while pair_idx < n_pairs and seed < seed_max:
        attempts += 1
        try:
            inst, sol, meta = gen_fn(
                base_seed=seed, max_attempts=1, verbose=False, **tier_kwargs,
            )
        except RuntimeError:
            seed += 1
            continue

        pair_id = f"{archetype}_{difficulty}_{pair_idx:02d}"
        pair_dir = archetype_out_dir / "pairs" / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)

        inst.save(str(pair_dir / "instance.pkl"))
        sol.save(str(pair_dir / "baseline_solution.pkl"))

        # Per-catchment solidity for shape_niceness secondary scores.
        # Cheap; computed for all archetypes for cross-archetype analysis.
        solidity_dict = _per_catchment_solidity(inst, sol)

        # Globally-clustered-sites for the cluster archetype's ground
        # truth (also useful as a cross-archetype diagnostic).
        cluster_radius = float(meta.get(
            "cluster_radius", _CLUSTER_RADIUS_CAP))
        cluster_radius = min(cluster_radius, _CLUSTER_RADIUS_CAP)
        cluster_min_sites = int(meta.get("cluster_min_sites", 4))
        clustered = _globally_clustered_sites(
            inst, sol,
            radius=cluster_radius,
            min_sites=cluster_min_sites,
        )

        meta_full = dict(meta)
        meta_full["pair_id"] = pair_id
        meta_full["difficulty"] = difficulty
        meta_full["archetype"] = archetype
        meta_full["tier_kwargs"] = tier_kwargs
        if solidity_dict:
            meta_full["per_catchment_solidity"] = {
                str(k): v for k, v in solidity_dict.items()
            }
        meta_full["globally_clustered_sites"] = clustered
        meta_full["clustered_sites_radius"] = float(cluster_radius)
        meta_full["clustered_sites_min_sites"] = int(cluster_min_sites)

        # Generate vague + precise query texts using the partner's
        # templates. The partner's _build_query_texts function reads
        # the same metadata fields the harness already populates
        # (cluster_center, coverage_gap_center+affected_precincts,
        # worst_culprit_site, worst_catchment_site).
        try:
            vague, precise, vidx, pidx = _build_query_texts(
                archetype, inst, meta_full, template_rng)
            meta_full["vague_text"] = vague
            meta_full["precise_text"] = precise
            meta_full["vague_template_idx"] = int(vidx)
            meta_full["precise_template_idx"] = int(pidx)
        except Exception as e:
            print(f"    WARN: query-text generation failed for "
                  f"{pair_id}: {e}")
            vidx = pidx = -1

        # Write the unified per-pair metadata as `query_metadata.json`
        # (the filename partner's run_dataset.py reads). Same content
        # that used to live in `meta.json`, plus the vague/precise
        # query texts.
        with open(pair_dir / "query_metadata.json", "w") as f:
            json.dump(_make_json_safe(meta_full), f, indent=2)

        # Pre-render canonical view(s) into pair_dir/views/.
        for renderer_name in renderer_names:
            try:
                _render_to_disk(inst, sol, pair_dir, renderer_name)
            except Exception as e:
                print(f"    render({renderer_name}) failed: {e}")

        gt = gt_fn(meta_full)
        for task in tasks:
            gt_rows.append({
                "pair_id": pair_id,
                "archetype": archetype,
                "difficulty": difficulty,
                "task": task,
                "answer_json": json.dumps(_make_json_safe(gt[task])),
                # source_dir is relative to harness root so eval_perception.py
                # can resolve it via HERE / source_dir.
                "source_dir": str(pair_dir.relative_to(HARNESS_ROOT)),
            })

        # Per-archetype index.json record. Matches the schema produced
        # by instance_generator/dataset_generator.py so partner's
        # run_dataset.py reads it without modification. `pair_dir` is
        # relative to the per-archetype directory.
        pair_records.append({
            "pair_id": pair_id,
            "pair_dir": f"pairs/{pair_id}",
            "archetype": archetype,
            "difficulty": difficulty,
            "base_seed": int(meta.get("base_seed", seed)),
            "vague_template_idx": int(vidx),
            "precise_template_idx": int(pidx),
        })

        if verbose:
            extra = ""
            if archetype == "contiguity":
                n_split = meta.get("n_split_sites_baseline", 0)
                excess = meta.get("total_disjoint_excess_voters_baseline", 0)
                extra = f"  n_split={n_split} excess={excess}v"
            elif archetype == "shape_niceness":
                mean_npi = float(meta.get("mean_npi_baseline", float("nan")))
                max_npi = float(meta.get("max_npi_baseline", float("nan")))
                worst = meta.get("worst_catchment_site")
                extra = (f"  mean_npi={mean_npi:.2f} "
                         f"max_npi={max_npi:.2f} worst_site={worst}")
            elif archetype == "cluster":
                size = meta.get("cluster_size", 0)
                ratio = float(meta.get("cluster_density_ratio", 0.0))
                n_global = len(clustered)
                extra = (f"  cluster_size={size} density={ratio:.2f}× "
                         f"global_clustered={n_global}")
            elif archetype == "coverage_gap":
                cmax = float(meta.get("current_max_distance_km", 0.0))
                best_cand = meta.get("best_candidate_idx")
                imp = float(meta.get("best_improvement_km", 0.0))
                extra = (f"  max_dist={cmax:.2f}km best_cand={best_cand} "
                         f"Δmax={imp:.2f}km")
            print(f"  [{difficulty}] pair {pair_idx:02d}: seed={seed}{extra}")

        pair_idx += 1
        seed += 1

    if verbose and pair_idx < n_pairs:
        print(f"  WARNING: only {pair_idx}/{n_pairs} pairs for "
              f"{archetype}/{difficulty} ({attempts} seeds tried)")

    return gt_rows, pair_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetypes", nargs="+",
                    default=list(ARCHETYPE_CONFIG.keys()),
                    choices=list(ARCHETYPE_CONFIG.keys()))
    ap.add_argument("--difficulties", nargs="+",
                    default=["easy", "med", "hard"],
                    choices=["easy", "med", "hard"])
    ap.add_argument("--n_per_tier", type=int, default=10,
                    help="Pairs per (archetype, difficulty) cell.")
    ap.add_argument("--seed_start", type=int, default=1)
    ap.add_argument("--seed_max", type=int, default=2000)
    ap.add_argument("--sampling_seed", type=int, default=42,
                    help=("RNG seed for vague/precise template "
                          "selection. Holds template choice "
                          "deterministic across regenerations."))
    ap.add_argument("--out_dir", default=str(HERE),
                    help="eval_set/ root directory.")
    ap.add_argument("--full_dataset_dir", default="full_dataset",
                    help=("Subdirectory under --out_dir that holds the "
                          "per-archetype dataset structure. Partner's "
                          "run_dataset.py points at this directory's "
                          "per-archetype subdirs."))
    ap.add_argument("--renderers", nargs="+", default=["v2", "v2_no_markers"],
                    help=("Renderer module name(s) to pre-render at "
                          "generation time. Each is written to "
                          "<pair_dir>/views/<name>.png. "
                          "Default: v2 + v2_no_markers."))
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    full_dataset_root = out_root / args.full_dataset_dir
    full_dataset_root.mkdir(parents=True, exist_ok=True)

    template_rng = np.random.default_rng(args.sampling_seed)

    all_gt_rows: List[Dict[str, Any]] = []
    archetype_summary: Dict[str, Any] = {}

    for archetype in args.archetypes:
        if archetype not in ARCHETYPE_CONFIG:
            print(f"  skipping {archetype} — not yet wired up.")
            continue
        cfg = ARCHETYPE_CONFIG[archetype]
        tiers = cfg["tiers"]
        archetype_dir = full_dataset_root / archetype
        archetype_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {archetype} -> {archetype_dir} ===")

        archetype_pair_records: List[Dict[str, Any]] = []
        archetype_gt_rows: List[Dict[str, Any]] = []

        for difficulty in args.difficulties:
            if difficulty not in tiers:
                continue
            print(f"\n--- {archetype}/{difficulty} "
                  f"(kwargs={tiers[difficulty]}) ---")
            seed_offset = 7919 * list(tiers).index(difficulty)
            seed_start = args.seed_start + seed_offset
            seed_attempts = cfg.get("seed_attempts")
            seed_max = (
                seed_start + int(seed_attempts)
                if seed_attempts is not None
                else args.seed_max + seed_offset
            )
            gt_rows, pair_records = _generate_one_tier(
                archetype=archetype,
                difficulty=difficulty,
                tier_kwargs=tiers[difficulty],
                n_pairs=args.n_per_tier,
                # Spread seeds so different tiers don't collide.
                seed_start=seed_start,
                seed_max=seed_max,
                archetype_out_dir=archetype_dir,
                renderer_names=args.renderers,
                template_rng=template_rng,
            )
            archetype_gt_rows.extend(gt_rows)
            archetype_pair_records.extend(pair_records)

        # Per-archetype index.json — matches partner's
        # instance_generator/dataset_generator.py schema so that
        # `python run_dataset.py --dataset_dir <archetype_dir>` works
        # against this output directly.
        idx = {
            "n_pairs": len(archetype_pair_records),
            "n_pairs_requested": args.n_per_tier * len(args.difficulties),
            "archetype": archetype,
            "sampling_seed": args.sampling_seed,
            "tiers_used": list(args.difficulties),
            "n_per_tier": args.n_per_tier,
            "pairs": archetype_pair_records,
        }
        with open(archetype_dir / "index.json", "w") as f:
            json.dump(_make_json_safe(idx), f, indent=2)
        print(f"\nWrote {archetype_dir / 'index.json'}  "
              f"({len(archetype_pair_records)} pairs)")

        archetype_summary[archetype] = {
            "n_pairs": len(archetype_pair_records),
            "path": str(archetype_dir.relative_to(out_root)),
        }
        all_gt_rows.extend(archetype_gt_rows)

    # Top-level index.json across archetypes.
    summary = {
        "n_per_archetype": args.n_per_tier * len(args.difficulties),
        "n_per_tier": args.n_per_tier,
        "tiers": list(args.difficulties),
        "archetypes": archetype_summary,
    }
    with open(full_dataset_root / "index.json", "w") as f:
        json.dump(_make_json_safe(summary), f, indent=2)
    print(f"\nWrote top-level {full_dataset_root / 'index.json'}")

    # Harness-side ground-truth CSV — one row per (pair, task), with
    # source_dir pointing into full_dataset/. eval_perception.py reads
    # this. Pairs in CSV use forward slashes for cross-platform
    # compatibility.
    gt_path = out_root / "ground_truth.csv"
    with open(gt_path, "w", newline="") as f:
        fieldnames = ["pair_id", "archetype", "difficulty",
                       "task", "answer_json", "source_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_gt_rows:
            row["source_dir"] = str(row["source_dir"]).replace("\\", "/")
            writer.writerow(row)

    n_pairs = len({r["pair_id"] for r in all_gt_rows})
    print(f"\nWrote {gt_path}  ({n_pairs} pairs, {len(all_gt_rows)} task rows)")
    print()
    print("Layout:")
    print(f"  {out_root}/")
    print(f"    ground_truth.csv               <- harness Phase 1 row index")
    print(f"    full_dataset/")
    print(f"      index.json                   <- top-level multi-archetype")
    for archetype, info in archetype_summary.items():
        print(f"      {archetype}/")
        print(f"        index.json               <- {info['n_pairs']} pairs "
              f"(partner's run_dataset.py reads this)")
        print(f"        pairs/<pair_id>/")
        print(f"          instance.pkl, baseline_solution.pkl,")
        print(f"          query_metadata.json, views/v2.png, views/v2_no_markers.png")


if __name__ == "__main__":
    main()
