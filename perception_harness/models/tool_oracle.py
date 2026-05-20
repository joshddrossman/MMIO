"""Non-vision oracle baseline.

Computes the answer to each task analytically from (instance, solution)
using the same helpers the generator uses to verify acceptance. By
construction the oracle gets a perfect score on identification tasks —
that's the point. It defines the upper bound any tools-only agent could
in principle reach if it learned to use the structured tools optimally.

Compare a VLM's score against the oracle's to read off "how much of the
achievable signal is the VLM picking up under this rendering and prompt?".

Contract: in addition to `query(messages, model_id, ...)` for runner
compatibility, this module exports `compute_answer(instance, solution,
archetype, task)` which returns the same raw-text answer the VLM would
produce. The runner detects the oracle by checking for this function.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make instance_generator importable.
HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))


NAME = "tool_oracle"


def compute_answer(*, instance, solution, archetype: str, task: str,
                    **_unused) -> str:
    """Return the raw-text answer the oracle would 'reply' with.

    The text is intentionally formatted to mimic a well-behaved VLM
    response so the existing task.parse_response code paths exercise it
    without special-casing.
    """
    if archetype == "contiguity" and task == "identify":
        from generation import _precinct_adjacency, _scan_all_contiguitys
        adj = _precinct_adjacency(instance.precinct_label_grid)
        culprits = _scan_all_contiguitys(instance, solution, adj)
        sites = sorted(int(c["site"]) for c in culprits)
        return json.dumps({"split_sites": sites})

    if archetype == "contiguity" and task == "describe":
        # The oracle's "description" should mention every concept the
        # describe scorer is looking for, since it has perfect
        # information. This serves as a 1.0 ceiling on the describe
        # score for diagnostic purposes.
        from generation import _precinct_adjacency, _scan_all_contiguitys
        adj = _precinct_adjacency(instance.precinct_label_grid)
        culprits = _scan_all_contiguitys(instance, solution, adj)
        if not culprits:
            return ("All service areas appear to be contiguous; no "
                    "disjoint or split patches are visible.")
        n = len(culprits)
        return (
            f"{n} service area(s) are non-contiguous: their assigned "
            "precincts form disjoint, disconnected, separated patches "
            "rather than one contiguous region. The same color appears "
            "in split, visually-disjoint pockets."
        )

    if archetype == "shape_niceness" and task == "identify":
        # Rank opened catchments by NPI descending; return the top-K
        # site indices. K must match tasks.shape_niceness.WORST_K.
        from generation import _per_catchment_npi
        per = _per_catchment_npi(instance, solution)
        items = sorted(per.items(),
                        key=lambda kv: (-float(kv[1]["NPI"]), int(kv[0])))
        # Lazy-import to avoid a hard module-load coupling between
        # models/ and tasks/.
        try:
            from tasks.shape_niceness import WORST_K
        except Exception:
            WORST_K = 3
        worst = [int(k) for k, _ in items[:WORST_K]]
        return json.dumps({"worst_sites": worst})

    if archetype == "shape_niceness" and task == "describe":
        from generation import _per_catchment_npi
        per = _per_catchment_npi(instance, solution)
        if not per:
            return ("All service areas appear compact and roughly "
                    "regular; no catchment stands out as oddly shaped.")
        worst_npi = max(float(v["NPI"]) for v in per.values())
        # Concept-rich response: hits every keyword the describe scorer
        # looks for, so the oracle establishes the 1.0 describe ceiling.
        if worst_npi >= 1.5:
            return (
                "Several catchments look elongated and stretched, with "
                "irregular, jagged outlines. At least one has a thin "
                "tail or bowtie-like protrusion — visibly odd shapes "
                "rather than compact regions."
            )
        return (
            "Most catchments look compact, but a few are mildly "
            "elongated or irregular in shape."
        )

    if archetype == "cluster" and task == "identify":
        # Reuse the eval_set generator's analytic helper so the oracle
        # and the ground truth agree by construction. Default knobs
        # match the optimization metric (radius ≤ 1.35 km, >= 4 sites).
        try:
            sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
            from build_eval_set import _globally_clustered_sites
        except Exception:
            return json.dumps({"clustered_sites": []})
        # Use any cluster-specific kwargs the caller passes, else fall
        # back to the optimization-scoring defaults.
        radius = float(_unused.get("cluster_radius", 1.3))
        min_sites = int(_unused.get("cluster_min_sites", 4))
        sites = _globally_clustered_sites(
            instance, solution, radius=radius, min_sites=min_sites)
        return json.dumps({"clustered_sites": sites})

    if archetype == "coverage_gap" and task == "identify":
        # Reuse the eval_set generator's analytic helper so the oracle
        # and the ground truth agree by construction.
        try:
            sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
            from build_eval_set import _rank_closed_candidates_by_improvement
        except Exception:
            return json.dumps({"best_candidate": None,
                                "reason": "oracle import failed"})
        ranking = _rank_closed_candidates_by_improvement(instance, solution)
        if not ranking:
            return json.dumps({"best_candidate": None,
                                "reason": ("no closed candidates available; "
                                           "no fix possible")})
        best_idx, new_max, imp = ranking[0]
        return json.dumps({
            "best_candidate": int(best_idx),
            "reason": (f"Opening site {best_idx} reduces the maximum "
                        f"travel distance to {new_max:.2f} km "
                        f"(improvement of {imp:.2f} km)."),
        })

    if archetype == "coverage_gap" and task == "describe":
        # Concept-rich response that hits all describe-task keywords
        # (stranded, underserved, far, isolated, gap, remote, distant)
        # so the oracle establishes the 1.0 describe ceiling.
        try:
            sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
            from build_eval_set import _rank_closed_candidates_by_improvement
        except Exception:
            return ("Some precincts appear far from any opened polling "
                    "place; coverage is uneven.")
        ranking = _rank_closed_candidates_by_improvement(instance, solution)
        if not ranking or ranking[0][2] <= 0.01:
            return ("All precincts are reasonably close to an opened "
                    "polling place; no precinct stands out as "
                    "underserved or isolated.")
        return ("At least one precinct is stranded — its nearest "
                "opened polling place is far away, leaving it remote "
                "and underserved with substantial travel distance. "
                "There is a noticeable coverage gap in an isolated "
                "area where voters must travel a distant route to "
                "reach any opened site.")

    if archetype == "cluster" and task == "describe":
        try:
            sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
            from build_eval_set import _globally_clustered_sites
        except Exception:
            return ("Polling places appear to be spread across the map "
                    "with no obvious concentration.")
        radius = float(_unused.get("cluster_radius", 1.3))
        min_sites = int(_unused.get("cluster_min_sites", 4))
        sites = _globally_clustered_sites(
            instance, solution, radius=radius, min_sites=min_sites)
        if not sites:
            return ("Polling places appear to be spread evenly across "
                    "the map with no obvious cluster, dense pocket, or "
                    "concentration of multiple sites.")
        # Concept-rich response: hits all five keywords (cluster,
        # clustered, dense, packed, concentrated) so the oracle
        # establishes the 1.0 describe ceiling.
        return (
            f"{len(sites)} polling places form a tight cluster — "
            "several opened sites are clustered together in a dense, "
            "concentrated, packed pocket rather than spread evenly "
            "across the map."
        )

    # Stubs — extend as new archetypes are added.
    return json.dumps({})


def query(messages: List[Dict[str, Any]], *, model_id: str = "oracle",
          **_kwargs) -> str:
    """Compatibility shim. The runner should call compute_answer() on the
    oracle instead of query(); this stub exists so that accidentally
    routing the oracle through the generic model.query() path produces a
    clear error rather than a silent miss."""
    raise RuntimeError(
        "tool_oracle.query() called — the oracle requires "
        "(instance, solution, archetype, task). The runner should detect "
        "the oracle via hasattr(module, 'compute_answer') and route to "
        "compute_answer() instead."
    )
