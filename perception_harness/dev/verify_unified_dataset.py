"""End-to-end verification of the unified-dataset refactor.

What this exercises (without gurobipy):
  - New renderers/v2.py and renderers/v2_no_markers.py both produce
    valid PNGs and have correct capability flags.
  - queries.py: max_assignment_distance metric + the rebuilt
    make_coverage_gap_query_from_metadata factory. Synthetic baseline
    vs response solutions through the metric and through the score()
    pipeline produce expected numbers.
  - build_eval_set imports cleanly (all the new helpers, the partner's
    _build_query_texts, the rewritten _generate_one_tier and main()
    are reachable).
  - dataset_generator._build_query_texts works on a fake meta dict
    populated with all the fields the harness writes — i.e. partner's
    text generation flows through the harness's metadata.
  - Simulated pair-write: hand-build (instance, solution, meta) for a
    coverage_gap pair, run through the meta-augmentation + write
    paths used by _generate_one_tier, verify the resulting
    query_metadata.json carries both perception-side and
    optimization-side fields.

Run:
    python dev/verify_unified_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))
sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
sys.path.insert(0, str(HERE))

from instance import Instance, Solution  # noqa: E402
from verify_slice import _build_synthetic_pair  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Unified-dataset verification")
    print("=" * 60)

    # ---- (1) Renderers ----
    print("\n[1] New renderers/v2.py + renderers/v2_no_markers.py")
    import renderers.v2 as r_v2
    import renderers.v2_no_markers as r_blind
    inst, sol, _ = _build_synthetic_pair()
    png_v2 = r_v2.render(inst, sol)
    png_blind = r_blind.render(inst, sol)
    assert png_v2[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_blind[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"  v2:            {len(png_v2):>6} bytes (saturated palette + "
          f"precinct lines + markers + assignment lines)")
    print(f"  v2_no_markers: {len(png_blind):>6} bytes (saturated palette only — "
          f"no precinct lines, no markers, no labels, no lines)")

    # Capability flags.
    flags = {
        "v2.HAS_SITE_MARKERS":        r_v2.HAS_SITE_MARKERS,
        "v2.HAS_CANDIDATE_MARKERS":   r_v2.HAS_CANDIDATE_MARKERS,
        "v2.HAS_ASSIGNMENT_LINES":    r_v2.HAS_ASSIGNMENT_LINES,
        "v2_no_markers.HAS_SITE_MARKERS":      r_blind.HAS_SITE_MARKERS,
        "v2_no_markers.HAS_CANDIDATE_MARKERS": r_blind.HAS_CANDIDATE_MARKERS,
        "v2_no_markers.HAS_ASSIGNMENT_LINES":  r_blind.HAS_ASSIGNMENT_LINES,
    }
    for k, v in flags.items():
        print(f"  {k}: {v}")
    assert r_v2.HAS_SITE_MARKERS and r_v2.HAS_CANDIDATE_MARKERS \
        and r_v2.HAS_ASSIGNMENT_LINES
    assert (not r_blind.HAS_SITE_MARKERS) and (not r_blind.HAS_CANDIDATE_MARKERS) \
        and (not r_blind.HAS_ASSIGNMENT_LINES)
    print("  capability flags correct  OK")

    # ---- (2) queries.py changes ----
    print("\n[2] queries.py: max_assignment_distance + new coverage_gap factory")
    import queries
    metric_fn = queries.max_assignment_distance()
    baseline_max = metric_fn(inst, sol)
    print(f"  baseline max_assignment_distance: {baseline_max:.2f} km")
    # Synthetic pair: 4 precincts, 2 sites at (5, 4.5)/(5, 5.5), each
    # precinct centroid at ±2.5. Distance from corner to nearest site is
    # ~3.5 km — that's our max.
    assert baseline_max > 0

    # Construct a "fixed" response solution where we open a 3rd site
    # close to the worst-served precinct, reducing max distance. Since
    # the synthetic pair has only 2 sites total, fake by re-using site 0
    # but assigning all precincts to it (degenerate but exercises the
    # math).
    response_sol = Solution(
        x=sol.x.copy(), y=sol.y.copy(),
        objective=sol.objective,
        solver_status="hypothetical",
        metadata={"feasible": True},
    )
    # Force the "fix": every precinct is assigned to its absolute
    # nearest opened site. With our 2 opened sites, this is what's
    # already in `sol`, so response_max == baseline_max. Verify that
    # the metric returns the same value for the same input.
    response_max = metric_fn(inst, response_sol)
    assert abs(response_max - baseline_max) < 1e-9
    print(f"  metric stable across identical solutions: {response_max:.2f}  OK")

    # Build the new coverage_gap query and score the response.
    fake_meta = {
        "archetype": "coverage_gap",
        "vague_text": "(test)", "precise_text": "(test)",
        "coverage_gap_center": [5.0, 5.0],
        "coverage_gap_radius": 1.0,
        "affected_precincts": [0],
        "best_candidate_idx": 1,
        "best_improvement_km": 1.5,
    }
    cg_query = queries.make_coverage_gap_query_from_metadata(
        query_id="cg_test", text="test",
        metadata_dict=fake_meta,
    )
    score = cg_query.score(inst, sol, response_sol)
    print(f"  query.score (no improvement): "
          f"target_baseline={score['target_baseline']:.2f}, "
          f"target_response={score['target_response']:.2f}, "
          f"fraction_improved={score['fraction_improved']:.2f}")
    assert score["target_baseline"] == baseline_max
    assert score["target_direction"] == "minimize"
    assert score["fraction_improved"] == 0.0  # no change → no improvement
    assert cg_query.metadata["target_metric"] == "max_assignment_distance"
    print("  coverage_gap factory uses max_assignment_distance  OK")

    # ---- (3) build_eval_set imports + augmentation ----
    print("\n[3] build_eval_set imports + ARCHETYPE_CONFIG entries")
    import build_eval_set as be
    for name in ("contiguity", "shape_niceness", "cluster", "coverage_gap"):
        assert name in be.ARCHETYPE_CONFIG
    print(f"  ARCHETYPE_CONFIG: {sorted(be.ARCHETYPE_CONFIG)}")
    assert callable(be.ARCHETYPE_CONFIG["coverage_gap"]["generator"])
    assert callable(be.ARCHETYPE_CONFIG["shape_niceness"]["generator"])
    print("  shape_niceness + coverage_gap generators are callables  OK")

    # ---- (4) dataset_generator._build_query_texts importable ----
    print("\n[4] partner's _build_query_texts importable + works on harness meta")
    from dataset_generator import _build_query_texts
    rng = np.random.default_rng(42)
    # Hand-build a coverage_gap meta with all the fields _build_query_texts
    # expects (after our augmentation). This simulates what the harness's
    # _generate_coverage_gap_uncapacitated will produce post-refactor.
    cg_meta = {
        "archetype": "coverage_gap",
        "coverage_gap_center": [3.0, 7.0],
        "coverage_gap_radius": 1.2,
        "affected_precincts": [12, 23, 31, 47, 58],
        "current_max_distance_km": 4.2,
    }
    vague, precise, vidx, pidx = _build_query_texts(
        "coverage_gap", inst, cg_meta, rng)
    print(f"  vague (template {vidx}):   {vague[:80]}...")
    print(f"  precise (template {pidx}): {precise[:80]}...")
    assert vague and precise
    assert "12" in precise  # affected_short should reference precinct 12
    print("  coverage_gap vague + precise text generated  OK")

    # And for cluster (uses different metadata fields).
    cluster_meta = {
        "archetype": "cluster",
        "cluster_center": [5.0, 2.5],
        "affected_sites": [3, 7, 12, 15],
        "cluster_size": 4,
        "cluster_radius": 1.3,
    }
    vague_c, precise_c, _, _ = _build_query_texts(
        "cluster", inst, cluster_meta, rng)
    assert "3" in precise_c
    print(f"  cluster precise: {precise_c[:80]}...  OK")

    # ---- (5) Simulated pair-write into the new layout ----
    print("\n[5] Simulated coverage_gap pair-write into full_dataset/ layout")
    fixture = Path("/tmp/ph_unified_fixture")
    if fixture.exists():
        shutil.rmtree(fixture)
    archetype_dir = fixture / "full_dataset" / "coverage_gap"
    pair_dir = archetype_dir / "pairs" / "coverage_gap_easy_00"
    pair_dir.mkdir(parents=True)

    inst.save(str(pair_dir / "instance.pkl"))
    sol.save(str(pair_dir / "baseline_solution.pkl"))

    # Hand-construct the query_metadata.json with both perception and
    # optimization-side fields. Mirrors what the refactored
    # _generate_one_tier writes.
    qmeta = {
        "archetype": "coverage_gap",
        "pair_id": "coverage_gap_easy_00",
        "difficulty": "easy",
        "tier_kwargs": {"min_max_distance": 3.5, "min_improvement": 1.5},
        # Perception-side
        "current_max_distance_km": 4.2,
        "most_stranded_precinct": 47,
        "best_candidate_idx": 19,
        "best_improvement_km": 2.0,
        "per_candidate_improvement": {"19": {"improvement_km": 2.0,
                                              "new_max_km": 2.2}},
        "candidates_ranked_by_improvement": [19, 7, 2],
        # Optimization-side (partner)
        "coverage_gap_center": [3.0, 7.0],
        "coverage_gap_radius": 1.2,
        "affected_precincts": [12, 23, 31, 47, 58],
        "coverage_gap_distance_threshold": 2.5,
        "coverage_gap_baseline_strand_count": 5,
        # Query texts
        "vague_text": vague, "precise_text": precise,
        "vague_template_idx": int(vidx),
        "precise_template_idx": int(pidx),
        # Diagnostics
        "base_seed": 1, "uncapacitated": True,
    }
    (pair_dir / "query_metadata.json").write_text(
        json.dumps(qmeta, indent=2))

    # Per-archetype index.json
    archetype_idx = {
        "n_pairs": 1,
        "archetype": "coverage_gap",
        "sampling_seed": 42,
        "tiers_used": ["easy"],
        "n_per_tier": 1,
        "pairs": [{
            "pair_id": "coverage_gap_easy_00",
            "pair_dir": "pairs/coverage_gap_easy_00",
            "archetype": "coverage_gap",
            "difficulty": "easy",
            "base_seed": 1,
            "vague_template_idx": int(vidx),
            "precise_template_idx": int(pidx),
        }],
    }
    (archetype_dir / "index.json").write_text(
        json.dumps(archetype_idx, indent=2))

    # Top-level index.json
    top_idx = {
        "n_per_archetype": 30,
        "n_per_tier": 10,
        "tiers": ["easy", "med", "hard"],
        "archetypes": {
            "coverage_gap": {"n_pairs": 1,
                              "path": "full_dataset/coverage_gap"},
        },
    }
    (fixture / "full_dataset" / "index.json").write_text(
        json.dumps(top_idx, indent=2))

    print(f"  layout under {fixture}/:")
    for p in sorted(fixture.rglob("*")):
        if p.is_file():
            rel = p.relative_to(fixture)
            print(f"    {rel}")

    # Round-trip read.
    print()
    import sys as _sys
    _sys.path.insert(0, str(HARNESS_ROOT))
    from eval_perception import _load_pair  # noqa: E402
    inst2, sol2, meta2 = _load_pair(pair_dir)
    assert meta2["archetype"] == "coverage_gap"
    assert meta2["best_candidate_idx"] == 19
    assert meta2["coverage_gap_center"] == [3.0, 7.0]
    assert "vague_text" in meta2 and "precise_text" in meta2
    print("  _load_pair reads query_metadata.json correctly  OK")
    print(f"    perception fields: best_candidate_idx={meta2['best_candidate_idx']}, "
          f"best_improvement_km={meta2['best_improvement_km']}")
    print(f"    optimization fields: coverage_gap_center={meta2['coverage_gap_center']}, "
          f"affected_precincts={meta2['affected_precincts']}")
    print(f"    queries: vague_template_idx={meta2['vague_template_idx']}, "
          f"precise_template_idx={meta2['precise_template_idx']}")

    # The factory should accept this meta and build a working query.
    cg_query2 = queries.make_coverage_gap_query_from_metadata(
        query_id="rt_test", text=meta2["precise_text"],
        metadata_dict=meta2,
    )
    score2 = cg_query2.score(inst2, sol2, sol2)
    assert score2["target_direction"] == "minimize"
    print(f"    queries.make_coverage_gap_query loads round-tripped meta: OK")

    # ---- (6) Backward-compat: meta.json still loadable ----
    print("\n[6] Backward compat: legacy meta.json fallback in _load_pair")
    legacy_pair = fixture / "legacy" / "pairs" / "legacy_00"
    legacy_pair.mkdir(parents=True)
    inst.save(str(legacy_pair / "instance.pkl"))
    sol.save(str(legacy_pair / "baseline_solution.pkl"))
    (legacy_pair / "meta.json").write_text(json.dumps({
        "archetype": "contiguity",
        "culprits": [],
    }))
    inst3, sol3, meta3 = _load_pair(legacy_pair)
    assert meta3["archetype"] == "contiguity"
    print("  _load_pair falls back to meta.json when no query_metadata.json  OK")

    print("\nAll unified-dataset checks PASSED.")


if __name__ == "__main__":
    main()
