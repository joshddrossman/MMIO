"""End-to-end verification of the coverage_gap wiring.

Builds a synthetic instance with:
  - 4 precincts in a 2×2 layout (BL, BR, UL, UR)
  - 2 opened sites in the BL and BR (south side), so UR is far from any
    opened site
  - 1 closed candidate in the UR (the obvious best-fix candidate)
  - 1 closed candidate in the centre (mediocre fix)
  - 1 closed candidate in the BL again (useless fix — UR still stranded)

The analytic best closed candidate is the UR one, by construction. The
oracle, ground truth function, and tasks/coverage_gap all agree on it.

Also checks: is_valid_view rejects v2_no_markers; the runner skip-row
behaves correctly for a (coverage_gap, v2_no_markers) cell.

Run:
    python dev/verify_coverage_gap.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
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


def _build_coverage_gap_pair():
    """Construct a synthetic coverage_gap pair with an unambiguous best fix.

    Layout: 6×6 grid, 3 precincts arranged so only ONE is stranded:
      precinct 0 — left half               centroid (2, 5)
      precinct 1 — bottom-right quadrant   centroid (8, 2)
      precinct 2 — top-right quadrant      centroid (8, 8)  ← stranded

    Opened sites: site 0 at (2, 5) covering precinct 0; site 1 at
    (8, 2) covering precinct 1. Precinct 2 is then 6 km from its
    nearest opened site (site 1).

    Closed candidates:
      site 2 at (8, 8)  — co-located with precinct 2 → BEST (Δ ≈ 6 km)
      site 3 at (5, 5)  — centre → MODERATE (Δ ≈ 1.76 km)
      site 4 at (2, 5)  — co-located with site 0 → USELESS (Δ = 0)
    """
    G = 6
    grid = np.zeros((G, G), dtype=np.int32)
    # Left half (cols 0..2 across all rows) → precinct 0
    grid[:, :3] = 0
    # Bottom-right quadrant → precinct 1
    grid[:3, 3:] = 1
    # Top-right quadrant → precinct 2 (the stranded precinct)
    grid[3:, 3:] = 2

    bounds = (0.0, 0.0, 10.0, 10.0)
    xs = np.linspace(bounds[0], bounds[2], G)
    ys = np.linspace(bounds[1], bounds[3], G)
    centroids = np.array([
        [2.0, 5.0],   # precinct 0 — left half
        [8.0, 2.0],   # precinct 1 — bottom-right
        [8.0, 8.0],   # precinct 2 — top-right (stranded)
    ])
    cell_w = float(xs[1] - xs[0]); cell_h = float(ys[1] - ys[0])
    areas = np.array([
        18 * cell_w * cell_h,   # precinct 0 = 18 cells (3×6)
        9 * cell_w * cell_h,    # precinct 1 = 9 cells (3×3)
        9 * cell_w * cell_h,    # precinct 2 = 9 cells (3×3)
    ])
    voters = np.full(3, 1000, dtype=int)

    site_locations = np.array([
        [2.0, 5.0],   # site 0 — opened, at precinct 0
        [8.0, 2.0],   # site 1 — opened, at precinct 1
        [8.0, 8.0],   # site 2 — closed, at precinct 2 (BEST FIX)
        [5.0, 5.0],   # site 3 — closed, centre (MODERATE)
        [2.0, 5.0],   # site 4 — closed, at site 0 (USELESS)
    ])
    site_capacity = np.full(5, 50_000, dtype=int)  # uncapacitated
    site_types = ["school"] * 5
    distance_matrix = np.linalg.norm(
        centroids[:, None, :] - site_locations[None, :, :], axis=2)

    inst = Instance(
        bounds=bounds, precinct_label_grid=grid,
        grid_xs=xs, grid_ys=ys,
        precinct_centroids=centroids, precinct_areas=areas,
        precinct_voters=voters,
        site_locations=site_locations,
        site_capacity=site_capacity,
        site_types=site_types,
        distance_matrix=distance_matrix,
        K=5, seed=0, params={"synthetic": True},
    )
    # Sites 0, 1 opened. Each precinct → its nearest opened site:
    #   precinct 0 (2, 5) → site 0   (distance 0)
    #   precinct 1 (8, 2) → site 1   (distance 0)
    #   precinct 2 (8, 8) → site 1   (distance 6 — STRANDED)
    x = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    y = np.zeros((3, 5), dtype=np.int8)
    y[0, 0] = 1
    y[1, 1] = 1
    y[2, 1] = 1
    sol = Solution(
        x=x, y=y,
        objective=float((voters[:, None] * distance_matrix * y).sum()),
        solver_status="synthetic",
        metadata={"feasible": True},
    )
    return inst, sol


def main():
    print("=" * 60)
    print("Perception harness — coverage_gap verification")
    print("=" * 60)

    print("\n[1] _rank_closed_candidates_by_improvement on synthetic pair")
    import build_eval_set as be
    inst, sol = _build_coverage_gap_pair()
    ranking = be._rank_closed_candidates_by_improvement(inst, sol)
    print(f"  ranking (best first): {ranking}")
    # Expected: site 2 (at precinct 2) gives Δ=6 (drops max from 6 to 0);
    # site 3 (centre) gives Δ≈1.76 (drops max to 4.24); site 4 useless.
    best_idx, best_new_max, best_imp = ranking[0]
    print(f"  best candidate: site {best_idx}, new_max={best_new_max:.2f}, "
           f"improvement={best_imp:.2f}km")
    assert best_idx == 2, f"expected best=site 2, got {best_idx}"
    assert abs(best_imp - 6.0) < 0.01, f"expected best Δ=6.0, got {best_imp}"
    assert ranking[1][0] == 3, f"expected 2nd-best=site 3, got {ranking[1][0]}"
    assert abs(ranking[1][2] - 1.76) < 0.05
    assert ranking[2][0] == 4 and ranking[2][2] == 0.0
    print(f"  ranking matches expected (site 2 > site 3 > site 4)  OK")

    print("\n[2] _generate_coverage_gap_uncapacitated against the synthetic")
    # The wrapper would generate a fresh instance via gurobi — we can't
    # test that path in the sandbox. Instead, exercise the helpers and
    # ground-truth function with hand-built meta.
    fake_meta = {
        "archetype": "coverage_gap",
        "current_max_distance_km": 5.0,
        "most_stranded_precinct": 3,
        "most_stranded_distance_km": 5.0,
        "best_candidate_idx": 2,
        "best_candidate_new_max_km": 5.0,
        "best_improvement_km": 0.0,
        "per_candidate_improvement": {
            "2": {"improvement_km": 0.0, "new_max_km": 5.0},
            "3": {"improvement_km": 0.0, "new_max_km": 5.0},
            "4": {"improvement_km": 0.0, "new_max_km": 5.0},
        },
        "candidates_ranked_by_improvement": [2, 3, 4],
    }
    # That degenerate case hits the "best_imp <= 0" guard. Let me build a
    # more realistic meta.
    real_meta = {
        "archetype": "coverage_gap",
        "current_max_distance_km": 6.0,
        "most_stranded_precinct": 2,
        "most_stranded_distance_km": 6.0,
        "best_candidate_idx": 2,
        "best_candidate_new_max_km": 0.0,
        "best_improvement_km": 6.0,
        "per_candidate_improvement": {
            "2": {"improvement_km": 6.0, "new_max_km": 0.0},
            "3": {"improvement_km": 1.76, "new_max_km": 4.24},
            "4": {"improvement_km": 0.0, "new_max_km": 6.0},
        },
        "candidates_ranked_by_improvement": [2, 3, 4],
    }
    gt = be._coverage_gap_ground_truth(real_meta)
    print(f"  ground_truth identify keys: {sorted(gt['identify'].keys())}")
    assert gt["identify"]["best_candidate"] == 2
    assert gt["identify"]["top3_candidates"] == [2, 3, 4]
    assert "concepts" in gt["describe"]
    print("  ground_truth populated with best_candidate, top3, concepts  OK")

    print("\n[3] tasks/coverage_gap parse + score")
    import tasks.coverage_gap as cg

    # parse
    raw = '{"best_candidate": 2, "reason": "site at UR best serves stranded precinct"}'
    parsed = cg.parse_response(raw, "identify")
    print(f"  parsed: {parsed}")
    assert parsed["best_candidate"] == 2
    assert parsed["reasons"] == {2: "site at UR best serves stranded precinct"}

    # parse — bare integer (legacy)
    parsed_bare = cg.parse_response('{"best_candidate": 2}', "identify")
    assert parsed_bare["best_candidate"] == 2 and parsed_bare["reasons"] == {}
    print(f"  bare-format also parses  OK")

    # parse — list of length 1
    parsed_list = cg.parse_response('{"best_candidate": [2], "reason": "x"}',
                                      "identify")
    assert parsed_list["best_candidate"] == 2
    print(f"  one-element-list format also parses  OK")

    # parse — invalid index
    parsed_bad = cg.parse_response('{"best_candidate": "abc"}', "identify")
    assert parsed_bad["best_candidate"] is None
    print(f"  invalid index returns None  OK")

    # score
    s_best = cg.score({"best_candidate": 2}, gt["identify"], "identify")
    s_mid = cg.score({"best_candidate": 3}, gt["identify"], "identify")
    s_bad = cg.score({"best_candidate": 4}, gt["identify"], "identify")
    s_invalid = cg.score({"best_candidate": 99}, gt["identify"], "identify")
    s_none = cg.score({"best_candidate": None}, gt["identify"], "identify")
    print(f"  score: site 2 (best)     = {s_best:.3f}  (expect 1.0)")
    print(f"  score: site 3 (moderate) = {s_mid:.3f}  (expect 1.76/6.0 ≈ 0.293)")
    print(f"  score: site 4 (no help)  = {s_bad:.3f}  (expect 0.0)")
    print(f"  score: site 99 (invalid) = {s_invalid:.3f}  (expect 0.0)")
    print(f"  score: None              = {s_none:.3f}  (expect 0.0)")
    assert s_best == 1.0
    assert abs(s_mid - 1.76 / 6.0) < 1e-9
    assert s_bad == 0.0
    assert s_invalid == 0.0
    assert s_none == 0.0

    # secondary scores
    sec_best = cg.secondary_scores({"best_candidate": 2}, gt["identify"],
                                     "identify")
    sec_mid = cg.secondary_scores({"best_candidate": 3}, gt["identify"],
                                    "identify")
    sec_bad = cg.secondary_scores({"best_candidate": 4}, gt["identify"],
                                    "identify")
    print(f"\n  secondary (site 2): {sec_best}")
    print(f"  secondary (site 3): {sec_mid}")
    print(f"  secondary (site 4): {sec_bad}")
    assert sec_best["exact_match"] == 1.0 and sec_best["rank"] == 1.0
    assert sec_mid["exact_match"] == 0.0 and sec_mid["rank"] == 2.0
    assert sec_bad["exact_match"] == 0.0 and sec_bad["rank"] == 3.0
    assert sec_best["in_top3_by_improvement"] == 1.0
    assert sec_mid["in_top3_by_improvement"] == 1.0
    assert sec_bad["in_top3_by_improvement"] == 1.0  # still in top 3 by rank

    print("\n[4] is_valid_view gate")
    assert cg.is_valid_view({"has_visual": True, "has_candidate_markers": True}) is True
    assert cg.is_valid_view({"has_visual": True, "has_candidate_markers": False}) is False
    assert cg.is_valid_view({"has_visual": False, "has_candidate_markers": False}) is True
    assert cg.is_valid_view(None) is True
    print("  visual+candidate markers   -> valid   OK")
    print("  visual+no candidate markers -> INVALID OK")
    print("  no_visual                  -> valid   OK")

    print("\n[5] tool_oracle coverage_gap branches")
    import models.tool_oracle as oracle
    raw_id = oracle.compute_answer(instance=inst, solution=sol,
                                    archetype="coverage_gap", task="identify")
    print(f"  oracle identify raw: {raw_id[:100]}...")
    oracle_parsed = json.loads(raw_id)
    assert oracle_parsed["best_candidate"] == 2
    print(f"  oracle picked site 2  OK")

    raw_de = oracle.compute_answer(instance=inst, solution=sol,
                                    archetype="coverage_gap", task="describe")
    s = cg.score({"text": raw_de}, gt["describe"], "describe")
    print(f"  oracle describe -> '{raw_de[:80]}...' (concept score {s:.2f})")
    assert s == 1.0, f"oracle describe should hit all 7 concepts, got {s}"

    print("\n[6] eval_set ARCHETYPE_CONFIG entry")
    cfg = be.ARCHETYPE_CONFIG["coverage_gap"]
    assert callable(cfg["generator"])
    assert cfg["generator"] is be._generate_coverage_gap_uncapacitated
    assert cfg["tasks"] == ["identify", "describe"]
    assert set(cfg["tiers"]) == {"easy", "med", "hard"}
    for t, kw in cfg["tiers"].items():
        assert "min_max_distance" in kw and "min_improvement" in kw
    print(f"  coverage_gap entry wired correctly  OK")
    print(f"  tiers: {cfg['tiers']}")

    print("\n[7] Renderer flags + view_info gating in runner")
    import renderers.v2 as r_v2
    import renderers.v2_no_markers as r_blind
    import renderers.v2_legend as r_lg
    import renderers.v2_patch_labels as r_pl
    print(f"  v2.HAS_CANDIDATE_MARKERS:           {r_v2.HAS_CANDIDATE_MARKERS}")
    print(f"  v2_no_markers.HAS_CANDIDATE_MARKERS: {r_blind.HAS_CANDIDATE_MARKERS}")
    print(f"  v2_legend.HAS_CANDIDATE_MARKERS:    {r_lg.HAS_CANDIDATE_MARKERS}")
    print(f"  v2_patch_labels.HAS_CANDIDATE_MARKERS: {r_pl.HAS_CANDIDATE_MARKERS}")
    assert r_v2.HAS_CANDIDATE_MARKERS is True
    assert r_blind.HAS_CANDIDATE_MARKERS is False
    assert r_lg.HAS_CANDIDATE_MARKERS is False
    assert r_pl.HAS_CANDIDATE_MARKERS is False

    print("\n[8] Runner skip-row behaviour for invalid (coverage_gap, v2_no_markers)")
    fixture = Path("/tmp/ph_coverage_gap_fixture")
    if fixture.exists():
        shutil.rmtree(fixture)
    pair_dir = fixture / "pairs" / "coverage_gap_synth_00"
    pair_dir.mkdir(parents=True)
    inst.save(str(pair_dir / "instance.pkl"))
    sol.save(str(pair_dir / "baseline_solution.pkl"))
    real_meta_full = dict(real_meta)
    real_meta_full["pair_id"] = "coverage_gap_synth_00"
    real_meta_full["difficulty"] = "easy"
    (pair_dir / "meta.json").write_text(json.dumps(real_meta_full, indent=2))
    # Use csv.writer for proper escaping of the nested-dict answer_json.
    gt_csv = fixture / "ground_truth.csv"
    with open(gt_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "archetype", "difficulty",
            "task", "answer_json", "source_dir",
        ])
        writer.writeheader()
        writer.writerow({
            "pair_id": "coverage_gap_synth_00",
            "archetype": "coverage_gap",
            "difficulty": "easy",
            "task": "identify",
            "answer_json": json.dumps(gt["identify"]),
            "source_dir": str(pair_dir),
        })

    out = Path("/tmp/ph_coverage_gap_results")
    if out.exists():
        shutil.rmtree(out)
    proc = subprocess.run(
        ["python3", "eval_perception.py",
         "--ground_truth", str(gt_csv),
         "--renderers", "v2", "v2_no_markers",
         "--prompts", "with_attribution",
         "--models", "tool_oracle",
         "--out_dir", str(out)],
        capture_output=True, text=True, cwd=str(HARNESS_ROOT),
    )
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        raise AssertionError(f"runner exited {proc.returncode}")
    with open(out / "per_question.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"  runner produced {len(rows)} row(s):")
    for row in rows:
        print(f"    renderer={row['renderer']:<14} "
              f"score={row['score']!r:<8} "
              f"error={row['error']!r}")
    rows_by_renderer = {r["renderer"]: r for r in rows}
    assert "v2" in rows_by_renderer, (
        f"missing v2 row; got renderers: {list(rows_by_renderer)}\n"
        f"runner stdout:\n{proc.stdout}")
    assert "v2_no_markers" in rows_by_renderer
    v2_row = rows_by_renderer["v2"]
    blind_row = rows_by_renderer["v2_no_markers"]
    assert v2_row["error"] == "", f"v2 row error: {v2_row['error']!r}"
    assert float(v2_row["score"]) == 1.0, (
        f"v2 score: {v2_row['score']!r}")
    print(f"  v2 cell: ran, score={v2_row['score']}  OK")
    assert blind_row["error"] == "invalid_view_for_task"
    assert blind_row["score"] == ""
    print(f"  v2_no_markers cell: skipped with "
          f"error='{blind_row['error']}'  OK")

    print("\nAll coverage_gap checks PASSED.")


if __name__ == "__main__":
    main()
