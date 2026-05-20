"""Verification for the cluster perception wiring.

Tests:
  (a) tasks/cluster.py — format_question across regimes, parse, score
  (b) tasks/cluster.is_valid_view — gate semantics
  (c) eval_set ground-truth and global-clustered-sites helper
  (d) tool_oracle cluster identify + describe
  (e) runner skip-row behaviour for invalid (cluster, v2_no_markers) cell

The cluster check builds a synthetic instance + solution where five
opened sites sit packed inside a 1 km box and one site sits alone on
the far side. The five clustered sites are the expected ground truth;
the lone site is not.

Run:
    python dev/verify_cluster.py
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Synthetic cluster pair builder
# ---------------------------------------------------------------------------
def _build_cluster_pair():
    """6 opened sites: 5 packed inside a 1 km box (the cluster) + 1 alone.

    Geometry doesn't need to be realistic — only opened-site coordinates
    matter for cluster identify. We slap in a trivial precinct grid so
    Instance is well-formed.
    """
    G = 4
    grid = np.zeros((G, G), dtype=np.int32)
    # 4 precincts: top-left, top-right, bottom-left, bottom-right
    grid[:G // 2, :G // 2] = 0
    grid[:G // 2, G // 2:] = 1
    grid[G // 2:, :G // 2] = 2
    grid[G // 2:, G // 2:] = 3

    bounds = (0.0, 0.0, 10.0, 10.0)
    xs = np.linspace(bounds[0], bounds[2], G)
    ys = np.linspace(bounds[1], bounds[3], G)
    centroids = np.array([[2.5, 2.5], [7.5, 2.5], [2.5, 7.5], [7.5, 7.5]])

    cell_w = float(xs[1] - xs[0])
    cell_h = float(ys[1] - ys[0])
    areas = np.full(4, (G // 2) ** 2 * cell_w * cell_h)
    voters = np.array([1000] * 4, dtype=int)

    # Five clustered sites inside a 1 km box around (5, 5).
    # Plus one lone site at (1, 1).
    site_locations = np.array([
        [5.0, 5.0],   # site 0 — cluster centre
        [5.4, 5.0],   # site 1 — 0.4 km east
        [5.0, 5.4],   # site 2 — 0.4 km north
        [4.6, 5.0],   # site 3 — 0.4 km west
        [5.0, 4.6],   # site 4 — 0.4 km south
        [1.0, 1.0],   # site 5 — far away (lone)
    ])
    site_capacity = np.array([5000] * 6, dtype=int)
    site_types = ["school"] * 6
    distance_matrix = np.linalg.norm(
        centroids[:, None, :] - site_locations[None, :, :], axis=2)

    inst = Instance(
        bounds=bounds,
        precinct_label_grid=grid,
        grid_xs=xs, grid_ys=ys,
        precinct_centroids=centroids,
        precinct_areas=areas,
        precinct_voters=voters,
        site_locations=site_locations,
        site_capacity=site_capacity,
        site_types=site_types,
        distance_matrix=distance_matrix,
        K=6, seed=0,
        params={"synthetic": True},
    )
    # All 6 sites opened. Trivial assignment: all precincts -> site 0.
    x = np.ones(6, dtype=np.int8)
    y = np.zeros((4, 6), dtype=np.int8)
    y[:, 0] = 1
    sol = Solution(
        x=x, y=y,
        objective=float((voters[:, None] * distance_matrix * y).sum()),
        solver_status="synthetic",
        metadata={"feasible": True},
    )
    meta = {
        "archetype": "cluster",
        "cluster_center": [5.0, 5.0],
        "cluster_radius": 1.3,
        "cluster_min_sites": 4,
        "affected_sites": [0, 1, 2, 3, 4],
        "cluster_size": 5,
    }
    return inst, sol, meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Perception harness — cluster verification")
    print("=" * 60)

    print("\n[a] tasks/cluster — format_question / parse / score")
    import tasks.cluster as cl
    assert cl.CLUSTER_RADIUS == 1.3 and cl.CLUSTER_MIN_SITES == 4

    # Three view-info regimes that cluster supports:
    vi_visual_markers = {"has_visual": True, "has_site_markers": True}
    vi_no_visual = {"has_visual": False, "has_site_markers": False}
    vi_visual_no_markers = {"has_visual": True, "has_site_markers": False}

    qm = cl.format_question({}, "identify", view_info=vi_visual_markers)
    assert "red circles" in qm and "1.3 km" in qm
    print("  format_question(visual+markers): OK")

    qn = cl.format_question({}, "identify", view_info=vi_no_visual)
    assert "list_sites" in qn and "1.3 km" in qn
    print("  format_question(no_visual): OK")

    parsed = cl.parse_response(
        'Answer: {"clustered_sites": [0, 1, 2, 3, 4]}', "identify")
    assert parsed == {"clustered_sites": [0, 1, 2, 3, 4]}
    parsed = cl.parse_response(
        '{"clustered_sites": [0, 0, 1, "junk", 2]}', "identify")
    assert parsed == {"clustered_sites": [0, 1, 2]}  # dedup + skip junk
    print("  parse_response: dedup + junk-tolerance OK")

    truth = {"clustered_sites": [0, 1, 2, 3, 4]}
    assert cl.score({"clustered_sites": [0, 1, 2, 3, 4]},
                     truth, "identify") == 1.0
    assert abs(cl.score({"clustered_sites": [0, 1, 2]},
                          truth, "identify") - 6 / 8) < 1e-9
    assert cl.score({"clustered_sites": [99]},
                     truth, "identify") == 0.0
    print("  score(identify): full / partial / miss OK")

    print("\n[b] is_valid_view — gate semantics")
    assert cl.is_valid_view(vi_visual_markers) is True
    assert cl.is_valid_view(vi_no_visual) is True
    assert cl.is_valid_view(vi_visual_no_markers) is False
    assert cl.is_valid_view(None) is True  # default safe
    print("  visual+markers       -> valid   OK")
    print("  no_visual            -> valid   OK")
    print("  visual+no_markers    -> INVALID OK")

    print("\n[c] eval_set ground truth + global helper")
    import build_eval_set as be
    inst, sol, meta = _build_cluster_pair()

    sites = be._globally_clustered_sites(
        inst, sol, radius=1.3, min_sites=4)
    print(f"  _globally_clustered_sites(synthetic) -> {sites}")
    assert sites == [0, 1, 2, 3, 4], sites

    # Inject globally_clustered_sites into meta and check ground-truth fn.
    meta_full = dict(meta)
    meta_full["globally_clustered_sites"] = sites
    gt = be.ARCHETYPE_CONFIG["cluster"]["ground_truth_fn"](meta_full)
    assert gt["identify"]["clustered_sites"] == [0, 1, 2, 3, 4]
    assert "concepts" in gt["describe"]
    print(f"  ground_truth_fn -> {gt['identify']}")
    print("  cluster ground-truth uses globally_clustered_sites  OK")

    # Tier dict shape.
    tiers = be.ARCHETYPE_CONFIG["cluster"]["tiers"]
    assert set(tiers) == {"easy", "med", "hard"}
    for t, kw in tiers.items():
        assert "cluster_min_sites" in kw and "cluster_density_factor" in kw
    print(f"  tiers: {list(tiers)}")

    print("\n[d] tool_oracle cluster identify + describe")
    import models.tool_oracle as oracle
    raw = oracle.compute_answer(instance=inst, solution=sol,
                                  archetype="cluster", task="identify")
    parsed = json.loads(raw)
    assert parsed == {"clustered_sites": [0, 1, 2, 3, 4]}, parsed
    print(f"  oracle identify -> {parsed['clustered_sites']}  OK")

    raw = oracle.compute_answer(instance=inst, solution=sol,
                                  archetype="cluster", task="describe")
    s = cl.score({"text": raw}, gt["describe"], "describe")
    print(f"  oracle describe -> '{raw[:80]}...' (concept score {s:.2f})")
    assert s == 1.0, f"oracle describe should hit all concepts, got {s}"

    print("\n[e] runner skips invalid (cluster, v2_no_markers) cell")
    fixture = Path("/tmp/ph_cluster_fixture")
    if fixture.exists():
        shutil.rmtree(fixture)
    pair_dir = fixture / "pairs" / "cluster_synth_00"
    pair_dir.mkdir(parents=True)
    inst.save(str(pair_dir / "instance.pkl"))
    sol.save(str(pair_dir / "baseline_solution.pkl"))
    meta_full["pair_id"] = "cluster_synth_00"
    meta_full["difficulty"] = "easy"
    meta_full["archetype"] = "cluster"
    (pair_dir / "meta.json").write_text(json.dumps(meta_full, indent=2))
    gt_csv = fixture / "ground_truth.csv"
    gt_csv.write_text(
        "pair_id,archetype,difficulty,task,answer_json,source_dir\n"
        "cluster_synth_00,cluster,easy,identify,"
        '"{""clustered_sites"": [0, 1, 2, 3, 4]}",'
        f"{pair_dir}\n")

    out = Path("/tmp/ph_cluster_results")
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
    import csv
    with open(out / "per_question.csv") as f:
        rows = list(csv.DictReader(f))
    rows_by_renderer = {r["renderer"]: r for r in rows}

    # v2 (markers) should run and score 1.0.
    v2_row = rows_by_renderer["v2"]
    assert v2_row["error"] == "" and float(v2_row["score"]) == 1.0
    print(f"  v2 cell: ran, score={v2_row['score']}  OK")

    # v2_no_markers should be skipped with the validity error.
    blind_row = rows_by_renderer["v2_no_markers"]
    assert blind_row["error"] == "invalid_view_for_task"
    assert blind_row["score"] == ""
    print(f"  v2_no_markers cell: skipped with "
          f"error='{blind_row['error']}'  OK")

    print("\nAll cluster checks PASSED.")


if __name__ == "__main__":
    main()
