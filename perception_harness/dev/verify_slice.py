"""End-to-end smoke test for the perception harness.

Builds a minimal synthetic (Instance, Solution) pair WITHOUT calling the
MILP, so it runs in environments without gurobipy. The synthetic pair is
constructed to have a known non-contiguous catchment, then we run the
oracle through the runner and assert that:

  - the eval_set fixture is found,
  - the renderer produces non-empty PNG bytes,
  - the oracle's identify response F1 is exactly 1.0,
  - the per_question.csv and aggregate.json files are written.

Run:
    python dev/verify_slice.py
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))

from instance import Instance, Solution  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic pair builder
# ---------------------------------------------------------------------------
def _build_synthetic_pair() -> tuple:
    """Return (instance, solution, meta) with an engineered non-contiguous
    catchment.

    Layout: 4 precincts arranged in a 2×2 grid on a 20×20 raster.
        precinct 0 = upper-left quadrant
        precinct 1 = upper-right quadrant
        precinct 2 = lower-left quadrant
        precinct 3 = lower-right quadrant

    Two opened sites:
        site 0 — assigned precincts {0, 3} (UL + LR, diagonal → split)
        site 1 — assigned precincts {1, 2} (UR + LL, diagonal → split)

    Both catchments are non-contiguous on the 4-connected adjacency graph
    (UL/LR and UR/LL share a corner, not an edge). The oracle should
    report split_sites = [0, 1].
    """
    G = 20
    half = G // 2
    grid = np.zeros((G, G), dtype=np.int32)
    grid[:half, :half] = 0          # UL
    grid[:half, half:] = 1          # UR
    grid[half:, :half] = 2          # LL
    grid[half:, half:] = 3          # LR

    bounds = (0.0, 0.0, 10.0, 10.0)
    xs = np.linspace(bounds[0], bounds[2], G)
    ys = np.linspace(bounds[1], bounds[3], G)

    # Precinct centroids — placed at quadrant centers.
    centroids = np.array([
        [2.5, 7.5],   # UL
        [7.5, 7.5],   # UR
        [2.5, 2.5],   # LL
        [7.5, 2.5],   # LR
    ])
    cell_area = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    areas = np.full(4, (G // 2) ** 2 * cell_area)
    voters = np.array([1000, 1000, 1000, 1000], dtype=int)

    # Two candidate sites — both opened. Locations don't matter much for
    # the topology check; place them centrally.
    site_locations = np.array([[5.0, 5.5], [5.0, 4.5]])
    site_capacity = np.array([5000, 5000], dtype=int)
    site_types = ["school", "school"]

    distance_matrix = np.linalg.norm(
        centroids[:, None, :] - site_locations[None, :, :], axis=2)

    inst = Instance(
        bounds=bounds,
        precinct_label_grid=grid,
        grid_xs=xs,
        grid_ys=ys,
        precinct_centroids=centroids,
        precinct_areas=areas,
        precinct_voters=voters,
        site_locations=site_locations,
        site_capacity=site_capacity,
        site_types=site_types,
        distance_matrix=distance_matrix,
        K=2,
        seed=0,
        params={"synthetic": True},
    )

    # Solution: x = [1, 1] (both opened). y assigns:
    #   precinct 0 -> site 0
    #   precinct 1 -> site 1
    #   precinct 2 -> site 1
    #   precinct 3 -> site 0
    x = np.array([1, 1], dtype=np.int8)
    y = np.zeros((4, 2), dtype=np.int8)
    y[0, 0] = 1; y[3, 0] = 1
    y[1, 1] = 1; y[2, 1] = 1
    objective = float((voters[:, None] * distance_matrix * y).sum())
    sol = Solution(
        x=x, y=y, objective=objective,
        solver_status="synthetic",
        metadata={"feasible": True},
    )

    # Engineered metadata mimicking what the contiguity generator would
    # produce. Both sites are culprits.
    meta = {
        "archetype": "contiguity",
        "pair_id": "contiguity_synth_00",
        "difficulty": "easy",
        "culprits": [
            {"site": 0, "n_components": 2, "components": [[0], [3]],
             "component_voters": [1000, 1000],
             "largest_component_voters": 1000,
             "disjoint_excess_voters": 1000,
             "worst_pair_separation_km": 7.07},
            {"site": 1, "n_components": 2, "components": [[1], [2]],
             "component_voters": [1000, 1000],
             "largest_component_voters": 1000,
             "disjoint_excess_voters": 1000,
             "worst_pair_separation_km": 7.07},
        ],
        "n_split_sites_baseline": 2,
        "total_disjoint_excess_voters_baseline": 2000,
        "worst_culprit_site": 0,
        "synthetic": True,
    }
    return inst, sol, meta


# ---------------------------------------------------------------------------
# Fixture write
# ---------------------------------------------------------------------------
def _write_fixture() -> Path:
    """Write the synthetic pair + ground_truth.csv to a temp eval_set dir."""
    fixture_root = HERE / "_verify_fixture"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    pair_dir = fixture_root / "pairs" / "contiguity_synth_00"
    pair_dir.mkdir(parents=True, exist_ok=True)

    inst, sol, meta = _build_synthetic_pair()
    inst.save(str(pair_dir / "instance.pkl"))
    sol.save(str(pair_dir / "baseline_solution.pkl"))
    with open(pair_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    gt_csv = fixture_root / "ground_truth.csv"
    rel_dir = pair_dir.relative_to(HARNESS_ROOT)
    with open(gt_csv, "w") as f:
        f.write("pair_id,archetype,difficulty,task,answer_json,source_dir\n")
        f.write(
            'contiguity_synth_00,contiguity,easy,identify,'
            '"{""split_sites"": [0, 1]}",'
            f'{rel_dir}\n')
        f.write(
            'contiguity_synth_00,contiguity,easy,describe,'
            '"{""concepts"": [""disjoint"", ""split"", ""disconnected"", '
            '""non-contiguous"", ""separated""]}",'
            f'{rel_dir}\n')
    return gt_csv


# ---------------------------------------------------------------------------
# Per-component checks
# ---------------------------------------------------------------------------
def _check_renderer():
    print("\n[1/4] renderer.base.render() ...")
    sys.path.insert(0, str(HARNESS_ROOT))
    import renderers.base as renderer
    inst, sol, _ = _build_synthetic_pair()
    png = renderer.render(inst, sol)
    assert isinstance(png, (bytes, bytearray)), "render() must return bytes"
    assert len(png) > 200, "PNG payload looks empty"
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG payload"
    print(f"  OK  ({len(png)} bytes, valid PNG header)")


def _check_oracle():
    print("\n[2/4] models.tool_oracle.compute_answer() ...")
    sys.path.insert(0, str(HARNESS_ROOT))
    import models.tool_oracle as oracle
    import tasks.contiguity as task
    inst, sol, _ = _build_synthetic_pair()
    raw = oracle.compute_answer(
        instance=inst, solution=sol,
        archetype="contiguity", task="identify")
    parsed = task.parse_response(raw, "identify")
    assert parsed == {"split_sites": [0, 1]}, \
        f"expected [0,1], got {parsed}"
    s = task.score(parsed, {"split_sites": [0, 1]}, "identify")
    assert s == 1.0, f"oracle F1 must be 1.0, got {s}"
    print(f"  OK  (raw={raw!r}, score={s})")


def _check_runner(gt_csv: Path) -> Path:
    print("\n[3/4] eval_perception.py --models tool_oracle ...")
    out_dir = HERE / "_verify_results"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable, str(HARNESS_ROOT / "eval_perception.py"),
        "--ground_truth", str(gt_csv),
        "--models", "tool_oracle",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(HARNESS_ROOT))
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        raise AssertionError(f"runner exited {proc.returncode}")
    print("  OK  (runner exited 0)")
    return out_dir


def _check_outputs(out_dir: Path):
    print("\n[4/4] outputs ...")
    pq = out_dir / "per_question.csv"
    agg = out_dir / "aggregate.json"
    assert pq.exists(), f"missing {pq}"
    assert agg.exists(), f"missing {agg}"
    rows = pq.read_text().strip().splitlines()
    assert len(rows) >= 3, f"expected header+2 rows, got {len(rows)}"
    summary = json.loads(agg.read_text())
    cells = summary["cells"]
    identify_cells = [c for c in cells if c["task"] == "identify"]
    assert identify_cells and identify_cells[0]["mean_score"] == 1.0, \
        f"oracle identify cell mean_score must be 1.0, got {cells}"
    describe_cells = [c for c in cells if c["task"] == "describe"]
    assert describe_cells and describe_cells[0]["mean_score"] == 1.0, \
        f"oracle describe should hit all concepts, got {describe_cells}"
    print(f"  OK  (per_question rows={len(rows) - 1}, "
          f"identify_mean={identify_cells[0]['mean_score']}, "
          f"describe_mean={describe_cells[0]['mean_score']})")


def main():
    print("=" * 60)
    print("Perception harness — vertical-slice smoke test")
    print("=" * 60)
    gt_csv = _write_fixture()
    print(f"  fixture written: {gt_csv}")
    _check_renderer()
    _check_oracle()
    out_dir = _check_runner(gt_csv)
    _check_outputs(out_dir)
    print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
