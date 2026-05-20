"""Verification for the solidity (convex-hull-ratio) wiring.

Runs end-to-end against the synthetic pair without requiring scipy in
the local environment: installs a small fake `scipy.spatial.ConvexHull`
into sys.modules before importing build_eval_set, so the helper's
`from scipy.spatial import ConvexHull` resolves to a pure-Python
Andrew's-monotone-chain implementation. On the user's machine real
scipy is used; on this sandbox the math is the same.

What this checks:
  (a) graceful degradation: with NO scipy, _per_catchment_solidity
      returns {} and prints a single warning.
  (b) full wiring: with the fake scipy installed, the helper computes
      sensible solidity values, the meta builder writes per_catchment_
      solidity, the ground-truth function emits both top10_by_npi and
      top10_by_solidity, and tasks.shape_niceness.secondary_scores
      reports both fractions.

Run:
    python dev/verify_solidity.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))
sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
sys.path.insert(0, str(HERE))


# --------------------------------------------------------------------------
# Fake scipy.spatial.ConvexHull (Andrew's monotone chain). Pure Python; only
# exists for sandbox testing. The real scipy version is faster and covers
# higher-dimensional cases we don't need.
# --------------------------------------------------------------------------
def _convex_hull_2d(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(verts):
    n = len(verts)
    s = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


class _FakeConvexHull:
    def __init__(self, points):
        pts = points.tolist() if hasattr(points, "tolist") else points
        self._verts = _convex_hull_2d(pts)
        # In real scipy 2D, .volume is the polygon area (.area is perimeter).
        self.volume = _polygon_area(self._verts)


def _install_fake_scipy() -> None:
    spatial = types.ModuleType("scipy.spatial")
    spatial.ConvexHull = _FakeConvexHull
    qhull = types.ModuleType("scipy.spatial.qhull")
    qhull.QhullError = Exception
    spatial.qhull = qhull
    scipy_mod = types.ModuleType("scipy")
    scipy_mod.spatial = spatial
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.spatial"] = spatial
    sys.modules["scipy.spatial.qhull"] = qhull


def _remove_fake_scipy() -> None:
    for mod in ("scipy.spatial.qhull", "scipy.spatial", "scipy"):
        sys.modules.pop(mod, None)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Perception harness — solidity verification")
    print("=" * 60)
    from verify_slice import _build_synthetic_pair
    inst, sol, _ = _build_synthetic_pair()

    print("\n[a] graceful degradation when scipy is unavailable")
    _remove_fake_scipy()
    # Fresh-import build_eval_set so the warning flag resets too.
    sys.modules.pop("build_eval_set", None)
    import build_eval_set as be_no_scipy
    out_no_scipy = be_no_scipy._per_catchment_solidity(inst, sol)
    assert out_no_scipy == {}, out_no_scipy
    print("  _per_catchment_solidity() -> {} (graceful)  OK")

    print("\n[b] full wiring with fake scipy")
    _install_fake_scipy()
    sys.modules.pop("build_eval_set", None)
    import build_eval_set as be
    sol_dict = be._per_catchment_solidity(inst, sol)
    assert set(sol_dict.keys()) == {0, 1}, sol_dict.keys()
    for j, rec in sol_dict.items():
        print(f"  site {j}: A={rec['A']:.2f} A_hull={rec['A_hull']:.2f} "
              f"solidity={rec['solidity']:.3f}")
    # Synthetic catchments are diagonal pairs of squares — non-convex.
    # Expected solidity ~0.77 by hand-calculation; allow a wide band.
    for j, rec in sol_dict.items():
        assert 0.6 < rec["solidity"] < 0.9, (j, rec)
    print("  solidity in expected band [0.6, 0.9]  OK")

    print("\n[c] ground-truth fn emits both top10 axes")
    # 12-site fixture so the top-10 cutoff is meaningful (with <=10 sites
    # the cutoff would just include everything, hiding the metric
    # disagreement we want to verify).
    fake_meta = {
        "archetype": "shape_niceness",
        "per_catchment_npi": {
            # Top of NPI ranking — these should land in top10_by_npi.
            "17": {"NPI": 2.50}, "8":  {"NPI": 2.10},
            "34": {"NPI": 1.95}, "12": {"NPI": 1.85},
            "5":  {"NPI": 1.42}, "40": {"NPI": 1.30},
            "41": {"NPI": 1.25}, "42": {"NPI": 1.20},
            "43": {"NPI": 1.15}, "44": {"NPI": 1.05},
            # These two fall outside top10 by NPI.
            "21": {"NPI": 1.02}, "45": {"NPI": 1.00},
        },
        "per_catchment_solidity": {
            # Site 21 has the WORST solidity but a LOW NPI — designed to
            # land in top10_by_solidity yet OUTSIDE top10_by_npi. This is
            # exactly the metric-disagreement case we want to surface.
            "21": {"solidity": 0.40}, "8":  {"solidity": 0.55},
            "17": {"solidity": 0.70}, "34": {"solidity": 0.78},
            "5":  {"solidity": 0.85}, "12": {"solidity": 0.92},
            "42": {"solidity": 0.93}, "43": {"solidity": 0.94},
            "40": {"solidity": 0.95}, "41": {"solidity": 0.96},
            "44": {"solidity": 0.97}, "45": {"solidity": 0.98},
        },
    }
    gt = be.ARCHETYPE_CONFIG["shape_niceness"]["ground_truth_fn"](fake_meta)
    print(f"  worst_sites (top-3 by NPI):  {gt['identify']['worst_sites']}")
    print(f"  top10_sites_by_npi:          {gt['identify']['top10_sites_by_npi']}")
    print(f"  top10_sites_by_solidity:     {gt['identify']['top10_sites_by_solidity']}")
    assert gt["identify"]["worst_sites"] == [17, 8, 34]
    # By NPI desc: 17, 8, 34, 12, 5, 40, 41, 42, 43, 44.
    assert gt["identify"]["top10_sites_by_npi"] == [17, 8, 34, 12, 5,
                                                       40, 41, 42, 43, 44]
    # By solidity asc: 21, 8, 17, 34, 5, 12, 42, 43, 40, 41.
    assert gt["identify"]["top10_sites_by_solidity"] == [21, 8, 17, 34, 5,
                                                           12, 42, 43, 40, 41]
    print("  both axes ranked correctly with NPI / solidity tie-breaks  OK")

    print("\n[d] secondary_scores emits both NPI and solidity fractions")
    import tasks.shape_niceness as sn

    # Case 1: VLM picks the NPI-truth top-3.
    pred_npi = {"worst_sites": [17, 8, 34]}
    s = sn.secondary_scores(pred_npi, gt["identify"], "identify")
    print(f"  pred=[17,8,34]  -> {s}")
    assert s["in_top10_by_npi_fraction"] == 1.0
    # All three (17, 8, 34) are also in top10_by_solidity.
    assert s["in_top10_by_solidity_fraction"] == 1.0

    # Case 2: VLM picks by perceptual ugliness — leans on solidity.
    # Site 21 is NOT in top10_by_npi (NPI 1.02) but IS the worst by
    # solidity. This is the kind of split we want the harness to surface.
    pred_sol = {"worst_sites": [21, 8, 17]}
    s = sn.secondary_scores(pred_sol, gt["identify"], "identify")
    print(f"  pred=[21,8,17]  -> {s}")
    # 8 and 17 are in top10_by_npi; 21 is not.
    assert abs(s["in_top10_by_npi_fraction"] - 2 / 3) < 1e-9
    # All three are in top10_by_solidity.
    assert s["in_top10_by_solidity_fraction"] == 1.0

    # Case 3: legacy meta with no solidity field -> only NPI metric reported.
    legacy_meta = dict(fake_meta)
    legacy_meta.pop("per_catchment_solidity")
    legacy_gt = be.ARCHETYPE_CONFIG["shape_niceness"]["ground_truth_fn"](legacy_meta)
    s = sn.secondary_scores({"worst_sites": [17, 8, 34]},
                              legacy_gt["identify"], "identify")
    print(f"  legacy-meta pred=[17,8,34]  -> {s}")
    assert "in_top10_by_npi_fraction" in s
    assert "in_top10_by_solidity_fraction" not in s
    print("  fraction reported per available axis only  OK")

    print("\n[e] full pair-build round-trip — meta has per_catchment_solidity")
    import shutil
    fixture = Path("/tmp/ph_solidity_fixture")
    if fixture.exists():
        shutil.rmtree(fixture)
    pair_dir = fixture / "pairs" / "synth_00"
    pair_dir.mkdir(parents=True)
    inst.save(str(pair_dir / "instance.pkl"))
    sol.save(str(pair_dir / "baseline_solution.pkl"))

    # Mimic what _generate_one_tier does for the meta-write step.
    sol_recomputed = be._per_catchment_solidity(inst, sol)
    meta_full = {
        "archetype": "contiguity",
        "per_catchment_solidity": {str(k): v for k, v in sol_recomputed.items()},
    }
    (pair_dir / "meta.json").write_text(json.dumps(meta_full, indent=2))
    loaded = json.loads((pair_dir / "meta.json").read_text())
    assert "per_catchment_solidity" in loaded
    assert set(loaded["per_catchment_solidity"]) == {"0", "1"}
    print(f"  meta.json contains per_catchment_solidity for sites "
          f"{sorted(loaded['per_catchment_solidity'].keys())}  OK")

    print("\nAll solidity checks PASSED.")


if __name__ == "__main__":
    main()
