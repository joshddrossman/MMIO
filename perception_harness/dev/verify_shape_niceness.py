"""Verification for the shape_niceness perception pieces.

Smoke-tests parse/score on the task module, the oracle's compute_answer
branches, and the eval-set generator's ARCHETYPE_CONFIG entry. Run
without API keys and without gurobipy; uses the synthetic pair from
verify_slice for the oracle path.

Run:
    python dev/verify_shape_niceness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_ROOT = HERE.parent
PROJECT_ROOT = HARNESS_ROOT.parent
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "instance_generator"))
sys.path.insert(0, str(HARNESS_ROOT / "eval_set"))
sys.path.insert(0, str(HERE))


def main() -> None:
    print("=" * 60)
    print("Perception harness — shape_niceness verification")
    print("=" * 60)

    # -- (1) tasks/shape_niceness ----------------------------------------
    print("\n[1] tasks/shape_niceness — format_question / parse / score")
    import tasks.shape_niceness as sn
    assert sn.WORST_K == 3
    print(f"  WORST_K = {sn.WORST_K}")

    q_id = sn.format_question({}, "identify")
    assert "worst_sites" in q_id and "JSON" in q_id and str(sn.WORST_K) in q_id
    print("  format_question(identify): JSON spec + K present  OK")

    q_de = sn.format_question({}, "describe")
    assert "shape" in q_de.lower() and "compact" in q_de.lower()
    print("  format_question(describe): mentions shape/compact concepts  OK")

    # parse_response with various model output styles.
    cases = [
        ('{"worst_sites": [12, 7, 3]}', [12, 7, 3]),
        ('Some preamble. {"worst_sites": [12, 7, 3]}', [12, 7, 3]),
        ('{"worst_sites": [3, 3, "oops", 7]}', [3, 7]),       # dedup + junk
        # Markdown-fenced — backticks expressed as literal chars.
        ("Here is my answer:\n"
         + chr(96) * 3 + "json\n"
         + '{"worst_sites": [1, 2, 3]}\n'
         + chr(96) * 3, [1, 2, 3]),
        ('Reasoning: site 5 looks bad.\nFinal: {"worst_sites": [5, 9, 11]}',
         [5, 9, 11]),
    ]
    for raw, expected in cases:
        parsed = sn.parse_response(raw, "identify")
        assert parsed == {"worst_sites": expected}, (raw[:60], parsed)
    print(f"  parse_response(identify): {len(cases)} cases all OK")

    # score(identify) — F1 on top-K sets.
    truth = {"worst_sites": [3, 7, 12]}
    assert sn.score({"worst_sites": [3, 7, 12]}, truth, "identify") == 1.0
    assert sn.score({"worst_sites": [12, 7, 3]}, truth, "identify") == 1.0
    assert abs(sn.score({"worst_sites": [3, 7, 99]}, truth, "identify")
                - 2 / 3) < 1e-9
    assert sn.score({"worst_sites": []}, truth, "identify") == 0.0
    assert sn.score({"worst_sites": [99, 100, 101]}, truth, "identify") == 0.0
    print("  score(identify): full / partial / empty / miss all correct")

    # score(describe) — concept-keyword fraction.
    truth_de = {"concepts": ["elongated", "jagged", "thin"]}
    assert abs(sn.score({"text": "These shapes are very elongated and jagged."},
                          truth_de, "describe") - 2 / 3) < 1e-9
    assert sn.score({"text": "all compact"}, truth_de, "describe") == 0.0
    print("  score(describe): keyword fraction correct")

    # -- (2) oracle shape_niceness branches ------------------------------
    print("\n[2] oracle shape_niceness branches against synthetic pair")
    from verify_slice import _build_synthetic_pair
    import models.tool_oracle as oracle

    inst, sol, _ = _build_synthetic_pair()
    raw_id = oracle.compute_answer(instance=inst, solution=sol,
                                     archetype="shape_niceness",
                                     task="identify")
    parsed_id = json.loads(raw_id)
    print(f"  oracle identify raw: {raw_id}")
    assert isinstance(parsed_id.get("worst_sites"), list)
    # Synthetic pair has 2 opened sites with disjoint catchments. NPI is
    # finite for both; oracle returns up to K=3, so we expect exactly
    # those 2 sites in some order.
    assert set(parsed_id["worst_sites"]) == {0, 1}
    print("  oracle identify -> includes both opened sites  OK")

    raw_de = oracle.compute_answer(instance=inst, solution=sol,
                                     archetype="shape_niceness",
                                     task="describe")
    print(f"  oracle describe raw: {raw_de[:90]}...")
    parsed_de = sn.parse_response(raw_de, "describe")
    truth_de_full = {"concepts": ["elongated", "stretched", "jagged",
                                    "irregular", "thin", "tail",
                                    "bowtie", "odd"]}
    s = sn.score(parsed_de, truth_de_full, "describe")
    print(f"  oracle describe scored against describe-task: {s:.2f}")
    # Synthetic catchments are disjoint pairs of squares — high NPI;
    # oracle picks the elongated-description branch and should hit
    # every concept keyword.
    assert s == 1.0, f"oracle describe should be 1.0, got {s}"
    print("  oracle describe hits all 8 concept keywords  OK")

    # -- (3) eval_set generator config -----------------------------------
    print("\n[3] eval_set/build_eval_set.py — ARCHETYPE_CONFIG entry")
    import build_eval_set as be
    assert "shape_niceness" in be.ARCHETYPE_CONFIG
    cfg = be.ARCHETYPE_CONFIG["shape_niceness"]
    assert cfg["tasks"] == ["identify", "describe"]
    assert cfg["generator"] == "generate_shape_niceness_instance"
    assert set(cfg["tiers"]) == {"easy", "med", "hard"}
    for tier_name, kw in cfg["tiers"].items():
        assert "min_mean_npi" in kw and "min_max_npi" in kw, (tier_name, kw)
    print(f"  tiers: {list(cfg['tiers'])}")
    print("  thresholds (mean / max NPI):")
    for t, kw in cfg["tiers"].items():
        print(f"    {t}: mean >= {kw['min_mean_npi']}, "
               f"max >= {kw['min_max_npi']}")

    meta = {
        "archetype": "shape_niceness",
        "per_catchment_npi": {
            "5":  {"NPI": 1.42}, "8":  {"NPI": 2.10},
            "12": {"NPI": 1.85}, "17": {"NPI": 2.50},
            "21": {"NPI": 1.10}, "34": {"NPI": 1.95},
        },
    }
    gt = cfg["ground_truth_fn"](meta)
    # Sorted by NPI desc: 17 (2.50), 8 (2.10), 34 (1.95), 12 (1.85), ...
    assert gt["identify"] == {"worst_sites": [17, 8, 34]}, gt["identify"]
    assert "concepts" in gt["describe"]
    print(f"  ground_truth_fn(meta) identify -> {gt['identify']['worst_sites']}  OK")

    # Ground truth tie-break: equal NPI, lower site index wins.
    meta_tie = {
        "per_catchment_npi": {
            "5":  {"NPI": 2.0}, "3": {"NPI": 2.0},
            "9":  {"NPI": 1.5}, "1": {"NPI": 1.0},
        },
    }
    gt_tie = cfg["ground_truth_fn"](meta_tie)
    assert gt_tie["identify"]["worst_sites"][:2] == [3, 5]
    print(f"  tie-break (equal NPI -> lower index): {gt_tie['identify']['worst_sites'][:2]}  OK")

    print("\nAll shape_niceness checks PASSED.")


if __name__ == "__main__":
    main()
