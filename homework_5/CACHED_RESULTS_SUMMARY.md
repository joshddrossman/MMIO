# Cached evaluation summary (`full_dataset/`)

All numbers below are read from on-disk **`aggregate.json`** files under  
`/Users/cat2510/MMIO/full_dataset/<archetype>/results_<modality>_vague/`.  
Each run has **30 pairs** (`n_completed: 30`). **Query type is vague only** (no `results_*_precise` trees in this snapshot).

## Headline comparison (multimodal vs tools-only)

| Archetype     | Modality   | Success rate | Mean fraction improved | Mean Δ assignment distance |
| ------------- | ---------- | ------------ | ----------------------- | -------------------------- |
| cluster       | multimodal | **70%** (21/30) | 0.711 | +392 |
| cluster       | tools_only | **60%** (18/30) | 0.607 | +277 |
| coverage_gap  | multimodal | **56.7%** (17/30) | 0.250 | +102 |
| coverage_gap  | tools_only | **53.3%** (16/30) | 0.218 | +211 |
| contiguity    | multimodal | **20%** (6/30)  | 0.150 | +79 |
| contiguity    | tools_only | **10%** (3/30)  | 0.094 | +25 |
| shape_niceness| multimodal | **0%** (0/30)   | 0.006 | +141 |
| shape_niceness| tools_only | **0%** (0/30)   | 0.006 | +178 |

**Takeaways for a homework writeup**

- **Cluster:** Multimodal edges tools-only on success rate and mean primary improvement; mean travel-distance cost (secondary metric) is somewhat higher for multimodal in this batch.
- **Coverage_gap:** Modest multimodal advantage on success; tools-only shows larger mean distance delta (check per-pair validity when interpreting).
- **Contiguity:** Both are hard; **multimodal doubles success rate** vs tools-only here (6 vs 3 successes on 30).
- **Shape_niceness:** Neither modality clears the success bar in this 30-pair slice (0 successes); mean fractional improvement is near zero — good place to discuss limits of the agent and scoring threshold.

---

## Example A — `cluster_med_00` (dense cluster; vague query)

Same engineered instance; **multimodal** and **tools-only** both reach **success = true** and full primary improvement (`fraction_improved = 1.0`).

| Field | Multimodal | Tools-only |
| ----- | ---------- | ---------- |
| `n_view_solution` | 3 | 0 (tool unavailable) |
| Wall-clock `elapsed_sec` | ~46 | ~9 |
| `total_tokens` | ~114.6k | ~102.4k |

**Illustrates:** maps help qualitative alignment but cost more time/tokens; tools-only can still win on easy cluster structure with pure structured probes.

---

## Example B — `cluster_med_01` (partial fix; vague query)

| Field | Multimodal | Tools-only |
| ----- | ---------- | ---------- |
| Success | **false** | **false** |
| `fraction_improved` | 0.0 | 0.2 |
| `selection_reason` (multimodal) | Baseline selected: improved candidates **invalid** under guards | (resolve selected; still below success threshold) |

**Illustrates:** multimodal can propose a primary fix that **violates guards**, so superscoring falls back to baseline; tools-only may show small numeric movement without meeting the archetype success rule.

---

## Example C — `contiguity_med_00` (split catchments; vague query)

| Field | Multimodal | Tools-only |
| ----- | ---------- | ---------- |
| Success | **true** | **false** |
| `n_view_solution` | 2 | 0 |
| `elapsed_sec` | ~84.5 | ~24.8 |
| Discontiguity target | 2 → 0 (per success) | Fails success criterion |

**Illustrates:** the clearest **modality gap** in this dataset: seeing coloured catchment patches helps the agent fix a **contiguity** violation that tools-only misses on the identical pair.

---

## Files to cite in the PDF

- Per-archetype rollups: `full_dataset/<arch>/results_multimodal_vague/aggregate.json` and `.../results_tools_only_vague/aggregate.json`.
- Per-pair detail: same folders, `<pair_id>.json` (e.g. `cluster_med_00.json`).
- Pair assets: `full_dataset/<arch>/pairs/<pair_id>/query_metadata.json`, `views/*.png`.

Homework scripts and `benchmark_tasks.json` now resolve pairs under **`full_dataset/`** (not `out/full_dataset`).
