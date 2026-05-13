# Multimodal Interactive Optimization for Polling Place Location

Class project on multimodal interactive optimization. A stakeholder issues
a critique of a polling-place location plan; an LLM-based optimization
agent inspects the current solution (via tools, optionally also via a
rendered map), and iteratively submits proposals that the system applies
by re-solving a MILP — or makes local-search-style edits that update the
solution directly. The agent is evaluated on a four-archetype benchmark
of *emergent solution properties*.

The hypothesis under study: **multimodal access (rendered map + structured**  
**tools) outperforms tools-only access, and the gap is largest when the**  
**stakeholder's query is vague rather than precise**.

The four archetypes form a difficulty gradient:


| Archetype        | Property                                                        | Why it might be hard for tools-only                                                                                                 |
| ---------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `cluster`        | Too many opened sites in a small region                         | Easy: spatial statistics on opened-site coordinates. **Baseline.**                                                                  |
| `coverage_gap`   | Interior pocket of voters with anomalously bad access           | Easy-ish: coverage-radius scan over opened sites. **Baseline.**                                                                     |
| `contiguity`     | Opened sites whose catchments are non-contiguous                | Mid: requires the precinct adjacency graph + a connected-components search. The agent has to *think* to do this.                    |
| `shape_niceness` | Catchments that are visibly "ugly" — elongated, jagged, bowtied | Hard: no clean formal characterisation. Vision sidesteps the formalisation problem entirely. **The strongest case for multimodal.** |


## Setting

A synthetic county with ~80 precincts and ~40 candidate polling-place
sites, generated procedurally with non-uniform demographic distributions
(Black-majority neighbourhoods south, predominantly white suburban
neighbourhoods north — used only for the underlying voter density and
not for any archetype query), a river barrier crossable only at bridges,
and deliberately under-supplied candidate sites on the south side. The
optimizer's stated objective is to minimise total voter-weighted travel
distance subject to capacity and a budget on opened sites. That objective
ignores the four emergent properties above; the agent's job is to detect
violations the stakeholder cares about and propose interventions.

## Optimization formulation

Capacitated facility location with assignment:

```
min   Σ_{i,j} (w_i · v_i) · d_ij · y_ij
s.t.  Σ_j y_ij = 1                    for all i        (each precinct assigned)
      y_ij ≤ x_j                       for all i, j     (assign only to opened sites)
      Σ_i v_i · y_ij ≤ cap_j · x_j    for all j        (capacity)
      Σ_j x_j ≤ K                                       (budget)
      x, y ∈ {0, 1}
```

`x_j ∈ {0,1}` whether site `j` is opened; `y_ij ∈ {0,1}` whether precinct
`i` is assigned to site `j`; `v_i` voter count; `w_i` per-precinct weight
multiplier (default 1.0, settable by the agent's soft action); `d_ij`
distance with a river-crossing penalty when the segment doesn't cross at
a bridge; `K = 18` opened sites. Solved with `gurobipy`.

## File structure

```
instance_generator/
├── instance.py                 Instance and Solution dataclasses.
├── generation.py               Procedural generation + per-archetype
│                                 engineered-instance generators
│                                 (cluster, coverage_gap, contiguity,
│                                 shape_niceness) plus the NPI helper
│                                 used by the shape archetype.
├── solver.py                   gurobipy MILP. FixingConstraints encodes
│                                 the agent's hard fixings + soft weights.
├── metrics.py                  Reference metrics on a solution.
├── rendering.py                Layered map renderer; render_all_pair_views
│                                 helper that produces the canonical
│                                 PNG set per dataset pair.
├── agent_tools.py              Tools the agent calls during a session:
│                                 list_sites, list_precincts_in_region,
│                                 get_site_at, get_precinct_at,
│                                 get_precinct_centroids,
│                                 get_current_assignments,
│                                 get_distance_matrix_data,
│                                 get_precinct_adjacency_data,
│                                 local edit helpers, plus Proposal /
│                                 apply_proposal and view_solution_png.
├── queries.py                  ArchetypeQuery + GuardSpec dataclasses,
│                                 the four archetype factories,
│                                 ARCHETYPE_FACTORIES dispatch table.
├── main.py                     Single-instance generator (one archetype).
├── dataset_generator.py        Per-archetype dataset generator. Each
│                                 pair carries BOTH a vague_text and a
│                                 precise_text in its query_metadata.json.
├── generate_full_dataset.py    Top-level orchestrator: generate N pairs
│                                 for all 4 archetypes in one command.
├── run_dataset.py              Run the agent across a dataset, score
│                                 per pair, aggregate. modality x query_type
│                                 experimental matrix.
├── test_agent.py               Single-shot agent runner against one pair.
├── requirements.txt            numpy, matplotlib, gurobipy, openai,
│                                 python-dotenv.
└── README.md                   (this file)
```

## Agent action space

The agent has three solution-flow actions: `resolve` for MILP re-solves,
local-edit tools for frozen direct edits, and `submit_proposal` to indicate
that the reasoning loop is done.

**MILP-based actions** — bundled in a `resolve` call; the system re-solves
the MILP under these fixings and makes the result the new current solution.
Each `resolve` call replaces prior fixings; include every fixing you still
want to preserve.


| Action                           | Type | Effect on the MILP                                                                      |
| -------------------------------- | ---- | --------------------------------------------------------------------------------------- |
| `force_open[j]`                  | hard | x_j = 1                                                                                 |
| `force_close[j]`                 | hard | x_j = 0                                                                                 |
| `force_assign[i, j]`             | hard | y_ij = 1 (and implicitly x_j = 1)                                                       |
| `precinct_weight_multipliers[i]` | soft | objective term `v_i · d_ij` becomes `(m_i · v_i) · d_ij`; m_i clamped to `[0.0, 100.0]` |


**Local-edit actions** — modify the current solution directly without
re-solving the MILP.


| Tool                                                   | Effect                                                                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `force_assign(precinct_index, site_index)`             | Reassign a precinct to an already-opened site while the rest of the solution stays put. Objective and capacity feasibility are recomputed without re-solving. |
| `swap_assignments(precinct_a_index, precinct_b_index)` | Swap two precincts' assigned sites while the rest of the solution stays put. Objective and capacity feasibility are recomputed without re-solving.            |


**Final submission** — `submit_proposal(rationale)` ends the session. It
does not take fixings; benchmark scoring uses the best feasible solution
explored during the session.

## Agent tools


| Tool                                                                                     | Purpose                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `view_solution(layers, show_precinct_labels)`                                            | Render the current solution as a PNG. The agent can request canonical view types (`plain`, `baseline`, `population_density`). The dispatcher attaches a structured payload of opened-site coordinates so the agent never has to OCR labels for positional reasoning. (Disabled in tools-only mode.) |
| `list_sites(opened_only)`                                                                | Structured list of candidate sites (index, x, y, type, capacity, opened, load).                                                                                                                                                                                                                     |
| `list_precincts_in_region(polygon)`                                                      | Precincts whose centroid lies inside a polygon, returning index, centroid x/y, and voters.                                                                                                                                                                                                          |
| `get_site_at(x, y)`                                                                      | Site nearest a coordinate.                                                                                                                                                                                                                                                                          |
| `get_precinct_at(x, y)`                                                                  | Precinct containing a coordinate, returning index, centroid x/y, and voters.                                                                                                                                                                                                                        |
| `get_precinct_centroids(precinct_indices)`                                               | Precinct centroid coordinates in km, plus voters. If `precinct_indices` is omitted, returns all precinct centroids.                                                                                                                                                                                 |
| `get_precinct_adjacency()`                                                               | Per-precinct list of spatially-neighbouring precincts. Use with connected-components on an opened site's assigned-precinct subgraph to detect non-contiguous catchments. Especially useful for the contiguity archetype in tools-only mode.                                                         |
| `get_current_assignments(precinct_indices)`                                              | Current precinct-to-polling-place assignments, including assigned site, assigned distance, weighted distance, and coordinates. If `precinct_indices` is omitted, returns all precincts.                                                                                                             |
| `get_distance_matrix(precinct_indices, site_indices, opened_only)`                       | Precinct-to-site travel distance matrix in km. Supports optional precinct/site slices, or `opened_only=true` to return only currently opened sites.                                                                                                                                                 |
| `resolve(force_open, force_close, force_assign, precinct_weight_multipliers, rationale)` | MILP re-solve under fixings. The result becomes the current solution if feasible.                                                                                                                                                                                                                   |
| `force_assign(precinct_index, site_index)`                                               | Local-edit reassignment. See above.                                                                                                                                                                                                                                                                 |
| `swap_assignments(precinct_a_index, precinct_b_index)`                                   | Local-edit swap. See above.                                                                                                                                                                                                                                                                         |
| `submit_proposal(rationale)`                                                             | Final submission. Ends the reasoning loop; scoring uses the best feasible solution explored during the session.                                                                                                                                                                                     |


## Iterative-proposal flow

The agent can call `resolve` multiple times per session. Each feasible
resolve applies tentatively: the system re-solves the MILP under the
fixings, updates the "current solution" that subsequent tool calls see,
and (in multimodal mode) sends back a fresh rendered view. The agent can
inspect, refine with another `resolve`, make local edits with
`force_assign` / `swap_assignments`, and then call
`submit_proposal(rationale)` once satisfied. The runner retains every
feasible solution produced by the baseline, feasible resolves, and feasible
local edits, then superscores those solutions and reports the best one.

## Map layers per dataset pair

`render_all_pair_views` writes a canonical PNG set per pair, with
strict layer partitioning so each view shows exactly its relevant context:


| View                          | Layers                                                                                                      | Used for                                                 |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `plain.png`                   | precincts, candidate sites                                                                                  | reference / inspection                                   |
| `baseline_text_only.png`      | precincts, candidate sites, opened sites, assignments (with categorical service-area fill by assigned site) | every archetype's primary visual                         |
| `baseline_text_annotated.png` | same + annotation polygon                                                                                   | precise condition (when archetype has an anchor polygon) |
| `population_density.png`      | density heatmap, opened sites                                                                               | contextual reference                                     |


The categorical service-area fill on `baseline_text_only.png` is the key
visual hook for the contiguity and shape_niceness archetypes — disjoint
same-coloured patches show non-contiguous catchments at a glance, and
elongated / bowtied shapes show ugly catchment geometry.

## Query types: vague vs. precise

Each pair's metadata carries **both** a `vague_text` and a `precise_text`.

- **Vague**: stakeholder-style critique with no specific entity reference.
*"Some service regions look weird. Make them more compact."* The agent
must localise the issue from the rendered map (multimodal) or from
probing structured tools (tools-only).
- **Precise**: names the offending site index or region centre.
*"Polling place 17's catchment is unusually elongated. Make it more
compact."* The agent still has to reason about the fix, but
identification is given.

Plus the modality knob (multimodal vs. tools-only), this gives a 2×2
experimental matrix per archetype:


|                   | Modality: multimodal                                 | Modality: tools-only                                              |
| ----------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| **Vague query**   | agent gets rendered map + critique text              | agent gets only structured tools + critique text                  |
| **Precise query** | agent gets map + critique text that names the entity | agent gets structured tools + critique text that names the entity |


`run_dataset.py` exposes `--no_visual` (modality) and `--query_type`
(vague/precise) and auto-names the results subdirectory by both knobs.

## Per-archetype scoring

Every score dict carries:

- **Primary target**: archetype-specific (see below) — `target_baseline`,
`target_response`, `raw_improvement`, `fraction_improved`.
- **Secondary target**: `final_assignment_distance` (the response
solution's voter-weighted total distance) plus
`assignment_distance_delta`. Two agents that fix the property equally
well can still differ on solution quality — this exposes that.
- **Guards**: total_weighted_distance and p90_voter_distance, with
configurable degradation budgets.
- **Binary success**: archetype-specific (see below). Combined with
feasibility AND all guards passed.

`run_dataset.py` uses **superscoring**: during an agent run, the runner
retains the baseline, every feasible `resolve` result, and every feasible
local-edit result. After the agent calls `submit_proposal(rationale)` (or
the loop otherwise ends), the runner scores all retained feasible solutions
and reports the best one according to success, validity, primary-target
improvement, and then assignment-distance tie-breaks. In other words,
`submit_proposal` is a stop signal, not a declaration that the current
solution is the only scored solution.


| Archetype        | Target metric (minimize)                                                                                                                                                                | Success criterion                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `cluster`        | **GLOBAL count of opened sites in any dense cluster** (union of opened sites appearing inside any radius-neighbourhood with ≥ `cluster_min_sites` opened sites within `cluster_radius`) | `target_response == 0` (no dense radius-neighbourhood of the threshold size exists) |
| `coverage_gap`   | **GLOBAL count of "stranded" precincts** (nearest-open-site distance > a per-pair calibrated threshold)                                                                                 | `fraction_improved >= 0.5`                                                          |
| `contiguity`     | **Count of discontiguous catchments**                                                                                                                                                   | `fraction_improved >= 0.5` (at least half of the split sites repaired)              |
| `shape_niceness` | **Mean Polsby-Popper-derived NPI** across opened catchments                                                                                                                             | `fraction_improved >= 0.2` (a 20% NPI reduction = visibly nicer shapes)             |


Definitions:

- **Dense cluster (cluster archetype)**: a radius-neighbourhood is dense
iff it contains at least `cluster_min_sites` opened sites within
`cluster_radius` of its centre, including itself. The target metric
counts the union of opened sites appearing inside any qualifying
radius-neighbourhood — global, so an agent who breaks the engineered
cluster but inadvertently creates a tight cluster elsewhere is
penalised. New generated cluster datasets use a tighter default
`cluster_radius` of 1.35 km, and scoring caps the effective cluster
radius at 1.35 km for older metadata as well.
- **Stranded precinct (coverage_gap archetype)**: a precinct whose
nearest-open-site distance exceeds a per-pair calibrated threshold.
The threshold is set at generation time so the baseline has ~5
stranded precincts (the engineered affected set plus a few
naturally-bad ones). The metric is global — closing a site that was
serving its neighbourhood creates new strandings and is penalised.
Coverage-gap generation records local-anomaly uniqueness diagnostics,
but uniqueness is not required by default. It does require a viable
fix path: at least one closed candidate facility must lie near the
centre of the annotated gap region, within half the annotation radius
by default.
- **NPI** (normalised perimeter index) = P / (2 · √(πA)), where A is the
catchment area and P is its perimeter, computed on the rasterised
precinct label grid. NPI = 1.0 for a circle and grows with elongation
/ jaggedness. Mean across opened catchments is the solution-level metric.
- **Discontiguous catchment**: an opened site whose assigned-precinct
subgraph (using 4-connected precinct adjacency) has more than one
connected component.

A response that improves the target while violating a guard is *not*
valid; this prevents trivial "fixes" that improve one metric at the
cost of catastrophic degradation elsewhere.

## Visual reasoning design choices

1. **Categorical service-area fill on the assignments view.** Each precinct
  is filled with the colour of its assigned opened site. Disjoint
   same-colour patches make non-contiguous catchments visually obvious;
   elongated patches make ugly shapes obvious.
2. **Auto-attached structured payload alongside images.** When the agent
  calls `view_solution`, the dispatcher attaches a JSON list of opened
   sites (index, x, y, load) so the agent never has to OCR labels for
   positional reasoning. The visual is for *patterns*; the structured
   tools are for *coordinates and indices*.
3. **System prompt mandates verification with structured tools.** The
  agent is explicitly told that VLMs are unreliable at reading map
   labels, and to use structured tools for any positional or index
   question.
4. **Site labels rendered with stroke + 10pt bold** so they survive VLM
  image downsampling.
5. **Precinct outlines always drawn** under every layer combination.
6. **The adjacency tool is available in BOTH modalities.** Tools-only
  agents need adjacency to detect non-contiguous catchments at all;
   multimodal agents may use it as verification. The asymmetry the
   experiment cares about is in *discovery effort*, not raw information
   availability.

## Running things

Install (gurobipy needs a license; academic license is free):

```sh
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

### Single-instance smoke

```sh
python main.py --archetype cluster
python main.py --archetype coverage_gap --base_seed 7
python main.py --archetype contiguity
python main.py --archetype shape_niceness
```

### Build the four-archetype dataset

```sh
# 25 pairs per archetype × 4 archetypes = 100 pairs total
python generate_full_dataset.py --root out/full_dataset --n_per_archetype 25

# Or per-archetype:
python dataset_generator.py --archetype contiguity --n_pairs 25
```

### Run the agent across the dataset (2×2 matrix)

```sh
# All four cells per archetype, written to results_<modality>_<query_type>/
# Optional: --temperature 0 --seed 42 for reproducibility on models that
# accept temperature=0 (omit --temperature for gpt-5-mini — API may reject it).
for arch in cluster coverage_gap contiguity shape_niceness; do
  for qt in vague precise; do
    for vis in "" "--no_visual"; do
      python run_dataset.py \
          --dataset_dir out/full_dataset/$arch \
          --query_type $qt $vis \
          --model gpt-5-mini \
          --temperature 0 --seed 42
    done
  done
done
```

**zsh:** unquoted `$EXTRA` is *not* word-split by default. If you use a variable for extra flags, either put them on the line explicitly (as above), use an array — `extra=(--temperature 0 --seed 42); python run_dataset.py ... "${extra[@]}"` — or force splitting with `${=EXTRA}`.

```sh
# Or one cell at a time:
python run_dataset.py --dataset_dir out/full_dataset/cluster \
                      --query_type vague --first_n 5 --model gpt-5-mini
```



Command **to re-run with a saved log**

```
cd /Users/cat2510/MMIO 

python test_agent.py \

  --pair_dir out/full_dataset/cluster/pairs/03 \

  --query_type vague \

  --model gpt-5-mini \

  --save_log /verbose_logs/cluster_03_vague_log.json 
```

Then inspect the full trajectory:

```
python3 -c "import json

log = json.load(open('/verbose_logs/cluster_03_vague_log.json'))

for e in log:

    ev = e.get('event')

    if ev in ('tool_call', 'resolve_applied', 'local_edit', 'solution_submitted', 'trajectory_summary'):

        print(json.dumps(e, indent=2))"
```

Or, to see just the resolve outcome and the view_solution purposes:

```
python3 -c " import json log = json.load(open('/verbose_logs/cluster_03_vague_log.json'))

for e in log:

    ev = e.get('event')

    if ev == 'resolve_applied':

        print('RESOLVE:', json.dumps({k:v for k,v in e.items() if k != 'proposal'}, indent=2))

        print('  fixings:', e.get('proposal', {}).get('force_close'), e.get('proposal', {}).get('force_open'))

    elif ev == 'tool_call' and e.get('name') == 'view_solution':

        print('VIEW_SOLUTION: purpose=', e.get('view_purpose'), 'layers=', e.get('layers'))

    elif ev == 'trajectory_summary':

        print('SUMMARY:', json.dumps(e, indent=2))

"
```



The key thing to look for in the log: after the resolve, `explore_scores[1].valid = false` tells you the guard failed. The `rationale` field in the `resolve_applied` event will show the agent thought it was closing a cluster site, but `affected_sites` in the pair metadata shows the actual cluster was `[1, 8, 33, 36]` — the agent closed site 5 by mistake after mis-reading the visual.

### Single-pair smoke against the agent

```sh
python test_agent.py --pair_dir out/full_dataset/shape_niceness/pairs/00 \
                     --query_type precise
python test_agent.py --pair_dir out/full_dataset/contiguity/pairs/03 \
                     --query_type vague --no_visual
```

## Aggregate results

Each per-cell `results_<modality>_<query_type>/aggregate.json` includes:

- Headline: `mean_fraction_improved`, `n_success`, `fraction_success`.
- Solution quality: `mean_baseline_assignment_distance`,
`mean_final_assignment_distance`, `mean_assignment_distance_delta`.
- Per-pair drilldown: target values, success flag, action counts,
elapsed seconds, baseline + final assignment distance.

Each per-pair result JSON also includes a `superscore` block with the
selected explored solution (`selected_source`, `selected_iteration`,
`selected_resolve_index`), the number of feasible explored solutions, and
compact score summaries for all feasible explored candidates.