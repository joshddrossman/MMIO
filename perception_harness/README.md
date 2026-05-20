# Perception harness

A small experimental environment for measuring how well a VLM extracts
specific perceptual insights from the rendered polling-place maps,
independent of any optimization reasoning. The goal is to disentangle
*visual perception* from *optimization reasoning* so we can test
rendering and prompting changes cheaply, then port the winners back into
the multimodal optimization agent.

## Why

In `instance_generator/`, the multimodal optimization agent's score
conflates two failure modes: (1) the VLM didn't see the bowtie-shaped
catchment, and (2) the VLM saw it but proposed a bad fix. End-to-end
runs can't tell those apart. This harness isolates (1).

## How it's organised

```
perception_harness/
├── eval_set/
│   ├── build_eval_set.py     generate graded pairs + ground_truth.csv
│   ├── ground_truth.csv      one row per (pair, task)
│   └── pairs/<pair_id>/      instance.pkl + baseline_solution.pkl + meta.json
│
├── renderers/                each exports render(instance, solution) -> PNG bytes
├── prompts/                  each exports build(task_name, question, image_b64) -> messages
├── tasks/                    each exports format_question / parse_response / score
├── models/                   each exports query(messages, model_id) -> raw text
│                              (tool_oracle additionally exports compute_answer)
│
├── eval_perception.py        main runner (cross-product loop)
└── results/<run_name>/
    ├── config.json
    ├── per_question.csv
    └── aggregate.json
```

Each axis (renderer, prompt, model, task) is a separate module so
variants are independently swappable. Adding a new rendering variant is
a single file in `renderers/`; adding a new prompt strategy is a single
file in `prompts/`. Nothing else changes.

## Vertical slice — what's currently wired up

- **One archetype:** `contiguity`. The most tractable starting point
  because the ground truth (which sites have non-contiguous catchments)
  is a finite set with a clean F1 score.
- **Two tasks:** `identify` (return the set of split-site indices) and
  `describe` (free-form, scored on concept-keyword overlap).
- **One renderer:** `base` — the canonical view the optimization agent
  currently sees.
- **One prompt:** `zero_shot`.
- **Two models:** `tool_oracle` (analytic upper bound, no API key) and
  `openai_vlm:gpt-4o`.

## Running it

Install (in addition to instance_generator's deps):

```sh
pip install -r requirements.txt
```

Generate the eval set (15 contiguity pairs across 3 difficulty tiers):

```sh
cd perception_harness
python eval_set/build_eval_set.py --archetypes contiguity --n_per_tier 5
```

Sanity-check with the oracle (no API key needed; should report 1.0
across the board on the identify task):

```sh
python eval_perception.py --models tool_oracle --first_n 6
```

Run the real VLM:

```sh
export OPENAI_API_KEY=sk-...
python eval_perception.py \
    --renderers base \
    --prompts zero_shot \
    --models openai_vlm:gpt-4o \
    --archetypes contiguity \
    --difficulties easy med hard
```

Each invocation drops a timestamped subdirectory under `results/`.
`per_question.csv` is pandas-friendly; `aggregate.json` rolls up mean
score per (archetype, difficulty, task, renderer, prompt, model) cell.

## Adding a variant

A new rendering: `renderers/<name>.py` with a `render(instance,
solution, **kwargs) -> bytes` function. Then
`--renderers base <name>` in the runner.

A new prompt: `prompts/<name>.py` with a `build(task_name, question,
image_b64) -> list[dict]` function returning OpenAI-format messages.
Then `--prompts zero_shot <name>`.

A new model provider: `models/<name>.py` with `query(messages,
model_id, **kwargs) -> str`. Then `--models <name>:<model_id>`.

A new archetype task: `tasks/<archetype>.py` with `format_question`,
`parse_response`, and `score`. Add a difficulty-tier dict and a
ground-truth derivation to `eval_set/build_eval_set.py`.

## Transfer back to the optimization agent

Renderer wins port cleanly: swap one import in
`instance_generator/agent_tools.view_solution_png` to use the new
renderer.

Prompt wins port less cleanly because the agent's system prompt is
already long. Design prompt variants as short paragraphs that could
plausibly be appended to the per-tool-call text returned by
`dispatch_tool` for `view_solution`, rather than as wholesale
replacements.
