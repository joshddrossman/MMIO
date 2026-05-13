"""True optimum / MIP regret — **not** derivable from ``*_result.json`` alone.

Result JSON (and ``explored_scores_full``) only describe states the agent **actually
visited**. A regret vs the **global** optimum needs:

- ``Instance`` / ``Solution`` objects (e.g. ``<pair_dir>/instance.pkl`` and baseline),
- the full discrete action space you allow the optimiser (opens/closes/assignments),
- and a solver (MIP / CP-SAT / enumeration) or a proven lower bound.

Suggested integration (future work):

1. Load ``instance`` + ``baseline`` from the pair directory used in ``run_dataset``.
2. Build a mathematical programme for the archetype’s primary metric (or a
   surrogate) subject to your feasibility + guard rules.
3. Compare optimum primary value to ``score['target_response']`` on the selected
   solution — that is **global** regret, independent of the agent’s search tree.

Until that exists, use ``oracle_max_feasible_fraction_improved`` in ``PairRecord``
(guard-agnostic ceiling **within logged explores**) as an empirical upper bound
that can still be below the true optimum.
"""
