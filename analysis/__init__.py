"""Post-API benchmark analysis (sweeps, plotting helpers for notebooks)."""

from .shape_niceness_replay_metrics import (
    audit_shape_niceness_dataset,
    rescore_one_result_json,
    rescore_shape_niceness_dataset,
)
from .sweep import (
    ARCHETYPE_ORDER,
    MODALITY_ORDER,
    OFFICIAL_FRACTION_THRESHOLDS,
    PairRecord,
    discover_result_paths,
    ecdf,
    load_all_records,
    load_pair_record,
    official_fraction_threshold,
    group_paths,
    record_success_for_threshold_plot,
    records_by_bucket,
    selection_gap_summary,
    sweep_offline_reselect_pass_same_tau,
    sweep_selected_and_oracle,
    sweep_selected_valid_and_feasible_oracle,
    tau_grid,
)

__all__ = [
    "audit_shape_niceness_dataset",
    "rescore_one_result_json",
    "rescore_shape_niceness_dataset",
    "ARCHETYPE_ORDER",
    "MODALITY_ORDER",
    "OFFICIAL_FRACTION_THRESHOLDS",
    "PairRecord",
    "discover_result_paths",
    "ecdf",
    "load_all_records",
    "load_pair_record",
    "official_fraction_threshold",
    "group_paths",
    "record_success_for_threshold_plot",
    "records_by_bucket",
    "selection_gap_summary",
    "sweep_offline_reselect_pass_same_tau",
    "sweep_selected_and_oracle",
    "sweep_selected_valid_and_feasible_oracle",
    "tau_grid",
]
