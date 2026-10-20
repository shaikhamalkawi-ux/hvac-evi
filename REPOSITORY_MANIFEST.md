# Repository manifest

This repository is the publication-facing reproducibility companion for:

**Fault-Class Coverage Versus Unseen-Equipment Diagnosis: Fixed-Budget Training-Unit Selection in 41 Air-Handling Units**

## Executable code
- `scripts/run_field_external_validation_A4.py` — complete 41-AHU real-operational external-validation runner.
- `scripts/run_release_1_4_extension.py` — archived fixed core-extension analysis script.

## Environment
- `environment/requirements_field.txt`
- `environment/requirements_core_release_1_4_2.txt`

## Protocol and source audit
- `protocol/FIELD_VALIDATION_PROTOCOL.md`
- `protocol/WANG_SOURCE_AUDIT.md`
- `protocol/SOURCE_AND_CLAIM_BOUNDARIES.md`

## Field-validation results
- `results/field/RESULT_SUMMARY.md`
- `results/field/INDEPENDENT_QA_REPORT.md`
- `results/field/within_building_primary_metrics.csv`
- `results/field/within_building_primary_contrasts.csv`
- `results/field/within_building_20seed_sensitivity_summary.csv`
- `results/field/cross_building_transfer_metrics.csv`
- `results/field/cross_building_support_metrics.csv`
- `results/field/primary_class_counts.csv`
- `results/field/common_feature_mapping.csv`
- `results/field/SOURCE_PROVENANCE.json`
- `results/field/STATE_LOCK.json`

## Core derived results
- `results/core/ornl_classwise_recall_primary.csv`
- `results/core/boiler_classwise_recall_primary.csv`
- `results/core/predist_support_violation_decomposition.csv`

## Raw-data boundary
Third-party raw data are not redistributed. Exact Wang v3 SHA-256 values are in `data/WANG_RAW_SOURCE_SHA256.txt`. Retrieval/version details are documented in the protocol files.

## Citation and licence
- `CITATION.cff`
- `LICENSE`

## V37 matched source replacement — additive public release (2026-10-20)
- `v37_time_matched/README.md` — protocol, scope, admitted outcomes and replay/rights boundary.
- `v37_time_matched/protocol_calendar_matched.json` — internally frozen, not externally preregistered.
- `v37_time_matched/code/` — allocation, scoring and independent-test scripts; `verify_public_tables.py` works without private inputs.
- `v37_time_matched/results/` — aggregate results and per-unit/class summary tables (not raw source or selected-index artifacts).
- `v37_time_matched/schema_gate/`, `v37_time_matched/qa/` and `v37_time_matched/SHA256SUMS.txt` — metadata-only schema and audit hashes.

The complete V37 experiment cannot be rerun from this public repository alone. Third-party raw observations and private source-derived detailed arrays are excluded pending rights checks. The Zenodo draft is separately shared with journal reviewers, not published here.
