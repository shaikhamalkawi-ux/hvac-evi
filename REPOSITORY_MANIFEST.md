# Repository manifest

This repository is the publication-facing reproducibility companion for:

**Beyond Aggregate Scores in HVAC Fault Diagnosis: Independent-Unit Evaluation and Real-Operational Validation Across 41 Air-Handling Units**

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
