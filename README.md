# HVAC-EVI

Reproducibility materials for the manuscript:

**Fault-Class Coverage Versus Unseen-Equipment Diagnosis: Fixed-Budget Training-Unit Selection in 41 Air-Handling Units**

Authors: Ghassan Malkawi, Ahmed Elsayed, Mohammed Alhagyan, and Bakeel Hussein.

This repository preserves the earlier validation evidence and the additive V37 public code/summary audit.  It is intended to make the evaluation design, source boundaries, fixed model settings, derived inputs, result tables, and validation code auditable.

## V38 original Hospital-source date audit (October 2026)

The V38 editorial revision separately audited the exact public [Wang Figshare v3 dataset](https://doi.org/10.6084/m9.figshare.27147678.v3). The hospital CSV contains 66,048 raw rows all labelled with **2023 dates** (1 January through 21 December). The original published data-descriptor describes **2022**. This discrepancy is **unresolved**: we did not redefine the physical collection year or silently rewrite timestamps. The main V38 journal manuscript reports the issue and retains previously locked results.

The public audit script and summary are in [`v38_hospital_provenance/`](v38_hospital_provenance/). Obtain the original hospital CSV from the source and run:

```bash
pip install -r v38_hospital_provenance/requirements.txt
python v38_hospital_provenance/verify_hospital_source_audit.py /path/to/hosptial_scientific_data.csv
```

This also reproduces the 161 repeated AHU-time keys, 322 included rows and 33 conflicting-label keys in the admitted three-class subset. This source audit does **not** rerun models or confirm physical repairs. No original raw data, private row indices, or Zenodo reviewer tokens appear here.

## V37 public code and aggregate-result update (20 October 2026)

The research manuscript now includes a **retrospective time-matched source-AHU replacement sensitivity** (3,240 additional fixed-model fits). The public update provides the analysis scripts, frozen protocol, metadata-only source schema, and aggregate result/QA tables in **[`v37_time_matched/`](v37_time_matched/)**. Run the publicly self-contained, standard-library checker with:

```bash
python v37_time_matched/code/verify_public_tables.py
```

**Critical availability boundary:** This checks public table arithmetic and file hashes; it does not reconstruct model fits. The full analysis code in this folder requires frozen V25 input/loader artifacts that have **not** been published here pending rights review. The independent public AHU source RBC-Nesbitt was inspected only for compatibility and was **not** added as a new performance dataset. The Zenodo reviewer draft is separate and remains unpublished; **no reviewer preview token is in this public repository**. Do not infer independent physical-fault confirmation, causal equipment effects, or energy savings.

The earlier repository files remain available unchanged; they document earlier fixed validation experiments and are not a complete public V37 numerical replay.

## Earlier validation scope

The study separates four quantities that should not be collapsed into a single HVAC fault-diagnosis score:

1. aggregate classification performance;
2. weakest-class recall;
3. sensitivity to the declared independent evaluation unit;
4. source-defined target-domain support.

The strongest external analysis uses **307,375 hourly observations from 41 real-operational AHUs** across auditorium, hospital, and office buildings. The primary row-wise/AHU-disjoint macro-F1 values are:

| Building | Row-wise macro-F1 | AHU-disjoint macro-F1 |
|---|---:|---:|
| Auditorium | 0.950 | 0.914 |
| Hospital | 0.970 | 0.526 |
| Office | 0.940 | 0.913 |

The repository also preserves the source-specific boiler sensor-tier analysis and PreDist target-support decomposition.

## Repository structure

- `scripts/run_field_external_validation_A4.py` — real-operational 41-AHU external validation.
- `scripts/run_release_1_4_extension.py` — fixed PreDist support-decomposition and classwise extension.
- `environment/` — recorded Python-package environments.
- `protocol/` — source, protocol, and claim-boundary documentation.
- `results/field/` — derived field-validation metrics and QA outputs.
- `results/core/` — derived ORNL/boiler/PreDist analytical tables.
- `data/` — source hashes and publication-safe provenance records. The large historical core-input tables remain in the archived Release 1.5 reproducibility package.
- `manifests/` — SHA-256 manifests from the archived reproducibility package.

## Raw-data boundary

Raw third-party datasets are **not redistributed** in this repository.

### Wang real-operational AHU data

- Article DOI: `10.1038/s41597-025-05825-9`
- Dataset DOI: `10.6084/m9.figshare.27147678.v3`
- Article/version used: Figshare article 27147678, version 3.

Expected raw-source SHA-256 values are recorded in `data/WANG_RAW_SOURCE_SHA256.txt`.

### ORNL, LBNL, and PreDist

The corresponding source/version information and analytical boundaries are documented in the manuscript supplement and in `protocol/SOURCE_AND_CLAIM_BOUNDARIES.md`. Derived analytical inputs needed by the public scripts are included where redistribution is permitted.

## Fixed external-validation model

The real-operational external validation uses the fixed probe:

- training-only median imputation;
- `DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=20260802)`;
- four-fold row-wise stratified OOF evaluation;
- four-fold AHU-disjoint stratified-group OOF evaluation;
- primary classes: `Normal`, `RATSF`, `SFF`;
- common eight-sensor semantic intersection documented in the script;
- 20 fixed split seeds `20260802...20260821` for sensitivity only.

No hyperparameter tuning, post-result threshold selection, or deep-learning model search is performed.

## Reproducing the field validation

1. Create a Python 3.13 environment.
2. Install the field requirements:
   ```bash
   pip install -r environment/requirements_field.txt
   ```
3. Run:
   ```bash
   python scripts/run_field_external_validation_A4.py --help
   ```
4. The runner can retrieve or use the exact Figshare v3 source files and writes derived results without redistributing the raw CSVs.

## Reproducing the fixed core extension

```bash
pip install -r environment/requirements_core_release_1_4_2.txt
python scripts/run_release_1_4_extension.py
```

The preserved extension script reads historical derived inputs from the original Release 1.4.2/1.5 reproducibility archive. The principal expected core result tables are mirrored under `results/core/`, but the full historical input lineage is not duplicated here.

## Claim boundary

This repository supports retrospective external validation on published real-operational data. It does **not** establish:

- a universal random-split penalty;
- a universal minimum sensor set;
- a universal target-support threshold;
- prospective repair-confirmed fault detection;
- realised energy or maintenance savings;
- causal building/manufacturer effects;
- operational certification or deployment readiness.

## Citation

See `CITATION.cff`. Until the manuscript receives a publication DOI, cite the manuscript title and this repository URL:

https://github.com/shaikhamalkawi-ux/hvac-evi

## Licence

Original code and documentation in this repository are provided under the MIT License. Third-party data are not included and remain subject to their original licences and terms.
