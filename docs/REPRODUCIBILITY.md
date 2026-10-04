# Reproducibility guide

## Two computational components

### 1. Fixed core extension
`scripts/run_release_1_4_extension.py` reproduces:
- PreDist support-mechanism decomposition;
- ORNL classwise recall comparison;
- boiler classwise recall comparison;
- PreDist transfer-score/support points.

It reads only derived package-relative inputs in `data/core_inputs/`.

### 2. Real-operational external validation
`scripts/run_field_external_validation_A4.py` implements the 41-AHU field-validation protocol:
- row-wise vs AHU-disjoint four-fold OOF evaluation;
- 20-seed split sensitivity;
- six ordered cross-building transfers;
- min-max, 1st–99th, and 5th–95th source-defined target support;
- schema/data gates;
- stored predictions, class metrics, fold logs, and source hashes.

The raw Wang CSV files are not committed to this repository.

## Expected fixed external results

Primary row-wise/AHU-disjoint macro-F1:

| Building | Row-wise | AHU-disjoint | Difference |
|---|---:|---:|---:|
| Auditorium | 0.950472 | 0.914224 | 0.036248 |
| Hospital | 0.969644 | 0.525912 | 0.443732 |
| Office | 0.939565 | 0.913178 | 0.026387 |

Cross-building results:

| Direction | Macro-F1 | Min-max support | 1st–99th | 5th–95th |
|---|---:|---:|---:|---:|
| Auditorium→Hospital | 0.554029 | 0.943949 | 0.702525 | 0.101437 |
| Auditorium→Office | 0.550469 | 0.984746 | 0.643362 | 0.142330 |
| Hospital→Auditorium | 0.668175 | 0.338999 | 0.251877 | 0.157890 |
| Hospital→Office | 0.412014 | 0.242000 | 0.139898 | 0.030846 |
| Office→Auditorium | 0.672395 | 0.991618 | 0.734075 | 0.457943 |
| Office→Hospital | 0.446287 | 0.917396 | 0.734255 | 0.564088 |

These values are verification targets, not tuning objectives.

## Independent QA preserved in the archived package

- within-building metric recomputation: 6/6 PASS;
- cross-building class-table metric recomputation: 6/6 PASS;
- primary fold records with all three primary labels in training: 24/24 PASS.

## Important

Do not interpret split-seed percentiles as population confidence intervals. Do not infer a universal random-split penalty from the direction or magnitude of any single source.
