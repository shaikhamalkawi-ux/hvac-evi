# Prespecified Protocol — Independent Real-Operational External Validation

**Protocol ID:** `HVAC_EVI_FieldExternalValidation_Protocol01`

**Purpose:** determine whether the earlier HVAC-EVI findings survive, narrow, or fail to reproduce on independent real-operational AHU data.

## Scientific question

Does a favourable HVAC-FDD score support the same conclusion when evaluation is made equipment-independent and when a target building is checked for source-domain support before a cross-building transport interpretation?

The study does **not** ask which classifier is best.

## External data

Exact source: Wang, *Scientific Data* (2025), DOI `10.1038/s41597-025-05825-9`; Figshare `10.6084/m9.figshare.27147678.v3`.

Three buildings:
- auditorium: 13 AHUs;
- hospital: 8 AHUs;
- office: 20 AHUs.

## Primary label universe

`{Normal, RATSF, SFF}`.

## Primary feature universe

The final source-resolved common intersection contains eight sensor concepts:
1. set-point temperature;
2. return-air temperature;
3. supply-air temperature;
4. supply fan;
5. valve position;
6. heating-supply temperature;
7. cooling-supply temperature;
8. cooling pump.

The exact hospital source does not provide a supported heating-pump equivalent; heating-pump semantics are excluded from the final common intersection.

## Fixed model probe

- `SimpleImputer(strategy='median', keep_empty_features=True)` fitted within training data only;
- `DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, class_weight='balanced', random_state=20260802)`.

No tuning, model search, ensemble, or deep-learning comparison is used.

## Analysis A — within-building equipment independence

Row-wise OOF:
`StratifiedKFold(n_splits=4, shuffle=True, random_state=20260802)`.

AHU-disjoint OOF:
`StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260802)`, groups = AHU identity.

Primary metric: macro-F1 on fixed labels `[Normal, RATSF, SFF]`.

Secondary metrics:
- minimum-class recall;
- mean class recall / balanced accuracy;
- accuracy;
- per-class precision, recall, F1.

Primary contrast:
`row-wise macro-F1 − AHU-disjoint macro-F1`.

Secondary split sensitivity:
20 fixed seeds `20260802…20260821`.

## Analysis B — six ordered cross-building transfers

For every ordered pair among auditorium, hospital, and office:
1. fit the same fixed decision tree on all source-building rows;
2. predict all target-building rows;
3. report target macro-F1, minimum-class recall, and per-class metrics;
4. compare descriptively with the target building's local AHU-disjoint OOF macro-F1.

## Analysis C — source-defined target support

For the same six ordered pairs and eight common features:
1. fit median imputation on source rows only;
2. calculate source feature intervals under min–max, 1st–99th percentile, and 5th–95th percentile bounds;
3. a target row is supported only if all imputed target features lie inside the corresponding source intervals;
4. report supported N/fraction and violation summaries.

Support is a conservative evidence-coverage diagnostic. It is not a probability of transfer success, density estimate, causal transportability test, or deployment certificate.

## Claim boundary

The Wang data are independently collected real-operational labelled field data. This protocol is an external validation analysis of existing field observations. It is not a field experiment or field intervention performed by the HVAC-EVI authors.

The source labels are expert/criteria-based operational annotations and are not upgraded to repair-confirmed failures.
