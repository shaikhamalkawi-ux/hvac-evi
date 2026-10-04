# HVAC_EVI_FieldExternalValidation_Protocol01 (A2_v3_common_feature_intersection) — observed result summary

This file is generated from the prespecified protocol. It reports observations only; it does not create a post-hoc pass/fail threshold.

## Data admitted by the schema gate

- **Auditorium**: Normal=60219, RATSF=1734, SFF=6525.
- **Hospital**: Normal=53757, RATSF=8835, SFF=1921.
- **Office**: Normal=119402, RATSF=6999, SFF=47983.

## Equipment-independence contrast

- **Auditorium**: row-wise macro-F1=0.9505; AHU-disjoint macro-F1=0.9142; row-minus-AHU-disjoint Δ=+0.0362. Minimum-class recall: 0.9637 vs 0.9596.
- **Hospital**: row-wise macro-F1=0.9696; AHU-disjoint macro-F1=0.5259; row-minus-AHU-disjoint Δ=+0.4437. Minimum-class recall: 0.9883 vs 0.0544.
- **Office**: row-wise macro-F1=0.9396; AHU-disjoint macro-F1=0.9132; row-minus-AHU-disjoint Δ=+0.0264. Minimum-class recall: 0.9173 vs 0.9096.

## Cross-building transfer and source-defined target support

- **auditorium->hospital**: target macro-F1=0.5540; minimum-class recall=0.4147; supported target rows=60897/64513 (min–max), 45322/64513 (1st–99th), 6544/64513 (5th–95th).
- **auditorium->office**: target macro-F1=0.5505; minimum-class recall=0.3393; supported target rows=171724/174384 (min–max), 112192/174384 (1st–99th), 24820/174384 (5th–95th).
- **hospital->auditorium**: target macro-F1=0.6682; minimum-class recall=0.3907; supported target rows=23214/68478 (min–max), 17248/68478 (1st–99th), 10812/68478 (5th–95th).
- **hospital->office**: target macro-F1=0.4120; minimum-class recall=0.1353; supported target rows=42201/174384 (min–max), 24396/174384 (1st–99th), 5379/174384 (5th–95th).
- **office->auditorium**: target macro-F1=0.6724; minimum-class recall=0.4855; supported target rows=67904/68478 (min–max), 50268/68478 (1st–99th), 31359/68478 (5th–95th).
- **office->hospital**: target macro-F1=0.4463; minimum-class recall=0.1114; supported target rows=59184/64513 (min–max), 47369/64513 (1st–99th), 36391/64513 (5th–95th).

## Interpretation boundary

- These are independent real-operational external validation results on published expert-labelled AHU data; they are **not** a prospective field intervention designed by the HVAC-EVI authors.
- The support fraction is a conservative source-coverage diagnostic on the eight common documented sensor concepts. It is **not** a probability of transfer success and has no post-hoc acceptance threshold.
- The source labels are expert/ASHRAE-guideline-based operational annotations. They should not be relabelled as repair-confirmed physical failures unless independent repair confirmation is available.
- Whether the field results strengthen, narrow, or fail to reproduce the Release 1.4.2 pattern must be decided from these prespecified outputs after QA, not by changing the protocol after seeing results.
