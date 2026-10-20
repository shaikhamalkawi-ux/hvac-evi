# HVAC-EVI V37 — Time-Matched Donor Replacement (Public Audit)

Companion **source-independent code and aggregate-result** release for:

*Fault-Class Coverage Versus Unseen-Equipment Diagnosis: Fixed-Budget Training-Unit Selection in 41 Air-Handling Units* (research revision V37; not a published article).

Authors (paper): Ghassan Malkawi, Ahmed Elsayed, Mohammed Alhagyan, and Bakeel Hussein.

## Scientific scope

The primary observational cohort was 307,375 expert-labelled hourly records from 41 air-handling units (AHUs) in one auditorium, one hospital and one office building, with Normal, RATSF and SFF labels. The **added V37 sensitivity** uses a fixed 384-row, 128-per-class training budget and tests replacing 64 records per class from one source AHU with 64 same-time-key labelled records from a second source AHU, retaining 64 distinct records from the original source. Two fixed models and ten paired allocations per eligible target AHU were executed. No prospective data or independent building is introduced.

Time-key designs:

- **Exact calendar date and hour:** 13/13 auditorium, 7/8 hospital, 20/20 office targets. Hospital **N06 is ineligible** for the strict RATSF overlap gate.
- **Year-month plus hour of day:** all 41 targets are eligible, but calendar dates are not matched.

Executed evidence: 810 paired allocations, 3,240 model fits. Independently archived checks reported 810 allocation audits, 2,430 per-class time-histogram comparisons, 9,720 prediction-cell checks and 48 full model refits. The six building-by-model mean differences in class-and-AHU-standardised recall are positive for **each** time-key design, but individual class recalls sometimes decrease. Strict hospital mean differences exclude the dominant RATSF AHU; at high support thresholds, some hospital results become undefined or reverse sign.

These are retrospective within-building source replacements. Matching clock times does **not** isolate the causal effect of equipment identity; the selected AHU-day representation still changes. There is no independent repair-adjudicated ground truth, prospective clinical/field deployment, maintenance cost or energy savings result.

## Directory guide

- `protocol_calendar_matched.json` — frozen *internal* analysis protocol; **not externally preregistered**.
- `code/` — time-matched allocation and diagnostic scripts. `verify_public_tables.py` runs independently on the public aggregate tables.
- `results/` — aggregate Q, class components, pooled macro-F1, threshold and omit-one-AHU summaries, **not source-level time series or predictions**.
- `schema_gate/` — metadata-only raw source column mapping, class counts and audit records.
- `qa/` — stored independent test counters and reported source SHA-256 identifiers.
- `SHA256SUMS.txt` — integrity manifest for this exact public directory.

## How to audit without the private research inputs

From the repository root, run:

```bash
python v37_time_matched/code/verify_public_tables.py
```

This **recalculates cross-file aggregate identities and checks integrity** using Python's standard library. It does **not** re-execute the 3,240 fits, reconstruct the original feature matrices, or independently substantiate the recorded model results.

## Full-fit reproduction boundary

The other scripts in `code/` are archived research execution code, and currently reference the frozen V25 package structure and its historical `source_loader_A4` module, which are **not** included here. They are provided for method inspection, not advertised as a self-contained out-of-the-box full replay. The exact frozen V25 input archive needed for the original independent verifier is `HVAC_EVI_V25_Executed_Robustness_Evidence_20261011.zip`, SHA-256 `eb46d628ed2e4fb5d3c425e7beafdf2609c1c98c5b32c30eca1217fdc1db3652` (not redistributed). Raw source: Wang Figshare v3, DOI [10.6084/m9.figshare.27147678.v3](https://doi.org/10.6084/m9.figshare.27147678.v3). Per-record selected indices, detailed predictions and third-party raw datasets are **not published here** while applicable redistribution rights are being audited.

A separate **unpublished** Zenodo draft may be shared privately by the corresponding author with editors/reviewers; its preview token does not appear anywhere in this public repository. This public GitHub material supplements but does not replace the journal reviewer package.

## Citation and license

Cite the study by its four-author research title and this repository (no manuscript publication DOI has been assigned). Original repository code/docs are under the existing MIT license; third-party data remain governed by source-specific terms.
