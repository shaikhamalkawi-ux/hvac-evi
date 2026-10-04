# Source audit — Wang real-operational AHU data

## Exact source

Wang, S. (2025). *Real operational labeled data of air handling units from office, auditorium, and hospital buildings*. Scientific Data, 12, 1481. DOI `10.1038/s41597-025-05825-9`.

Dataset DOI: `10.6084/m9.figshare.27147678.v3`.

## Source files

- `auditorium_scientific_data.csv`
- `hosptial_scientific_data.csv` (source spelling)
- `office_scientific_data.csv`
- optional provenance notebook: `FDD_processing.ipynb`

## Label provenance

The descriptor reports that:
- operational/fault labels were created using stated criteria and ASHRAE-informed guidance adapted to installation context;
- four annotators with at least 20 years of HVAC-FDD experience manually labelled data;
- two additional experts cross-checked a randomly selected annotated subset.

This is stronger than unreviewed auto-labelling but is not equivalent to repair-confirmed fault ground truth.

## Data-quality boundary

The source reports annual sensor calibration, missing-data review, duplicate handling, and hourly aggregation from one-minute measurements. The HVAC-EVI external probe retains train-only median imputation for remaining missing values.

## Admission decision

Admitted for:
- equipment-independent evaluation within each building;
- cross-building score comparison on common classes/features;
- source-defined target-support diagnostics.

Not admitted as:
- an author-run field trial;
- repair-confirmed fault validation;
- energy-savings validation;
- operator-action or maintenance-outcome validation.

Raw CSV files are not redistributed in this repository.
