# V38 public Hospital source-year and duplicate-label audit

The manuscript **Fault-Class Coverage Versus Unseen-Equipment Diagnosis: Fixed-Budget Training-Unit Selection in 41 Air-Handling Units** audits one provenance discrepancy in its primary 41-AHU study. This code is **not a new model experiment**.

Download the exact original `hosptial_scientific_data.csv` from https://doi.org/10.6084/m9.figshare.27147678.v3 and run:

```bash
pip install -r requirements.txt
python verify_hospital_source_audit.py /path/to/hosptial_scientific_data.csv
```

The independent source audit verifies the exact original source SHA256 and observes **66,048** raw hospital records, eight AHUs, 344 observed calendar dates, and as-exported dates **2023-01-01 to 2023-12-21**. The published Wang data paper describes collection in **2022**; the physical collection year remains **unresolved**. Three-class subset = **64,513** rows, with **161** duplicated AHU/time keys, **322** rows, and **33** key pairs with conflicting expert labels. This audit uses no private data or confidential reviewer links, and does not change prior fixed-model results, infer causal effects, or establish technician-confirmed faults.
