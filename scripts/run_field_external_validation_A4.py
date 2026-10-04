#!/usr/bin/env python3
"""Prospectively specified external field validation for HVAC-EVI.

This script downloads the exact Wang Figshare v3 dataset (article 27147678),
performs a schema/data gate, and ONLY if the gate passes runs the predeclared
field validation analyses. It does not tune models, choose thresholds from
results, or alter the locked HVAC-EVI Release 1.4.2 baseline.

Primary analyses
----------------
1) Within-building equipment-independence contrast:
   row-wise stratified 4-fold OOF versus AHU-disjoint stratified-group 4-fold OOF,
   using a fixed DecisionTree probe and a fixed 3-class universe
   {Normal, RATSF, SFF}.
2) Cross-building transfer/support:
   six ordered building pairs, fixed DecisionTree trained on all source rows,
   tested on all target rows, plus source-defined target-support fractions under
   min-max, 1st-99th, and 5th-95th bounds on the same common sensor features.

Secondary analysis
------------------
20-seed split sensitivity for the within-building row/group contrast. This is
predeclared to characterize fold randomness only; it is not a separate claim.

Raw Figshare CSVs are NOT redistributed in the result ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# Immutable protocol constants
# -----------------------------------------------------------------------------
PROTOCOL_ID = "HVAC_EVI_FieldExternalValidation_Protocol01"
RUNNER_REVISION = "A4_RAW_HEADER_EXACT_ALIAS"
AMENDMENT_ID = "A2_v3_common_feature_intersection"
ARTICLE_ID = 27147678
ARTICLE_VERSION = 3
DATA_DOI = "10.6084/m9.figshare.27147678.v3"
SOURCE_ARTICLE_DOI = "10.1038/s41597-025-05825-9"
SEED = 20260802
N_FOLDS = 4
SECONDARY_SEEDS = list(range(SEED, SEED + 20))
PRIMARY_LABELS = ["Normal", "RATSF", "SFF"]
BUILDINGS = ["auditorium", "hospital", "office"]
EXPECTED_AHU_COUNTS = {"auditorium": 13, "hospital": 8, "office": 20}
EXPECTED_FILES = {
    "auditorium": "auditorium_scientific_data.csv",
    "hospital": "hosptial_scientific_data.csv",  # source filename contains this typo
    "office": "office_scientific_data.csv",
}
OPTIONAL_PROVENANCE_FILES = {"FDD_processing.ipynb"}
BOUNDS = [
    ("minmax", 0.00, 1.00),
    ("p01_p99", 0.01, 0.99),
    ("p05_p95", 0.05, 0.95),
]
BASELINE_STATE_LOCK = {
    "scientific_baseline": "HVAC-EVI Release 1.4.2 — Energy and Buildings Findings-First Final Micro-Closure",
    "main_pdf_sha256": "6d41f154324dd114794948bd62937c89ed612346e96a1033b8ea6fd6684a3f70",
    "supplement_pdf_sha256": "102e9d8e05160fa7e1eec130010951aaa0725b8da8c94fc1a3ff3312e9d330c7",
    "evidence_register_sha256": "3a931eba9eedd59a4f908c9ab9226fc8df365b01657a5868845c4a3caeed1f8e",
    "journal_package_sha256": "fe9ffed61ded6c4ff1152d3c2a01e119c9e28782d7cc9c44d4e4866687bce709",
    "reproducibility_package_sha256": "f8d03e1e51ceaefd23fde9d72481b17fd9e760696e14e935f50574a032115cc1",
    "locked_findings": {
        "ORNL_DecisionTree_row_macro_f1": 0.835,
        "ORNL_DecisionTree_complete_day_macro_f1": 0.475,
        "ORNL_grouped_rmin_approx": 0.005,
        "Boiler_all22_macro_f1_approx": 0.758,
        "Boiler_basic15_macro_f1_approx": 0.748,
        "Boiler_all22_rmin_approx": 0.342,
        "Boiler_basic15_rmin_approx": 0.113,
        "active_empirical_cards": 100,
        "active_status_counts": {"Admitted": 2, "Reportable under stated boundary": 92, "Hold": 6, "Remove": 0},
        "lineage_records": 151,
        "all_transfer_cards": "Hold",
    },
}

# Protocol Amendment A2: the exact Figshare v3 hospital CSV does not expose a
# heating-pump column. Because the analysis must not equate the hospital column
# "Heat Exchanger" with "Heating pump 1" without source evidence, the common
# cross-building feature universe is the exact eight-feature intersection below.
# This correction was made after a schema blocker and before any model fitting or
# result inspection. Features are source-defined, not selected by predictive performance.
SENSOR_ALIASES: Mapping[str, Sequence[str]] = {
    "set_point_temperature": ["set point temperature"],
    "return_air_temperature": ["return air temperature", "return temperature"],
    "supply_air_temperature": ["supply air temperature"],
    "supply_fan": ["supply fan speed", "supply fan"],
    "valve_position": ["valve position"],
    "heating_supply_temperature": ["heating supply temperature 1", "heating supply temperature1", "heating supply temperature"],
    "cooling_supply_temperature": ["cooling supply temperature 1", "cooling supply temperature"],
    "cooling_pump": ["cooling pump 1", "cooling pump"],
}

EXCLUDED_SCHEMA_SEMANTICS = {
    "heating_pump": {
        "reason": "Exact Figshare v3 hospital CSV has no heating-pump column; 'Heat Exchanger' is not treated as a synonym without source evidence.",
        "status": "excluded_before_results_due_to_source_schema_mismatch",
    }
}

LABEL_COLUMN_ALIASES = [
    "labeling", "label", "labels", "target", "target class", "target_class", "class",
    "fault", "fault class", "fault_class", "condition", "operational condition",
    "operation condition", "operational_condition", "operation_condition",
]
AHU_COLUMN_ALIASES = [
    "ahu", "ahu id", "ahu_id", "ahu name", "ahu_name", "ahu number",
    "ahu_number", "ahu no", "ahu_no", "air handling unit",
]
TIME_COLUMN_ALIASES = [
    "timestamp", "datetime", "date time", "date_time", "time", "date",
]

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def norm_text(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_col(x: Any) -> str:
    return norm_text(x)


def canonical_label(x: Any) -> str | None:
    n = norm_text(x)
    if not n or n == "nan":
        return None
 