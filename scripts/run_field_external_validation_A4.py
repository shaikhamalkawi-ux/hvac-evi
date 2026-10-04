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
    compact = n.replace(" ", "")
    if compact in {"normal", "normalcondition", "normaloperation"} or n.startswith("normal"):
        return "Normal"
    if compact == "ratsf" or ("return" in n and "air" in n and "temperature" in n and "fault" in n):
        return "RATSF"
    if compact == "satsf" or ("supply" in n and "air" in n and "temperature" in n and "fault" in n):
        return "SATSF"
    if compact == "sff" or ("supply" in n and "fan" in n and "fault" in n):
        return "SFF"
    if compact == "vpf" or ("valve" in n and "position" in n and "fault" in n):
        return "VPF"
    if compact == "cpf" or ("cooling" in n and "pump" in n and "fault" in n):
        return "CPF"
    if compact == "hpf" or ("heating" in n and "pump" in n and "fault" in n):
        return "HPF"
    # The data descriptor refers once to cooling supply temperature fault; preserve
    # it as a documented non-primary label if it appears.
    if "cooling" in n and "supply" in n and "temperature" in n and "fault" in n:
        return "CSTF"
    return None


def resolve_by_alias(columns: Sequence[str], aliases: Sequence[str], role: str, exact_only: bool = False) -> str:
    normalized = {c: norm_col(c) for c in columns}
    alias_norm = [norm_col(a) for a in aliases]
    exact = [c for c, n in normalized.items() if n in alias_norm]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Ambiguous {role}: multiple exact alias matches: {exact}")
    if exact_only:
        raise ValueError(f"No exact alias match for {role}. Columns: {list(columns)}")
    # Token-contained aliases, longest aliases first to reduce accidental matches.
    candidates: List[Tuple[int, str]] = []
    for c, n in normalized.items():
        for a in sorted(alias_norm, key=len, reverse=True):
            atoks = a.split()
            ntoks = n.split()
            if all(t in ntoks for t in atoks):
                candidates.append((len(atoks), c))
                break
    if not candidates:
        raise ValueError(f"Could not resolve {role} from documented aliases. Columns: {list(columns)}")
    max_score = max(s for s, _ in candidates)
    best = sorted({c for s, c in candidates if s == max_score})
    if len(best) != 1:
        raise ValueError(f"Ambiguous {role}: {best}")
    return best[0]


def resolve_sensor_columns(df: pd.DataFrame, excluded: set[str]) -> Dict[str, str]:
    """Resolve the eight cross-building measurement points by exact source names.

    Amendment A1 deliberately disables token/fuzzy matching for sensor variables.
    The first exact alias in each ordered list is selected. This prevents aggregate
    pump metrics or secondary channels from being substituted for the documented
    common measurement points merely because their names contain the same tokens.
    """
    cols = [c for c in df.columns if c not in excluded]
    normalized = {c: norm_col(c) for c in cols}
    mapping: Dict[str, str] = {}
    used: set[str] = set()
    for semantic, aliases in SENSOR_ALIASES.items():
        candidate = None
        for alias in aliases:
            an = norm_col(alias)
            matches = [c for c, n in normalized.items() if n == an]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous exact source mapping for sensor:{semantic} alias={alias!r}: {matches}")
            if len(matches) == 1:
                candidate = matches[0]
                break
        if candidate is None:
            raise ValueError(
                f"Could not resolve sensor:{semantic} by the published cross-building exact aliases {list(aliases)}. "
                f"Columns: {list(cols)}"
            )
        if candidate in used:
            raise ValueError(f"One column mapped to multiple sensor concepts: {candidate}")
        # The source paper reports some missing sensor entries; reject only a column
        # that is not predominantly numeric/coercible rather than requiring completeness.
        numeric = pd.to_numeric(df[candidate], errors="coerce")
        observed = int(numeric.notna().sum())
        if observed == 0:
            raise ValueError(f"Resolved sensor {semantic!r} -> {candidate!r}, but it has no observed numeric values.")
        mapping[semantic] = candidate
        used.add(candidate)
    return mapping

def fixed_model(seed: int = SEED) -> Pipeline:
    # Exact DecisionTree probe inherited from the locked HVAC-EVI computational core,
    # with the current matched-reanalysis seed.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def fixed_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    rec = recall_score(yt, yp, labels=PRIMARY_LABELS, average=None, zero_division=0)
    f1c = f1_score(yt, yp, labels=PRIMARY_LABELS, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(f1c)),
        "balanced_accuracy": float(np.mean(rec)),
        "min_class_recall": float(np.min(rec)),
        "mean_class_recall": float(np.mean(rec)),
        "accuracy": float(accuracy_score(yt, yp)),
    }


def class_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    pre = precision_score(yt, yp, labels=PRIMARY_LABELS, average=None, zero_division=0)
    rec = recall_score(yt, yp, labels=PRIMARY_LABELS, average=None, zero_division=0)
    f1c = f1_score(yt, yp, labels=PRIMARY_LABELS, average=None, zero_division=0)
    support = pd.Series(yt).value_counts()
    return pd.DataFrame({
        "class_label": PRIMARY_LABELS,
        "support": [int(support.get(l, 0)) for l in PRIMARY_LABELS],
        "precision": pre.astype(float),
        "recall": rec.astype(float),
        "f1": f1c.astype(float),
    })


def run_oof(X: pd.DataFrame, y: pd.Series, groups: pd.Series, design: str, seed: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(str)
    groups = groups.reset_index(drop=True).astype(str)
    if design == "row_wise":
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    elif design == "ahu_disjoint":
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        raise ValueError(design)

    pred = np.empty(len(y), dtype=object)
    filled = np.zeros(len(y), dtype=bool)
    fold_log: List[Dict[str, Any]] = []
    for fold, (tr, te) in enumerate(split_iter, 1):
        model = fixed_model(seed)
        train_labels = sorted(pd.Series(y.iloc[tr]).unique().tolist())
        test_labels = sorted(pd.Series(y.iloc[te]).unique().tolist())
        missing_train = sorted(set(PRIMARY_LABELS) - set(train_labels))
        model.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = model.predict(X.iloc[te]).astype(str)
        filled[te] = True
        fold_log.append({
            "fold": fold,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_ahus": int(groups.iloc[tr].nunique()),
            "test_ahus": int(groups.iloc[te].nunique()),
            "train_labels": train_labels,
            "test_labels": test_labels,
            "missing_primary_labels_from_train": missing_train,
            "test_ahu_values": sorted(groups.iloc[te].unique().tolist()),
        })
    if not filled.all():
        raise RuntimeError(f"OOF design {design} did not fill all predictions")
    return pred.astype(str), fold_log


def support_diagnostic(source: pd.DataFrame, target: pd.DataFrame, features: Sequence[str], loq: float, hiq: float) -> Dict[str, Any]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    tr = imp.fit_transform(source[list(features)])
    te = imp.transform(target[list(features)])
    if np.isnan(tr).all(axis=0).any():
        bad = [features[i] for i, b in enumerate(np.isnan(tr).all(axis=0)) if b]
        raise ValueError(f"All-missing source features after coercion: {bad}")
    lo = np.quantile(tr, loq, axis=0) if loq > 0 else np.min(tr, axis=0)
    hi = np.quantile(tr, hiq, axis=0) if hiq < 1 else np.max(tr, axis=0)
    inside = (te >= lo) & (te <= hi)
    violations = (~inside).sum(axis=1)
    return {
        "N": int(len(te)),
        "supported": int((violations == 0).sum()),
        "support_fraction": float((violations == 0).mean()),
        "n0": int((violations == 0).sum()),
        "n1": int((violations == 1).sum()),
        "n2": int((violations == 2).sum()),
        "n3_5": int(((violations >= 3) & (violations <= 5)).sum()),
        "nGT5": int((violations > 5).sum()),
        "median_violations": float(np.median(violations)),
        "mean_violations": float(np.mean(violations)),
        "q25_violations": float(np.quantile(violations, 0.25)),
        "q75_violations": float(np.quantile(violations, 0.75)),
        "feature_failure_counts": {features[j]: int((~inside[:, j]).sum()) for j in range(len(features))},
    }


def write_manifest(root: Path, manifest_path: Path, exclude: Iterable[Path] = ()) -> None:
    excluded = {p.resolve() for p in exclude}
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.resolve() in excluded:
            continue
        rows.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def make_result_zip(result_root: Path, zip_path: Path) -> None:
    # Explicitly exclude the raw source directory. Source file hashes and metadata
    # remain in the package for provenance without redistributing the CSVs.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(result_root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(result_root)
            if rel.parts and rel.parts[0] == "raw_source_not_for_redistribution":
                continue
            if p == zip_path:
                continue
            zf.write(p, rel.as_posix())


def blocker(result_root: Path, stage: str, message: str, details: Any = None) -> None:
    obj = {
        "protocol_id": PROTOCOL_ID,
        "protocol_amendment": AMENDMENT_ID,
        "status": "BLOCKED",
        "stage": stage,
        "message": message,
        "details": details,
    }
    json_dump(obj, result_root / "BLOCKER_REPORT.json")
    raise RuntimeError(f"{stage}: {message}")

# -----------------------------------------------------------------------------
# Figshare download / source audit
# -----------------------------------------------------------------------------
def fetch_figshare_metadata() -> Tuple[Dict[str, Any], str]:
    urls = [
        f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{ARTICLE_VERSION}",
        f"https://api.figshare.com/v2/articles/{ARTICLE_ID}",
    ]
    errors = []
    for url in urls:
        try:
            r = requests.get(url, timeout=60)
            if r.ok:
                meta = r.json()
                version = meta.get("version")
                # The version-specific endpoint may omit a numeric 'version'; if the
                # generic endpoint reports a version, it must equal the protocol lock.
                if url.endswith(str(ARTICLE_ID)) and version is not None and int(version) != ARTICLE_VERSION:
                    errors.append(f"{url}: latest version is {version}, protocol requires v{ARTICLE_VERSION}")
                    continue
                return meta, url
            errors.append(f"{url}: HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not retrieve exact Figshare metadata: " + " | ".join(errors))


def download_source(raw_dir: Path) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta, api_url = fetch_figshare_metadata()
    files = meta.get("files", [])
    by_name = {f.get("name"): f for f in files if f.get("name")}
    missing = [name for name in EXPECTED_FILES.values() if name not in by_name]
    if missing:
        raise RuntimeError(f"Exact v{ARTICLE_VERSION} metadata is missing expected source files: {missing}; available={list(by_name)}")
    selected_names = list(EXPECTED_FILES.values()) + [n for n in OPTIONAL_PROVENANCE_FILES if n in by_name]
    downloaded = []
    for name in selected_names:
        fmeta = by_name[name]
        url = fmeta.get("download_url")
        if not url:
            raise RuntimeError(f"No download_url for {name}")
        path = raw_dir / name
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with path.open("wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        downloaded.append({
            "name": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "figshare_file_id": fmeta.get("id"),
            "figshare_reported_size": fmeta.get("size"),
            "figshare_computed_md5": fmeta.get("computed_md5") or fmeta.get("supplied_md5"),
            "download_url": url,
        })
    return {
        "protocol_article_id": ARTICLE_ID,
        "protocol_version": ARTICLE_VERSION,
        "data_doi": DATA_DOI,
        "source_article_doi": SOURCE_ARTICLE_DOI,
        "figshare_api_url_used": api_url,
        "metadata_