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
        "metadata_title": meta.get("title"),
        "metadata_version": meta.get("version"),
        "metadata_published_date": meta.get("published_date"),
        "metadata_modified_date": meta.get("modified_date"),
        "metadata_license": meta.get("license"),
        "metadata_defined_type_name": meta.get("defined_type_name"),
        "downloaded_files": downloaded,
    }

# -----------------------------------------------------------------------------
# Data gate and analyses
# -----------------------------------------------------------------------------
def load_and_gate(raw_dir: Path, result_root: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]], List[str]]:
    datasets: Dict[str, pd.DataFrame] = {}
    schemas: Dict[str, Dict[str, Any]] = {}
    resolved_semantics_by_building: Dict[str, Dict[str, str]] = {}
    inventory_rows = []

    for building in BUILDINGS:
        path = raw_dir / EXPECTED_FILES[building]
        if not path.exists():
            blocker(result_root, "source_files", f"Missing local source file {path.name}")
        df = pd.read_csv(path, low_memory=False)
        if len(df) == 0:
            blocker(result_root, "schema_gate", f"{building}: CSV is empty")
        columns = list(df.columns)

        try:
            label_col = resolve_by_alias(columns, LABEL_COLUMN_ALIASES, f"{building}:label")
            ahu_col = resolve_by_alias(columns, AHU_COLUMN_ALIASES, f"{building}:AHU")
        except Exception as e:
            blocker(result_root, "schema_gate", f"{building}: could not resolve label/AHU columns", {"error": str(e), "columns": columns})

        time_col = None
        try:
            time_col = resolve_by_alias(columns, TIME_COLUMN_ALIASES, f"{building}:time")
        except Exception:
            # Timestamp is useful provenance but not required for the prespecified
            # equipment-independence or cross-building analyses.
            time_col = None

        excluded = {label_col, ahu_col}
        if time_col:
            excluded.add(time_col)
        try:
            sensor_map = resolve_sensor_columns(df, excluded)
        except Exception as e:
            blocker(result_root, "schema_gate", f"{building}: common documented sensor mapping failed", {"error": str(e), "columns": columns})

        # Canonical labels are defined from source terminology, not from model performance.
        canonical = df[label_col].map(canonical_label)
        raw_labels = df[label_col].astype(str)
        unmapped_examples = sorted(raw_labels[canonical.isna()].dropna().unique().tolist())[:30]
        df = df.copy()
        df["__label_canonical__"] = canonical
        df["__ahu_canonical__"] = df[ahu_col].astype(str).str.strip()
        for semantic, col in sensor_map.items():
            df[f"__sensor__{semantic}"] = pd.to_numeric(df[col], errors="coerce")

        # Gate on documented number of independent AHU units.
        n_ahu = int(df["__ahu_canonical__"].nunique())
        if n_ahu != EXPECTED_AHU_COUNTS[building]:
            blocker(
                result_root,
                "schema_gate",
                f"{building}: resolved AHU count {n_ahu} does not match published count {EXPECTED_AHU_COUNTS[building]}",
                {"ahu_col": ahu_col, "ahu_values": sorted(df["__ahu_canonical__"].unique().tolist())},
            )

        primary = df[df["__label_canonical__"].isin(PRIMARY_LABELS)].copy()
        counts = primary["__label_canonical__"].value_counts().to_dict()
        missing_primary = [l for l in PRIMARY_LABELS if int(counts.get(l, 0)) == 0]
        if missing_primary:
            blocker(result_root, "class_gate", f"{building}: primary source-defined labels missing: {missing_primary}", counts)

        # Every common sensor must contain observed numeric information in every building.
        all_missing = []
        for semantic in SENSOR_ALIASES:
            c = f"__sensor__{semantic}"
            if primary[c].notna().sum() == 0:
                all_missing.append(semantic)
        if all_missing:
            blocker(result_root, "feature_gate", f"{building}: primary data have all-missing common sensors: {all_missing}")

        datasets[building] = primary
        resolved_semantics_by_building[building] = sensor_map
        schemas[building] = {
            "source_file": EXPECTED_FILES[building],
            "raw_rows_loaded": int(len(df)),
            "primary_three_class_rows": int(len(primary)),
            "raw_column_count": int(len(columns)),
            "raw_columns": columns,
            "label_column": label_col,
            "ahu_column": ahu_col,
            "time_column": time_col,
            "resolved_sensor_columns": sensor_map,
            "resolved_ahu_count": n_ahu,
            "canonical_primary_class_counts": {l: int(counts.get(l, 0)) for l in PRIMARY_LABELS},
            "canonical_all_label_counts": {str(k): int(v) for k, v in df["__label_canonical__"].value_counts(dropna=False).to_dict().items()},
            "unmapped_raw_label_examples": unmapped_examples,
            "primary_missing_fraction_by_sensor": {
                semantic: float(primary[f"__sensor__{semantic}"].isna().mean()) for semantic in SENSOR_ALIASES
            },
        }
        for c in columns:
            inventory_rows.append({
                "building": building,
                "column": c,
                "normalized_column": norm_col(c),
                "dtype": str(df[c].dtype),
                "missing_fraction": float(df[c].isna().mean()),
                "n_unique": int(df[c].nunique(dropna=True)),
            })

    # The semantic feature universe is fixed by the eight source-documented common
    # sensor concepts; per-building raw names may differ but map to these same concepts.
    common_features = [f"__sensor__{semantic}" for semantic in SENSOR_ALIASES]

    pd.DataFrame(inventory_rows).to_csv(result_root / "schema_column_inventory.csv", index=False)
    json_dump(schemas, result_root / "schema_audit.json")
    json_dump(resolved_semantics_by_building, result_root / "resolved_sensor_mapping.json")
    pd.DataFrame([
        {"building": b, "class_label": l, "count": int((datasets[b]["__label_canonical__"] == l).sum())}
        for b in BUILDINGS for l in PRIMARY_LABELS
    ]).to_csv(result_root / "primary_class_counts.csv", index=False)

    feature_rows = []
    for semantic in SENSOR_ALIASES:
        row = {"semantic_feature": semantic}
        for b in BUILDINGS:
            row[b] = resolved_semantics_by_building[b][semantic]
        feature_rows.append(row)
    pd.DataFrame(feature_rows).to_csv(result_root / "common_feature_mapping.csv", index=False)
    return datasets, schemas, common_features


def within_building_analysis(datasets: Dict[str, pd.DataFrame], features: Sequence[str], result_root: Path, run_multiseed: bool) -> Dict[str, Any]:
    summary_rows = []
    class_rows = []
    fold_records: Dict[str, Any] = {}
    prediction_rows = []
    local_group_macro: Dict[str, float] = {}

    for building in BUILDINGS:
        df = datasets[building].reset_index(drop=True)
        X = df[list(features)]
        y = df["__label_canonical__"].astype(str)
        g = df["__ahu_canonical__"].astype(str)
        for design in ["row_wise", "ahu_disjoint"]:
            pred, fold_log = run_oof(X, y, g, design, SEED)
            met = fixed_metrics(y, pred)
            summary_rows.append({"building": building, "design": design, "seed": SEED, **met})
            cm = class_metrics(y, pred)
            cm.insert(0, "building", building)
            cm.insert(1, "design", design)
            class_rows.extend(cm.to_dict("records"))
            fold_records[f"{building}/{design}/seed_{SEED}"] = fold_log
            if design == "ahu_disjoint":
                local_group_macro[building] = met["macro_f1"]
            prediction_rows.extend([
                {
                    "building": building,
                    "design": design,
                    "row_index_within_source_csv": int(i),
                    "ahu": str(g.iloc[i]),
                    "true_label": str(y.iloc[i]),
                    "predicted_label": str(pred[i]),
                }
                for i in range(len(df))
            ])

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(result_root / "within_building_primary_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(result_root / "within_building_primary_class_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(result_root / "within_building_primary_oof_predictions.csv", index=False)
    json_dump(fold_records, result_root / "within_building_primary_fold_log.json")

    # Matched row-minus-group contrasts on the same fixed model / seed / folds count.
    contrast = []
    for b in BUILDINGS:
        r = summ[(summ.building == b) & (summ.design == "row_wise")].iloc[0]
        g = summ[(summ.building == b) & (summ.design == "ahu_disjoint")].iloc[0]
        contrast.append({
            "building": b,
            "row_macro_f1": float(r.macro_f1),
            "ahu_disjoint_macro_f1": float(g.macro_f1),
            "delta_row_minus_ahu_disjoint_macro_f1": float(r.macro_f1 - g.macro_f1),
            "row_min_class_recall": float(r.min_class_recall),
            "ahu_disjoint_min_class_recall": float(g.min_class_recall),
            "delta_row_minus_ahu_disjoint_min_recall": float(r.min_class_recall - g.min_class_recall),
        })
    pd.DataFrame(contrast).to_csv(result_root / "within_building_primary_contrasts.csv", index=False)

    multiseed_rows = []
    if run_multiseed:
        for seed in SECONDARY_SEEDS:
            for building in BUILDINGS:
                df = datasets[building].reset_index(drop=True)
                X = df[list(features)]
                y = df["__label_canonical__"].astype(str)
                gr = df["__ahu_canonical__"].astype(str)
                m = {}
                for design in ["row_wise", "ahu_disjoint"]:
                    pred, _ = run_oof(X, y, gr, design, seed)
                    m[design] = fixed_metrics(y, pred)
                multiseed_rows.append({
                    "building": building,
                    "seed": seed,
                    "row_macro_f1": m["row_wise"]["macro_f1"],
                    "ahu_disjoint_macro_f1": m["ahu_disjoint"]["macro_f1"],
                    "delta_row_minus_ahu_disjoint_macro_f1": m["row_wise"]["macro_f1"] - m["ahu_disjoint"]["macro_f1"],
                    "row_min_class_recall": m["row_wise"]["min_class_recall"],
                    "ahu_disjoint_min_class_recall": m["ahu_disjoint"]["min_class_recall"],
                    "delta_row_minus_ahu_disjoint_min_recall": m["row_wise"]["min_class_recall"] - m["ahu_disjoint"]["min_class_recall"],
                })
        pd.DataFrame(multiseed_rows).to_csv(result_root / "within_building_20seed_split_sensitivity.csv", index=False)
        ms = pd.DataFrame(multiseed_rows)
        ms_summary = ms.groupby("building").agg(
            n_seeds=("seed", "count"),
            median_delta_macro_f1=("delta_row_minus_ahu_disjoint_macro_f1", "median"),
            min_delta_macro_f1=("delta_row_minus_ahu_disjoint_macro_f1", "min"),
            max_delta_macro_f1=("delta_row_minus_ahu_disjoint_macro_f1", "max"),
            median_delta_min_recall=("delta_row_minus_ahu_disjoint_min_recall", "median"),
            min_delta_min_recall=("delta_row_minus_ahu_disjoint_min_recall", "min"),
            max_delta_min_recall=("delta_row_minus_ahu_disjoint_min_recall", "max"),
        ).reset_index()
        ms_summary.to_csv(result_root / "within_building_20seed_sensitivity_summary.csv", index=False)

    # Primary figure: show measured score under row-wise and AHU-disjoint OOF.
    c = pd.DataFrame(contrast)
    x = np.arange(len(BUILDINGS)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width/2, c["row_macro_f1"], width, label="Row-wise 4-fold OOF")
    ax.bar(x + width/2, c["ahu_disjoint_macro_f1"], width, label="AHU-disjoint 4-fold OOF")
    ax.set_xticks(x); ax.set_xticklabels([b.title() for b in BUILDINGS])
    ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1")
    ax.set_title("External field validation: row-wise versus equipment-independent evaluation")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(result_root / "Figure_Field_Independence_Contrast.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return {"local_ahu_disjoint_macro_f1": local_group_macro}


def cross_building_analysis(datasets: Dict[str, pd.DataFrame], features: Sequence[str], result_root: Path, local_group_macro: Mapping[str, float]) -> None:
    transfer_rows = []
    class_rows = []
    support_rows = []
    support_details: Dict[str, Any] = {}

    for source in BUILDINGS:
        src = datasets[source].reset_index(drop=True)
        Xs = src[list(features)]
        ys = src["__label_canonical__"].astype(str)
        for target in BUILDINGS:
            if source == target:
                continue
            tgt = datasets[target].reset_index(drop=True)
            Xt = tgt[list(features)]
            yt = tgt["__label_canonical__"].astype(str)
            model = fixed_model(SEED)
            model.fit(Xs, ys)
            pred = model.predict(Xt).astype(str)
            met = fixed_metrics(yt, pred)
            direction = f"{source}->{target}"
            transfer_rows.append({
                "source": source,
                "target": target,
                "direction": direction,
                "target_rows": int(len(tgt)),
                **met,
                "target_local_ahu_disjoint_macro_f1": float(local_group_macro[target]),
                "transfer_minus_target_local_group_macro_f1": float(met["macro_f1"] - local_group_macro[target]),
            })
            cm = class_metrics(yt, pred)
            cm.insert(0, "source", source); cm.insert(1, "target", target); cm.insert(2, "direction", direction)
            class_rows.extend(cm.to_dict("records"))

            for bn, loq, hiq in BOUNDS:
                r = support_diagnostic(src, tgt, features, loq, hiq)
                support_rows.append({
                    "source": source,
                    "target": target,
                    "direction": direction,
                    "bounds": bn,
                    **{k: v for k, v in r.items() if k != "feature_failure_counts"},
                })
                support_details[f"{direction}/{bn}"] = r

    transfer = pd.DataFrame(transfer_rows)
    support = pd.DataFrame(support_rows)
    transfer.to_csv(result_root / "cross_building_transfer_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(result_root / "cross_building_transfer_class_metrics.csv", index=False)
    support.to_csv(result_root / "cross_building_support_metrics.csv", index=False)
    json_dump(support_details, result_root / "cross_building_support_details.json")

    merged = transfer.merge(
        support[support.bounds == "minmax"][["direction", "support_fraction", "supported", "N", "median_violations", "nGT5"]],
        on="direction",
        how="left",
    )
    merged.to_csv(result_root / "cross_building_transfer_vs_minmax_support.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for _, r in merged.iterrows():
        ax.scatter(r.support_fraction, r.macro_f1, s=60)
        ax.annotate(r.direction, (r.support_fraction, r.macro_f1), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    ax.set_xlabel("Target support fraction under source min–max bounds")
    ax.set_ylabel("Cross-building target macro-F1")
    ax.set_title("Cross-building score and target support answer different questions")
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(result_root / "Figure_CrossBuilding_Score_vs_Support.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_result_summary(result_root: Path, run_multiseed: bool) -> None:
    within = pd.read_csv(result_root / "within_building_primary_contrasts.csv")
    transfer = pd.read_csv(result_root / "cross_building_transfer_metrics.csv")
    support = pd.read_csv(result_root / "cross_building_support_metrics.csv")
    class_counts = pd.read_csv(result_root / "primary_class_counts.csv")

    lines = [
        f"# {PROTOCOL_ID} ({AMENDMENT_ID}) — observed result summary",
        "",
        "This file is generated from the prespecified protocol. It reports observations only; it does not create a post-hoc pass/fail threshold.",
        "",
        "## Data admitted by the schema gate",
        "",
    ]
    for b in BUILDINGS:
        sub = class_counts[class_counts.building == b]
        counts = ", ".join(f"{r.class_label}={int(r['count'])}" for _, r in sub.iterrows())
        lines.append(f"- **{b.title()}**: {counts}.")
    lines += ["", "## Equipment-independence contrast", ""]
    for _, r in within.iterrows():
        lines.append(
            f"- **{r.building.title()}**: row-wise macro-F1={r.row_macro_f1:.4f}; "
            f"AHU-disjoint macro-F1={r.ahu_disjoint_macro_f1:.4f}; "
            f"row-minus-AHU-disjoint Δ={r.delta_row_minus_ahu_disjoint_macro_f1:+.4f}. "
            f"Minimum-class recall: {r.row_min_class_recall:.4f} vs {r.ahu_disjoint_min_class_recall:.4f}."
        )
    if run_multiseed and (result_root / "within_building_20seed_sensitivity_summary.csv").exists():
        lines += ["", "The 20-seed split-sensitivity file characterizes fold randomness; it is secondary support, not a separate contribution."]

    lines += ["", "## Cross-building transfer and source-defined target support", ""]
    minmax = support[support.bounds == "minmax"]
    p01 = support[support.bounds == "p01_p99"]
    p05 = support[support.bounds == "p05_p95"]
    for _, t in transfer.iterrows():
        d = t.direction
        a = minmax[minmax.direction == d].iloc[0]
        b = p01[p01.direction == d].iloc[0]
        c = p05[p05.direction == d].iloc[0]
        lines.append(
            f"- **{d}**: target macro-F1={t.macro_f1:.4f}; minimum-class recall={t.min_class_recall:.4f}; "
            f"supported target rows={int(a.supported)}/{int(a.N)} (min–max), "
            f"{int(b.supported)}/{int(b.N)} (1st–99th), "
            f"{int(c.supported)}/{int(c.N)} (5th–95th)."
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- These are independent real-operational external validation results on published expert-labelled AHU data; they are **not** a prospective field intervention designed by the HVAC-EVI authors.",
        "- The support fraction is a conservative source-coverage diagnostic on the eight common documented sensor concepts. It is **not** a probability of transfer success and has no post-hoc acceptance threshold.",
        "- The source labels are expert/ASHRAE-guideline-based operational annotations. They should not be relabelled as repair-confirmed physical failures unless independent repair confirmation is available.",
        "- Whether the field results strengthen, narrow, or fail to reproduce the Release 1.4.2 pattern must be decided from these prespecified outputs after QA, not by changing the protocol after seeing results.",
        "",
    ]
    (result_root / "RESULT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    result_root = Path(args.output_dir).resolve()
    if result_root.exists():
        shutil.rmtree(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    raw_dir = result_root / "raw_source_not_for_redistribution"
    raw_dir.mkdir(parents=True, exist_ok=True)

    json_dump({
        "protocol_id": PROTOCOL_ID,
        "protocol_amendment": AMENDMENT_ID,
        "protocol_status": "PRESPECIFIED_ANALYSIS_WITH_PRE_RESULT_SCHEMA_AMENDMENT",
        "article_id": ARTICLE_ID,
        "article_version": ARTICLE_VERSION,
        "data_doi": DATA_DOI,
        "source_article_doi": SOURCE_ARTICLE_DOI,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "secondary_seeds": SECONDARY_SEEDS,
        "primary_labels": PRIMARY_LABELS,
        "primary_feature_semantics": list(SENSOR_ALIASES),
        "excluded_schema_semantics": EXCLUDED_SCHEMA_SEMANTICS,
        "model": {
            "pipeline": ["SimpleImputer(strategy=median, keep_empty_features=True)", "DecisionTreeClassifier"],
            "max_depth": 10,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "random_state": SEED,
        },
        "support_bounds": [x[0] for x in BOUNDS],
        "baseline_state_lock": BASELINE_STATE_LOCK,
        "raw_data_redistribution": "excluded_from_result_zip",
    }, result_root / "STATE_LOCK.json")

    try:
        if args.local_data_dir:
            local = Path(args.local_data_dir).resolve()
            for name in EXPECTED_FILES.values():
                src = local / name
                if not src.exists():
                    blocker(result_root, "local_source", f"Missing local source file: {src}")
                shutil.copy2(src, raw_dir / name)
            prov = {
                "mode": "local_data_dir",
                "local_data_dir": str(local),
                "data_doi_expected": DATA_DOI,
                "downloaded_files": [
                    {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in sorted(raw_dir.glob("*.csv"))
                ],
            }
        else:
            prov = download_source(raw_dir)
        json_dump(prov, result_root / "SOURCE_PROVENANCE.json")

        datasets, schemas, common_features = load_and_gate(raw_dir, result_root)
        gate = {
            "status": "PASS",
            "buildings": BUILDINGS,
            "resolved_ahu_counts": {b: int(datasets[b]["__ahu_canonical__"].nunique()) for b in BUILDINGS},
            "primary_labels": PRIMARY_LABELS,
            "common_sensor_semantics": [f.replace("__sensor__", "") for f in common_features],
            "excluded_schema_semantics": EXCLUDED_SCHEMA_SEMANTICS,
        }
        json_dump(gate, result_root / "DATA_GATE_RESULT.json")

        within = within_building_analysis(datasets, common_features, result_root, run_multiseed=not args.no_multiseed)
        cross_building_analysis(datasets, common_features, result_root, within["local_ahu_disjoint_macro_f1"])
        write_result_summary(result_root, run_multiseed=not args.no_multiseed)

        # Compact machine-readable completion record.
        json_dump({
            "protocol_id": PROTOCOL_ID,
            "protocol_amendment": AMENDMENT_ID,
            "runner_revision": RUNNER_REVISION,
            "status": "COMPLETE",
            "data_gate": "PASS",
            "scientific_baseline_changed": False,
            "raw_source_included_in_result_zip": False,
            "next_decision": "Independent QA must decide whether results justify opening HVAC-EVI Release 1.5; do not update the manuscript automatically.",
        }, result_root / "RUN_STATUS.json")

        manifest = result_root / "MANIFEST_SHA256.csv"
        write_manifest(result_root, manifest, exclude=[manifest])
        zip_path = result_root.parent / "HVAC_EVI_FieldExternalValidation_A4_ResultPackage.zip"
        if zip_path.exists():
            zip_path.unlink()
        make_result_zip(result_root, zip_path)
        return zip_path
    except Exception as e:
        if not (result_root / "BLOCKER_REPORT.json").exists():
            json_dump({
                "protocol_id": PROTOCOL_ID,
                "protocol_amendment": AMENDMENT_ID,
                "runner_revision": RUNNER_REVISION,
                "status": "BLOCKED",
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }, result_root / "BLOCKER_REPORT.json")
        manifest = result_root / "MANIFEST_SHA256.csv"
        try:
            write_manifest(result_root, manifest, exclude=[manifest])
        except Exception:
            pass
        zip_path = result_root.parent / "HVAC_EVI_FieldExternalValidation_A4_BLOCKED_ResultPackage.zip"
        if zip_path.exists():
            zip_path.unlink()
        make_result_zip(result_root, zip_path)
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="HVAC_EVI_FieldExternalValidation_A4_Result")
    p.add_argument("--local-data-dir", default=None, help="Use already downloaded exact Figshare v3 CSVs instead of internet download")
    p.add_argument("--no-multiseed", action="store_true", help="Skip the predeclared 20-seed secondary split sensitivity (primary analyses still run)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        z = run(args)
        print(f"RESULT_PACKAGE={z}")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
