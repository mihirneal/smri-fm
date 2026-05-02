#!/usr/bin/env python
"""Predict baseline MCI conversion to AD within 36 months from ADNI features."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from path_utils import resolve_from_repo

SYNTHSEG_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "synthseg_ridge_baseline" / "scripts"
sys.path.insert(0, str(SYNTHSEG_SCRIPT_DIR))

from run_brainage_models import build_synthseg_features  # noqa: E402


DEFAULT_ADNI_PROGNOSIS = "../smri-dataset/ADNI/qc/adni_ood_eval/adni_mci_ad_prognosis_baseline.csv"
DEFAULT_ADNI_BRAINAGE = "../smri-dataset/ADNI/qc/adni_ood_eval/mci_ad_dlbs_brainage_transfer_predictions.csv"
DEFAULT_OUTPUT_DIR = "../smri-dataset/ADNI/qc/adni_ood_eval"
BAG_COLUMN = "evaluated_BAG"
TARGET = "mci_to_ad_36mo"
NUMERIC_BASE_COLUMNS = ["AgeCog", "EduYrsEstCap", "HandednessScore", "BMI"]
CATEGORICAL_BASE_COLUMNS = ["Sex", "Race", "Ethnicity", "adni_phase", "Manufacturer", "MagneticFieldStrength"]
VARIANTS = {
    "bag_only": {"demographics": False, "synthseg": False, "bag": True},
    "synthseg_only": {"demographics": False, "synthseg": True, "bag": False},
    "synthseg_bag": {"demographics": False, "synthseg": True, "bag": True},
    "demographics_only": {"synthseg": False, "bag": False},
    "demographics_synthseg": {"synthseg": True, "bag": False},
    "demographics_bag": {"synthseg": False, "bag": True},
    "demographics_synthseg_bag": {"synthseg": True, "bag": True},
}


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown variant(s): {','.join(unknown)}")
    return variants


def load_bag_predictions(path: Path, feature_set: str, model_name: str) -> pd.DataFrame:
    predictions = pd.read_csv(path, low_memory=False)
    selected = predictions[
        predictions["feature_set"].eq(feature_set) & predictions["model_name"].eq(model_name)
    ].copy()
    if selected.empty:
        available = predictions[["feature_set", "model_name"]].drop_duplicates()
        raise ValueError(
            f"No BAG rows for {feature_set}/{model_name} in {path}. "
            f"Available: {available.to_dict(orient='records')}"
        )
    selected[BAG_COLUMN] = pd.to_numeric(selected["corrected_residual"], errors="coerce")
    return selected[["subject_id", "wave", BAG_COLUMN]].drop_duplicates(["subject_id", "wave"])


def build_frame(master: pd.DataFrame, bag: pd.DataFrame) -> pd.DataFrame:
    synthseg, _ = build_synthseg_features(master)
    synthseg = synthseg.add_prefix("img_")
    frame = pd.concat([master.reset_index(drop=True), synthseg.reset_index(drop=True)], axis=1)
    frame = frame.merge(bag, on=["subject_id", "wave"], how="left")
    for col in [*NUMERIC_BASE_COLUMNS, BAG_COLUMN, TARGET, *[c for c in frame.columns if c.startswith("img_")]]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in CATEGORICAL_BASE_COLUMNS:
        if col in frame.columns:
            frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    return frame[frame[TARGET].notna()].copy()


def feature_columns(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    spec = VARIANTS[variant]
    if spec.get("demographics", True):
        numeric = [
            col
            for col in NUMERIC_BASE_COLUMNS
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
        ]
        categorical = [col for col in CATEGORICAL_BASE_COLUMNS if col in df.columns]
    else:
        numeric = []
        categorical = []
    if spec["synthseg"]:
        numeric.extend(col for col in df.columns if col.startswith("img_synthseg_norm__"))
    if spec["bag"]:
        numeric.append(BAG_COLUMN)
    numeric = [
        col
        for col in dict.fromkeys(numeric)
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    return numeric, categorical


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=13,
                ),
            ),
        ]
    )


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def run_grouped_cv(df: pd.DataFrame, variant: str, folds: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = df[df[TARGET].notna()].copy()
    y = rows[TARGET].astype(int).to_numpy()
    groups = rows["subject_id"].astype(str).to_numpy()
    n_splits = min(folds, np.unique(groups).size, np.bincount(y).min())
    if n_splits < 2:
        raise ValueError(f"Need at least two folds and both classes for {variant}.")
    numeric, categorical = feature_columns(rows, variant)
    model = make_model(numeric, categorical)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=13)
    prob = np.full(len(rows), np.nan, dtype=float)
    fold_numbers = np.full(len(rows), -1, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_idx, test_idx) in enumerate(cv.split(rows, y, groups), start=1):
            fitted = clone(model).fit(rows.iloc[train_idx], y[train_idx])
            prob[test_idx] = fitted.predict_proba(rows.iloc[test_idx])[:, 1]
            fold_numbers[test_idx] = fold
    pred = rows[
        [
            col
            for col in [
                "subject_id",
                "wave",
                "PTID",
                "AgeMRI",
                "AgeCog",
                "Sex",
                "Race",
                "Ethnicity",
                "Manufacturer",
                "MagneticFieldStrength",
                "mci_to_ad_36mo",
                "months_to_ad",
                "max_followup_months",
            ]
            if col in rows.columns
        ]
    ].copy()
    pred["feature_variant"] = variant
    pred["predicted_probability"] = prob
    pred["predicted_label"] = (prob >= 0.5).astype(int)
    pred["outer_fold"] = fold_numbers
    return pred, classification_metrics(y, prob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adni-prognosis", type=Path, default=Path(DEFAULT_ADNI_PROGNOSIS))
    parser.add_argument("--adni-brainage", type=Path, default=Path(DEFAULT_ADNI_BRAINAGE))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="mci_ad")
    parser.add_argument("--bag-feature-set", default="synthseg")
    parser.add_argument("--bag-model-name", default="ElasticNet")
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("bag_only,synthseg_only,synthseg_bag"),
    )
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", low_memory=False)


def main() -> None:
    args = parse_args()
    adni_prognosis = resolve_from_repo(args.adni_prognosis)
    adni_brainage = resolve_from_repo(args.adni_brainage)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master = read_table(adni_prognosis)
    bag = load_bag_predictions(adni_brainage, args.bag_feature_set, args.bag_model_name)
    frame = build_frame(master, bag)

    predictions = []
    summaries = []
    for variant in args.variants:
        print(f"Running MCI->AD prognosis {variant}")
        pred, metrics = run_grouped_cv(frame, variant, args.folds)
        predictions.append(pred)
        summaries.append({"feature_variant": variant, **metrics})
    predictions_df = pd.concat(predictions, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(["auc", "average_precision"], ascending=False)
    pred_path = output_dir / f"{args.output_prefix}_prognosis_predictions.csv"
    metrics_path = output_dir / f"{args.output_prefix}_prognosis_metrics.csv"
    json_path = output_dir / f"{args.output_prefix}_prognosis_metrics.json"
    predictions_df.to_csv(pred_path, index=False)
    summary_df.to_csv(metrics_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "adni_prognosis": str(adni_prognosis),
                "adni_brainage": str(adni_brainage),
                "target": TARGET,
                "bag_source": {"feature_set": args.bag_feature_set, "model_name": args.bag_model_name},
                "variants": VARIANTS,
                "folds_requested": args.folds,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
