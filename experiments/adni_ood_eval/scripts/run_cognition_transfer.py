#!/usr/bin/env python
"""Train DLBS cognition baselines and evaluate ADNI fluid-like cognition targets."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold

from path_utils import resolve_from_repo

SYNTHSEG_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "synthseg_ridge_baseline" / "scripts"
sys.path.insert(0, str(SYNTHSEG_SCRIPT_DIR))

from run_brainage_models import build_synthseg_features, regression_metrics  # noqa: E402
from run_cognition_bag_models import (  # noqa: E402
    build_model_factories,
    make_model,
    parse_float_list,
    parse_int_tuple,
    parse_models,
)


DEFAULT_DLBS_MASTER = "../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv"
DEFAULT_DLBS_COGNITION = "../smri-dataset/DLBS/derivatives/master_tables/dlbs_cognitive_composites.tsv"
DEFAULT_DLBS_BRAINAGE = "../smri-dataset/DLBS/qc/brainage_tabular_models/predictions.csv"
DEFAULT_ADNI_TEST = "../smri-dataset/ADNI/qc/adni_ood_eval/adni_healthy_cn_subset_200.csv"
DEFAULT_ADNI_BRAINAGE = "../smri-dataset/ADNI/qc/adni_ood_eval/healthy_adni_brainage_transfer_predictions.csv"
DEFAULT_OUTPUT_DIR = "../smri-dataset/ADNI/qc/adni_ood_eval"

NUMERIC_BASE_COLUMNS = ["AgeCog", "EduYrsEstCap", "HandednessScore", "BMI"]
CATEGORICAL_BASE_COLUMNS = ["Sex_model", "Race_model", "Ethnicity_model", "phase_model"]
BAG_COLUMN = "evaluated_BAG"
TARGET_MAPPINGS = {
    "cog_fluid_z": "ADNI_EF",
}
ADNI_NATIVE_TARGETS = ["ADNI_EF", "ADNI_EF2", "ADNI_MEM", "ADNI_LAN", "ADNI_VS"]
VARIANTS = {
    "agecog_only": {"demographics": False, "synthseg": False, "bag": False},
    "agecog_synthseg": {"demographics": False, "synthseg": True, "bag": False},
    "demographics_only": {"demographics": True, "synthseg": False, "bag": False},
    "demographics_synthseg": {"demographics": True, "synthseg": True, "bag": False},
    "demographics_bag": {"demographics": True, "synthseg": False, "bag": True},
    "demographics_synthseg_bag": {"demographics": True, "synthseg": True, "bag": True},
}


def parse_targets(value: str) -> list[str]:
    targets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [target for target in targets if target not in TARGET_MAPPINGS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown target(s): {','.join(unknown)}")
    return targets


def parse_native_targets(value: str) -> list[str]:
    targets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [target for target in targets if target not in ADNI_NATIVE_TARGETS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown ADNI target(s): {','.join(unknown)}")
    return targets


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown variant(s): {','.join(unknown)}")
    return variants


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 2 or valid.std(ddof=0) == 0:
        return numeric * np.nan
    return (numeric - valid.mean()) / valid.std(ddof=0)


def normalize_sex(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"m", "male"}:
        return "M"
    if text in {"f", "female"}:
        return "F"
    return "Unknown"


def normalize_ethnicity(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "1.0", "hispanic", "hispanic or latino"}:
        return "Hispanic"
    if text in {"0", "0.0", "2", "2.0", "not hispanic", "not hispanic or latino"}:
        return "Not Hispanic"
    return "Unknown"


def normalize_race(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"5", "5.0", "white"}:
        return "White"
    if "unknown" in text or "not reported" in text:
        return "Unknown"
    return "Non-White"


def normalize_handedness(value: object) -> float:
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return np.nan
    score = float(score)
    if abs(score) > 4:
        return score / 25.0
    return score


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


def build_modeling_frame(
    master: pd.DataFrame,
    cognition: pd.DataFrame | None = None,
    bag: pd.DataFrame | None = None,
) -> pd.DataFrame:
    synthseg, _ = build_synthseg_features(master)
    synthseg = synthseg.add_prefix("img_")
    base_cols = [
        "dataset",
        "subject_id",
        "wave",
        "AgeCog",
        "Sex",
        "Race",
        "Ethnicity",
        "EduYrsEstCap",
        "HandednessScore",
        "BMI",
        "adni_phase",
        "baseline_phase",
        "diagnosis",
        "baseline_diagnosis",
        "mci_to_ad_36mo",
        *ADNI_NATIVE_TARGETS,
    ]
    available = [col for col in base_cols if col in master.columns]
    frame = pd.concat([master[available].reset_index(drop=True), synthseg.reset_index(drop=True)], axis=1)
    if cognition is not None:
        frame = frame.merge(cognition, on=["dataset", "subject_id", "wave"], how="inner")
    if bag is not None:
        frame = frame.merge(bag, on=["subject_id", "wave"], how="left")
    frame["Sex_model"] = frame["Sex"].map(normalize_sex) if "Sex" in frame.columns else np.nan
    frame["Race_model"] = frame["Race"].map(normalize_race) if "Race" in frame.columns else np.nan
    frame["Ethnicity_model"] = (
        frame["Ethnicity"].map(normalize_ethnicity) if "Ethnicity" in frame.columns else np.nan
    )
    if "adni_phase" in frame.columns:
        frame["phase_model"] = frame["adni_phase"]
    elif "wave" in frame.columns:
        frame["phase_model"] = frame["wave"]
    else:
        frame["phase_model"] = np.nan
    if "HandednessScore" in frame.columns:
        frame["HandednessScore"] = frame["HandednessScore"].map(normalize_handedness)
    for col in [*NUMERIC_BASE_COLUMNS, *[c for c in frame.columns if c.startswith("img_")], BAG_COLUMN]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def feature_columns(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str]]:
    spec = VARIANTS[variant]
    numeric = ["AgeCog"]
    categorical: list[str] = []
    if spec["demographics"]:
        numeric = list(NUMERIC_BASE_COLUMNS)
        categorical = [col for col in CATEGORICAL_BASE_COLUMNS if col in df.columns]
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


def run_transfer_for_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dlbs_target: str,
    adni_target: str,
    model_name: str,
    estimator_factory,
    variants: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_rows = train[train[dlbs_target].notna() & train["AgeCog"].notna()].copy()
    test_rows = test[test[adni_target].notna() & test["AgeCog"].notna()].copy()
    if len(train_rows) < 10 or len(test_rows) < 10:
        return pd.DataFrame(), []

    y_train_s = zscore(train_rows[dlbs_target])
    y_test_s = zscore(test_rows[adni_target])
    train_rows = train_rows[y_train_s.notna()].copy()
    test_rows = test_rows[y_test_s.notna()].copy()
    y_train = y_train_s.dropna().to_numpy(dtype=float)
    y_test = y_test_s.dropna().to_numpy(dtype=float)

    prediction_frames = []
    summaries = []
    for variant in variants:
        numeric, categorical = feature_columns(train_rows, variant)
        for col in numeric:
            if col not in test_rows.columns:
                test_rows[col] = np.nan
        model = make_model(numeric, categorical, estimator_factory())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fitted = clone(model).fit(train_rows, y_train)
            pred = fitted.predict(test_rows)
        metrics = regression_metrics(y_test, pred)
        summaries.append(
            {
                "analysis": "dlbs_to_adni",
                "dlbs_target": dlbs_target,
                "adni_target": adni_target,
                "model_name": model_name,
                "feature_variant": variant,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                **metrics,
            }
        )
        out_cols = [
            col
            for col in [
                "dataset",
                "subject_id",
                "wave",
                "AgeCog",
                "Sex",
                "Race",
                "Ethnicity",
                "adni_phase",
                "baseline_phase",
                "diagnosis",
                "baseline_diagnosis",
                "mci_to_ad_36mo",
            ]
            if col in test_rows.columns
        ]
        out = test_rows[out_cols].copy()
        out["analysis"] = "dlbs_to_adni"
        out["dlbs_target"] = dlbs_target
        out["adni_target"] = adni_target
        out["model_name"] = model_name
        out["feature_variant"] = variant
        out["target_value_z"] = y_test
        out["predicted_value_z"] = pred
        out["residual_z"] = pred - y_test
        out["absolute_error_z"] = np.abs(out["residual_z"])
        prediction_frames.append(out)
    return pd.concat(prediction_frames, ignore_index=True), summaries


def run_adni_native_cv(
    frame: pd.DataFrame,
    adni_target: str,
    model_name: str,
    estimator_factory,
    variants: list[str],
    folds: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows = frame[frame[adni_target].notna() & frame["AgeCog"].notna()].copy()
    if len(rows) < 10:
        return pd.DataFrame(), []
    y_s = zscore(rows[adni_target])
    rows = rows[y_s.notna()].copy()
    y = y_s.dropna().to_numpy(dtype=float)
    groups = rows["subject_id"].astype(str).to_numpy()
    n_splits = min(folds, np.unique(groups).size)
    if n_splits < 2:
        return pd.DataFrame(), []

    prediction_frames = []
    summaries = []
    cv = GroupKFold(n_splits=n_splits)
    for variant in variants:
        numeric, categorical = feature_columns(rows, variant)
        pred = np.full(len(rows), np.nan, dtype=float)
        fold_numbers = np.full(len(rows), -1, dtype=int)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            for fold, (train_idx, test_idx) in enumerate(cv.split(rows, y, groups), start=1):
                model = make_model(numeric, categorical, estimator_factory())
                fitted = clone(model).fit(rows.iloc[train_idx], y[train_idx])
                pred[test_idx] = fitted.predict(rows.iloc[test_idx])
                fold_numbers[test_idx] = fold
        metrics = regression_metrics(y, pred)
        summaries.append(
            {
                "analysis": "adni_native_cv",
                "dlbs_target": np.nan,
                "adni_target": adni_target,
                "model_name": model_name,
                "feature_variant": variant,
                "n_train": int(len(y)),
                "n_test": int(len(y)),
                "folds": int(n_splits),
                **metrics,
            }
        )
        out_cols = [
            col
            for col in [
                "dataset",
                "subject_id",
                "wave",
                "AgeCog",
                "Sex",
                "Race",
                "Ethnicity",
                "adni_phase",
                "baseline_phase",
                "diagnosis",
                "baseline_diagnosis",
                "mci_to_ad_36mo",
            ]
            if col in rows.columns
        ]
        out = rows[out_cols].copy()
        out["analysis"] = "adni_native_cv"
        out["dlbs_target"] = np.nan
        out["adni_target"] = adni_target
        out["model_name"] = model_name
        out["feature_variant"] = variant
        out["target_value_z"] = y
        out["predicted_value_z"] = pred
        out["residual_z"] = pred - y
        out["absolute_error_z"] = np.abs(out["residual_z"])
        out["outer_fold"] = fold_numbers
        prediction_frames.append(out)
    return pd.concat(prediction_frames, ignore_index=True), summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlbs-master", type=Path, default=Path(DEFAULT_DLBS_MASTER))
    parser.add_argument("--dlbs-cognition", type=Path, default=Path(DEFAULT_DLBS_COGNITION))
    parser.add_argument("--dlbs-brainage", type=Path, default=Path(DEFAULT_DLBS_BRAINAGE))
    parser.add_argument("--adni-test", type=Path, default=Path(DEFAULT_ADNI_TEST))
    parser.add_argument("--adni-brainage", type=Path, default=Path(DEFAULT_ADNI_BRAINAGE))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="healthy_adni")
    parser.add_argument("--bag-feature-set", default="synthseg")
    parser.add_argument("--bag-model-name", default="ElasticNet")
    parser.add_argument("--targets", type=parse_targets, default=list(TARGET_MAPPINGS))
    parser.add_argument("--native-targets", type=parse_native_targets, default=parse_native_targets("ADNI_EF,ADNI_EF2,ADNI_MEM"))
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("agecog_only,agecog_synthseg,demographics_only,demographics_synthseg,demographics_bag,demographics_synthseg_bag"),
    )
    parser.add_argument("--models", type=parse_models, default=parse_models("Ridge,ElasticNet,MLP"))
    parser.add_argument("--native-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=parse_float_list("0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000"),
    )
    parser.add_argument(
        "--l1-ratios",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.25,0.5,0.75,0.9,0.95"),
    )
    parser.add_argument("--elasticnet-max-iter", type=int, default=20000)
    parser.add_argument("--mlp-hidden-layer-sizes", type=parse_int_tuple, default=(64, 32))
    parser.add_argument("--mlp-alpha", type=float, default=0.001)
    parser.add_argument("--mlp-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-max-iter", type=int, default=1000)
    parser.add_argument("--boosting-estimators", type=int, default=300)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", low_memory=False)


def main() -> None:
    args = parse_args()
    dlbs_master = resolve_from_repo(args.dlbs_master)
    dlbs_cognition = resolve_from_repo(args.dlbs_cognition)
    dlbs_brainage = resolve_from_repo(args.dlbs_brainage)
    adni_test = resolve_from_repo(args.adni_test)
    adni_brainage = resolve_from_repo(args.adni_brainage)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_master = read_table(dlbs_master)
    train_cognition = read_table(dlbs_cognition)
    test_master = read_table(adni_test)
    train_bag = load_bag_predictions(dlbs_brainage, args.bag_feature_set, args.bag_model_name)
    test_bag = load_bag_predictions(adni_brainage, args.bag_feature_set, args.bag_model_name)
    train = build_modeling_frame(train_master, train_cognition, train_bag)
    test = build_modeling_frame(test_master, None, test_bag)
    factories, skipped_models = build_model_factories(args)
    if not factories:
        raise SystemExit(f"No requested cognition models are available. Skipped: {skipped_models}")

    predictions = []
    comparison_rows = []
    for dlbs_target in args.targets:
        adni_target = TARGET_MAPPINGS[dlbs_target]
        for model_name, factory in factories.items():
            print(f"Training DLBS {dlbs_target} / {model_name}; testing ADNI {adni_target}")
            pred, summaries = run_transfer_for_target(
                train, test, dlbs_target, adni_target, model_name, factory, args.variants
            )
            if not pred.empty:
                predictions.append(pred)
            comparison_rows.extend(summaries)
    for adni_target in args.native_targets:
        for model_name, factory in factories.items():
            print(f"Running ADNI-native CV {adni_target} / {model_name}")
            pred, summaries = run_adni_native_cv(
                test, adni_target, model_name, factory, args.variants, args.native_folds
            )
            if not pred.empty:
                predictions.append(pred)
            comparison_rows.extend(summaries)

    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["adni_target", "mae", "model_name", "feature_variant"], kind="stable"
    )
    pred_path = output_dir / f"{args.output_prefix}_cognition_transfer_predictions.csv"
    metrics_path = output_dir / f"{args.output_prefix}_cognition_transfer_metrics.csv"
    json_path = output_dir / f"{args.output_prefix}_cognition_transfer_metrics.json"
    predictions_df.to_csv(pred_path, index=False)
    comparison.to_csv(metrics_path, index=False)
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "dlbs_master": str(dlbs_master),
        "dlbs_cognition": str(dlbs_cognition),
        "dlbs_brainage": str(dlbs_brainage),
        "adni_test": str(adni_test),
        "adni_brainage": str(adni_brainage),
        "bag_source": {"feature_set": args.bag_feature_set, "model_name": args.bag_model_name},
        "target_mappings": TARGET_MAPPINGS,
        "native_targets_available": args.native_targets,
        "target_scaling": "DLBS and ADNI targets are z-scored within their own modeling tables before transfer metrics.",
        "adni_ef_direction": "Higher ADNI_EF is better cognition.",
        "variants": VARIANTS,
        "models_requested": args.models,
        "models_run": sorted(factories),
        "models_skipped": skipped_models,
    }
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
