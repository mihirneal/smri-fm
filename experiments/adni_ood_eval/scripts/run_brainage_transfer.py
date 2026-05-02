#!/usr/bin/env python
"""Train DLBS tabular brain-age models and evaluate them on ADNI."""

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

from path_utils import resolve_from_repo

SYNTHSEG_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "synthseg_ridge_baseline" / "scripts"
sys.path.insert(0, str(SYNTHSEG_SCRIPT_DIR))

from run_brainage_models import (  # noqa: E402
    apply_age_bias_correction,
    build_feature_bundle,
    build_model_factories,
    canonicalize_feature_sets,
    canonicalize_models,
    feature_summary,
    fit_age_bias_correction,
    parse_float_list,
    parse_int_tuple,
    regression_metrics,
)


DEFAULT_DLBS_MASTER = "../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv"
DEFAULT_ADNI_TEST = "../smri-dataset/ADNI/qc/adni_ood_eval/adni_healthy_cn_subset_200.csv"
DEFAULT_OUTPUT_DIR = "../smri-dataset/ADNI/qc/adni_ood_eval"


def align_test_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:
    return test_x.reindex(columns=train_x.columns)


def run_transfer(train_bundle, test_bundle, model_name: str, estimator) -> tuple[pd.DataFrame, dict[str, object]]:
    x_train = train_bundle.x.to_numpy(dtype=float)
    y_train = train_bundle.y.to_numpy(dtype=float)
    x_test_df = align_test_features(train_bundle.x, test_bundle.x)
    x_test = x_test_df.to_numpy(dtype=float)
    y_test = test_bundle.y.to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = clone(estimator)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        raw_test_pred = model.predict(x_test)

    correction = fit_age_bias_correction(y_train, train_pred)
    corrected_test_pred = apply_age_bias_correction(y_test, raw_test_pred, correction)

    pred_df = test_bundle.meta.reset_index(drop=True).copy()
    pred_df["feature_set"] = test_bundle.name
    pred_df["model_name"] = model_name
    pred_df["true_age"] = y_test
    pred_df["raw_predicted_age"] = raw_test_pred
    pred_df["corrected_predicted_age"] = corrected_test_pred
    pred_df["raw_residual"] = raw_test_pred - y_test
    pred_df["corrected_residual"] = corrected_test_pred - y_test
    pred_df["raw_absolute_error"] = np.abs(pred_df["raw_residual"])
    pred_df["corrected_absolute_error"] = np.abs(pred_df["corrected_residual"])
    pred_df["corrected_BAG"] = pred_df["corrected_residual"]

    report = {
        "raw": regression_metrics(y_test, raw_test_pred),
        "corrected": regression_metrics(y_test, corrected_test_pred),
        "train_metrics_raw": regression_metrics(y_train, train_pred),
        "age_bias_correction": correction,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_features": int(train_bundle.x.shape[1]),
        "n_test_features_original": int(test_bundle.x.shape[1]),
        "n_aligned_features": int(x_test_df.shape[1]),
        "test_missing_fraction_max": float(x_test_df.isna().mean().max()),
    }
    return pred_df, report


def group_breakdowns(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "diagnosis",
        "baseline_diagnosis",
        "mci_to_ad_36mo",
        "Sex",
        "Race",
        "Ethnicity",
        "Manufacturer",
        "MagneticFieldStrength",
        "adni_phase",
        "baseline_phase",
        "age_bin",
    ]:
        if col not in predictions.columns:
            continue
        for value, group in predictions.groupby(col, dropna=False):
            if len(group) < 2:
                continue
            for pred_col, prefix in [
                ("raw_predicted_age", "raw"),
                ("corrected_predicted_age", "corrected"),
            ]:
                metrics = regression_metrics(
                    group["true_age"].to_numpy(dtype=float),
                    group[pred_col].to_numpy(dtype=float),
                )
                rows.append({"variable": col, "value": value, "prediction": prefix, **metrics})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlbs-master", type=Path, default=Path(DEFAULT_DLBS_MASTER))
    parser.add_argument("--adni-test", type=Path, default=Path(DEFAULT_ADNI_TEST))
    parser.add_argument("--adni-healthy-train", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="healthy_adni")
    parser.add_argument(
        "--feature-sets",
        type=canonicalize_feature_sets,
        default=canonicalize_feature_sets("synthseg"),
    )
    parser.add_argument(
        "--models",
        type=canonicalize_models,
        default=canonicalize_models("Ridge,ElasticNet,MLP"),
    )
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
    adni_test = resolve_from_repo(args.adni_test)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_table(dlbs_master)
    test_df = read_table(adni_test)
    train_sources = ["DLBS"]
    if args.adni_healthy_train is not None:
        healthy_train_path = resolve_from_repo(args.adni_healthy_train)
        healthy_train = read_table(healthy_train_path)
        test_subjects = set(test_df["subject_id"].astype(str))
        healthy_train = healthy_train[~healthy_train["subject_id"].astype(str).isin(test_subjects)].copy()
        train_df = pd.concat([train_df, healthy_train], ignore_index=True, sort=False)
        train_sources.append("ADNI_stable_CN_excluding_test_subjects")

    factories, skipped_models = build_model_factories(args)
    if not factories:
        raise SystemExit(f"No requested brain-age models are available. Skipped: {skipped_models}")

    predictions = []
    metrics: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "dlbs_master": str(dlbs_master),
        "adni_test": str(adni_test),
        "train_sources": train_sources,
        "feature_sets": {},
        "models_requested": args.models,
        "models_run": sorted(factories),
        "models_skipped": skipped_models,
        "results": {},
        "age_bias_correction": "Fit raw_predicted_age - true_age on all training rows, then subtract alpha * true_age + beta from ADNI predictions.",
    }

    for feature_set in args.feature_sets:
        train_bundle = build_feature_bundle(train_df, feature_set)
        test_bundle = build_feature_bundle(test_df, feature_set)
        metrics["feature_sets"][feature_set] = {
            "train": feature_summary(train_bundle),
            "test": feature_summary(test_bundle),
        }
        metrics["results"][feature_set] = {}
        for model_name, factory in factories.items():
            print(f"Training {'+'.join(train_sources)} {feature_set} / {model_name}; testing ADNI")
            pred_df, report = run_transfer(train_bundle, test_bundle, model_name, factory())
            predictions.append(pred_df)
            metrics["results"][feature_set][model_name] = report

    predictions_df = pd.concat(predictions, ignore_index=True)
    meta_cols = [
        col
        for col in [
            "row_index",
            "subset_index",
            "subject_id",
            "wave",
            "PTID",
            "RID",
            "scan_date",
            "diagnosis",
            "baseline_diagnosis",
            "mci_to_ad_36mo",
            "Sex",
            "Race",
            "Ethnicity",
            "Manufacturer",
            "ManufacturersModelName",
            "MagneticFieldStrength",
            "adni_phase",
            "baseline_phase",
            "age_bin",
            "ADNI_EF",
            "ADNI_EF2",
            "ADNI_MEM",
        ]
        if col in test_df.columns
    ]
    predictions_df = predictions_df.merge(
        test_df[meta_cols],
        on=["subject_id", "wave"],
        how="left",
        suffixes=("", "_test"),
    )
    pred_path = output_dir / f"{args.output_prefix}_brainage_transfer_predictions.csv"
    metrics_path = output_dir / f"{args.output_prefix}_brainage_transfer_metrics.csv"
    group_path = output_dir / f"{args.output_prefix}_brainage_transfer_group_metrics.csv"
    json_path = output_dir / f"{args.output_prefix}_brainage_transfer_metrics.json"
    predictions_df.to_csv(pred_path, index=False)

    comparison_rows = []
    for feature_set, model_reports in metrics["results"].items():
        for model_name, report in model_reports.items():
            row = {"feature_set": feature_set, "model_name": model_name}
            row.update({f"raw_{key}": value for key, value in report["raw"].items()})
            row.update({f"corrected_{key}": value for key, value in report["corrected"].items()})
            row["n_train"] = report["n_train"]
            row["n_test"] = report["n_test"]
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["corrected_mae", "raw_mae", "feature_set", "model_name"], kind="stable"
    )
    comparison.to_csv(metrics_path, index=False)
    group_breakdowns(predictions_df).to_csv(group_path, index=False)
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
