#!/usr/bin/env python
"""Tabular DLBS brain-age models from SynthSeg and FreeSurfer features."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from path_utils import resolve_from_repo


ICV_COL = "synthseg_vol_total_intracranial"
FS_ONLY_SUBCORTICAL_VOLUME_BASES = {
    "fs_SubcorticalVolume_LhChoroidPVol",
    "fs_SubcorticalVolume_RhChoroidPVol",
    "fs_SubcorticalVolume_CCPosteriorVol",
    "fs_SubcorticalVolume_CCMidPosteriorVol",
    "fs_SubcorticalVolume_CCCentralVol",
    "fs_SubcorticalVolume_CCMidAnteriorVol",
    "fs_SubcorticalVolume_CCAnteriorVol",
}
FS_GLOBAL_THICKNESS_BASES = {
    "fs_GlobalVariables_LhMeanThick",
    "fs_GlobalVariables_RhMeanThick",
}
FS_GLOBAL_AREA_BASES = {
    "fs_GlobalVariables_LhWhiteSurfArea",
    "fs_GlobalVariables_RhWhiteSurfArea",
}
FS_GLOBAL_VOLUME_BASES = {
    "fs_GlobalVariables_WMHypointensitiesVol",
    "fs_GlobalVariables_NonWMHypointensitiesVol",
}
MODEL_ALIASES = {
    "ridge": "Ridge",
    "ridgecv": "Ridge",
    "elasticnet": "ElasticNet",
    "elasticnetcv": "ElasticNet",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "mlpregressor": "MLP",
}
FEATURE_SET_ALIASES = {
    "synthseg": "synthseg",
    "freesurfer": "freesurfer",
    "fs": "freesurfer",
    "combined": "combined",
    "synthseg+freesurfer": "combined",
    "synthseg_freesurfer": "combined",
}


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    x: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    source_columns: list[str]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return tuple(parts)


def canonicalize_models(value: str) -> list[str]:
    models = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise argparse.ArgumentTypeError(f"Unknown model: {part}")
        model = MODEL_ALIASES[key]
        if model not in models:
            models.append(model)
    if not models:
        raise argparse.ArgumentTypeError("Expected at least one model.")
    return models


def canonicalize_feature_sets(value: str) -> list[str]:
    feature_sets = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in FEATURE_SET_ALIASES:
            raise argparse.ArgumentTypeError(f"Unknown feature set: {part}")
        feature_set = FEATURE_SET_ALIASES[key]
        if feature_set not in feature_sets:
            feature_sets.append(feature_set)
    if not feature_sets:
        raise argparse.ArgumentTypeError("Expected at least one feature set.")
    return feature_sets


def is_freesurfer_cortical_thickness_col(col: str) -> bool:
    if not col.startswith("fs_CorticalThickness_"):
        return False
    base = canonical_freesurfer_col(col)
    return base.endswith("Thick") or base == "fs_CorticalThickness_Thickness"


def is_freesurfer_surface_area_col(col: str) -> bool:
    if not col.startswith("fs_SurfaceArea_"):
        return False
    base = canonical_freesurfer_col(col)
    return base.endswith("Area") or base == "fs_SurfaceArea_Area"


def is_freesurfer_fs_only_subcortical_volume_col(col: str) -> bool:
    if not col.startswith("fs_SubcorticalVolume_"):
        return False
    return canonical_freesurfer_col(col) in FS_ONLY_SUBCORTICAL_VOLUME_BASES


def is_freesurfer_global_thickness_col(col: str) -> bool:
    return canonical_freesurfer_col(col) in FS_GLOBAL_THICKNESS_BASES


def is_freesurfer_global_area_col(col: str) -> bool:
    return canonical_freesurfer_col(col) in FS_GLOBAL_AREA_BASES


def is_freesurfer_global_volume_col(col: str) -> bool:
    return canonical_freesurfer_col(col) in FS_GLOBAL_VOLUME_BASES


def canonical_freesurfer_col(col: str) -> str:
    return re.sub(r"_[xy]$", "", col)


def synthseg_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col for col in df.columns if col.startswith("synthseg_vol_") and col != ICV_COL
    )


def freesurfer_thickness_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if is_freesurfer_cortical_thickness_col(col)
        or is_freesurfer_global_thickness_col(col)
    )


def freesurfer_area_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if is_freesurfer_surface_area_col(col) or is_freesurfer_global_area_col(col)
    )


def freesurfer_volume_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if is_freesurfer_fs_only_subcortical_volume_col(col)
        or is_freesurfer_global_volume_col(col)
    )


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return out.dropna(axis=1, how="all")


def wave_specific_freesurfer_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Collapse FreeSurfer wave-specific merge columns into one feature per measure."""
    if not columns:
        return pd.DataFrame(index=df.index)

    wave_to_suffix = {"wave1": "_x", "wave2": "_y", "wave3": ""}
    grouped: dict[str, list[str]] = {}
    for col in columns:
        grouped.setdefault(canonical_freesurfer_col(col), []).append(col)

    wave = df["wave"].astype(str) if "wave" in df.columns else pd.Series("", index=df.index)
    out = pd.DataFrame(index=df.index)
    for base_col, source_cols in sorted(grouped.items()):
        numeric_sources = {
            col: pd.to_numeric(df[col], errors="coerce")
            for col in sorted(source_cols, key=lambda col: (col.endswith("_x"), col.endswith("_y"), col))
        }
        values = pd.Series(np.nan, index=df.index, dtype=float)
        for wave_name, suffix in wave_to_suffix.items():
            source_col = f"{base_col}{suffix}"
            if source_col in numeric_sources:
                mask = wave.eq(wave_name)
                values.loc[mask] = numeric_sources[source_col].loc[mask]

        fallback = pd.DataFrame(numeric_sources).bfill(axis=1).iloc[:, 0]
        values = values.fillna(fallback)
        out[base_col] = values

    return out.dropna(axis=1, how="all")


def build_synthseg_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = synthseg_feature_columns(df)
    if ICV_COL not in df.columns:
        raise ValueError(f"Missing required iCV column: {ICV_COL}")
    icv = pd.to_numeric(df[ICV_COL], errors="coerce").replace(0, np.nan)
    x = numeric_frame(df, columns).div(icv, axis=0)
    x.columns = [f"synthseg_norm__{col.removeprefix('synthseg_vol_')}" for col in x.columns]
    return x.dropna(axis=1, how="all"), columns


def build_freesurfer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    thickness_cols = freesurfer_thickness_columns(df)
    area_cols = freesurfer_area_columns(df)
    volume_cols = freesurfer_volume_columns(df)
    thickness = wave_specific_freesurfer_frame(df, thickness_cols)
    areas = wave_specific_freesurfer_frame(df, area_cols)
    volumes = wave_specific_freesurfer_frame(df, volume_cols)
    if ICV_COL in df.columns:
        icv = pd.to_numeric(df[ICV_COL], errors="coerce").replace(0, np.nan)
        if not areas.empty:
            areas = areas.div(np.power(icv, 2.0 / 3.0), axis=0)
            areas.columns = [f"fs_area_norm__{col}" for col in areas.columns]
        if not volumes.empty:
            volumes = volumes.div(icv, axis=0)
            volumes.columns = [f"fs_vol_norm__{col}" for col in volumes.columns]
    else:
        if not areas.empty:
            areas.columns = [f"fs_area_raw__{col}" for col in areas.columns]
        if not volumes.empty:
            volumes.columns = [f"fs_vol_raw__{col}" for col in volumes.columns]
    if not thickness.empty:
        thickness.columns = [f"fs_thick__{col}" for col in thickness.columns]
    x = pd.concat([thickness, areas, volumes], axis=1).dropna(axis=1, how="all")
    return x, thickness_cols + area_cols + volume_cols


def build_feature_bundle(df: pd.DataFrame, feature_set: str) -> FeatureBundle:
    if "AgeMRI" not in df.columns:
        raise ValueError("Input table must include AgeMRI.")
    if "subject_id" not in df.columns:
        raise ValueError("Input table must include subject_id.")
    base = df[df["AgeMRI"].notna()].copy()
    y = pd.to_numeric(base["AgeMRI"], errors="coerce")
    base = base[y.notna()].copy()
    y = y.loc[base.index].astype(float)

    source_columns: list[str] = []
    if feature_set == "synthseg":
        x, source_columns = build_synthseg_features(base)
    elif feature_set == "freesurfer":
        x, source_columns = build_freesurfer_features(base)
    elif feature_set == "combined":
        synthseg_x, synthseg_cols = build_synthseg_features(base)
        freesurfer_x, freesurfer_cols = build_freesurfer_features(base)
        x = pd.concat([synthseg_x, freesurfer_x], axis=1).dropna(axis=1, how="all")
        source_columns = synthseg_cols + freesurfer_cols
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if x.empty:
        raise ValueError(f"No usable features found for feature set: {feature_set}")

    meta_cols = [col for col in ["subject_id", "wave", "dataset", "synthseg_run"] if col in base.columns]
    meta = base.loc[:, meta_cols].copy()
    meta.insert(0, "row_index", base.index.to_numpy())
    return FeatureBundle(
        name=feature_set,
        x=x.astype(float),
        y=y,
        meta=meta,
        source_columns=source_columns,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    y_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if len(y_true) > 1 and y_std > 0 and pred_std > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 and y_std > 0 else float("nan"),
        "pearson_r": pearson,
        "bias": float(np.mean(residual)),
        "n": int(len(y_true)),
    }


def fit_age_bias_correction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    if len(y_true) > 1 and np.std(y_true) > 0:
        alpha, beta = np.polyfit(y_true, residual, deg=1)
    else:
        alpha, beta = 0.0, float(np.mean(residual))
    return {"alpha": float(alpha), "beta": float(beta)}


def apply_age_bias_correction(
    y_true: np.ndarray, y_pred: np.ndarray, correction: dict[str, float]
) -> np.ndarray:
    return y_pred - (correction["alpha"] * y_true + correction["beta"])


def make_standard_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def make_tree_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def build_model_factories(args: argparse.Namespace) -> tuple[dict[str, Callable[[], Pipeline]], dict[str, str]]:
    factories: dict[str, Callable[[], Pipeline]] = {}
    skipped: dict[str, str] = {}
    alphas = np.asarray(args.alphas, dtype=float)

    if "Ridge" in args.models:
        factories["Ridge"] = lambda: make_standard_pipeline(RidgeCV(alphas=alphas))
    if "ElasticNet" in args.models:
        factories["ElasticNet"] = lambda: make_standard_pipeline(
            ElasticNetCV(
                l1_ratio=args.l1_ratios,
                alphas=alphas,
                cv=args.inner_folds,
                max_iter=args.elasticnet_max_iter,
                random_state=args.seed,
            )
        )
    if "MLP" in args.models:
        factories["MLP"] = lambda: make_standard_pipeline(
            MLPRegressor(
                hidden_layer_sizes=args.mlp_hidden_layer_sizes,
                alpha=args.mlp_alpha,
                learning_rate_init=args.mlp_learning_rate,
                max_iter=args.mlp_max_iter,
                random_state=args.seed,
                early_stopping=False,
            )
        )
    if "XGBoost" in args.models:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            skipped["XGBoost"] = f"not importable: {exc}"
        else:
            factories["XGBoost"] = lambda: make_tree_pipeline(
                XGBRegressor(
                    n_estimators=args.boosting_estimators,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=args.seed,
                    n_jobs=1,
                )
            )
    if "LightGBM" in args.models:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            skipped["LightGBM"] = f"not importable: {exc}"
        else:
            factories["LightGBM"] = lambda: make_tree_pipeline(
                LGBMRegressor(
                    n_estimators=args.boosting_estimators,
                    learning_rate=0.05,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=args.seed,
                    n_jobs=1,
                    verbose=-1,
                )
            )
    return factories, skipped


def run_grouped_cv(
    bundle: FeatureBundle,
    model_name: str,
    estimator: Pipeline,
    folds: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x = bundle.x.to_numpy(dtype=float)
    y = bundle.y.to_numpy(dtype=float)
    groups = bundle.meta["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two subjects for grouped cross-validation.")

    cv = GroupKFold(n_splits=n_splits)
    raw_pred = np.full(len(y), np.nan, dtype=float)
    corrected_pred = np.full(len(y), np.nan, dtype=float)
    fold_numbers = np.full(len(y), -1, dtype=int)
    corrections: list[dict[str, float | int]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
            model = clone(estimator)
            model.fit(x[train_idx], y[train_idx])
            train_pred = model.predict(x[train_idx])
            test_pred = model.predict(x[test_idx])
            correction = fit_age_bias_correction(y[train_idx], train_pred)
            raw_pred[test_idx] = test_pred
            corrected_pred[test_idx] = apply_age_bias_correction(
                y[test_idx], test_pred, correction
            )
            fold_numbers[test_idx] = fold_index
            corrections.append({"fold": fold_index, **correction})

    pred_df = bundle.meta.reset_index(drop=True).copy()
    pred_df["feature_set"] = bundle.name
    pred_df["model_name"] = model_name
    pred_df["true_age"] = y
    pred_df["raw_predicted_age"] = raw_pred
    pred_df["corrected_predicted_age"] = corrected_pred
    pred_df["raw_residual"] = raw_pred - y
    pred_df["corrected_residual"] = corrected_pred - y
    pred_df["raw_absolute_error"] = np.abs(pred_df["raw_residual"])
    pred_df["corrected_absolute_error"] = np.abs(pred_df["corrected_residual"])
    pred_df["outer_fold"] = fold_numbers

    report = {
        "raw": regression_metrics(y, raw_pred),
        "corrected": regression_metrics(y, corrected_pred),
        "age_bias_corrections": corrections,
        "folds": int(n_splits),
    }
    return pred_df, report


def feature_summary(bundle: FeatureBundle) -> dict[str, object]:
    missing_fraction = bundle.x.isna().mean().sort_values(ascending=False)
    return {
        "n_rows": int(len(bundle.y)),
        "n_subjects": int(bundle.meta["subject_id"].nunique()),
        "n_features": int(bundle.x.shape[1]),
        "source_columns": bundle.source_columns,
        "features": list(bundle.x.columns),
        "missing_fraction_max": float(missing_fraction.max()) if len(missing_fraction) else 0.0,
        "top_missing_fraction": {
            key: float(value) for key, value in missing_fraction.head(20).items()
        },
    }


def build_top_corrected_bag_export(
    predictions: pd.DataFrame, comparison: pd.DataFrame, top_n: int = 2
) -> pd.DataFrame:
    top_models = comparison.head(top_n).loc[:, ["feature_set", "model_name"]].copy()
    top_models["model_rank"] = np.arange(1, len(top_models) + 1)
    export = predictions.merge(top_models, on=["feature_set", "model_name"], how="inner")
    export = export.rename(
        columns={
            "true_age": "AgeMRI",
            "raw_residual": "raw_BAG",
            "corrected_residual": "corrected_BAG",
        }
    )
    ordered_cols = [
        "model_rank",
        "feature_set",
        "model_name",
        "row_index",
        "subject_id",
        "wave",
        "dataset",
        "synthseg_run",
        "outer_fold",
        "AgeMRI",
        "raw_predicted_age",
        "corrected_predicted_age",
        "raw_BAG",
        "corrected_BAG",
        "raw_absolute_error",
        "corrected_absolute_error",
    ]
    available_cols = [col for col in ordered_cols if col in export.columns]
    export = export.loc[:, available_cols].sort_values(
        ["model_rank", "subject_id", "wave", "row_index"], kind="stable"
    )
    return export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv",
        help="DLBS master long table with AgeMRI, SynthSeg, and FreeSurfer columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="../smri-dataset/DLBS/qc/brainage_tabular_models",
        help="Directory for predictions and metrics.",
    )
    parser.add_argument(
        "--feature-sets",
        type=canonicalize_feature_sets,
        default=canonicalize_feature_sets("synthseg,freesurfer,combined"),
        help="Comma-separated: synthseg,freesurfer,combined.",
    )
    parser.add_argument(
        "--models",
        type=canonicalize_models,
        default=canonicalize_models("Ridge,ElasticNet,XGBoost,LightGBM,MLP"),
        help="Comma-separated: Ridge,ElasticNet,XGBoost,LightGBM,MLP.",
    )
    parser.add_argument("--folds", type=int, default=5)
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


def main() -> None:
    args = parse_args()
    input_path = resolve_from_repo(args.input)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t" if input_path.suffix == ".tsv" else ",")
    bundles = {feature_set: build_feature_bundle(df, feature_set) for feature_set in args.feature_sets}
    factories, skipped_models = build_model_factories(args)
    if not factories:
        raise SystemExit(f"No requested models are available. Skipped: {skipped_models}")

    predictions: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "input": str(input_path),
        "feature_sets": {},
        "models_requested": args.models,
        "models_run": sorted(factories),
        "models_skipped": skipped_models,
        "results": {},
        "age_bias_correction": "fit raw_predicted_age - true_age as alpha * true_age + beta on each training fold; subtract fitted bias from test predictions",
    }

    for feature_set, bundle in bundles.items():
        metrics["feature_sets"][feature_set] = feature_summary(bundle)
        metrics["results"][feature_set] = {}
        for model_name, factory in factories.items():
            print(f"Running {feature_set} / {model_name}")
            pred_df, report = run_grouped_cv(bundle, model_name, factory(), args.folds)
            predictions.append(pred_df)
            metrics["results"][feature_set][model_name] = report

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    comparison_rows = []
    for feature_set, model_reports in metrics["results"].items():
        for model_name, report in model_reports.items():
            row = {"feature_set": feature_set, "model_name": model_name}
            row.update({f"raw_{key}": value for key, value in report["raw"].items()})
            row.update({f"corrected_{key}": value for key, value in report["corrected"].items()})
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["corrected_mae", "raw_mae", "feature_set", "model_name"], kind="stable"
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    top_corrected_bag = build_top_corrected_bag_export(predictions_df, comparison, top_n=2)
    top_corrected_bag.to_csv(output_dir / "corrected_bag_top2_models.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "feature_summary.json").write_text(
        json.dumps(metrics["feature_sets"], indent=2) + "\n"
    )

    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'comparison.csv'}")
    print(f"Wrote {output_dir / 'corrected_bag_top2_models.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
