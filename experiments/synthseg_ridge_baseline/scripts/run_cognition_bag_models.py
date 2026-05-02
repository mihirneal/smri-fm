#!/usr/bin/env python
"""Predict DLBS cognitive composites from demographics and out-of-fold corrected BAG."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from path_utils import resolve_from_repo
from run_brainage_models import (
    ICV_COL,
    build_freesurfer_features,
    build_synthseg_features,
    wave_specific_freesurfer_frame,
)


COGNITIVE_TARGETS = [
    "cog_fluid_z",
    "cog_speed_z",
    "cog_working_memory_z",
    "cog_reasoning_z",
    "cog_vocabulary_z",
    "cog_verbal_fluency_z",
]
NUMERIC_BASE_COLUMNS = ["AgeCog", "EduYrsEstCap", "HandednessScore", "BMI"]
CATEGORICAL_BASE_COLUMNS = ["Sex", "Race", "Ethnicity", "wave"]
BAG_COLUMN = "corrected_BAG"
SYNTHSEG_GROUPS = {
    "synthseg_anatomical_core": {
        "exact": [
            "img_synthseg_norm__left_cerebral_cortex",
            "img_synthseg_norm__right_cerebral_cortex",
            "img_synthseg_norm__left_cerebral_white_matter",
            "img_synthseg_norm__right_cerebral_white_matter",
            "img_synthseg_norm__left_cerebellum_cortex",
            "img_synthseg_norm__right_cerebellum_cortex",
            "img_synthseg_norm__left_cerebellum_white_matter",
            "img_synthseg_norm__right_cerebellum_white_matter",
            "img_synthseg_norm__brain_stem",
            "img_synthseg_norm__csf",
        ],
        "prefix": [],
    },
    "synthseg_ventricles_csf": {
        "exact": [
            "img_synthseg_norm__3rd_ventricle",
            "img_synthseg_norm__4th_ventricle",
            "img_synthseg_norm__left_lateral_ventricle",
            "img_synthseg_norm__right_lateral_ventricle",
            "img_synthseg_norm__left_inferior_lateral_ventricle",
            "img_synthseg_norm__right_inferior_lateral_ventricle",
            "img_synthseg_norm__csf",
        ],
        "prefix": [],
    },
    "synthseg_mtl_ad": {
        "exact": [
            "img_synthseg_norm__left_hippocampus",
            "img_synthseg_norm__right_hippocampus",
            "img_synthseg_norm__left_amygdala",
            "img_synthseg_norm__right_amygdala",
            "img_synthseg_norm__left_inferior_lateral_ventricle",
            "img_synthseg_norm__right_inferior_lateral_ventricle",
            "img_synthseg_norm__ctx_lh_entorhinal",
            "img_synthseg_norm__ctx_rh_entorhinal",
            "img_synthseg_norm__ctx_lh_parahippocampal",
            "img_synthseg_norm__ctx_rh_parahippocampal",
            "img_synthseg_norm__ctx_lh_fusiform",
            "img_synthseg_norm__ctx_rh_fusiform",
            "img_synthseg_norm__ctx_lh_inferiortemporal",
            "img_synthseg_norm__ctx_rh_inferiortemporal",
            "img_synthseg_norm__ctx_lh_middletemporal",
            "img_synthseg_norm__ctx_rh_middletemporal",
            "img_synthseg_norm__ctx_lh_temporalpole",
            "img_synthseg_norm__ctx_rh_temporalpole",
        ],
        "prefix": [],
    },
    "synthseg_subcortical": {
        "exact": [
            "img_synthseg_norm__left_thalamus",
            "img_synthseg_norm__right_thalamus",
            "img_synthseg_norm__left_caudate",
            "img_synthseg_norm__right_caudate",
            "img_synthseg_norm__left_putamen",
            "img_synthseg_norm__right_putamen",
            "img_synthseg_norm__left_pallidum",
            "img_synthseg_norm__right_pallidum",
            "img_synthseg_norm__left_accumbens_area",
            "img_synthseg_norm__right_accumbens_area",
            "img_synthseg_norm__left_ventral_DC",
            "img_synthseg_norm__right_ventral_DC",
            "img_synthseg_norm__brain_stem",
        ],
        "prefix": [],
    },
    "synthseg_cerebellum_brainstem": {
        "exact": [
            "img_synthseg_norm__brain_stem",
            "img_synthseg_norm__left_cerebellum_cortex",
            "img_synthseg_norm__right_cerebellum_cortex",
            "img_synthseg_norm__left_cerebellum_white_matter",
            "img_synthseg_norm__right_cerebellum_white_matter",
        ],
        "prefix": [],
    },
    "synthseg_cortical_parcels": {
        "exact": [],
        "prefix": ["img_synthseg_norm__ctx_lh_", "img_synthseg_norm__ctx_rh_"],
    },
}
FS_EXPLORATORY_VOLUME_BASES = [
    "fs_SubcorticalVolume_LhVesselVol",
    "fs_SubcorticalVolume_RhVesselVol",
    "fs_SubcorticalVolume_FifthVentVol",
    "fs_SubcorticalVolume_OpticChiasmVol",
]
FREESURFER_GROUPS = {
    "fs_cortical_thickness": {
        "exact": [],
        "prefix": ["img_fs_thick__"],
    },
    "fs_cortical_surface_area": {
        "exact": [],
        "prefix": ["img_fs_area_norm__"],
    },
    "fs_wmh_hypointensities": {
        "exact": [
            "img_fs_vol_norm__fs_GlobalVariables_WMHypointensitiesVol",
            "img_fs_vol_norm__fs_GlobalVariables_NonWMHypointensitiesVol",
        ],
        "prefix": [],
    },
    "fs_corpus_callosum": {
        "exact": [
            "img_fs_vol_norm__fs_SubcorticalVolume_CCAnteriorVol",
            "img_fs_vol_norm__fs_SubcorticalVolume_CCMidAnteriorVol",
            "img_fs_vol_norm__fs_SubcorticalVolume_CCCentralVol",
            "img_fs_vol_norm__fs_SubcorticalVolume_CCMidPosteriorVol",
            "img_fs_vol_norm__fs_SubcorticalVolume_CCPosteriorVol",
        ],
        "prefix": [],
    },
    "fs_choroid_plexus": {
        "exact": [
            "img_fs_vol_norm__fs_SubcorticalVolume_LhChoroidPVol",
            "img_fs_vol_norm__fs_SubcorticalVolume_RhChoroidPVol",
        ],
        "prefix": [],
    },
    "fs_exploratory": {
        "exact": [
            "img_fs_vol_norm_exploratory__fs_SubcorticalVolume_LhVesselVol",
            "img_fs_vol_norm_exploratory__fs_SubcorticalVolume_RhVesselVol",
            "img_fs_vol_norm_exploratory__fs_SubcorticalVolume_FifthVentVol",
            "img_fs_vol_norm_exploratory__fs_SubcorticalVolume_OpticChiasmVol",
        ],
        "prefix": [],
    },
}
VARIANT_SPECS = {
    "demographics_only": {
        "include_demographics": True,
        "feature_blocks": [],
        "include_bag": False,
        "reference_variant": None,
    },
    "demographics_synthseg": {
        "include_demographics": True,
        "feature_blocks": ["synthseg"],
        "include_bag": False,
        "reference_variant": "demographics_only",
    },
    "demographics_freesurfer": {
        "include_demographics": True,
        "feature_blocks": ["freesurfer"],
        "include_bag": False,
        "reference_variant": "demographics_only",
    },
    "demographics_bag": {
        "include_demographics": True,
        "feature_blocks": [],
        "include_bag": True,
        "reference_variant": "demographics_only",
    },
    "agecog_only": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": [],
        "include_bag": False,
        "reference_variant": None,
    },
    "agecog_synthseg": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": ["synthseg"],
        "include_bag": False,
        "reference_variant": "agecog_only",
    },
    "agecog_freesurfer": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": ["freesurfer"],
        "include_bag": False,
        "reference_variant": "agecog_only",
    },
    "agecog_mri": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": False,
        "reference_variant": "agecog_only",
    },
    "agecog_bag": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": [],
        "include_bag": True,
        "reference_variant": "agecog_only",
    },
    "agecog_mri_bag": {
        "include_demographics": False,
        "numeric_columns": ["AgeCog"],
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": True,
        "reference_variant": "agecog_mri",
    },
    "synthseg_only": {
        "include_demographics": False,
        "feature_blocks": ["synthseg"],
        "include_bag": False,
        "reference_variant": None,
    },
    "freesurfer_only": {
        "include_demographics": False,
        "feature_blocks": ["freesurfer"],
        "include_bag": False,
        "reference_variant": None,
    },
    "mri_only": {
        "include_demographics": False,
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": False,
        "reference_variant": "demographics_only",
    },
    "mri_bag": {
        "include_demographics": False,
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": True,
        "reference_variant": "mri_only",
    },
    "demographics_all_mri": {
        "include_demographics": True,
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": False,
        "reference_variant": "demographics_only",
    },
    "demographics_all_mri_bag": {
        "include_demographics": True,
        "feature_blocks": ["synthseg", "freesurfer"],
        "include_bag": True,
        "reference_variant": "demographics_all_mri",
    },
}
for group_name in SYNTHSEG_GROUPS:
    VARIANT_SPECS[f"{group_name}_only"] = {
        "include_demographics": False,
        "feature_blocks": [group_name],
        "include_bag": False,
        "reference_variant": None,
    }
    VARIANT_SPECS[f"demographics_{group_name}"] = {
        "include_demographics": True,
        "feature_blocks": [group_name],
        "include_bag": False,
        "reference_variant": f"{group_name}_only",
    }
for group_name in FREESURFER_GROUPS:
    VARIANT_SPECS[f"{group_name}_only"] = {
        "include_demographics": False,
        "feature_blocks": [group_name],
        "include_bag": False,
        "reference_variant": None,
    }
    VARIANT_SPECS[f"demographics_{group_name}"] = {
        "include_demographics": True,
        "feature_blocks": [group_name],
        "include_bag": False,
        "reference_variant": f"{group_name}_only",
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


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_targets(value: str) -> list[str]:
    targets = [part.strip() for part in value.split(",") if part.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("Expected at least one target.")
    return targets


def parse_models(value: str) -> list[str]:
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


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("Expected at least one feature variant.")
    unknown = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown feature variant(s): {','.join(unknown)}")
    return variants


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return tuple(parts)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson_r": pearson,
        "bias": float(np.mean(residual)),
        "n": int(len(y_true)),
    }


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical, categorical_columns))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model(numeric_columns: list[str], categorical_columns: list[str], estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("model", estimator),
        ]
    )


def transformed_feature_coefficient(model: Pipeline, feature_name: str) -> float:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return float("nan")
    preprocessor = model.named_steps["preprocessor"]
    names = list(preprocessor.get_feature_names_out())
    if feature_name not in names:
        return float("nan")
    index = names.index(feature_name)
    return float(estimator.coef_[index])


def transformed_coefficients(model: Pipeline) -> list[dict[str, float | str]]:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return []
    names = list(model.named_steps["preprocessor"].get_feature_names_out())
    coefs = np.asarray(estimator.coef_, dtype=float)
    return [
        {"transformed_feature": name, "coefficient": float(coef)}
        for name, coef in zip(names, coefs)
    ]


def selected_alpha(model: Pipeline) -> float:
    estimator = model.named_steps["model"]
    alpha = getattr(estimator, "alpha_", np.nan)
    try:
        return float(alpha)
    except TypeError:
        return float("nan")


def build_model_factories(args: argparse.Namespace) -> tuple[dict[str, Callable[[], object]], dict[str, str]]:
    factories: dict[str, Callable[[], object]] = {}
    skipped: dict[str, str] = {}
    alphas = np.asarray(args.alphas, dtype=float)

    if "Ridge" in args.models:
        factories["Ridge"] = lambda: RidgeCV(alphas=alphas)
    if "ElasticNet" in args.models:
        factories["ElasticNet"] = lambda: ElasticNetCV(
            l1_ratio=args.l1_ratios,
            alphas=alphas,
            cv=args.inner_folds,
            max_iter=args.elasticnet_max_iter,
            random_state=args.seed,
        )
    if "MLP" in args.models:
        factories["MLP"] = lambda: MLPRegressor(
            hidden_layer_sizes=args.mlp_hidden_layer_sizes,
            alpha=args.mlp_alpha,
            learning_rate_init=args.mlp_learning_rate,
            max_iter=args.mlp_max_iter,
            random_state=args.seed,
            early_stopping=False,
        )
    if "XGBoost" in args.models:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            skipped["XGBoost"] = f"not importable: {exc}"
        else:
            factories["XGBoost"] = lambda: XGBRegressor(
                n_estimators=args.boosting_estimators,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=args.seed,
                n_jobs=1,
            )
    if "LightGBM" in args.models:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            skipped["LightGBM"] = f"not importable: {exc}"
        else:
            factories["LightGBM"] = lambda: LGBMRegressor(
                n_estimators=args.boosting_estimators,
                learning_rate=0.05,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.seed,
                n_jobs=1,
                verbose=-1,
            )
    return factories, skipped


def load_modeling_table(args: argparse.Namespace) -> pd.DataFrame:
    master = pd.read_csv(args.master_table, sep="\t")
    cognition = pd.read_csv(args.cognition_table, sep="\t")
    bag = pd.read_csv(args.corrected_bag_csv)

    synthseg_features, _ = build_synthseg_features(master)
    synthseg_features = synthseg_features.add_prefix("img_")
    freesurfer_features, _ = build_freesurfer_features(master)
    freesurfer_features = freesurfer_features.add_prefix("img_")
    exploratory_source_cols = [
        col
        for base_col in FS_EXPLORATORY_VOLUME_BASES
        for col in (f"{base_col}_x", f"{base_col}_y", base_col)
        if col in master.columns
    ]
    exploratory_fs = wave_specific_freesurfer_frame(master, exploratory_source_cols)
    if not exploratory_fs.empty:
        icv = pd.to_numeric(master[ICV_COL], errors="coerce").replace(0, np.nan)
        exploratory_fs = exploratory_fs.div(icv, axis=0)
        exploratory_fs.columns = [
            f"img_fs_vol_norm_exploratory__{col}" for col in exploratory_fs.columns
        ]
    imaging = pd.concat(
        [
            master[["dataset", "subject_id", "wave"]].reset_index(drop=True),
            synthseg_features.reset_index(drop=True),
            freesurfer_features.reset_index(drop=True),
            exploratory_fs.reset_index(drop=True),
        ],
        axis=1,
    )
    imaging = imaging.drop_duplicates(["dataset", "subject_id", "wave"])

    demographics = master[
        ["dataset", "subject_id", "wave", *NUMERIC_BASE_COLUMNS, "Sex", "Race", "Ethnicity"]
    ].copy()
    demographics = demographics.drop_duplicates(["dataset", "subject_id", "wave"])
    cognition_cols = ["dataset", "subject_id", "wave", *args.targets]
    modeling = cognition[cognition_cols].merge(
        demographics, on=["dataset", "subject_id", "wave"], how="inner"
    )
    bag_cols = [
        "model_rank",
        "feature_set",
        "model_name",
        "dataset",
        "subject_id",
        "wave",
        "outer_fold",
        BAG_COLUMN,
    ]
    modeling = modeling.merge(
        bag[bag_cols], on=["dataset", "subject_id", "wave"], how="inner"
    )
    modeling = modeling.merge(imaging, on=["dataset", "subject_id", "wave"], how="left")
    numeric_cols = [*args.targets, *NUMERIC_BASE_COLUMNS, BAG_COLUMN]
    numeric_cols.extend(col for col in modeling.columns if col.startswith("img_"))
    for col in numeric_cols:
        if col in modeling.columns:
            modeling[col] = pd.to_numeric(modeling[col], errors="coerce")
    for col in ["Race", "Ethnicity", "Sex", "wave"]:
        modeling[col] = modeling[col].astype(object).where(modeling[col].notna(), np.nan)
    return modeling


def feature_columns_for_variant(df: pd.DataFrame, variant: str) -> list[str]:
    spec = VARIANT_SPECS[variant]
    columns = list(spec.get("numeric_columns", []))
    if spec["include_demographics"]:
        columns.extend(NUMERIC_BASE_COLUMNS)
    for block in spec["feature_blocks"]:
        if block == "synthseg":
            columns.extend(col for col in df.columns if col.startswith("img_synthseg_norm__"))
        elif block == "freesurfer":
            columns.extend(col for col in df.columns if col.startswith("img_fs_"))
        elif block in SYNTHSEG_GROUPS:
            group = SYNTHSEG_GROUPS[block]
            exact = [col for col in group["exact"] if col in df.columns]
            prefixed = [
                col
                for col in df.columns
                if any(col.startswith(prefix) for prefix in group["prefix"])
            ]
            columns.extend([*exact, *prefixed])
        elif block in FREESURFER_GROUPS:
            group = FREESURFER_GROUPS[block]
            exact = [col for col in group["exact"] if col in df.columns]
            prefixed = [
                col
                for col in df.columns
                if any(col.startswith(prefix) for prefix in group["prefix"])
            ]
            columns.extend([*exact, *prefixed])
    if spec["include_bag"]:
        columns.append(BAG_COLUMN)
    return list(dict.fromkeys(columns))


def categorical_columns_for_variant(variant: str) -> list[str]:
    return list(CATEGORICAL_BASE_COLUMNS) if VARIANT_SPECS[variant]["include_demographics"] else []


def coefficient_summary(values: list[float]) -> dict[str, float]:
    coeffs = np.asarray(values, dtype=float)
    valid_coeffs = coeffs[np.isfinite(coeffs)]
    if len(valid_coeffs) == 0:
        return {
            "bag_coefficient_mean": float("nan"),
            "bag_coefficient_std": float("nan"),
            "bag_coefficient_min": float("nan"),
            "bag_coefficient_max": float("nan"),
            "bag_coefficient_sign_consistency": float("nan"),
        }
    coeff_mean = float(np.mean(valid_coeffs))
    return {
        "bag_coefficient_mean": coeff_mean,
        "bag_coefficient_std": float(np.std(valid_coeffs, ddof=1)) if len(valid_coeffs) > 1 else 0.0,
        "bag_coefficient_min": float(np.min(valid_coeffs)),
        "bag_coefficient_max": float(np.max(valid_coeffs)),
        "bag_coefficient_sign_consistency": float(
            np.mean(np.sign(valid_coeffs) == np.sign(coeff_mean))
        ),
    }


def run_target_cv(
    df: pd.DataFrame,
    target: str,
    cognition_model: str,
    estimator_factory: Callable[[], object],
    alphas: list[float],
    folds: int,
    selected_variants: list[str],
) -> tuple[
    pd.DataFrame,
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float | str]],
]:
    rows = df[df[target].notna() & df["AgeCog"].notna()].copy()
    if rows.empty:
        raise ValueError(f"No usable rows for target: {target}")

    y = pd.to_numeric(rows[target], errors="coerce").to_numpy(dtype=float)
    groups = rows["subject_id"].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(f"Need at least two subjects for target: {target}")

    variant_predictions = {variant: np.full(len(rows), np.nan) for variant in selected_variants}
    fold_numbers = np.full(len(rows), -1)
    fold_coefficients: list[dict[str, float]] = []
    linear_coefficients: list[dict[str, float | str]] = []

    cv = GroupKFold(n_splits=n_splits)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for fold, (train_idx, test_idx) in enumerate(cv.split(rows, y, groups), start=1):
            train = rows.iloc[train_idx]
            test = rows.iloc[test_idx]
            fold_numbers[test_idx] = fold
            for variant in selected_variants:
                numeric_cols = feature_columns_for_variant(rows, variant)
                categorical = categorical_columns_for_variant(variant)
                model = make_model(numeric_cols, categorical, estimator_factory())
                fitted = model.fit(train, y[train_idx])
                variant_predictions[variant][test_idx] = fitted.predict(test)
                for coef_row in transformed_coefficients(fitted):
                    linear_coefficients.append(
                        {
                            "fold": int(fold),
                            "variant": variant,
                            "alpha": selected_alpha(fitted),
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            **coef_row,
                        }
                    )
                if VARIANT_SPECS[variant]["include_bag"]:
                    fold_coefficients.append(
                        {
                            "fold": int(fold),
                            "variant": variant,
                            "bag_coefficient": transformed_feature_coefficient(
                                fitted, "num__corrected_BAG"
                            ),
                            "alpha": selected_alpha(fitted),
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                        }
                    )

    out = rows[
        [
            "dataset",
            "subject_id",
            "wave",
            "model_rank",
            "feature_set",
            "model_name",
            BAG_COLUMN,
            *NUMERIC_BASE_COLUMNS,
            "Sex",
            "Race",
            "Ethnicity",
        ]
    ].copy()
    out = out.rename(columns={"feature_set": "bag_feature_set", "model_name": "bag_source_model"})
    out["target"] = target
    out["cognition_model"] = cognition_model
    out["target_value"] = y
    out["outer_fold"] = fold_numbers.astype(int)
    long_predictions = []
    metrics_by_variant = {}
    for variant, pred in variant_predictions.items():
        variant_out = out.copy()
        variant_out["feature_variant"] = variant
        variant_out["predicted"] = pred
        variant_out["residual"] = pred - y
        variant_out["absolute_error"] = np.abs(variant_out["residual"])
        long_predictions.append(variant_out)
        metrics_by_variant[variant] = regression_metrics(y, pred)

    comparison_rows = []
    for variant, metrics in metrics_by_variant.items():
        reference = VARIANT_SPECS[variant]["reference_variant"]
        row = {
            "feature_variant": variant,
            "reference_variant": reference,
            **metrics,
            "folds": int(n_splits),
        }
        if reference is None or reference not in metrics_by_variant:
            row["delta_r2"] = float("nan")
            row["delta_mae"] = float("nan")
        else:
            ref = metrics_by_variant[reference]
            row["delta_r2"] = float(metrics["r2"] - ref["r2"])
            row["delta_mae"] = float(ref["mae"] - metrics["mae"])
        variant_coefficients = [
            item["bag_coefficient"]
            for item in fold_coefficients
            if item["variant"] == variant
        ]
        row.update(coefficient_summary(variant_coefficients))
        comparison_rows.append(row)

    return (
        pd.concat(long_predictions, ignore_index=True),
        comparison_rows,
        fold_coefficients,
        linear_coefficients,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-table",
        type=Path,
        default=Path("../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv"),
    )
    parser.add_argument(
        "--cognition-table",
        type=Path,
        default=Path("../smri-dataset/DLBS/derivatives/master_tables/dlbs_cognitive_composites.tsv"),
    )
    parser.add_argument(
        "--corrected-bag-csv",
        type=Path,
        default=Path("../smri-dataset/DLBS/qc/brainage_tabular_models/corrected_bag_top2_models.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../smri-dataset/DLBS/qc/cognition_from_demographics_imaging_bag"),
    )
    parser.add_argument("--targets", type=parse_targets, default=COGNITIVE_TARGETS)
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants(
            "demographics_only,demographics_bag,mri_only,"
            "demographics_all_mri,demographics_all_mri_bag"
        ),
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=parse_models("Ridge,ElasticNet,XGBoost,LightGBM,MLP"),
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
    args.master_table = resolve_from_repo(args.master_table)
    args.cognition_table = resolve_from_repo(args.cognition_table)
    args.corrected_bag_csv = resolve_from_repo(args.corrected_bag_csv)
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modeling = load_modeling_table(args)
    modeling.to_csv(output_dir / "modeling_data.csv", index=False)
    factories, skipped_models = build_model_factories(args)
    if not factories:
        raise SystemExit(f"No requested cognition models are available. Skipped: {skipped_models}")
    predictions = []
    comparison_rows = []
    fold_rows = []
    linear_coef_rows = []
    selected_variants = args.variants
    uses_bag = any(VARIANT_SPECS[variant]["include_bag"] for variant in selected_variants)
    grouped = modeling.groupby(["model_rank", "feature_set", "model_name"], sort=True)
    for (model_rank, feature_set, model_name), bag_df in grouped:
        if not uses_bag and (model_rank, feature_set, model_name) != next(iter(grouped.groups)):
            continue
        for target in args.targets:
            for cognition_model, factory in factories.items():
                print(
                    f"Running rank={model_rank} {feature_set}/{model_name} "
                    f"target={target} cognition_model={cognition_model}"
                )
                pred, summaries, coefficients, linear_coefficients = run_target_cv(
                    bag_df,
                    target,
                    cognition_model,
                    factory,
                    args.alphas,
                    args.folds,
                    selected_variants,
                )
                predictions.append(pred)
                for summary in summaries:
                    comparison_rows.append(
                        {
                            "model_rank": int(model_rank),
                            "bag_feature_set": feature_set,
                            "target": target,
                            "bag_source_model": model_name,
                            "cognition_model": cognition_model,
                            **summary,
                        }
                    )
                for item in coefficients:
                    fold_rows.append(
                        {
                            "model_rank": int(model_rank),
                            "bag_feature_set": feature_set,
                            "bag_source_model": model_name,
                            "target": target,
                            "cognition_model": cognition_model,
                            **item,
                        }
                    )
                for item in linear_coefficients:
                    linear_coef_rows.append(
                        {
                            "model_rank": int(model_rank),
                            "bag_feature_set": feature_set,
                            "bag_source_model": model_name,
                            "target": target,
                            "cognition_model": cognition_model,
                            **item,
                        }
                    )

    predictions_df = pd.concat(predictions, ignore_index=True)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["target", "model_rank", "cognition_model", "feature_variant"], kind="stable"
    )
    fold_coefficients = pd.DataFrame(fold_rows)
    linear_coefficients = pd.DataFrame(linear_coef_rows)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    fold_coefficients.to_csv(output_dir / "bag_coefficients_by_fold.csv", index=False)
    linear_coefficients.to_csv(output_dir / "linear_coefficients_by_fold.csv", index=False)
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "master_table": str(args.master_table),
        "cognition_table": str(args.cognition_table),
        "corrected_bag_csv": str(args.corrected_bag_csv),
        "output_dir": str(output_dir),
        "targets": args.targets,
        "variants_requested": selected_variants,
        "models_requested": args.models,
        "models_run": sorted(factories),
        "models_skipped": skipped_models,
        "base_numeric_columns": NUMERIC_BASE_COLUMNS,
        "categorical_columns": CATEGORICAL_BASE_COLUMNS,
        "feature_variants": VARIANT_SPECS,
        "synthseg_groups": SYNTHSEG_GROUPS,
        "freesurfer_groups": FREESURFER_GROUPS,
        "bag_column": BAG_COLUMN,
        "model": "Configured estimator with median-imputed/scaled numeric columns and one-hot categorical columns",
        "comparison": comparison_rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'modeling_data.csv'}")
    print(f"Wrote {output_dir / 'comparison.csv'}")
    print(f"Wrote {output_dir / 'bag_coefficients_by_fold.csv'}")
    print(f"Wrote {output_dir / 'linear_coefficients_by_fold.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(
        comparison[
            [
                "model_rank",
                "bag_feature_set",
                "bag_source_model",
                "cognition_model",
                "target",
                "feature_variant",
                "r2",
                "delta_r2",
                "mae",
                "delta_mae",
                "bag_coefficient_mean",
                "bag_coefficient_std",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
