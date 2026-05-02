from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/teamspace/studios/this_studio/smri-dataset/DLBS/qc")


def cognition_rows(label: str, rel_path: str) -> pd.DataFrame:
    df = pd.read_csv(BASE / rel_path / "comparison.csv")
    df = df.copy()
    df["experiment"] = label
    df["mse"] = df["rmse"] ** 2
    keep = [
        "experiment",
        "target",
        "feature_variant",
        "cognition_model",
        "r2",
        "mse",
        "pearson_r",
        "mae",
        "rmse",
        "delta_r2",
        "n",
        "model_rank",
        "bag_feature_set",
        "bag_source_model",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep]


def brainage_rows() -> pd.DataFrame:
    df = pd.read_csv(BASE / "brainage_tabular_models_fs_core" / "comparison.csv")
    rows = []
    for _, row in df.iterrows():
        for stage in ["raw", "corrected"]:
            rows.append(
                {
                    "experiment": f"brain_age_{stage}",
                    "target": "AgeMRI",
                    "feature_variant": row["feature_set"],
                    "cognition_model": row["model_name"],
                    "r2": row[f"{stage}_r2"],
                    "mse": row[f"{stage}_rmse"] ** 2,
                    "pearson_r": row[f"{stage}_pearson_r"],
                    "mae": row[f"{stage}_mae"],
                    "rmse": row[f"{stage}_rmse"],
                    "delta_r2": pd.NA,
                    "n": row[f"{stage}_n"],
                    "model_rank": pd.NA,
                    "bag_feature_set": pd.NA,
                    "bag_source_model": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def best_per_variant(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["experiment", "target", "feature_variant", "r2"], ascending=[True, True, True, False])
        .groupby(["experiment", "target", "feature_variant"], as_index=False)
        .head(1)
    )


def main() -> None:
    frames = [
        brainage_rows(),
        cognition_rows("cognition_core_no_bag", "cognition_fs_core_no_bag"),
        cognition_rows("cognition_core_bag", "cognition_fs_core_bag"),
        cognition_rows("interpretability_synthseg_groups", "cognition_synthseg_interpretability_groups_top3"),
        cognition_rows("interpretability_freesurfer_groups", "cognition_freesurfer_interpretability_groups_top3"),
        cognition_rows(
            "interpretability_synthseg_groups_demo_baseline",
            "cognition_synthseg_interpretability_groups_top3_with_demo_baseline",
        ),
        cognition_rows(
            "interpretability_freesurfer_groups_demo_baseline",
            "cognition_freesurfer_interpretability_groups_top3_with_demo_baseline",
        ),
    ]
    all_metrics = pd.concat(frames, ignore_index=True)
    out = BASE / "all_experiment_metrics_detailed.csv"
    all_metrics.to_csv(out, index=False)

    best = best_per_variant(all_metrics)
    best_out = BASE / "all_experiment_metrics_best_per_variant.csv"
    best.to_csv(best_out, index=False)

    print(f"WROTE {out}")
    print(f"WROTE {best_out}")

    brain = all_metrics[
        all_metrics["experiment"].eq("brain_age_corrected")
        & all_metrics["cognition_model"].isin(["Ridge", "ElasticNet", "LightGBM", "XGBoost"])
    ].sort_values(["feature_variant", "cognition_model"])
    print("\nBRAIN_AGE_CORRECTED")
    print(
        brain[
            ["feature_variant", "cognition_model", "r2", "mse", "pearson_r", "mae", "rmse"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    core_variants = [
        "demographics_only",
        "demographics_bag",
        "demographics_synthseg",
        "demographics_freesurfer",
        "demographics_all_mri",
        "demographics_all_mri_bag",
        "synthseg_only",
        "freesurfer_only",
        "mri_only",
        "mri_bag",
    ]
    core = best[
        best["experiment"].isin(["cognition_core_no_bag", "cognition_core_bag"])
        & best["feature_variant"].isin(core_variants)
    ].sort_values(["target", "feature_variant", "experiment"])
    print("\nCORE_COGNITION_BEST")
    print(
        core[
            ["experiment", "target", "feature_variant", "cognition_model", "r2", "mse", "pearson_r"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    interp = pd.read_csv(BASE / "interpretability_all_group_summary.csv")
    interp = interp.sort_values(["target", "modality", "delta_r2_vs_demo"], ascending=[True, True, False])
    print("\nINTERPRETABILITY_DEMO_DELTA")
    print(
        interp[
            [
                "modality",
                "target",
                "group",
                "cognition_model",
                "r2",
                "demo_only_r2",
                "delta_r2_vs_demo",
                "group_only_r2",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
