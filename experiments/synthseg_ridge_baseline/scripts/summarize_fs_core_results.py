from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/teamspace/studios/this_studio/smri-dataset/DLBS/qc")


def fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def best_rows(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    rows = (
        df.sort_values(["target", "feature_variant", "r2"], ascending=[True, True, False])
        .groupby(["target", "feature_variant"], as_index=False)
        .head(1)
    )
    return rows[rows["feature_variant"].isin(variants)].sort_values(["target", "feature_variant"])


def main() -> None:
    brain = pd.read_csv(BASE / "brainage_tabular_models_fs_core" / "comparison.csv")
    brain_preds = pd.read_csv(BASE / "brainage_tabular_models_fs_core" / "predictions.csv")
    brain_extra = []
    for (feature_set, model_name), group in brain_preds.groupby(["feature_set", "model_name"]):
        age = group["true_age"]
        raw_resid = group["raw_residual"]
        corrected_resid = group["corrected_residual"]
        brain_extra.append(
            {
                "feature_set": feature_set,
                "model_name": model_name,
                "raw_bag_age_slope": raw_resid.cov(age) / age.var(),
                "corrected_bag_age_slope": corrected_resid.cov(age) / age.var(),
                "corrected_bag_age_corr": corrected_resid.corr(age),
                "raw_pred_age_std": group["raw_predicted_age"].std(),
                "corrected_pred_age_std": group["corrected_predicted_age"].std(),
                "true_age_std": age.std(),
            }
        )
    brain = brain.merge(pd.DataFrame(brain_extra), on=["feature_set", "model_name"], how="left")
    brain_cols = [
        "feature_set",
        "model_name",
        "raw_r2",
        "raw_mae",
        "raw_rmse",
        "raw_pearson_r",
        "raw_bag_age_slope",
        "corrected_r2",
        "corrected_mae",
        "corrected_rmse",
        "corrected_pearson_r",
        "corrected_bag_age_slope",
        "corrected_bag_age_corr",
        "raw_pred_age_std",
        "corrected_pred_age_std",
        "true_age_std",
    ]
    print("BRAINAGE_TOP8")
    print(fmt(brain.sort_values("corrected_r2", ascending=False).head(8)[brain_cols]))

    no_bag = pd.read_csv(BASE / "cognition_fs_core_no_bag" / "comparison.csv")
    bag = pd.read_csv(BASE / "cognition_fs_core_bag" / "comparison.csv")
    print(f"\nNO_BAG_ROWS {no_bag.shape}")
    print(f"BAG_ROWS {bag.shape}")

    no_bag_variants = [
        "synthseg_only",
        "freesurfer_only",
        "mri_only",
        "agecog_only",
        "agecog_synthseg",
        "agecog_freesurfer",
        "agecog_mri",
        "demographics_only",
        "demographics_synthseg",
        "demographics_freesurfer",
        "demographics_all_mri",
    ]
    print("\nNO_BAG_BEST_PER_TARGET_VARIANT")
    print(fmt(best_rows(no_bag, no_bag_variants)[["target", "feature_variant", "cognition_model", "r2", "mae"]]))

    bag_variants = [
        "mri_only",
        "mri_bag",
        "agecog_only",
        "agecog_bag",
        "agecog_mri",
        "agecog_mri_bag",
        "demographics_only",
        "demographics_bag",
        "demographics_all_mri",
        "demographics_all_mri_bag",
    ]
    bag_cols = [
        "target",
        "feature_variant",
        "cognition_model",
        "r2",
        "delta_r2",
        "mae",
        "delta_mae",
        "bag_coefficient_mean",
        "bag_coefficient_std",
    ]
    for rank in [1, 2]:
        print(f"\nBAG_RANK{rank}_BEST_PER_TARGET_VARIANT")
        rows = best_rows(bag[bag["model_rank"].eq(rank)], bag_variants)
        print(fmt(rows[bag_cols]))


if __name__ == "__main__":
    main()
