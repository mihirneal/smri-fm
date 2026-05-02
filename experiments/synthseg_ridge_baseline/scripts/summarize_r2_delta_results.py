from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/teamspace/studios/this_studio/smri-dataset/DLBS/qc")


def best(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["target", "feature_variant", "r2"], ascending=[True, True, False])
        .groupby(["target", "feature_variant"], as_index=False)
        .head(1)
        .copy()
    )


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    brain = pd.read_csv(BASE / "brainage_tabular_models_fs_core" / "comparison.csv")
    brain["delta_r2_corrected_minus_raw"] = brain["corrected_r2"] - brain["raw_r2"]
    print_table(
        "BRAIN_AGE",
        brain.sort_values("corrected_r2", ascending=False)[
            ["feature_set", "model_name", "raw_r2", "corrected_r2", "delta_r2_corrected_minus_raw"]
        ],
    )

    no = best(pd.read_csv(BASE / "cognition_fs_core_no_bag" / "comparison.csv"))
    bag = best(pd.read_csv(BASE / "cognition_fs_core_bag" / "comparison.csv"))

    demo_base = no[no["feature_variant"].eq("demographics_only")][["target", "r2"]].rename(
        columns={"r2": "base_r2"}
    )
    age_base = no[no["feature_variant"].eq("agecog_only")][["target", "r2"]].rename(
        columns={"r2": "base_r2"}
    )
    mri_base = no[no["feature_variant"].eq("mri_only")][["target", "r2"]].rename(columns={"r2": "base_r2"})

    demo_variants = [
        "demographics_only",
        "demographics_synthseg",
        "demographics_freesurfer",
        "demographics_all_mri",
    ]
    age_variants = ["agecog_only", "agecog_synthseg", "agecog_freesurfer", "agecog_mri"]
    mri_variants = ["synthseg_only", "freesurfer_only", "mri_only"]

    for title, variants, base_df in [
        ("COGNITION_DEMOGRAPHICS_FEATURES delta_vs_demographics_only", demo_variants, demo_base),
        ("COGNITION_AGECOG_FEATURES delta_vs_agecog_only", age_variants, age_base),
        ("COGNITION_MRI_ONLY_FEATURES delta_vs_mri_only", mri_variants, mri_base),
    ]:
        rows = no[no["feature_variant"].isin(variants)].merge(base_df, on="target", how="left")
        rows["delta_r2"] = rows["r2"] - rows["base_r2"]
        print_table(title, rows[["target", "feature_variant", "cognition_model", "r2", "delta_r2"]])

    bag_variants = [
        "demographics_only",
        "demographics_bag",
        "demographics_all_mri",
        "demographics_all_mri_bag",
        "mri_only",
        "mri_bag",
        "agecog_only",
        "agecog_bag",
        "agecog_mri",
        "agecog_mri_bag",
    ]
    for rank in [1, 2]:
        rows = bag[bag["model_rank"].eq(rank) & bag["feature_variant"].isin(bag_variants)]
        print_table(
            f"COGNITION_BAG_RANK{rank} delta_from_script",
            rows[
                [
                    "target",
                    "feature_variant",
                    "cognition_model",
                    "r2",
                    "delta_r2",
                    "bag_coefficient_mean",
                    "bag_coefficient_std",
                ]
            ],
        )


if __name__ == "__main__":
    main()
