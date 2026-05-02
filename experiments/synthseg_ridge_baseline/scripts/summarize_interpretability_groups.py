from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/teamspace/studios/this_studio/smri-dataset/DLBS/qc")


def summarize(modality: str, group_path: Path, demo_path: Path) -> pd.DataFrame:
    group_df = pd.read_csv(group_path)
    demo_df = pd.read_csv(demo_path)

    only = group_df[group_df["feature_variant"].str.endswith("_only")].copy()
    only["group"] = only["feature_variant"].str.replace("_only", "", regex=False)
    best_only = (
        only.sort_values(["target", "group", "r2"], ascending=[True, True, False])
        .groupby(["target", "group"], as_index=False)
        .head(1)
    )

    demo = demo_df[demo_df["feature_variant"].ne("demographics_only")].copy()
    demo["group"] = demo["feature_variant"].str.replace("demographics_", "", regex=False)
    demo_base = demo_df[demo_df["feature_variant"].eq("demographics_only")][
        ["target", "cognition_model", "r2"]
    ].rename(columns={"r2": "demo_only_r2"})
    demo = demo.merge(demo_base, on=["target", "cognition_model"], how="left")
    demo["delta_r2_vs_demo"] = demo["r2"] - demo["demo_only_r2"]
    best_demo = (
        demo.sort_values(["target", "group", "r2"], ascending=[True, True, False])
        .groupby(["target", "group"], as_index=False)
        .head(1)
    )

    out = best_demo[
        ["target", "group", "cognition_model", "r2", "demo_only_r2", "delta_r2_vs_demo", "mae"]
    ].merge(
        best_only[["target", "group", "cognition_model", "r2", "mae"]].rename(
            columns={
                "cognition_model": "group_only_model",
                "r2": "group_only_r2",
                "mae": "group_only_mae",
            }
        ),
        on=["target", "group"],
        how="left",
    )
    out.insert(0, "modality", modality)
    return out


def main() -> None:
    specs = [
        (
            "SynthSeg",
            BASE / "cognition_synthseg_interpretability_groups_top3" / "comparison.csv",
            BASE / "cognition_synthseg_interpretability_groups_top3_with_demo_baseline" / "comparison.csv",
        ),
        (
            "FreeSurfer",
            BASE / "cognition_freesurfer_interpretability_groups_top3" / "comparison.csv",
            BASE / "cognition_freesurfer_interpretability_groups_top3_with_demo_baseline" / "comparison.csv",
        ),
    ]
    summary = pd.concat([summarize(*spec) for spec in specs], ignore_index=True)
    summary = summary.sort_values(
        ["target", "modality", "delta_r2_vs_demo"], ascending=[True, True, False]
    )
    output = BASE / "interpretability_all_group_summary.csv"
    summary.to_csv(output, index=False)
    print(f"WROTE {output}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
