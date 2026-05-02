from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "synthseg_ridge_baseline" / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from run_brainage_models import (  # noqa: E402
    apply_age_bias_correction,
    build_feature_bundle,
    fit_age_bias_correction,
)


def _brainage_table(n_subjects: int = 12) -> pd.DataFrame:
    rows = []
    for idx in range(n_subjects):
        age = 45.0 + idx
        icv = 1000.0 + 10.0 * idx
        rows.append(
            {
                "dataset": "DLBS",
                "subject_id": f"sub-{idx:04d}",
                "wave": f"wave{idx % 3 + 1}",
                "AgeMRI": age,
                "synthseg_vol_total_intracranial": icv,
                "synthseg_vol_left_hippocampus": 20.0 + idx,
                "synthseg_vol_right_hippocampus": 22.0 + idx,
                "fs_CorticalThickness_HasData": 1,
                "fs_CorticalThickness_NumScores": 2,
                "fs_CorticalThickness_Thickness": 2.5,
                "fs_CorticalThickness_LhBanksstsThick": 2.0 + idx / 100.0,
                "fs_CorticalThickness_RhBanksstsThick": 2.1 + idx / 100.0,
                "fs_GMVolume_HasData": 1,
                "fs_GMVolume_NumScores": 2,
                "fs_GMVolume_Volume": 300.0,
                "fs_GMVolume_LhBanksstsVol": 100.0 + idx,
                "fs_GMVolume_RhBanksstsVol": 110.0 + idx,
            }
        )
    rows[1]["fs_GMVolume_RhBanksstsVol"] = np.nan
    return pd.DataFrame(rows)


def test_feature_builders_icv_normalize_volumes_and_keep_thickness_raw() -> None:
    df = _brainage_table()

    synthseg = build_feature_bundle(df, "synthseg")
    assert "synthseg_norm__left_hippocampus" in synthseg.x.columns
    assert np.isclose(synthseg.x.iloc[0]["synthseg_norm__left_hippocampus"], 20.0 / 1000.0)
    assert all("total_intracranial" not in col for col in synthseg.x.columns)

    freesurfer = build_feature_bundle(df, "freesurfer")
    assert "fs_thick__fs_CorticalThickness_LhBanksstsThick" in freesurfer.x.columns
    assert "fs_gmvol_norm__fs_GMVolume_LhBanksstsVol" in freesurfer.x.columns
    assert np.isclose(
        freesurfer.x.iloc[0]["fs_thick__fs_CorticalThickness_LhBanksstsThick"], 2.0
    )
    assert np.isclose(
        freesurfer.x.iloc[0]["fs_gmvol_norm__fs_GMVolume_LhBanksstsVol"], 100.0 / 1000.0
    )
    assert all("HasData" not in col and "NumScores" not in col for col in freesurfer.x.columns)
    assert all(not col.endswith("_Volume") for col in freesurfer.x.columns)

    combined = build_feature_bundle(df, "combined")
    assert combined.x.shape[1] == synthseg.x.shape[1] + freesurfer.x.shape[1]


def test_age_bias_correction_removes_train_linear_bias() -> None:
    y_true = np.array([40.0, 50.0, 60.0, 70.0])
    y_pred = y_true + (0.2 * y_true - 5.0)

    correction = fit_age_bias_correction(y_true, y_pred)
    corrected = apply_age_bias_correction(y_true, y_pred, correction)

    assert np.isclose(correction["alpha"], 0.2)
    assert np.isclose(correction["beta"], -5.0)
    assert np.allclose(corrected, y_true)


def test_smoke_brainage_tabular_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "master.tsv"
    output_dir = tmp_path / "out"
    _brainage_table(n_subjects=12).to_csv(input_path, sep="\t", index=False)

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = "/tmp/uv-cache"
    subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(SCRIPT_DIR / "run_brainage_models.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--feature-sets",
            "synthseg,freesurfer,combined",
            "--models",
            "Ridge,ElasticNet,MLP",
            "--folds",
            "3",
            "--inner-folds",
            "3",
            "--mlp-max-iter",
            "50",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    predictions = pd.read_csv(output_dir / "predictions.csv")
    comparison = pd.read_csv(output_dir / "comparison.csv")
    corrected_bag = pd.read_csv(output_dir / "corrected_bag_top2_models.csv")
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert {"synthseg", "freesurfer", "combined"} == set(comparison["feature_set"])
    assert {"Ridge", "ElasticNet", "MLP"} == set(comparison["model_name"])
    assert {
        "raw_predicted_age",
        "corrected_predicted_age",
        "raw_residual",
        "corrected_residual",
    } <= set(predictions.columns)
    assert {"model_rank", "AgeMRI", "raw_BAG", "corrected_BAG"} <= set(corrected_bag.columns)
    assert set(corrected_bag["model_rank"]) == {1, 2}
    assert corrected_bag.groupby(["model_rank", "subject_id", "wave"]).size().max() == 1
    assert metrics["models_skipped"] == {}
