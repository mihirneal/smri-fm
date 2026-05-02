from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "synthseg_cross_domain_cv" / "scripts"
)
sys.path.insert(0, str(EXPERIMENT_SCRIPTS_DIR))

from manifest import DLBS_DATASET, DLBS_DOMAIN, OPENNEURO_DOMAIN
from modeling import select_model_via_grouped_cv
from splits import build_domain_splits


def _write_synthseg_files(base: Path, stem: str, qc_score: float, tcv: float = 1000.0) -> tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    volumes_path = base / f"{stem}_desc-synthseg_volumes.tsv"
    qc_path = base / f"{stem}_desc-synthseg_qc.tsv"
    pd.DataFrame(
        [
            {"region": "left hippocampus", "volume_mm3": 25.0},
            {"region": "right hippocampus", "volume_mm3": 35.0},
            {"region": "TCV", "volume_mm3": tcv},
        ]
    ).to_csv(volumes_path, sep="\t", index=False)
    pd.DataFrame(
        [
            {"structure": "general white matter", "qc_score": qc_score},
            {"structure": "general grey matter", "qc_score": qc_score},
        ]
    ).to_csv(qc_path, sep="\t", index=False)
    return volumes_path, qc_path


def _make_manifest_row(
    root: Path,
    row_id: str,
    dataset: str,
    sub: str,
    ses: str,
    run: str,
    domain_label: str,
    synthseg_qc: float = 0.9,
) -> dict[str, object]:
    raw_rel = Path(dataset) / f"sub-{sub}" / "anat" / f"sub-{sub}_T1w.nii.gz"
    if domain_label == DLBS_DOMAIN:
        raw_rel = Path(f"sub-{sub}") / f"ses-{ses}" / "anat" / f"sub-{sub}_ses-{ses}_run-{run}_T1w.nii.gz"
    preproc_root = root / "processed" / ("openneuro" if domain_label == OPENNEURO_DOMAIN else "DLBS")
    synthseg_parent = preproc_root / "derivatives" / "synthseg" / raw_rel.parent
    volumes_path, qc_path = _write_synthseg_files(
        synthseg_parent,
        raw_rel.name.replace(".nii.gz", ""),
        synthseg_qc,
    )

    return {
        "row_id": row_id,
        "subject_id": f"{dataset}:{sub}",
        "dataset": dataset,
        "sub": sub,
        "ses": ses,
        "run": run,
        "age": float(20 + int(sub) % 50),
        "sex": "F" if int(sub) % 2 else "M",
        "raw_t1_path": str(root / "raw-placeholder" / raw_rel),
        "preproc_t1_path": str(root / "preproc-placeholder" / raw_rel),
        "brain_mask_path": str(root / "mask-placeholder" / raw_rel),
        "synthseg_volumes_tsv_path": str(volumes_path),
        "synthseg_qc_tsv_path": str(qc_path),
        "domain_label": domain_label,
        "source_relpath": str(raw_rel),
    }


def _build_experiment_manifest(tmp_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(1, 7):
        rows.append(
            _make_manifest_row(
                tmp_path,
                f"open-{idx}-a",
                f"ds0100{idx}",
                f"{idx}",
                "",
                "",
                OPENNEURO_DOMAIN,
            )
        )
        duplicate = _make_manifest_row(
            tmp_path,
            f"open-{idx}-b",
            f"ds0100{idx}",
            f"{idx}",
            "",
            "",
            OPENNEURO_DOMAIN,
        )
        duplicate["row_id"] = f"open-{idx}-b"
        rows.append(duplicate)

    for idx in range(1, 7):
        rows.append(
            _make_manifest_row(
                tmp_path,
                f"dlbs-{idx}",
                DLBS_DATASET,
                f"{2000 + idx}",
                f"wave{idx % 3 + 1}",
                "1",
                DLBS_DOMAIN,
            )
        )

    return pd.DataFrame(rows)


def test_select_model_via_grouped_cv_keeps_groups_in_single_fold() -> None:
    X = pd.DataFrame({"feature": np.linspace(0.0, 1.0, 10)})
    y = pd.Series(np.linspace(20.0, 60.0, 10))
    groups = pd.Series(["sub1", "sub1", "sub2", "sub2", "sub3", "sub3", "sub4", "sub4", "sub5", "sub5"])

    result = select_model_via_grouped_cv(
        X=X,
        y=y,
        groups=groups,
        model_name="DummyRegressor",
        seed=13,
        n_splits=5,
    )

    fold_df = pd.DataFrame({"group": groups, "fold": result.fold_assignments})
    assert result.n_splits == 5
    assert len(result.fold_scores) == 5
    assert fold_df.groupby("group")["fold"].nunique().max() == 1


def test_cross_domain_runner_rejects_subject_overlap(tmp_path: Path) -> None:
    manifest = _build_experiment_manifest(tmp_path)
    manifest.loc[0, "subject_id"] = manifest.loc[12, "subject_id"]
    manifest_path = tmp_path / "manifest_overlap.csv"
    manifest.to_csv(manifest_path, index=False)

    proc = subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "experiments/synthseg_cross_domain_cv/scripts/run_experiment.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "overlap" in proc.stderr.lower()


@pytest.mark.filterwarnings("ignore:.*ConvergenceWarning.*")
def test_cross_domain_runner_smoke(tmp_path: Path) -> None:
    manifest = _build_experiment_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    splits_path = tmp_path / "splits.csv"
    output_dir = tmp_path / "outputs"

    manifest.to_csv(manifest_path, index=False)
    build_domain_splits(manifest, seed=17, val_frac=0.2, test_frac=0.2).to_csv(
        splits_path, index=False
    )

    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "experiments/synthseg_cross_domain_cv/scripts/run_experiment.py",
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
            "--models",
            "DummyRegressor,LinearRegression,Ridge",
        ],
        check=True,
    )

    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert len(comparison) == 12
    assert set(comparison["source_domain"]) == {DLBS_DOMAIN, OPENNEURO_DOMAIN}
    assert set(comparison["target_evaluation"]) == {"dlbs_holdout", "openneuro_nodlbs_holdout"}
    assert set(
        zip(comparison["source_domain"], comparison["target_evaluation"], strict=False)
    ) == {
        (DLBS_DOMAIN, "dlbs_holdout"),
        (DLBS_DOMAIN, "openneuro_nodlbs_holdout"),
        (OPENNEURO_DOMAIN, "openneuro_nodlbs_holdout"),
        (OPENNEURO_DOMAIN, "dlbs_holdout"),
    }
    assert (comparison["cv_n_splits"] == 5).all()
    assert (comparison["correction_source"] == "source_dev_grouped_cv").all()
    assert {"raw_mae", "corrected_mae", "source_dev_cv_raw_mae", "source_dev_cv_corrected_mae"} <= set(
        comparison.columns
    )

    saved_manifest = pd.read_csv(output_dir / "manifest.csv")
    openneuro = saved_manifest[saved_manifest["domain_label"] == OPENNEURO_DOMAIN]
    assert (openneuro["dataset"] == DLBS_DATASET).sum() == 0

    run_dir = (
        output_dir
        / "openneuro_nodlbs__to__dlbs_holdout"
        / "synthseg"
        / "Ridge"
    )
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "source_dev_cv_predictions.csv").exists()
    assert (run_dir / "exclusions.csv").exists()

    cv_predictions = pd.read_csv(run_dir / "source_dev_cv_predictions.csv")
    assert cv_predictions.groupby("subject_id")["cv_fold"].nunique().max() == 1
    assert set(cv_predictions["cv_fold"]) == {1, 2, 3, 4, 5}

    metrics = json.loads((run_dir / "metrics.json").read_text())
    raw_cv_mae = np.mean(np.abs(cv_predictions["predicted_age_cv"] - cv_predictions["age"]))
    corrected_cv_mae = np.mean(
        np.abs(cv_predictions["predicted_age_cv_corrected"] - cv_predictions["age"])
    )
    assert metrics["source_dev_cv_raw_mae"] == pytest.approx(raw_cv_mae)
    assert metrics["source_dev_cv_corrected_mae"] == pytest.approx(corrected_cv_mae)
