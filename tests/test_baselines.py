from __future__ import annotations

from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from baselines.manifest import DLBS_DATASET, DLBS_DOMAIN, OPENNEURO_DOMAIN, build_manifest
from baselines.modeling import apply_bag_age_bias_correction, fit_bag_age_bias_correction
from baselines.radiomics import zscore_intensity_inside_mask
from baselines.splits import build_domain_splits, select_canonical_dlbs_rows
from baselines.synthseg import build_synthseg_feature_table, passes_qc_threshold, pivot_synthseg_features


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


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
    raw_path = root / "raw" / ("openneuro" if domain_label == OPENNEURO_DOMAIN else "DLBS") / raw_rel
    preproc_root = root / "processed" / ("openneuro" if domain_label == OPENNEURO_DOMAIN else "DLBS")
    if domain_label == OPENNEURO_DOMAIN:
        preproc_rel_parent = raw_rel.parent
        preproc_stem = raw_rel.name.replace("_T1w.nii.gz", "")
        synthseg_parent = preproc_root / "derivatives" / "synthseg" / preproc_rel_parent
    else:
        preproc_rel_parent = raw_rel.parent
        preproc_stem = raw_rel.name.replace("_T1w.nii.gz", "")
        synthseg_parent = preproc_root / "derivatives" / "synthseg" / preproc_rel_parent
    preproc_path = preproc_root / preproc_rel_parent / f"{preproc_stem}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    mask_path = preproc_root / preproc_rel_parent / f"{preproc_stem}_space-MNI152NLin2009cAsym_desc-brain_mask_T1w.nii.gz"
    _write_nifti(preproc_path, np.random.default_rng(abs(hash(row_id)) % 10000).normal(size=(8, 8, 8)))
    mask_data = np.zeros((8, 8, 8), dtype=np.uint8)
    mask_data[2:6, 2:6, 2:6] = 1
    _write_nifti(mask_path, mask_data)
    volumes_path, qc_path = _write_synthseg_files(synthseg_parent, raw_rel.name.replace(".nii.gz", ""), synthseg_qc)

    return {
        "row_id": row_id,
        "subject_id": f"{dataset}:{sub}",
        "dataset": dataset,
        "sub": sub,
        "ses": ses,
        "run": run,
        "age": float(20 + int(sub) % 50),
        "sex": "F" if int(sub) % 2 else "M",
        "raw_t1_path": str(raw_path),
        "preproc_t1_path": str(preproc_path),
        "brain_mask_path": str(mask_path),
        "synthseg_volumes_tsv_path": str(volumes_path),
        "synthseg_qc_tsv_path": str(qc_path),
        "domain_label": domain_label,
        "source_relpath": str(raw_rel),
    }


def test_build_manifest_excludes_ds004856_from_openneuro_domain(tmp_path: Path) -> None:
    images = pd.DataFrame(
        [
            {
                "dataset": "ds000001",
                "sub": "01",
                "ses": "",
                "run": "",
                "suffix": "T1w",
                "path": "ds000001/sub-01/anat/sub-01_T1w.nii.gz",
            },
            {
                "dataset": DLBS_DATASET,
                "sub": "1001",
                "ses": "wave1",
                "run": "1",
                "suffix": "T1w",
                "path": "ds004856/sub-1001/ses-wave1/anat/sub-1001_ses-wave1_run-1_T1w.nii.gz",
            },
        ]
    )
    participants = pd.DataFrame(
        [
            {"dataset": "ds000001", "sub": "01", "ses": "", "sex": "F", "age": 30},
            {"dataset": DLBS_DATASET, "sub": "1001", "ses": "wave1", "sex": "M", "age": 70},
        ]
    )
    include_filelist = tmp_path / "include.txt"
    include_filelist.write_text("\n".join(images["path"]) + "\n")
    images_csv = tmp_path / "images.csv"
    participants_csv = tmp_path / "participants.csv"
    images.to_csv(images_csv, index=False)
    participants.to_csv(participants_csv, index=False)

    manifest = build_manifest(
        images_csv=images_csv,
        participants_csv=participants_csv,
        include_filelist=include_filelist,
        openneuro_raw_root=tmp_path / "raw" / "openneuro",
        dlbs_raw_root=tmp_path / "raw" / "DLBS",
        openneuro_processed_root=tmp_path / "processed" / "openneuro",
        dlbs_processed_root=tmp_path / "processed" / "DLBS",
    )

    openneuro = manifest[manifest["domain_label"] == OPENNEURO_DOMAIN]
    assert (openneuro["dataset"] == DLBS_DATASET).sum() == 0
    assert set(manifest["domain_label"]) == {OPENNEURO_DOMAIN, DLBS_DOMAIN}


def test_subject_safe_split_generation() -> None:
    rows = []
    for idx in range(1, 7):
        rows.append(
            {
                "row_id": f"open-{idx}-a",
                "subject_id": f"open:{idx}",
                "domain_label": OPENNEURO_DOMAIN,
                "ses": "",
                "run": "",
                "source_relpath": f"a/{idx}",
            }
        )
        rows.append(
            {
                "row_id": f"open-{idx}-b",
                "subject_id": f"open:{idx}",
                "domain_label": OPENNEURO_DOMAIN,
                "ses": "followup",
                "run": "",
                "source_relpath": f"b/{idx}",
            }
        )
    for idx in range(1, 7):
        rows.append(
            {
                "row_id": f"dlbs-{idx}",
                "subject_id": f"dlbs:{idx}",
                "domain_label": DLBS_DOMAIN,
                "ses": f"wave{idx % 3 + 1}",
                "run": "1",
                "source_relpath": f"c/{idx}",
            }
        )
    manifest = pd.DataFrame(rows)
    splits = build_domain_splits(manifest, seed=7, val_frac=0.2, test_frac=0.2)
    subject_counts = splits.groupby(["domain_label", "subject_id"])["split"].nunique()
    assert subject_counts.max() == 1


def test_canonical_dlbs_selection_prefers_first_sorted_scan() -> None:
    manifest = pd.DataFrame(
        [
            {
                "row_id": "a",
                "subject_id": "ds004856:1001",
                "domain_label": DLBS_DOMAIN,
                "ses": "wave2",
                "run": "1",
                "source_relpath": "p2",
            },
            {
                "row_id": "b",
                "subject_id": "ds004856:1001",
                "domain_label": DLBS_DOMAIN,
                "ses": "wave1",
                "run": "2",
                "source_relpath": "p1",
            },
            {
                "row_id": "c",
                "subject_id": "ds004856:1001",
                "domain_label": DLBS_DOMAIN,
                "ses": "wave1",
                "run": "1",
                "source_relpath": "p0",
            },
        ]
    )
    canonical = select_canonical_dlbs_rows(manifest)
    assert canonical["row_id"].tolist() == ["c"]


def test_synthseg_pivot_and_normalization() -> None:
    volumes = pd.DataFrame(
        [
            {"region": "left caudate", "volume_mm3": 25.0},
            {"region": "TCV", "volume_mm3": 100.0},
        ]
    )
    features = pivot_synthseg_features(volumes)
    assert features["TCV"] == 100.0
    assert features["raw__left_caudate"] == 25.0
    assert features["norm__left_caudate"] == 0.25


def test_qc_threshold_filtering(tmp_path: Path) -> None:
    row = _make_manifest_row(tmp_path, "row-1", "ds000001", "01", "", "", OPENNEURO_DOMAIN, synthseg_qc=0.5)
    manifest = pd.DataFrame([row])
    feature_df, exclusions = build_synthseg_feature_table(manifest, qc_threshold=0.65)
    assert feature_df.empty
    assert exclusions.iloc[0]["reason"] == "qc_below_threshold_0.65"
    qc = pd.DataFrame([{"structure": "wm", "qc_score": 0.65}, {"structure": "gm", "qc_score": 0.70}])
    assert passes_qc_threshold(qc, 0.65)


def test_zscore_intensity_inside_mask() -> None:
    image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :, :1] = True
    normalized = zscore_intensity_inside_mask(image, mask)
    masked = normalized[mask]
    assert np.isclose(masked.mean(), 0.0)


def test_bag_age_bias_correction_reduces_linear_age_dependence() -> None:
    y_true = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    y_pred = 0.8 * y_true + 6.0
    correction = fit_bag_age_bias_correction(y_true, y_pred)
    corrected = apply_bag_age_bias_correction(y_true, y_pred, correction)
    corrected_gap = corrected - y_true
    assert np.isclose(correction["alpha"], -0.2)
    assert np.isclose(correction["beta"], 6.0)
    assert np.allclose(corrected_gap, 0.0, atol=1e-6)


@pytest.mark.filterwarnings("ignore:.*ConvergenceWarning.*")
def test_smoke_synthseg_cli_run(tmp_path: Path) -> None:
    manifest_rows = []
    for idx in range(1, 7):
        manifest_rows.append(_make_manifest_row(tmp_path, f"open-{idx}", f"ds0000{idx}", f"{idx}", "", "", OPENNEURO_DOMAIN))
    for idx in range(1, 7):
        manifest_rows.append(_make_manifest_row(tmp_path, f"dlbs-{idx}", DLBS_DATASET, f"{1000 + idx}", f"wave{idx % 3 + 1}", "1", DLBS_DOMAIN))
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = tmp_path / "manifest.csv"
    splits_path = tmp_path / "splits.csv"
    manifest.to_csv(manifest_path, index=False)
    build_domain_splits(manifest, seed=11, val_frac=0.2, test_frac=0.2).to_csv(splits_path, index=False)

    output_dir = tmp_path / "runs" / "synthseg"
    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-m",
            "baselines.cli",
            "run",
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
            "--feature-family",
            "synthseg",
            "--models",
            "DummyRegressor,LinearRegression",
        ],
        check=True,
    )
    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert set(comparison["target_evaluation"]) == {
        "openneuro_nodlbs_holdout",
        "dlbs",
        "dlbs_holdout",
    }
    assert {"DummyRegressor", "LinearRegression"} == set(comparison["model_name"])
    assert {"raw_mae", "corrected_mae", "age_bias_correction_alpha", "age_bias_correction_beta"} <= set(
        comparison.columns
    )


@pytest.mark.filterwarnings("ignore:.*ConvergenceWarning.*")
def test_smoke_raw_t1_cli_run(tmp_path: Path) -> None:
    pytest.importorskip("radiomics")
    pytest.importorskip("SimpleITK")

    manifest_rows = []
    for idx in range(1, 7):
        manifest_rows.append(_make_manifest_row(tmp_path, f"open-{idx}", f"ds0100{idx}", f"{idx}", "", "", OPENNEURO_DOMAIN))
    for idx in range(1, 7):
        manifest_rows.append(_make_manifest_row(tmp_path, f"dlbs-{idx}", DLBS_DATASET, f"{2000 + idx}", f"wave{idx % 3 + 1}", "1", DLBS_DOMAIN))
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = tmp_path / "manifest.csv"
    splits_path = tmp_path / "splits.csv"
    manifest.to_csv(manifest_path, index=False)
    build_domain_splits(manifest, seed=19, val_frac=0.2, test_frac=0.2).to_csv(splits_path, index=False)

    output_dir = tmp_path / "runs" / "raw_t1"
    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-m",
            "baselines.cli",
            "run",
            "--manifest",
            str(manifest_path),
            "--splits",
            str(splits_path),
            "--output-dir",
            str(output_dir),
            "--feature-family",
            "raw_t1",
            "--models",
            "DummyRegressor,LinearRegression",
        ],
        check=True,
    )
    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert not comparison.empty
    assert (output_dir / "openneuro_nodlbs__to__dlbs" / "raw_t1" / "DummyRegressor" / "predictions.csv").exists()
