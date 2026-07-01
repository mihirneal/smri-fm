from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Nifti, Value

DIAGNOSIS_NAMES = ["CN", "PD", "Prodromal", "SWEDD"]
SEX_NAMES = ["Female", "Male"]

COHORT_TO_DIAGNOSIS = {
    "Healthy Control": "CN",
    "Parkinson's Disease": "PD",
    "Prodromal": "Prodromal",
    "SWEDD": "SWEDD",
}

SEX_TO_LABEL = {
    "0": "Female",
    "1": "Male",
}

T1W_RE = re.compile(
    r"^sub-(?P<subject>.+?)_ses-(?P<session>\d{8})(?P<run>_run-[^_]+)?_T1w\.nii(?:\.gz)?$"
)


@dataclass(frozen=True)
class CohortConfig:
    seed: int = 4466
    limit: int = 1000
    min_age: float = 18.0
    max_age: float = 120.0


def ppmi_features() -> Features:
    return Features({
        "sample_id": Value("string"),
        "participant_id": Value("string"),
        "session_id": Value("string"),
        "scan_date": Value("string"),
        "age": Value("float32"),
        "sex": ClassLabel(names=SEX_NAMES),
        "diagnosis": ClassLabel(names=DIAGNOSIS_NAMES),
        "nifti": Nifti(),
    })


def _read_zip_csv(zip_path: Path, filename: str) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if not name.startswith("__MACOSX/") and name.endswith(f"/{filename}")
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected exactly one {filename} in {zip_path}, got {matches}")
        with archive.open(matches[0]) as handle:
            return pd.read_csv(handle, low_memory=False)


def _normalize_patno(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip()
    if text.startswith("sub-"):
        text = text[4:]
    return text


def _normalize_code(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_birth_month(value) -> pd.Timestamp:
    return pd.to_datetime(value, format="%m/%Y", errors="coerce")


def _f(value) -> float:
    return float(value) if value is not None and pd.notna(value) else float("nan")


def _discover_t1w(bids_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(bids_root.glob("sub-*/ses-*/anat/*T1w.nii.gz")):
        match = T1W_RE.match(path.name)
        if match is None:
            continue
        participant_id = f"sub-{match.group('subject')}"
        session_id = f"ses-{match.group('session')}"
        scan_date = pd.to_datetime(match.group("session"), format="%Y%m%d", errors="coerce")
        if pd.isna(scan_date):
            continue
        sample_id = path.name.removesuffix(".nii.gz").removesuffix(".nii")
        rows.append({
            "sample_id": sample_id,
            "participant_id": participant_id,
            "patno": _normalize_patno(participant_id),
            "session_id": session_id,
            "scan_date": scan_date,
            "run": match.group("run") or "",
            "local_path": str(path),
            "path": f"images/{path.name}",
        })
    if not rows:
        raise ValueError(f"no T1w NIfTI files found under {bids_root}")
    return pd.DataFrame(rows).sort_values(
        ["participant_id", "scan_date", "run", "local_path"]
    ).reset_index(drop=True)


def _load_demographics(subject_characteristics_zip: Path) -> pd.DataFrame:
    raw = _read_zip_csv(subject_characteristics_zip, "Demographics_30Jun2026.csv")
    df = pd.DataFrame({
        "patno": raw["PATNO"].map(_normalize_patno),
        "birthdt": raw["BIRTHDT"].map(_parse_birth_month),
        "sex": raw["SEX"].map(_normalize_code).map(SEX_TO_LABEL),
    })
    return (
        df.dropna(subset=["patno"])
        .sort_values(["patno", "birthdt"])
        .drop_duplicates("patno", keep="first")
        .reset_index(drop=True)
    )


def _load_participant_status(subject_characteristics_zip: Path) -> pd.DataFrame:
    raw = _read_zip_csv(subject_characteristics_zip, "Participant_Status_30Jun2026.csv")
    cohort = raw["COHORT_DEFINITION"].astype(str).str.strip()
    df = pd.DataFrame({
        "patno": raw["PATNO"].map(_normalize_patno),
        "diagnosis": cohort.map(COHORT_TO_DIAGNOSIS),
        "cohort_definition": cohort,
        "enroll_status": raw["ENROLL_STATUS"],
    })
    return (
        df.dropna(subset=["patno"])
        .drop_duplicates("patno", keep="first")
        .reset_index(drop=True)
    )


def _attach_metadata(
    scans: pd.DataFrame,
    *,
    demographics: pd.DataFrame,
    participant_status: pd.DataFrame,
) -> pd.DataFrame:
    manifest = scans.merge(demographics, on="patno", how="left")
    manifest = manifest.merge(participant_status, on="patno", how="left")
    manifest["age"] = (manifest["scan_date"] - manifest["birthdt"]).dt.days / 365.25
    manifest = manifest.dropna(subset=["age", "sex", "diagnosis"])
    manifest = manifest[manifest["diagnosis"].isin(DIAGNOSIS_NAMES)]
    manifest = manifest[manifest["sex"].isin(SEX_NAMES)]
    return manifest.reset_index(drop=True)


def _one_scan_per_subject(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.sort_values(["participant_id", "scan_date", "run", "local_path"])
        .drop_duplicates("participant_id", keep="first")
        .sort_values(["diagnosis", "participant_id"])
        .reset_index(drop=True)
    )


def _filter_eligible(manifest: pd.DataFrame, *, config: CohortConfig) -> pd.DataFrame:
    valid = (
        manifest["age"].between(config.min_age, config.max_age, inclusive="both")
        & manifest["local_path"].map(lambda value: Path(value).is_file())
    )
    return manifest.loc[valid].reset_index(drop=True)


def _balanced_sample(manifest: pd.DataFrame, *, config: CohortConfig) -> pd.DataFrame:
    if config.limit <= 0:
        raise ValueError("limit must be positive")

    groups = {label: manifest[manifest["diagnosis"] == label].copy() for label in DIAGNOSIS_NAMES}
    groups = {label: frame for label, frame in groups.items() if not frame.empty}
    if not groups:
        raise ValueError("no labelled PPMI rows are eligible")

    total_available = sum(len(frame) for frame in groups.values())
    if total_available < config.limit:
        raise ValueError(
            f"only {total_available} eligible one-scan-per-subject rows; need {config.limit}"
        )

    rng = np.random.default_rng(config.seed)
    shuffled = {}
    for label, frame in groups.items():
        order = rng.permutation(len(frame))
        shuffled[label] = frame.iloc[order].reset_index(drop=True)

    quotas = {label: 0 for label in groups}
    labels = [label for label in DIAGNOSIS_NAMES if label in groups]
    while sum(quotas.values()) < config.limit:
        candidates = [label for label in labels if quotas[label] < len(groups[label])]
        if not candidates:
            break
        label = min(candidates, key=lambda item: (quotas[item], labels.index(item)))
        quotas[label] += 1

    selected = pd.concat(
        [shuffled[label].iloc[:count] for label, count in quotas.items() if count],
        ignore_index=True,
    )
    return selected.sort_values("sample_id").reset_index(drop=True)


def build_manifest(
    *,
    bids_root: Path,
    subject_characteristics_zip: Path,
    config: CohortConfig = CohortConfig(),
) -> pd.DataFrame:
    scans = _discover_t1w(bids_root)
    demographics = _load_demographics(subject_characteristics_zip)
    participant_status = _load_participant_status(subject_characteristics_zip)
    manifest = _attach_metadata(
        scans,
        demographics=demographics,
        participant_status=participant_status,
    )
    manifest = _one_scan_per_subject(manifest)
    manifest = _filter_eligible(manifest, config=config)
    manifest = _balanced_sample(manifest, config=config)
    return manifest.reset_index(drop=True)


def manifest_report(manifest: pd.DataFrame) -> dict:
    return {
        "total_scans": int(len(manifest)),
        "total_subjects": int(manifest["participant_id"].nunique()),
        "diagnosis": {
            label: int((manifest["diagnosis"] == label).sum()) for label in DIAGNOSIS_NAMES
        },
        "sex": {label: int((manifest["sex"] == label).sum()) for label in SEX_NAMES},
        "age": {
            "min": float(manifest["age"].min()),
            "mean": float(manifest["age"].mean()),
            "max": float(manifest["age"].max()),
        },
        "tasks": {
            "pd_cn": {
                "CN": int((manifest["diagnosis"] == "CN").sum()),
                "PD": int((manifest["diagnosis"] == "PD").sum()),
            },
            "diagnosis": {
                label: int((manifest["diagnosis"] == label).sum()) for label in DIAGNOSIS_NAMES
            },
        },
    }


def _generate_samples(records):
    for record in records:
        local_path = Path(record["local_path"])
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        yield {
            "sample_id": record["sample_id"],
            "participant_id": record["participant_id"],
            "session_id": record["session_id"],
            "scan_date": str(record["scan_date"])[:10],
            "age": _f(record["age"]),
            "sex": record["sex"],
            "diagnosis": record["diagnosis"],
            "nifti": {"path": record["path"], "bytes": local_path.read_bytes()},
        }


def _write_readme(output_dir: Path, report: dict) -> None:
    diagnosis = report["diagnosis"]
    sex = report["sex"]
    text = f"""---
title: PPMI Mini
tags:
- medical-imaging
- mri
- nifti
- parkinsons
- ppmi
---

# PPMI Mini

A 1,000-subject PPMI sMRI evaluation subset derived from local BIDS T1w images
and PPMI subject-characteristics metadata.

## Cohort

- {report["total_subjects"]:,} subjects / {report["total_scans"]:,} scans
- One T1w scan per subject
- Sex: {sex.get("Female", 0)} Female / {sex.get("Male", 0)} Male
- Diagnosis: CN {diagnosis.get("CN", 0)} / PD {diagnosis.get("PD", 0)} / Prodromal {diagnosis.get("Prodromal", 0)} / SWEDD {diagnosis.get("SWEDD", 0)}

## Features

The dataset has a single `eval` split with embedded NIfTI image bytes in `nifti`.
Brain masks and SynthSeg volumes are intentionally omitted for v0.1.
"""
    (output_dir / "README.md").write_text(text + "\n")


def build_dataset(
    *,
    bids_root: Path,
    subject_characteristics_zip: Path,
    output_dir: Path,
    num_proc: int = 8,
    max_shard_size: str = "1GB",
    cohort_config: CohortConfig = CohortConfig(),
) -> DatasetDict:
    eval_path = output_dir / "eval"
    if eval_path.exists():
        raise FileExistsError(f"dataset output already exists: {eval_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        bids_root=bids_root,
        subject_characteristics_zip=subject_characteristics_zip,
        config=cohort_config,
    )

    source_manifest = manifest.drop(columns=["local_path", "birthdt"], errors="ignore")
    source_manifest.to_parquet(output_dir / "source_manifest.parquet", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    report = manifest_report(manifest)
    (output_dir / "manifest_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    _write_readme(output_dir, report)

    records = manifest.to_dict("records")
    eval_ds = Dataset.from_generator(
        _generate_samples,
        features=ppmi_features(),
        gen_kwargs={"records": records},
        num_proc=min(num_proc, max(1, len(records))),
        split="eval",
        fingerprint=f"ppmi-mini-v0.1-eval-{cohort_config.seed}-{len(records)}",
        writer_batch_size=4,
    )
    dataset = DatasetDict({"eval": eval_ds})
    dataset.save_to_disk(output_dir, max_shard_size=max_shard_size, num_proc=num_proc)
    return dataset


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build PPMI mini eval dataset as local Arrow shards"
    )
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--subject-characteristics-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4466)
    parser.add_argument("--num-proc", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()

    dataset = build_dataset(
        bids_root=args.bids_root,
        subject_characteristics_zip=args.subject_characteristics_zip,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        max_shard_size=args.max_shard_size,
        cohort_config=CohortConfig(seed=args.seed, limit=args.limit),
    )
    print(dataset)


if __name__ == "__main__":
    cli()
