#!/usr/bin/env python
"""Build AABC/HCPA DLBS-style tables and a representative test subset."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from path_utils import resolve_from_repo


DEFAULT_METADATA = "/teamspace/studios/this_studio/AABC2_subjects_2026_04_30_10_04_46.csv"
DEFAULT_AABC_ROOT = "/teamspace/studios/this_studio/aabc"
DEFAULT_OUTPUT_DIR = "../smri-dataset/AABC/qc/aabc_ood_eval"
SUPPORTED_WAVES = ("V1", "V2", "V3")
ID_EVENT_RE = re.compile(r"^(?P<subject>HCA\d+)_V(?P<wave>\d+)$")


def parse_id_event(value: str) -> tuple[str | None, str | None]:
    match = ID_EVENT_RE.match(str(value))
    if match is None:
        return None, None
    return f"sub-{match.group('subject')}", f"V{match.group('wave')}"


def parse_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() == "90 or older":
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def education_to_years(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    mapping = {
        "primary level education": 6.0,
        "some secondary level education": 10.0,
        "secondary level education or equivalent": 12.0,
        "some college / associates degree / trade or vocational training": 14.0,
        "four years college or undergraduate degree": 16.0,
        "graduate degree": 18.0,
    }
    return mapping.get(text, np.nan)


def harmonize_handedness(value: object) -> float:
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return np.nan
    if abs(float(score)) > 4:
        return float(score) / 25.0
    return float(score)


def clean_synthseg_region(region: str) -> str:
    cleaned = str(region).strip().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def synthseg_paths(synthseg_dir: Path, subject_id: str, wave: str) -> tuple[Path, Path]:
    stem = f"{subject_id}_ses-{wave}_T1w"
    return synthseg_dir / f"{stem}_volumes.csv", synthseg_dir / f"{stem}_qc.csv"


def load_synthseg_row(volumes_path: Path, qc_path: Path) -> dict[str, float | str]:
    volumes = pd.read_csv(volumes_path, index_col=0)
    if volumes.empty:
        raise ValueError(f"Empty SynthSeg volumes file: {volumes_path}")
    row = volumes.iloc[0]
    out: dict[str, float | str] = {"synthseg_source_file": str(volumes_path)}
    for region, value in row.items():
        out[f"synthseg_vol_{clean_synthseg_region(region)}"] = pd.to_numeric(
            value, errors="coerce"
        )

    if qc_path.exists():
        qc = pd.read_csv(qc_path, index_col=0)
        if not qc.empty:
            qc_values = qc.iloc[0].apply(pd.to_numeric, errors="coerce")
            out["synthseg_qc_min"] = float(qc_values.min(skipna=True))
            out["synthseg_qc_mean"] = float(qc_values.mean(skipna=True))
            for name, value in qc_values.items():
                out[f"synthseg_qc_{clean_synthseg_region(name)}"] = value
    return out


def build_master_table(
    metadata_path: Path,
    synthseg_dir: Path,
    include_waves: tuple[str, ...] = SUPPORTED_WAVES,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    parsed = metadata["id_event"].map(parse_id_event)
    metadata["subject_id"] = [item[0] for item in parsed]
    metadata["wave"] = [item[1] for item in parsed]
    metadata["AgeMRI"] = metadata["age_open"].map(parse_age).astype(float)
    metadata["AgeCog"] = metadata["AgeMRI"]

    metadata = metadata[
        metadata["study"].isin(["HCA", "AABC"])
        & metadata["wave"].isin(include_waves)
        & metadata["subject_id"].notna()
        & metadata["AgeMRI"].notna()
    ].copy()

    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        volumes_path, qc_path = synthseg_paths(synthseg_dir, row["subject_id"], row["wave"])
        if not volumes_path.exists():
            continue
        base = {
            "dataset": "AABC_HCPA",
            "source_study": row["study"],
            "id_event": row["id_event"],
            "subject_id": row["subject_id"],
            "wave": row["wave"],
            "AgeMRI": row["AgeMRI"],
            "AgeCog": row["AgeCog"],
            "Sex": row.get("sex", np.nan),
            "Race": row.get("race", np.nan),
            "Ethnicity": row.get("ethnicity", np.nan),
            "site": row.get("site", np.nan),
            "scanner": row.get("scanner", np.nan),
            "EduYrsEstCap": education_to_years(row.get("education", np.nan)),
            "Education": row.get("education", np.nan),
            "HandednessScore": harmonize_handedness(row.get("iihand_edinburgh", np.nan)),
            "BMI": pd.to_numeric(row.get("bmi", np.nan), errors="coerce"),
            "MR_QC_Issue_Codes": row.get("MR_QC_Issue_Codes", np.nan),
            "Bulk_Imaging": row.get("Bulk_Imaging", np.nan),
            "IDPs": row.get("IDPs", np.nan),
        }
        cognition = cognition_fields(row)
        rows.append({**base, **cognition, **load_synthseg_row(volumes_path, qc_path)})

    if not rows:
        raise ValueError("No eligible AABC/HCPA rows with T1 SynthSeg volumes were found.")
    return pd.DataFrame(rows).sort_values(["subject_id", "wave"], kind="stable")


def cognition_fields(row: pd.Series) -> dict[str, object]:
    columns = [
        "FluidIQ_Tr35_60y",
        "CrystIQ_Tr35_60y",
        "Memory_Tr35_60y",
        "moca_sum",
        "trail1",
        "trail2",
        "ravlt_immediate_recall",
        "ravlt_learning_score",
        "tlbx_lswm_uncorrected_standard_score",
        "tlbx_dccs_uncorrected_standard_score",
        "tlbx_pcps_uncorrected_standard_score",
        "tlbx_pv_uncorrected_standard_score",
        "tlbx_psm_uncorrected_standard_score",
        "tlbx_orr_uncorrected_standard_score",
    ]
    return {col: pd.to_numeric(row.get(col, np.nan), errors="coerce") for col in columns}


def add_age_bins(df: pd.DataFrame) -> pd.Series:
    return pd.cut(
        df["AgeMRI"],
        bins=[35, 45, 55, 65, 75, 90],
        labels=["36-45", "46-55", "56-65", "66-75", "76-89"],
        include_lowest=True,
        right=True,
    ).astype(str)


def select_representative_subset(
    df: pd.DataFrame,
    n: int,
    seed: int,
    strata_columns: tuple[str, ...] = ("age_bin", "Sex", "site", "scanner", "wave"),
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("--n must be positive.")
    if len(df) < n:
        raise ValueError(f"Requested {n} rows but only {len(df)} eligible rows are available.")

    work = df.copy()
    if "age_bin" not in work.columns:
        work["age_bin"] = add_age_bins(work)
    for col in strata_columns:
        work[col] = work[col].astype(object).where(work[col].notna(), "missing")

    rng = np.random.default_rng(seed)
    grouped = work.groupby(list(strata_columns), dropna=False, sort=True)
    counts = grouped.size().rename("count").reset_index()
    counts["expected"] = counts["count"] / counts["count"].sum() * n
    counts["quota"] = np.floor(counts["expected"]).astype(int)
    remaining = n - int(counts["quota"].sum())
    counts["remainder"] = counts["expected"] - counts["quota"]
    if remaining > 0:
        order = counts.sort_values(["remainder", "count"], ascending=False).head(remaining).index
        counts.loc[order, "quota"] += 1

    selected_indices: list[int] = []
    for _, quota_row in counts.iterrows():
        quota = int(min(quota_row["quota"], quota_row["count"]))
        if quota == 0:
            continue
        mask = np.ones(len(work), dtype=bool)
        for col in strata_columns:
            mask &= work[col].to_numpy() == quota_row[col]
        choices = work.index[mask].to_numpy()
        selected_indices.extend(rng.choice(choices, size=quota, replace=False).tolist())

    if len(selected_indices) < n:
        remaining_indices = work.index.difference(selected_indices).to_numpy()
        fill = rng.choice(remaining_indices, size=n - len(selected_indices), replace=False).tolist()
        selected_indices.extend(fill)
    elif len(selected_indices) > n:
        selected_indices = rng.choice(np.asarray(selected_indices), size=n, replace=False).tolist()

    subset = work.loc[selected_indices].sort_values(
        ["AgeMRI", "Sex", "site", "scanner", "wave", "subject_id"], kind="stable"
    )
    subset.insert(0, "subset_index", np.arange(1, len(subset) + 1))
    return subset


def distribution_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in ["source_study", "wave", "Sex", "Race", "Ethnicity", "site", "scanner", "age_bin"]:
        if col not in df.columns:
            continue
        counts = df[col].fillna("missing").value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "table": label,
                    "variable": col,
                    "value": value,
                    "count": int(count),
                    "fraction": float(count / len(df)) if len(df) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_outputs(master: pd.DataFrame, subset: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_dir / "aabc_hcpa_master_long.tsv", sep="\t", index=False)
    subset.to_csv(output_dir / f"aabc_hcpa_subset_{len(subset)}.csv", index=False)

    diagnostics = pd.concat(
        [distribution_summary(master, "eligible_full"), distribution_summary(subset, "subset")],
        ignore_index=True,
    )
    diagnostics.to_csv(output_dir / "subset_distribution_diagnostics.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "metadata": str(args.metadata),
        "aabc_root": str(args.aabc_root),
        "synthseg_dir": str(args.synthseg_dir),
        "n_eligible": int(len(master)),
        "n_subset": int(len(subset)),
        "n_subjects_eligible": int(master["subject_id"].nunique()),
        "n_subjects_subset": int(subset["subject_id"].nunique()),
        "waves": sorted(master["wave"].dropna().unique().tolist()),
        "subset_waves": sorted(subset["wave"].dropna().unique().tolist()),
        "age_eligible": master["AgeMRI"].describe().to_dict(),
        "age_subset": subset["AgeMRI"].describe().to_dict(),
        "strata": ["age_bin", "Sex", "site", "scanner", "wave"],
        "top_coded_age_policy": "Rows with age_open='90 or older' are excluded.",
    }
    (output_dir / "subset_distribution_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(DEFAULT_METADATA))
    parser.add_argument("--aabc-root", type=Path, default=Path(DEFAULT_AABC_ROOT))
    parser.add_argument("--synthseg-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.metadata = resolve_from_repo(args.metadata)
    args.aabc_root = resolve_from_repo(args.aabc_root)
    args.synthseg_dir = (
        resolve_from_repo(args.synthseg_dir)
        if args.synthseg_dir is not None
        else args.aabc_root / "processed" / "derivatives" / "synthseg"
    )
    output_dir = resolve_from_repo(args.output_dir)

    master = build_master_table(args.metadata, args.synthseg_dir)
    master["age_bin"] = add_age_bins(master)
    subset = select_representative_subset(master, args.n, args.seed)
    write_outputs(master, subset, output_dir, args)
    print(f"Wrote {output_dir / 'aabc_hcpa_master_long.tsv'}")
    print(f"Wrote {output_dir / f'aabc_hcpa_subset_{len(subset)}.csv'}")
    print(f"Wrote {output_dir / 'subset_distribution_diagnostics.csv'}")


if __name__ == "__main__":
    main()
