#!/usr/bin/env python
"""Build ADNI DLBS-style tables, healthy CN subset, and MCI-to-AD prognosis set."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from path_utils import resolve_from_repo


DEFAULT_ADNI_ROOT = "/teamspace/studios/this_studio/ADNI-bids"
DEFAULT_DXSUM = "/teamspace/studios/this_studio/DXSUM_01May2026.csv"
DEFAULT_PTDEMOG = "/teamspace/studios/this_studio/PTDEMOG_01May2026.csv"
DEFAULT_NEUROPSYCH = "/teamspace/studios/this_studio/Neuropsychological-2.zip"
DEFAULT_MANIFEST = "/teamspace/studios/this_studio/missing_ADNI_5_01_2026.csv"
DEFAULT_OUTPUT_DIR = "../smri-dataset/ADNI/qc/adni_ood_eval"

SYNTHSEG_RE = re.compile(
    r"^(?P<subject_id>sub-[0-9]+S[0-9]+)_ses-(?P<session>[0-9]{8})_T1w(?P<run>_[0-9]+)?_volumes\.csv$"
)
DIAGNOSIS_LABELS = {1: "CN", 2: "MCI", 3: "AD"}
COGNITION_COLUMNS = ["ADNI_MEM", "ADNI_EF", "ADNI_LAN", "ADNI_VS", "ADNI_EF2"]


def clean_synthseg_region(region: str) -> str:
    cleaned = str(region).strip().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def subject_id_to_ptid(subject_id: str) -> str:
    raw = subject_id.removeprefix("sub-")
    return raw.replace("S", "_S_", 1)


def parse_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"^(\d+(?:\.\d+)?)", text)
    if match is None:
        return np.nan
    return float(match.group(1))


def normalize_sex(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in {"M", "MALE", "1", "1.0"}:
        return "M"
    if text in {"F", "FEMALE", "2", "2.0"}:
        return "F"
    return np.nan


def map_ethnicity(value: object) -> object:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return np.nan
    if int(numeric) == 1:
        return "Hispanic"
    if int(numeric) == 2:
        return "Not Hispanic"
    return "Unknown"


def map_race(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-4", "-4.0"}:
        return "Unknown"
    parts = {part.strip() for part in text.split("|") if part.strip()}
    if parts == {"5"}:
        return "White"
    if "5" in parts:
        return "Multiracial"
    return "Non-White"


def map_handedness(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or numeric < 0:
        return np.nan
    mapping = {1: 1.0, 2: -1.0, 3: 0.0}
    return mapping.get(int(numeric), np.nan)


def read_json_metadata(json_path: Path) -> dict[str, object]:
    if not json_path.exists():
        return {"json_exists": False}
    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError:
        return {"json_exists": False}
    return {
        "json_exists": True,
        "AgeMRI": parse_age(data.get("PatientAge")),
        "Sex": normalize_sex(data.get("PatientSex")),
        "PatientWeight": pd.to_numeric(data.get("PatientWeight"), errors="coerce"),
        "Manufacturer": data.get("Manufacturer"),
        "ManufacturersModelName": data.get("ManufacturersModelName"),
        "MagneticFieldStrength": pd.to_numeric(data.get("MagneticFieldStrength"), errors="coerce"),
        "ProtocolName": data.get("ProtocolName"),
        "SeriesDescription": data.get("SeriesDescription"),
    }


def load_synthseg_row(volumes_path: Path, qc_path: Path) -> dict[str, object]:
    volumes = pd.read_csv(volumes_path, index_col=0)
    if volumes.empty:
        raise ValueError(f"Empty SynthSeg volumes file: {volumes_path}")
    out: dict[str, object] = {"synthseg_source_file": str(volumes_path)}
    for region, value in volumes.iloc[0].items():
        out[f"synthseg_vol_{clean_synthseg_region(region)}"] = pd.to_numeric(value, errors="coerce")

    if qc_path.exists():
        qc = pd.read_csv(qc_path, index_col=0)
        if not qc.empty:
            qc_values = qc.iloc[0].apply(pd.to_numeric, errors="coerce")
            out["synthseg_qc_min"] = float(qc_values.min(skipna=True))
            out["synthseg_qc_mean"] = float(qc_values.mean(skipna=True))
            for name, value in qc_values.items():
                out[f"synthseg_qc_{clean_synthseg_region(name)}"] = value
    return out


def load_synthseg_scans(adni_root: Path, synthseg_dir: Path) -> pd.DataFrame:
    rows = []
    for volumes_path in sorted(synthseg_dir.glob("*_T1w*volumes.csv")):
        match = SYNTHSEG_RE.match(volumes_path.name)
        if match is None:
            continue
        subject_id = match.group("subject_id")
        session = match.group("session")
        run_label = (match.group("run") or "").removeprefix("_")
        run_number = pd.to_numeric(run_label, errors="coerce")
        stem = volumes_path.name.removesuffix("_volumes.csv")
        qc_path = volumes_path.with_name(f"{stem}_qc.csv")
        json_path = adni_root / subject_id / f"ses-{session}" / "anat" / f"{stem}.json"
        base = {
            "dataset": "ADNI",
            "subject_id": subject_id,
            "PTID": subject_id_to_ptid(subject_id),
            "session": session,
            "wave": f"ses-{session}",
            "scan_date": pd.to_datetime(session, format="%Y%m%d", errors="coerce"),
            "synthseg_run": run_label if run_label else "single",
            "synthseg_run_number": float(run_number) if pd.notna(run_number) else 0.0,
            "synthseg_qc_file": str(qc_path),
            "json_file": str(json_path),
        }
        rows.append({**base, **read_json_metadata(json_path), **load_synthseg_row(volumes_path, qc_path)})
    if not rows:
        raise ValueError(f"No ADNI T1w SynthSeg volume CSVs found in {synthseg_dir}")
    return pd.DataFrame(rows)


def deduplicate_t1_runs(scans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_cols = ["subject_id", "session", "synthseg_qc_mean", "synthseg_run_number"]
    work = scans.copy()
    if "synthseg_qc_mean" not in work.columns:
        work["synthseg_qc_mean"] = np.nan
    work = work.sort_values(
        sort_cols,
        ascending=[True, True, False, True],
        na_position="last",
        kind="stable",
    )
    keep_mask = ~work.duplicated(["subject_id", "session"], keep="first")
    kept = work[keep_mask].copy()
    dropped = work[~keep_mask].copy()
    dropped["exclusion_reason"] = "duplicate_t1_run_lower_qc_or_later_run"
    return kept, dropped


def load_dxsum(path: Path) -> pd.DataFrame:
    dx = pd.read_csv(path, low_memory=False)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx[dx["PTID"].notna() & dx["EXAMDATE"].notna() & dx["DIAGNOSIS"].isin([1, 2, 3])].copy()
    dx["diagnosis"] = dx["DIAGNOSIS"].astype(int).map(DIAGNOSIS_LABELS)
    return dx


def load_demographics(path: Path) -> pd.DataFrame:
    demog = pd.read_csv(path, low_memory=False)
    demog["VISDATE"] = pd.to_datetime(demog["VISDATE"], errors="coerce")
    demog = demog.sort_values(["PTID", "VISDATE"], kind="stable")
    baseline = demog[demog["VISCODE2"].astype(str).eq("bl")].drop_duplicates("PTID")
    fallback = demog.drop_duplicates("PTID")
    selected = pd.concat([baseline, fallback]).drop_duplicates("PTID", keep="first")
    out = selected[["PTID", "RID", "PHASE", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTHAND"]].copy()
    out = out.rename(columns={"PHASE": "adni_phase", "PTEDUCAT": "EduYrsEstCap"})
    out["RID"] = pd.to_numeric(out["RID"], errors="coerce")
    out["EduYrsEstCap"] = pd.to_numeric(out["EduYrsEstCap"], errors="coerce").where(lambda s: s.ge(0))
    out["Ethnicity"] = out["PTETHCAT"].map(map_ethnicity)
    out["Race"] = out["PTRACCAT"].map(map_race)
    out["HandednessScore"] = out["PTHAND"].map(map_handedness)
    return out.drop(columns=["PTETHCAT", "PTRACCAT", "PTHAND"])


def load_manifest(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["PTID", "scan_date", "manifest_group"])
    manifest = pd.read_csv(path, low_memory=False)
    manifest["scan_date"] = pd.to_datetime(manifest["Acq Date"], errors="coerce")
    manifest = manifest.rename(
        columns={"Subject": "PTID", "Group": "manifest_group", "Visit": "manifest_visit"}
    )
    keep = manifest[["PTID", "scan_date", "manifest_group", "manifest_visit"]].dropna(
        subset=["PTID", "scan_date"]
    )
    return keep.drop_duplicates(["PTID", "scan_date", "manifest_group", "manifest_visit"])


def load_uw_cognition(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("UWNPSYCHSUM_01May2026.csv") as handle:
            cog = pd.read_csv(handle, low_memory=False)
    cog["EXAMDATE"] = pd.to_datetime(cog["EXAMDATE"], errors="coerce")
    cog["RID"] = pd.to_numeric(cog["RID"], errors="coerce")
    for col in COGNITION_COLUMNS:
        cog[col] = pd.to_numeric(cog[col], errors="coerce")
    return cog[cog["RID"].notna() & cog["EXAMDATE"].notna()].copy()


def nearest_by_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
    left_date: str,
    right_date: str,
    max_days: int,
    suffix: str,
) -> pd.DataFrame:
    left_indexed = left.reset_index(drop=True).reset_index(names="left_row")
    candidates = left_indexed.merge(
        right,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffixes=("", suffix),
    )
    candidates[f"abs_days{suffix}"] = (
        candidates[left_date] - candidates[right_date]
    ).abs().dt.days
    candidates = candidates.sort_values(["left_row", f"abs_days{suffix}"], kind="stable")
    nearest = candidates.drop_duplicates("left_row", keep="first")
    matched = nearest[f"abs_days{suffix}"].le(max_days)
    right_cols = [col for col in right.columns if col not in right_keys]
    output_cols = ["left_row", *right_cols, f"abs_days{suffix}"]
    nearest = nearest[output_cols].copy()
    for col in right_cols:
        nearest.loc[~matched, col] = np.nan
    nearest.loc[~matched, f"abs_days{suffix}"] = np.nan
    out = left_indexed.merge(nearest, on="left_row", how="left").drop(columns=["left_row"])
    return out


def add_diagnosis(scans: pd.DataFrame, dx: pd.DataFrame, max_days: int) -> pd.DataFrame:
    dx_keep = dx[["PTID", "EXAMDATE", "VISCODE", "VISCODE2", "DIAGNOSIS", "diagnosis", "PHASE"]].copy()
    dx_keep = dx_keep.rename(
        columns={
            "EXAMDATE": "diagnosis_date",
            "VISCODE": "diagnosis_viscode",
            "VISCODE2": "diagnosis_viscode2",
            "DIAGNOSIS": "diagnosis_code",
            "PHASE": "diagnosis_phase",
        }
    )
    return nearest_by_date(
        scans,
        dx_keep,
        ["PTID"],
        ["PTID"],
        "scan_date",
        "diagnosis_date",
        max_days,
        "_diagnosis",
    )


def add_cognition(scans: pd.DataFrame, cognition: pd.DataFrame, max_days: int) -> pd.DataFrame:
    cog_keep = cognition[["RID", "EXAMDATE", "VISCODE", "VISCODE2", *COGNITION_COLUMNS]].copy()
    cog_keep = cog_keep.rename(
        columns={
            "EXAMDATE": "cognition_date",
            "VISCODE": "cognition_viscode",
            "VISCODE2": "cognition_viscode2",
        }
    )
    return nearest_by_date(
        scans,
        cog_keep,
        ["RID"],
        ["RID"],
        "scan_date",
        "cognition_date",
        max_days,
        "_cognition",
    )


def add_age_bins(df: pd.DataFrame) -> pd.Series:
    return pd.cut(
        df["AgeMRI"],
        bins=[55, 65, 70, 75, 80, 85, 95],
        labels=["55-65", "66-70", "71-75", "76-80", "81-85", "86-95"],
        include_lowest=True,
        right=True,
    ).astype(str)


def qc_filter(master: pd.DataFrame, min_age: float, max_age: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = master.copy()
    reasons = pd.Series("", index=work.index, dtype=object)
    reasons = reasons.mask(~work["json_exists"].fillna(False).astype(bool), reasons + "|missing_json")
    reasons = reasons.mask(work["AgeMRI"].isna(), reasons + "|missing_age")
    reasons = reasons.mask(work["AgeMRI"].lt(min_age) | work["AgeMRI"].gt(max_age), reasons + "|implausible_age")
    reasons = reasons.mask(~work["Sex"].isin(["M", "F"]), reasons + "|nonstandard_or_missing_sex")
    reasons = reasons.mask(work["synthseg_vol_total_intracranial"].isna(), reasons + "|missing_icv")
    keep = reasons.eq("")
    excluded = work[~keep].copy()
    excluded["exclusion_reason"] = reasons[~keep].str.strip("|")
    return work[keep].copy(), excluded


def baseline_records(dx: pd.DataFrame) -> pd.DataFrame:
    baseline = dx[dx["VISCODE2"].astype(str).eq("bl")].copy()
    return baseline.sort_values(["PTID", "EXAMDATE"], kind="stable").drop_duplicates("PTID")


def followup_months(dx: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base = baseline[["PTID", "EXAMDATE", "diagnosis"]].rename(
        columns={"EXAMDATE": "baseline_date", "diagnosis": "baseline_diagnosis"}
    )
    future = base.merge(dx[["PTID", "EXAMDATE", "diagnosis"]], on="PTID", how="left")
    future = future[future["EXAMDATE"].ge(future["baseline_date"])].copy()
    future["months_from_baseline"] = (
        future["EXAMDATE"] - future["baseline_date"]
    ).dt.days / 30.4375
    return future


def select_baseline_scan(
    master: pd.DataFrame,
    baseline: pd.DataFrame,
    diagnosis: str,
    max_days: int,
) -> pd.DataFrame:
    base = baseline[baseline["diagnosis"].eq(diagnosis)][["PTID", "EXAMDATE", "diagnosis", "PHASE"]].copy()
    base = base.rename(columns={"EXAMDATE": "baseline_date", "PHASE": "baseline_phase"})
    candidates = master.merge(base, on="PTID", how="inner", suffixes=("", "_baseline"))
    candidates["baseline_scan_abs_days"] = (candidates["scan_date"] - candidates["baseline_date"]).abs().dt.days
    candidates = candidates[candidates["baseline_scan_abs_days"].le(max_days)].copy()
    return candidates.sort_values(
        ["PTID", "baseline_scan_abs_days", "synthseg_qc_mean"], ascending=[True, True, False], kind="stable"
    ).drop_duplicates("PTID")


def stable_cn_subjects(dx: pd.DataFrame, min_followup_months: float) -> pd.DataFrame:
    baseline = baseline_records(dx)
    base_cn = baseline[baseline["diagnosis"].eq("CN")].copy()
    future = followup_months(dx, base_cn)
    status = future.groupby("PTID").agg(
        max_followup_months=("months_from_baseline", "max"),
        ever_mci_ad=("diagnosis", lambda s: bool(s.isin(["MCI", "AD"]).any())),
    )
    stable = status[~status["ever_mci_ad"] & status["max_followup_months"].ge(min_followup_months)]
    return base_cn.merge(stable.reset_index(), on="PTID", how="inner")


def prognosis_labels(dx: pd.DataFrame, min_nonconverter_followup_months: float) -> pd.DataFrame:
    baseline = baseline_records(dx)
    base_mci = baseline[baseline["diagnosis"].eq("MCI")].copy()
    future = followup_months(dx, base_mci)
    grouped = future.groupby("PTID").agg(
        max_followup_months=("months_from_baseline", "max"),
        converted_36mo=("diagnosis", lambda s: False),
    )
    converted = (
        future[future["months_from_baseline"].le(36)]
        .groupby("PTID")["diagnosis"]
        .apply(lambda s: bool(s.eq("AD").any()))
    )
    grouped["converted_36mo"] = converted.reindex(grouped.index).fillna(False).astype(bool)
    first_ad = future[future["diagnosis"].eq("AD")].groupby("PTID")["months_from_baseline"].min()
    grouped["months_to_ad"] = first_ad
    keep = grouped["converted_36mo"] | grouped["max_followup_months"].ge(min_nonconverter_followup_months)
    labels = grouped[keep].reset_index()
    labels["mci_to_ad_36mo"] = labels["converted_36mo"].astype(int)
    return base_mci.merge(labels, on="PTID", how="inner")


def select_representative_subset(
    df: pd.DataFrame,
    n: int,
    seed: int,
    strata_columns: tuple[str, ...] = (
        "age_bin",
        "Sex",
        "Manufacturer",
        "MagneticFieldStrength",
        "baseline_phase",
    ),
) -> pd.DataFrame:
    if len(df) <= n:
        subset = df.copy()
        subset.insert(0, "subset_index", np.arange(1, len(subset) + 1))
        return subset
    work = df.copy()
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
        selected_indices.extend(rng.choice(work.index[mask].to_numpy(), size=quota, replace=False).tolist())
    if len(selected_indices) < n:
        remaining_indices = work.index.difference(selected_indices).to_numpy()
        selected_indices.extend(rng.choice(remaining_indices, size=n - len(selected_indices), replace=False).tolist())
    subset = work.loc[selected_indices].sort_values(
        ["AgeMRI", "Sex", "Manufacturer", "MagneticFieldStrength", "subject_id"], kind="stable"
    )
    subset.insert(0, "subset_index", np.arange(1, len(subset) + 1))
    return subset


def distribution_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in [
        "diagnosis",
        "baseline_diagnosis",
        "mci_to_ad_36mo",
        "Sex",
        "Race",
        "Ethnicity",
        "Manufacturer",
        "MagneticFieldStrength",
        "adni_phase",
        "baseline_phase",
        "age_bin",
    ]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adni-root", type=Path, default=Path(DEFAULT_ADNI_ROOT))
    parser.add_argument("--synthseg-dir", type=Path, default=None)
    parser.add_argument("--dxsum", type=Path, default=Path(DEFAULT_DXSUM))
    parser.add_argument("--ptdemog", type=Path, default=Path(DEFAULT_PTDEMOG))
    parser.add_argument("--neuropsych-zip", type=Path, default=Path(DEFAULT_NEUROPSYCH))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--healthy-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-age", type=float, default=55)
    parser.add_argument("--max-age", type=float, default=95)
    parser.add_argument("--max-diagnosis-days", type=int, default=90)
    parser.add_argument("--max-cognition-days", type=int, default=90)
    parser.add_argument("--max-baseline-scan-days", type=int, default=180)
    parser.add_argument("--min-stable-cn-followup-months", type=float, default=12)
    parser.add_argument("--min-nonconverter-followup-months", type=float, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adni_root = resolve_from_repo(args.adni_root)
    synthseg_dir = (
        resolve_from_repo(args.synthseg_dir)
        if args.synthseg_dir is not None
        else adni_root / "derivatives" / "synthseg"
    )
    dxsum = resolve_from_repo(args.dxsum)
    ptdemog = resolve_from_repo(args.ptdemog)
    neuropsych_zip = resolve_from_repo(args.neuropsych_zip)
    manifest_path = resolve_from_repo(args.manifest) if args.manifest is not None else None
    output_dir = resolve_from_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scans_raw = load_synthseg_scans(adni_root, synthseg_dir)
    scans_dedup, duplicate_exclusions = deduplicate_t1_runs(scans_raw)
    dx = load_dxsum(dxsum)
    demog = load_demographics(ptdemog)
    cognition = load_uw_cognition(neuropsych_zip)
    manifest = load_manifest(manifest_path)

    master = scans_dedup.merge(demog, on="PTID", how="left", suffixes=("", "_demog"))
    if "RID_demog" in master.columns:
        master["RID"] = master["RID"].fillna(master["RID_demog"])
        master = master.drop(columns=["RID_demog"])
    master = add_diagnosis(master, dx, args.max_diagnosis_days)
    master = add_cognition(master, cognition, args.max_cognition_days)
    if not manifest.empty:
        master = master.merge(manifest, on=["PTID", "scan_date"], how="left")
    master["AgeCog"] = master["AgeMRI"]
    master["BMI"] = np.nan
    master["age_bin"] = add_age_bins(master)

    eligible, qc_exclusions = qc_filter(master, args.min_age, args.max_age)

    stable_cn = stable_cn_subjects(dx, args.min_stable_cn_followup_months)
    healthy = select_baseline_scan(eligible, stable_cn, "CN", args.max_baseline_scan_days)
    healthy["baseline_diagnosis"] = "CN"
    healthy_subset = select_representative_subset(healthy, args.healthy_n, args.seed)

    prognosis = prognosis_labels(dx, args.min_nonconverter_followup_months)
    mci_ad = select_baseline_scan(eligible, prognosis, "MCI", args.max_baseline_scan_days)
    label_cols = [
        "PTID",
        "mci_to_ad_36mo",
        "converted_36mo",
        "months_to_ad",
        "max_followup_months",
    ]
    mci_ad = mci_ad.drop(
        columns=[col for col in label_cols if col != "PTID" and col in mci_ad.columns],
        errors="ignore",
    )
    mci_ad = mci_ad.merge(prognosis[label_cols], on="PTID", how="inner")
    mci_ad["baseline_diagnosis"] = "MCI"
    mci_ad = mci_ad.sort_values(["mci_to_ad_36mo", "AgeMRI", "subject_id"], kind="stable")
    mci_ad.insert(0, "subset_index", np.arange(1, len(mci_ad) + 1))

    eligible.to_csv(output_dir / "adni_master_long.tsv", sep="\t", index=False)
    healthy.to_csv(output_dir / "adni_healthy_cn_baseline_eligible.csv", index=False)
    healthy_subset.to_csv(output_dir / f"adni_healthy_cn_subset_{len(healthy_subset)}.csv", index=False)
    mci_ad.to_csv(output_dir / "adni_mci_ad_prognosis_baseline.csv", index=False)
    duplicate_exclusions.to_csv(output_dir / "adni_duplicate_t1_exclusions.csv", index=False)
    qc_exclusions.to_csv(output_dir / "adni_qc_exclusions.csv", index=False)
    diagnostics = pd.concat(
        [
            distribution_summary(eligible, "eligible_all"),
            distribution_summary(healthy, "healthy_stable_cn_eligible"),
            distribution_summary(healthy_subset, "healthy_stable_cn_subset"),
            distribution_summary(mci_ad, "mci_to_ad_36mo"),
        ],
        ignore_index=True,
    )
    diagnostics.to_csv(output_dir / "adni_distribution_diagnostics.csv", index=False)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "adni_root": str(adni_root),
        "synthseg_dir": str(synthseg_dir),
        "dxsum": str(dxsum),
        "ptdemog": str(ptdemog),
        "neuropsych_zip": str(neuropsych_zip),
        "raw_t1_synthseg_rows": int(len(scans_raw)),
        "deduplicated_t1_rows": int(len(scans_dedup)),
        "eligible_rows": int(len(eligible)),
        "eligible_subjects": int(eligible["subject_id"].nunique()),
        "healthy_stable_cn_eligible_rows": int(len(healthy)),
        "healthy_stable_cn_subset_rows": int(len(healthy_subset)),
        "mci_ad_prognosis_rows": int(len(mci_ad)),
        "mci_ad_converters_36mo": int(mci_ad["mci_to_ad_36mo"].sum()) if len(mci_ad) else 0,
        "diagnosis_match_max_days": args.max_diagnosis_days,
        "cognition_match_max_days": args.max_cognition_days,
        "age_range": [args.min_age, args.max_age],
        "cognition_targets": COGNITION_COLUMNS,
    }
    (output_dir / "adni_table_build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output_dir / 'adni_master_long.tsv'}")
    print(f"Wrote {output_dir / f'adni_healthy_cn_subset_{len(healthy_subset)}.csv'}")
    print(f"Wrote {output_dir / 'adni_mci_ad_prognosis_baseline.csv'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
