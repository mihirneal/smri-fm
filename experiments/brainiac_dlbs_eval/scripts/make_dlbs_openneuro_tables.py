#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openpyxl",
#   "pandas",
# ]
# ///
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


DATASET = "DLBS"
OPENNEURO_S3 = "s3://openneuro.org/ds004856/derivatives"
COGNITION_FILES = [
    "Construct1_SpeedOfProcessing.xlsx",
    "Construct2_WorkingMemory.xlsx",
    "Construct5_Reasoning.xlsx",
    "Construct6_Vocabulary.xlsx",
    "Construct7_VerbalFluency.xlsx",
]
STRUCTURAL_FILE = "Template_Structural_MRI.xlsx"
BASE_COLUMNS = [
    "dataset",
    "subject_id",
    "wave",
    "AgeMRI",
    "AgeCog",
    "Sex",
    "Race",
    "Ethnicity",
    "EduComp",
    "EduYrsEstCap",
    "HandednessScore",
    "Height",
    "Weight",
    "BMI",
    "MMSE",
    "MRIW1toW2",
    "MRIW2toW3",
    "MRIW1toW3",
    "CogW1toW2",
    "CogW2toW3",
    "CogW1toW3",
]
PARTICIPANT_WAVE_COLUMNS = {
    "AgeMRI": "AgeMRI_W{wave}",
    "AgeCog": "AgeCog_W{wave}",
    "Height": "Height_W{wave}",
    "Weight": "Weight_W{wave}",
    "BMI": "BMI_W{wave}",
    "MMSE": "MMSE_W{wave}",
}
PARTICIPANT_CONSTANT_COLUMNS = [
    "Sex",
    "Race",
    "Ethnicity",
    "EduComp",
    "EduYrsEstCap",
    "HandednessScore",
    "MRIW1toW2",
    "MRIW2toW3",
    "MRIW1toW3",
    "CogW1toW2",
    "CogW2toW3",
    "CogW1toW3",
]
MISSING_VALUES = {"n/a", "NA", "xx", ""}
COGNITIVE_COMPOSITES = {
    "speed": {
        "file": "Construct1_SpeedOfProcessing.xlsx",
        "tasks": [
            {"name": "digcomp_total", "columns": ["DigCompTotal1"]},
            {"name": "digsym_total", "columns": ["DigSymTotal2"]},
            {"name": "nih_speed", "columns": ["NIHSpeedRaw3", "NIHSpeedUn3"]},
        ],
    },
    "working_memory": {
        "file": "Construct2_WorkingMemory.xlsx",
        "tasks": [
            {"name": "spatial_wm_total_errors", "columns": ["SptlWMTotErrs4"], "direction": -1},
            {"name": "letter_number_seq_total", "columns": ["LetNumSeqTot5"]},
            {"name": "operation_span_total", "columns": ["OSpanTot6"]},
            {"name": "nih_list_sort", "columns": ["NIHTBLstSrtRaw7", "NIHTBLstSrtUncrrctd7"]},
            {"name": "dms_total", "columns": ["DMSTot8"]},
            {"name": "srm_total", "columns": ["SRMTot9"]},
        ],
    },
    "reasoning": {
        "file": "Construct5_Reasoning.xlsx",
        "tasks": [
            {"name": "ravens_matrices", "columns": ["RavenNumCor20", "RavenAccAll20"]},
            {"name": "ets_letter_sets", "columns": ["EtsLsTOTAL21"]},
            {"name": "cantab_stockings", "columns": ["CantabSOCMinMov22"]},
            {"name": "everyday_problem_solving", "columns": ["Eps23"]},
        ],
    },
    "vocabulary": {
        "file": "Construct6_Vocabulary.xlsx",
        "tasks": [
            {"name": "ets_vocabulary", "columns": ["ETSVocab24"]},
            {"name": "shipley_vocabulary", "columns": ["ShipVocab25"]},
            {"name": "cantab_graded_naming", "columns": ["CantabGnt26"]},
            {"name": "nih_oral_reading", "columns": ["NIHOralReadUn27"]},
            {"name": "nih_picture_vocabulary", "columns": ["NIHPicVocabUn28"]},
        ],
    },
    "verbal_fluency": {
        "file": "Construct7_VerbalFluency.xlsx",
        "tasks": [
            {"name": "fas_total", "columns": ["WContOralAssocLetterTot29"]},
            {"name": "category_total", "columns": ["ContOralAssocCatTot30"]},
        ],
    },
}
FLUID_CONSTRUCTS = ["speed", "working_memory", "reasoning", "verbal_fluency"]


def clean_name(value: object) -> str:
    text = "unnamed" if value is None else str(value)
    text = text.strip().replace("#", "num")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def normalize_wave(value: object) -> str:
    match = re.search(r"([123])", str(value))
    if not match:
        raise ValueError(f"Could not parse wave from {value!r}")
    return f"wave{match.group(1)}"


def subject_from_snum(value: object) -> str:
    return f"sub-{int(float(value))}"


def read_table(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, na_values=list(MISSING_VALUES), keep_default_na=True, **kwargs)


def download_openneuro_derivatives(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in COGNITION_FILES:
        target = cache_dir / filename
        if not target.exists():
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--no-sign-request",
                    f"{OPENNEURO_S3}/cognition/{filename}",
                    str(target),
                ],
                check=True,
            )
    structural_target = cache_dir / STRUCTURAL_FILE
    if not structural_target.exists():
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                "--no-sign-request",
                f"{OPENNEURO_S3}/brainsummary/{STRUCTURAL_FILE}",
                str(structural_target),
            ],
            check=True,
        )


def build_participants_long(participants_path: Path) -> pd.DataFrame:
    participants = read_table(participants_path, sep="\t")
    rows = []
    for _, source in participants.iterrows():
        for wave_num in (1, 2, 3):
            row = {
                "dataset": DATASET,
                "subject_id": source["participant_id"],
                "wave": f"wave{wave_num}",
            }
            for output_col, input_template in PARTICIPANT_WAVE_COLUMNS.items():
                row[output_col] = source.get(input_template.format(wave=wave_num))
            for col in PARTICIPANT_CONSTANT_COLUMNS:
                row[col] = source.get(col)
            rows.append(row)
    return pd.DataFrame(rows)


def choose_scan(rows: list[dict[str, object]]) -> dict[str, object]:
    def key(row: dict[str, object]) -> tuple[int, str]:
        run = str(row.get("run") or "")
        return (0 if run == "1" else 1, str(row.get("source_file") or ""))

    return sorted(rows, key=key)[0]


def load_synthseg(synthseg_dir: Path) -> pd.DataFrame:
    pattern = re.compile(r"(sub-[^_/]+)_ses-(wave[123]).*?_run-(\d+)_T1w_(volumes|qc)\.csv$")
    by_scan: dict[tuple[str, str], list[dict[str, object]]] = {}
    for volume_path in sorted(synthseg_dir.rglob("*_volumes.csv")):
        match = pattern.search(volume_path.name)
        if not match:
            continue
        subject_id, wave, run, _ = match.groups()
        qc_path = volume_path.with_name(volume_path.name.replace("_volumes.csv", "_qc.csv"))
        row: dict[str, object] = {
            "subject_id": subject_id,
            "wave": wave,
            "synthseg_run": run,
            "synthseg_source_file": str(volume_path),
        }
        volumes = read_table(volume_path)
        if not volumes.empty:
            for col, value in volumes.iloc[0].items():
                if str(col).startswith("Unnamed"):
                    continue
                row[f"synthseg_vol_{clean_name(col)}"] = value
        if qc_path.exists():
            qc = read_table(qc_path)
            if not qc.empty:
                for col, value in qc.iloc[0].items():
                    if str(col).startswith("Unnamed"):
                        continue
                    row[f"synthseg_qc_{clean_name(col)}"] = value
        by_scan.setdefault((subject_id, wave), []).append(row)
    return pd.DataFrame(choose_scan(rows) for rows in by_scan.values())


def load_freesurfer(structural_xlsx: Path) -> pd.DataFrame:
    sheets = pd.read_excel(structural_xlsx, sheet_name=None, dtype=object)
    frames = []
    for sheet_name, df in sheets.items():
        wave = normalize_wave(sheet_name)
        feature_group = clean_name(re.sub(r"[-_]?W[123]$", "", sheet_name, flags=re.IGNORECASE))
        df = df.dropna(how="all")
        if "S#" not in df.columns:
            continue
        df = df[df["S#"].notna()].copy()
        out = pd.DataFrame(
            {
                "subject_id": df["S#"].map(subject_from_snum),
                "wave": wave,
            }
        )
        for col in df.columns:
            if col in {"S#", "AIRC_ID", "ConstructName", "Wave"}:
                continue
            out[f"fs_{feature_group}_{clean_name(col)}"] = df[col]
        frames.append(out)
    merged: pd.DataFrame | None = None
    for frame in frames:
        merged = frame if merged is None else merged.merge(frame, on=["subject_id", "wave"], how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["subject_id", "wave"])


def construct_prefix(path: Path) -> str:
    match = re.match(r"Construct(\d+)_(.+)\.xlsx", path.name)
    if not match:
        return clean_name(path.stem)
    number, name = match.groups()
    return f"cog_construct{number}_{clean_name(name)}"


def load_cognition(cognition_dir: Path) -> pd.DataFrame:
    frames = []
    duplicate_context = {
        "AgeInterval",
        "Sex",
        "Race",
        "Ethnicity",
        "HandednessScore",
        "MMSE",
        "CogW1toW2",
        "CogW2toW3",
        "CogW1toW3",
        "TakeHomeW1toW2",
        "TakeHomeW2toW3",
        "TakeHomeW1toW3",
        "MRIW1toW2",
        "MRIW2toW3",
        "MRIW1toW3",
        "PETAmyW1toW2",
        "PETAmyW2toW3",
        "PETAmyW1toW3",
        "EduComp5",
        "EduYrsEstCap5",
    }
    for filename in sorted(COGNITION_FILES):
        path = cognition_dir / filename
        if not path.exists():
            continue
        prefix = construct_prefix(path)
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)
        for sheet_name, df in sheets.items():
            wave = normalize_wave(sheet_name)
            if "S#" not in df.columns:
                continue
            df = df[df["S#"].notna()].copy()
            out = pd.DataFrame(
                {
                    "subject_id": df["S#"].map(subject_from_snum),
                    "wave": wave,
                }
            )
            seen: dict[str, int] = {}
            for col in df.columns:
                if col in {"S#", "Wave"} or col in duplicate_context:
                    continue
                if pd.isna(col):
                    continue
                clean = clean_name(col)
                seen[clean] = seen.get(clean, 0) + 1
                suffix = f"_{seen[clean]}" if seen[clean] > 1 else ""
                out[f"{prefix}_{clean}{suffix}"] = df[col]
            frames.append(out)
    merged: pd.DataFrame | None = None
    for frame in frames:
        merged = frame if merged is None else merged.merge(frame, on=["subject_id", "wave"], how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["subject_id", "wave"])


def to_numeric(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    values = values.mask(values.astype(str).isin(MISSING_VALUES))
    return pd.to_numeric(values, errors="coerce")


def winsorized_zscore(series: pd.Series, min_n_winsor: int = 50) -> pd.Series:
    values = to_numeric(series)
    valid = values.dropna()
    clipped = values.copy()
    if len(valid) >= min_n_winsor:
        lower = valid.quantile(0.01)
        upper = valid.quantile(0.99)
        clipped = values.clip(lower=lower, upper=upper)
    std = clipped.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    return (clipped - clipped.mean()) / std


def zscore(series: pd.Series) -> pd.Series:
    values = to_numeric(series)
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    return (values - values.mean()) / std


def zscore_by_wave(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for _, index in df.groupby("wave").groups.items():
        out.loc[index] = zscore(df.loc[index, col])
    return out


def first_usable_column(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, pd.Series | None]:
    for col in candidates:
        if col not in df.columns:
            continue
        values = to_numeric(df[col])
        if values.notna().sum() > 0:
            return col, values
    return None, None


def load_cognitive_composites(cognition_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    construct_frames = []
    provenance: dict[str, object] = {
        "pipeline": [
            "raw_or_uncorrected_task_score",
            "direction_correct_higher_is_better",
            "winsorize_task_within_wave_at_1st_99th_percentile_when_n_at_least_50",
            "task_zscore_within_wave",
            "average_available_task_zscores_into_construct",
            "final_construct_zscore_within_wave",
            "fluid_mean_of_speed_working_memory_reasoning_verbal_fluency",
            "final_fluid_zscore_within_wave",
        ],
        "fluid_constructs": FLUID_CONSTRUCTS,
        "constructs": {},
    }
    for construct, spec in COGNITIVE_COMPOSITES.items():
        workbook = cognition_dir / str(spec["file"])
        sheets = pd.read_excel(workbook, sheet_name=None, dtype=object)
        wave_frames = []
        construct_provenance = {
            "file": spec["file"],
            "tasks": {},
        }
        for sheet_name, source in sheets.items():
            wave = normalize_wave(sheet_name)
            if "S#" not in source.columns:
                continue
            source = source[source["S#"].notna()].copy()
            out = pd.DataFrame(
                {
                    "subject_id": source["S#"].map(subject_from_snum),
                    "wave": wave,
                }
            )
            task_zscores = []
            for task in spec["tasks"]:
                column, raw_values = first_usable_column(source, list(task["columns"]))
                task_name = str(task["name"])
                construct_provenance["tasks"].setdefault(
                    task_name,
                    {
                        "candidate_columns": task["columns"],
                        "direction": int(task.get("direction", 1)),
                        "selected_columns_by_wave": {},
                    },
                )
                if column is None:
                    construct_provenance["tasks"][task_name]["selected_columns_by_wave"][wave] = None
                    continue
                print(f"{construct} / {wave} / {task_name}: using {column}")
                direction = int(task.get("direction", 1))
                construct_provenance["tasks"][task_name]["selected_columns_by_wave"][wave] = column
                task_zscores.append(winsorized_zscore(raw_values * direction))
            n_tasks = (
                pd.concat(task_zscores, axis=1).notna().sum(axis=1)
                if task_zscores
                else pd.Series(0, index=source.index)
            )
            if task_zscores:
                composite = pd.concat(task_zscores, axis=1).mean(axis=1, skipna=True)
            else:
                composite = pd.Series(pd.NA, index=source.index, dtype="Float64")
            out[f"cog_{construct}_z"] = composite
            out[f"cog_{construct}_n_tasks"] = n_tasks.astype("Int64")
            wave_frames.append(out)
        if wave_frames:
            construct_frames.append(pd.concat(wave_frames, ignore_index=True))
        provenance["constructs"][construct] = construct_provenance
    merged: pd.DataFrame | None = None
    for frame in construct_frames:
        merged = frame if merged is None else merged.merge(frame, on=["subject_id", "wave"], how="outer")
    if merged is None:
        return pd.DataFrame(columns=["subject_id", "wave"]), provenance
    for construct in COGNITIVE_COMPOSITES:
        col = f"cog_{construct}_z"
        merged[col] = zscore_by_wave(merged, col)
    fluid_cols = [f"cog_{construct}_z" for construct in FLUID_CONSTRUCTS]
    merged["cog_fluid_n_constructs"] = merged[fluid_cols].notna().sum(axis=1).astype("Int64")
    fluid = merged[fluid_cols].mean(axis=1, skipna=True)
    merged["cog_fluid_z"] = fluid.where(merged["cog_fluid_n_constructs"] > 0)
    merged["cog_fluid_z"] = zscore_by_wave(merged, "cog_fluid_z")
    return merged, provenance


def write_outputs(df: pd.DataFrame, output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    ordered = BASE_COLUMNS + [col for col in df.columns if col not in BASE_COLUMNS]
    df = df.loc[:, ordered]
    df = df.sort_values(["subject_id", "wave"], kind="stable")
    df.to_csv(output_tsv, sep="\t", index=False, na_rep="NA")
    df.to_csv(output_tsv.with_suffix(".csv"), index=False, na_rep="NA")


def write_cognitive_outputs(df: pd.DataFrame, output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    score_cols = [
        "cog_speed_z",
        "cog_working_memory_z",
        "cog_reasoning_z",
        "cog_vocabulary_z",
        "cog_verbal_fluency_z",
        "cog_fluid_z",
    ]
    qc_cols = [
        "cog_speed_n_tasks",
        "cog_working_memory_n_tasks",
        "cog_reasoning_n_tasks",
        "cog_vocabulary_n_tasks",
        "cog_verbal_fluency_n_tasks",
        "cog_fluid_n_constructs",
    ]
    ordered = ["dataset", "subject_id", "wave"] + score_cols + qc_cols
    df = df.loc[:, ordered]
    df = df.sort_values(["subject_id", "wave"], kind="stable")
    df.to_csv(output_tsv, sep="\t", index=False, na_rep="NA")
    df.to_csv(output_tsv.with_suffix(".csv"), index=False, na_rep="NA")


def write_cognitive_provenance(provenance: dict[str, object], output_tsv: Path) -> Path:
    output_json = output_tsv.with_suffix(".json")
    output_json.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlbs-dir", type=Path, default=Path("../smri-dataset/DLBS"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("../smri-dataset/DLBS/metadata"),
        help="Directory for downloaded OpenNeuro derivative Excel files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../smri-dataset/DLBS/derivatives/master_tables/dlbs_master_long.tsv"),
    )
    parser.add_argument(
        "--cognition-output",
        type=Path,
        default=Path("../smri-dataset/DLBS/derivatives/master_tables/dlbs_cognitive_composites.tsv"),
    )
    parser.add_argument("--no-download", action="store_true", help="Use existing cached Excel files.")
    parser.add_argument(
        "--include-cognition",
        action="store_true",
        help="Include OpenNeuro cognitive construct workbook columns.",
    )
    parser.add_argument(
        "--merge-cognitive-composites",
        action="store_true",
        help="Also merge cognitive composite columns into the master table.",
    )
    parser.add_argument(
        "--no-cognitive-composites",
        action="store_true",
        help="Do not include robust cognitive construct composite scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_download:
        download_openneuro_derivatives(args.cache_dir)
    participants = build_participants_long(args.dlbs_dir / "participants.tsv")
    synthseg = load_synthseg(args.dlbs_dir / "derivatives" / "synthseg")
    freesurfer = load_freesurfer(args.cache_dir / STRUCTURAL_FILE)
    cognitive_composites = None
    cognitive_provenance = None
    if not args.no_cognitive_composites:
        cognitive_composites, cognitive_provenance = load_cognitive_composites(
            args.cache_dir
        )
        cognitive_composites = participants[["dataset", "subject_id", "wave"]].merge(
            cognitive_composites, on=["subject_id", "wave"], how="left"
        )
        write_cognitive_outputs(cognitive_composites, args.cognition_output)
        cognitive_provenance_path = write_cognitive_provenance(
            cognitive_provenance, args.cognition_output
        )
    master = participants.merge(synthseg, on=["subject_id", "wave"], how="left")
    master = master.merge(freesurfer, on=["subject_id", "wave"], how="left")
    if args.merge_cognitive_composites and cognitive_composites is not None:
        master = master.merge(
            cognitive_composites.drop(columns=["dataset"]), on=["subject_id", "wave"], how="left"
        )
    if args.include_cognition:
        cognition = load_cognition(args.cache_dir)
        master = master.merge(cognition, on=["subject_id", "wave"], how="left")
    write_outputs(master, args.output)
    print(f"rows={len(master)}")
    print(f"columns={len(master.columns)}")
    print(f"subjects={master['subject_id'].nunique()}")
    print(f"waves={','.join(sorted(master['wave'].dropna().unique()))}")
    print(f"output={args.output}")
    print(f"csv={args.output.with_suffix('.csv')}")
    if cognitive_composites is not None:
        print(f"cognition_output={args.cognition_output}")
        print(f"cognition_csv={args.cognition_output.with_suffix('.csv')}")
        print(f"cognition_provenance={cognitive_provenance_path}")
    print(
        "cognitive_composites="
        f"{'excluded' if args.no_cognitive_composites else 'separate_file'}"
    )
    print(f"cognitive_composites_merged={args.merge_cognitive_composites}")
    print(f"cognitive_construct_scores={'included' if args.include_cognition else 'excluded'}")
    print("wmh_features=not_found_on_openneuro_s3_or_local_derivatives")


if __name__ == "__main__":
    main()
