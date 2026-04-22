#!/usr/bin/env python
import argparse
import re

import pandas as pd

from path_utils import resolve_from_repo


WAVE_TO_AGE_COLUMN = {
    "wave1": "AgeMRI_W1",
    "wave2": "AgeMRI_W2",
    "wave3": "AgeMRI_W3",
}


def main():
    parser = argparse.ArgumentParser(
        description="Create a BrainIAC brain-age CSV for processed DLBS T1w images."
    )
    parser.add_argument(
        "--processed_dir",
        default="DLBS/processed_brainiac_100",
        help="Directory containing processed .nii.gz files.",
    )
    parser.add_argument(
        "--participants",
        default="DLBS/participants.tsv",
        help="DLBS participants.tsv file.",
    )
    parser.add_argument(
        "--output_csv",
        default="DLBS/brainage_100.csv",
        help="Output CSV with pat_id,label.",
    )
    parser.add_argument("--age_units", choices=["years", "months"], default="months")
    args = parser.parse_args()

    processed_dir = resolve_from_repo(args.processed_dir)
    participants_path = resolve_from_repo(args.participants)
    participants = pd.read_csv(participants_path, sep="\t", dtype=str).set_index("participant_id")

    rows = []
    skipped = []
    for image_path in sorted(processed_dir.glob("*.nii.gz")):
        pat_id = image_path.name[:-7]
        match = re.match(r"(?P<participant>sub-[^_]+)_ses-(?P<wave>wave[123])_", pat_id)
        if match is None:
            skipped.append((pat_id, "filename did not match DLBS pattern"))
            continue

        participant = match.group("participant")
        wave = match.group("wave")
        age_column = WAVE_TO_AGE_COLUMN[wave]

        if participant not in participants.index:
            skipped.append((pat_id, f"{participant} not in participants.tsv"))
            continue

        age = participants.loc[participant, age_column]
        if pd.isna(age) or str(age).lower() == "n/a":
            skipped.append((pat_id, f"{age_column} missing"))
            continue

        label = float(age)
        if args.age_units == "months":
            label *= 12.0

        rows.append({"pat_id": pat_id, "label": label})

    output_csv = resolve_from_repo(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} files")
        for pat_id, reason in skipped[:20]:
            print(f"  {pat_id}: {reason}")


if __name__ == "__main__":
    main()
