#!/usr/bin/env python
import argparse
import os

import pandas as pd

from path_utils import resolve_from_repo


def main():
    parser = argparse.ArgumentParser(
        description="Create a deterministic symlink subset of DLBS NIfTI images."
    )
    parser.add_argument(
        "--input_dir",
        default="DLBS/images",
        help="Directory containing DLBS .nii.gz files.",
    )
    parser.add_argument(
        "--output_dir",
        default="DLBS/images_100",
        help="Directory where symlinks will be created.",
    )
    parser.add_argument(
        "--manifest",
        default="DLBS/images_100_manifest.csv",
        help="CSV manifest to write.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of sorted images to select.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks/files in output_dir.",
    )
    args = parser.parse_args()

    input_dir = resolve_from_repo(args.input_dir)
    output_dir = resolve_from_repo(args.output_dir)
    manifest = resolve_from_repo(args.manifest)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    images = sorted(input_dir.glob("*.nii.gz"))[:args.n]
    if len(images) < args.n:
        raise SystemExit(f"Requested {args.n} files but found only {len(images)} in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, src in enumerate(images, start=1):
        dst = output_dir / src.name
        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                raise SystemExit(f"Output already exists: {dst}. Use --overwrite to replace it.")
            dst.unlink()
        os.symlink(src, dst)
        rows.append(
            {
                "index": index,
                "filename": src.name,
                "source_path": str(src),
                "symlink_path": str(dst),
            }
        )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Created {len(rows)} symlinks in {output_dir}")
    print(f"Wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
