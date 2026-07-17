#!/usr/bin/env python
"""Add compressed SynthSeg patch-count targets to existing MRI WDS shards."""

import argparse
from pathlib import Path

from data.anatomy_targets import (
    DEFAULT_VOCABULARY_PATH,
    as_3tuple,
    augment_wds_shard_with_anatomy,
    load_anatomy_vocabulary,
)
from data.mri_data import expand_urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input tar shards; glob and brace expressions are supported.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Current root containing <subset>/derivatives/synthseg dseg files.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=DEFAULT_VOCABULARY_PATH,
    )
    parser.add_argument("--patch-size", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        default=None,
        help="Optional expected dense image shape; metadata remains the source of truth.",
    )
    parser.add_argument("--target-suffix", default="anatomy.npz")
    parser.add_argument("--ignore-unknown", action="store_true")
    parser.add_argument("--replace-targets", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(path) for path in expand_urls(args.inputs)]
    if not inputs:
        raise FileNotFoundError(f"no input shards matched: {args.inputs}")

    patch_size = as_3tuple(
        args.patch_size[0] if len(args.patch_size) == 1 else args.patch_size,
        "patch_size",
    )
    img_size = None
    if args.img_size is not None:
        img_size = as_3tuple(
            args.img_size[0] if len(args.img_size) == 1 else args.img_size,
            "img_size",
        )
    vocabulary = load_anatomy_vocabulary(args.vocabulary)
    print(
        f"vocabulary={vocabulary.name!r}, classes={vocabulary.num_classes}, patch_size={patch_size}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = [path.name for path in inputs]
    if len(set(output_names)) != len(output_names):
        raise ValueError("input shards contain duplicate basenames; use separate invocations")

    totals = {"samples": 0, "targets_added": 0, "targets_kept": 0}
    for shard_idx, input_path in enumerate(inputs, start=1):
        output_path = args.output_dir / input_path.name
        stats = augment_wds_shard_with_anatomy(
            input_path,
            output_path,
            source_root=args.source_root,
            vocabulary=vocabulary,
            patch_size=patch_size,
            target_suffix=args.target_suffix,
            expected_img_size=img_size,
            ignore_unknown=args.ignore_unknown,
            replace_targets=args.replace_targets,
            overwrite_output=args.overwrite_output,
        )
        for key, value in stats.items():
            totals[key] += value
        print(
            f"[{shard_idx}/{len(inputs)}] {input_path} -> {output_path}: "
            f"{stats['targets_added']} added, {stats['targets_kept']} kept"
        )
    print(
        f"done: {totals['samples']} samples, {totals['targets_added']} targets added, "
        f"{totals['targets_kept']} existing targets kept"
    )


if __name__ == "__main__":
    main()
