#!/usr/bin/env python3
"""Rewrite sparse WDS MRI shards with int8-quantized brain voxel values."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


VALUE_SUFFIX = "image_values.npy"
MASK_SUFFIX = "img_mask.npy"
META_SUFFIX = "meta.json"
EXPECTED_SUFFIXES = {VALUE_SUFFIX, MASK_SUFFIX, META_SUFFIX}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FOMO sparse WebDataset shards from float16 image_values.npy "
            "to int8 image_values.npy and slim duplicated metadata."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/data/mihir-stuff/FOMO300/wds"),
        help="Source directory containing shard.*.tar files.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/mihir-stuff/FOMO300-int8/wds"),
        help="Destination directory for rewritten shards.",
    )
    parser.add_argument("--pattern", default="shard.*.tar", help="Shard filename glob.")
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help="Specific shard filename or path to convert. Can be passed multiple times.",
    )
    parser.add_argument("--start", type=int, default=None, help="First shard index to include.")
    parser.add_argument("--stop", type=int, default=None, help="Last shard index to include, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of shards to convert.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip shards that already exist in the destination directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output shards.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected shards without writing.")
    return parser.parse_args()


def split_member_name(name: str) -> tuple[str, str]:
    base = Path(name).name
    for suffix in EXPECTED_SUFFIXES:
        marker = "." + suffix
        if base.endswith(marker):
            return base[: -len(marker)], suffix
    return base, ""


def shard_index(path: Path) -> int | None:
    parts = path.name.split(".")
    if len(parts) >= 3 and parts[-1] == "tar" and parts[-2].isdigit():
        return int(parts[-2])
    return None


def select_shards(args: argparse.Namespace) -> list[Path]:
    if args.shard:
        shards = []
        for shard in args.shard:
            path = Path(shard)
            if not path.is_absolute():
                path = args.src / path
            shards.append(path)
    else:
        shards = sorted(args.src.glob(args.pattern))

    selected = []
    for shard in shards:
        index = shard_index(shard)
        if args.start is not None and (index is None or index < args.start):
            continue
        if args.stop is not None and (index is None or index > args.stop):
            continue
        selected.append(shard)

    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def load_npy(tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    return np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False)


def json_bytes(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def quantize_affine_per_sample(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    qmin = -128
    qmax = 127
    value_min = float(np.nanmin(values)) if values.size else 0.0
    value_max = float(np.nanmax(values)) if values.size else 0.0
    scale = (value_max - value_min) / float(qmax - qmin) if value_max > value_min else 1.0
    quantized = np.rint((values - value_min) / scale + qmin).clip(qmin, qmax)
    quantized = quantized.astype(np.int8, copy=False)
    metadata = {
        "scheme": "affine_per_sample_minmax",
        "domain": "normalized",
        "storage_dtype": "int8",
        "dequantized_dtype": "float16",
        "scale": scale,
        "value_min": value_min,
        "value_max": value_max,
        "qmin": qmin,
        "qmax": qmax,
        "dequantize": "value = (int8_value - qmin) * scale + value_min",
    }
    return quantized, metadata


def slim_metadata(meta: dict[str, Any], quantization: dict[str, Any]) -> dict[str, Any]:
    sparse = dict(meta.get("sparse_image") or {})
    sparse["values_dtype"] = "int8"
    sparse["values_normalized"] = True

    slim = {
        "key": meta.get("key"),
        "subset": meta.get("subset"),
        "modality": meta.get("modality"),
        "native_stem": meta.get("native_stem"),
        "raw_mean": meta.get("raw_mean"),
        "raw_std": meta.get("raw_std"),
        "sparse_image": sparse,
        "int8_quantization": quantization,
    }
    return {key: value for key, value in slim.items() if value is not None}


def grouped_members(tar: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    grouped: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    for member in tar.getmembers():
        if not member.isfile():
            continue
        key, suffix = split_member_name(member.name)
        if suffix:
            grouped[key][suffix] = member
    return dict(grouped)


def add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def convert_shard(src: Path, dst: Path, *, overwrite: bool) -> tuple[int, int, int]:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"output shard exists: {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=dst.name + ".", suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    samples = 0
    fp16_bytes = 0
    int8_bytes = 0
    try:
        with tarfile.open(src, "r") as in_tar, tarfile.open(tmp_path, "w") as out_tar:
            members = grouped_members(in_tar)
            for key in sorted(members):
                parts = members[key]
                missing = EXPECTED_SUFFIXES - parts.keys()
                if missing:
                    raise ValueError(f"{src}: sample {key} is missing {sorted(missing)}")

                values = load_npy(in_tar, parts[VALUE_SUFFIX])
                meta = json.load(in_tar.extractfile(parts[META_SUFFIX]))
                quantized, quant_meta = quantize_affine_per_sample(values)
                fp16_bytes += int(values.nbytes)
                int8_bytes += int(quantized.nbytes)

                slim_meta = slim_metadata(meta, quant_meta)

                add_file(out_tar, f"{key}.{VALUE_SUFFIX}", npy_bytes(quantized))
                mask_bytes = in_tar.extractfile(parts[MASK_SUFFIX]).read()
                add_file(out_tar, f"{key}.{MASK_SUFFIX}", mask_bytes)
                add_file(out_tar, f"{key}.{META_SUFFIX}", json_bytes(slim_meta))
                samples += 1

        tmp_path.replace(dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return samples, fp16_bytes, int8_bytes


def main() -> None:
    args = parse_args()
    shards = select_shards(args)
    if not shards:
        raise SystemExit("no shards selected")

    skipped_existing = 0
    if args.skip_existing and not args.overwrite:
        remaining = []
        for shard in shards:
            if (args.dst / shard.name).exists():
                skipped_existing += 1
            else:
                remaining.append(shard)
        shards = remaining

    print(f"source: {args.src}")
    print(f"destination: {args.dst}")
    print(f"selected shards: {len(shards)}")
    if skipped_existing:
        print(f"skipped existing shards: {skipped_existing}")
    if args.dry_run:
        for shard in shards:
            print(shard)
        return
    if not shards:
        print("no shards to convert")
        return

    total_samples = 0
    total_fp16_bytes = 0
    total_int8_bytes = 0
    skipped_during_run = 0
    for src in tqdm(shards, unit="shard"):
        dst = args.dst / src.name
        if args.skip_existing and not args.overwrite and dst.exists():
            skipped_during_run += 1
            continue
        samples, fp16_bytes, int8_bytes = convert_shard(src, dst, overwrite=args.overwrite)
        total_samples += samples
        total_fp16_bytes += fp16_bytes
        total_int8_bytes += int8_bytes

    saved = total_fp16_bytes - total_int8_bytes
    print(f"samples: {total_samples}")
    if skipped_during_run:
        print(f"skipped existing shards during run: {skipped_during_run}")
    print(f"image_values raw bytes fp16 -> int8: {total_fp16_bytes} -> {total_int8_bytes}")
    print(f"image_values raw bytes saved: {saved}")


if __name__ == "__main__":
    main()
