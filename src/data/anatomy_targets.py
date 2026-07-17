"""Utilities for building masked-anatomy targets from SynthSeg dseg files."""

import copy
import json
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


DEFAULT_VOCABULARY_PATH = Path(__file__).with_name("synthseg_2_parc_vocabulary.json")


@dataclass(frozen=True)
class AnatomyVocabulary:
    name: str
    values: tuple[int, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("anatomy vocabulary cannot be empty")
        if len(self.values) != len(self.names):
            raise ValueError("anatomy vocabulary values and names must have the same length")
        if len(set(self.values)) != len(self.values):
            raise ValueError("anatomy vocabulary contains duplicate label values")
        if any(value <= 0 for value in self.values):
            raise ValueError("anatomy vocabulary must exclude background and negative labels")

    @property
    def num_classes(self) -> int:
        return len(self.values)


def load_anatomy_vocabulary(
    path: str | Path = DEFAULT_VOCABULARY_PATH,
) -> AnatomyVocabulary:
    path = Path(path)
    with path.open() as file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(file)
        else:
            payload = json.load(file)

    vocabulary_name = path.stem
    if isinstance(payload, dict):
        vocabulary_name = str(payload.get("name", vocabulary_name))
        labels = payload.get("labels", payload.get("label_values"))
    else:
        labels = payload
    if not isinstance(labels, list):
        raise ValueError(f"{path} must contain a list under 'labels' or 'label_values'")

    values = []
    names = []
    for item in labels:
        if isinstance(item, dict):
            values.append(int(item["value"]))
            names.append(str(item.get("name", item["value"])))
        else:
            values.append(int(item))
            names.append(str(item))
    return AnatomyVocabulary(vocabulary_name, tuple(values), tuple(names))


def as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must have three values, got {tuple(value)}")
    return tuple(int(item) for item in value)


def center_crop_or_pad(
    volume: np.ndarray,
    target_shape: int | Sequence[int],
    *,
    fill_value: int = 0,
) -> np.ndarray:
    """Center-fit a 3D array without interpolation."""
    target_shape = as_3tuple(target_shape, "target_shape")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

    output = np.full(target_shape, fill_value, dtype=volume.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(volume.shape, target_shape):
        copy_size = min(source_size, target_size)
        source_start = max((source_size - target_size) // 2, 0)
        target_start = max((target_size - source_size) // 2, 0)
        source_slices.append(slice(source_start, source_start + copy_size))
        target_slices.append(slice(target_start, target_start + copy_size))
    output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def load_synthseg_dseg(
    path: str | Path,
    target_shape: int | Sequence[int],
) -> np.ndarray:
    """
    Load a SynthSeg dseg in the model's [depth, height, width] axis order.

    The preprocessing data path stores NIfTI arrays as [z, y, x], followed by
    the same center crop/pad operation used for sparse image volumes.
    """
    image = nib.load(path)
    labels = np.asanyarray(image.dataobj)
    labels = np.squeeze(labels)
    if labels.ndim != 3:
        raise ValueError(f"SynthSeg dseg must be 3D, got shape {labels.shape} from {path}")
    if not np.isfinite(labels).all():
        raise ValueError(f"SynthSeg dseg contains non-finite labels: {path}")

    rounded = np.rint(labels)
    if not np.array_equal(labels, rounded):
        raise ValueError(f"SynthSeg dseg contains non-integer labels: {path}")
    labels = rounded.astype(np.int32, copy=False).transpose(2, 1, 0)
    return center_crop_or_pad(labels, target_shape)


def patch_label_counts(
    labels: np.ndarray,
    label_values: Sequence[int],
    patch_size: int | Sequence[int],
    *,
    ignore_unknown: bool = False,
) -> np.ndarray:
    """
    Count labelled brain voxels per patch.

    Returns a dense [num_patches, num_classes] uint16 matrix. It is intended to
    be stored with ``np.savez_compressed``; the matrix is highly sparse.
    """
    patch_size = as_3tuple(patch_size, "patch_size")
    if labels.ndim != 3:
        raise ValueError(f"expected 3D labels, got shape {labels.shape}")
    if any(size % patch != 0 for size, patch in zip(labels.shape, patch_size)):
        raise ValueError(f"label shape {labels.shape} must be divisible by patch_size {patch_size}")

    label_values = np.asarray(tuple(int(value) for value in label_values), dtype=np.int64)
    if label_values.ndim != 1 or label_values.size == 0:
        raise ValueError("label_values must be a non-empty one-dimensional sequence")
    if np.unique(label_values).size != label_values.size:
        raise ValueError("label_values contains duplicates")

    sorted_order = np.argsort(label_values)
    sorted_values = label_values[sorted_order]
    flat_labels = labels.reshape(-1).astype(np.int64, copy=False)
    positions = np.searchsorted(sorted_values, flat_labels)
    clipped_positions = np.minimum(positions, sorted_values.size - 1)
    known = (positions < sorted_values.size) & (sorted_values[clipped_positions] == flat_labels)

    unknown = np.unique(flat_labels[(flat_labels != 0) & ~known])
    if unknown.size and not ignore_unknown:
        preview = ", ".join(str(value) for value in unknown[:12])
        suffix = "" if unknown.size <= 12 else f", ... ({unknown.size} total)"
        raise ValueError(f"dseg contains labels absent from the vocabulary: {preview}{suffix}")

    valid_flat_ids = np.flatnonzero(known)
    class_ids = sorted_order[positions[known]]

    grid_size = tuple(size // patch for size, patch in zip(labels.shape, patch_size))
    num_patches = int(np.prod(grid_size))
    num_classes = int(label_values.size)
    patch_voxels = int(np.prod(patch_size))
    if patch_voxels > np.iinfo(np.uint16).max:
        raise ValueError(f"patch volume {patch_voxels} does not fit in uint16")

    z, y, x = np.unravel_index(valid_flat_ids, labels.shape)
    patch_ids = (
        (z // patch_size[0]) * grid_size[1] * grid_size[2]
        + (y // patch_size[1]) * grid_size[2]
        + (x // patch_size[2])
    )
    flat_count_ids = patch_ids * num_classes + class_ids
    counts = np.bincount(
        flat_count_ids,
        minlength=num_patches * num_classes,
    ).reshape(num_patches, num_classes)
    return counts.astype(np.uint16, copy=False)


def anatomy_counts_from_dseg(
    dseg_path: str | Path,
    vocabulary: AnatomyVocabulary,
    target_shape: int | Sequence[int],
    patch_size: int | Sequence[int],
    *,
    ignore_unknown: bool = False,
) -> np.ndarray:
    labels = load_synthseg_dseg(dseg_path, target_shape)
    return patch_label_counts(
        labels,
        vocabulary.values,
        patch_size,
        ignore_unknown=ignore_unknown,
    )


def encode_anatomy_npz(
    counts: np.ndarray,
    vocabulary: AnatomyVocabulary,
) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        counts=counts,
        label_values=np.asarray(vocabulary.values, dtype=np.int32),
    )
    return buffer.getvalue()


def resolve_synthseg_dseg(meta: dict, source_root: str | Path) -> Path:
    """Resolve a dseg from a WDS sample's metadata and a current dataset root."""
    source_root = Path(source_root)
    direct_keys = ("synthseg_path", "synthseg_source_path", "dseg_path")
    for key in direct_keys:
        value = meta.get(key)
        if value and Path(value).exists():
            return Path(value)

    subset = str(meta.get("subset", ""))
    stems = []
    native_stem = meta.get("native_stem")
    if native_stem:
        stems.append(str(native_stem))
    image_path = meta.get("image_path")
    if image_path:
        image_name = Path(image_path).name.removesuffix(".nii.gz").removesuffix(".nii")
        stems.append(image_name.split("_space-", maxsplit=1)[0])
    stems = list(dict.fromkeys(stems))
    if not stems:
        raise KeyError("sample metadata must contain native_stem or image_path")

    roots = [source_root]
    if subset:
        roots.insert(0, source_root / subset)
    candidates = [
        root / "derivatives" / "synthseg" / f"{stem}_desc-synthseg_dseg.nii.gz"
        for root in roots
        for stem in stems
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"could not resolve SynthSeg dseg; tried:\n{attempted}")


def _sample_target_shape(meta: dict) -> tuple[int, int, int]:
    try:
        dense_shape = tuple(int(value) for value in meta["sparse_image"]["dense_shape"])
    except (KeyError, TypeError) as exc:
        raise KeyError("sample metadata is missing sparse_image.dense_shape") from exc
    if len(dense_shape) == 4:
        dense_shape = dense_shape[1:]
    if len(dense_shape) != 3:
        raise ValueError(f"expected 3D dense shape (with optional channel), got {dense_shape}")
    return dense_shape


def augment_wds_shard_with_anatomy(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_root: str | Path,
    vocabulary: AnatomyVocabulary,
    patch_size: int | Sequence[int],
    target_suffix: str = "anatomy.npz",
    expected_img_size: int | Sequence[int] | None = None,
    ignore_unknown: bool = False,
    replace_targets: bool = False,
    overwrite_output: bool = False,
) -> dict[str, int]:
    """Copy a WDS tar shard and add compressed patch anatomy counts per sample."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output shards must be different paths")
    if output_path.exists() and not overwrite_output:
        raise FileExistsError(f"output shard already exists: {output_path}")

    patch_size = as_3tuple(patch_size, "patch_size")
    expected_img_size = (
        None if expected_img_size is None else as_3tuple(expected_img_size, "expected_img_size")
    )
    target_suffix = target_suffix.removeprefix(".")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    stats = {"samples": 0, "targets_added": 0, "targets_kept": 0}
    try:
        with tarfile.open(input_path, "r:*") as source:
            members = source.getmembers()
            member_names = {member.name for member in members}
            write_mode = "w:gz" if output_path.name.endswith((".tar.gz", ".tgz")) else "w"
            with tarfile.open(temporary_path, write_mode, format=tarfile.PAX_FORMAT) as output:
                for member in members:
                    is_target = member.name.endswith(f".{target_suffix}")
                    if is_target and replace_targets:
                        continue

                    member_file = source.extractfile(member) if member.isfile() else None
                    data = member_file.read() if member_file is not None else None
                    output.addfile(
                        copy.copy(member),
                        None if data is None else BytesIO(data),
                    )

                    if not member.name.endswith(".meta.json"):
                        continue
                    stats["samples"] += 1
                    sample_key = member.name.removesuffix(".meta.json")
                    target_name = f"{sample_key}.{target_suffix}"
                    if target_name in member_names and not replace_targets:
                        stats["targets_kept"] += 1
                        continue
                    if data is None:
                        raise ValueError(f"metadata member has no data: {member.name}")

                    meta = json.loads(data)
                    target_shape = _sample_target_shape(meta)
                    if expected_img_size is not None and target_shape != expected_img_size:
                        raise ValueError(
                            f"{sample_key}: target shape {target_shape} does not match "
                            f"expected image size {expected_img_size}"
                        )
                    dseg_path = resolve_synthseg_dseg(meta, source_root)
                    counts = anatomy_counts_from_dseg(
                        dseg_path,
                        vocabulary,
                        target_shape,
                        patch_size,
                        ignore_unknown=ignore_unknown,
                    )
                    target_data = encode_anatomy_npz(counts, vocabulary)
                    target_info = tarfile.TarInfo(target_name)
                    target_info.size = len(target_data)
                    target_info.mode = 0o644
                    target_info.mtime = member.mtime
                    target_info.uid = member.uid
                    target_info.gid = member.gid
                    target_info.uname = member.uname
                    target_info.gname = member.gname
                    output.addfile(target_info, BytesIO(target_data))
                    stats["targets_added"] += 1

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return stats
