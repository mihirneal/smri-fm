import io
import json
import math
import multiprocessing
import os
import random
import tarfile
from collections import defaultdict
from functools import partial
from glob import glob
from pathlib import Path
from typing import Iterator, Sequence

import braceexpand
import numpy as np
import torch
import webdataset as wds
from torch import Tensor


SUBJECT_INDEX_VERSION = 1


def collate(
    samples: list[dict],
    *,
    include_meta: bool = True,
    include_anat: bool = False,
) -> dict[str, Tensor]:
    masks = [torch.as_tensor(sample["img_mask"]) for sample in samples]
    batch = {"img_mask": torch.stack(masks)}
    image_values = [
        torch.as_tensor(sample["image_values"], dtype=torch.float16) for sample in samples
    ]
    batch["image_values"] = torch.cat(image_values)

    if include_anat:
        batch["anatomy_counts"] = torch.as_tensor(
            np.stack([sample["anatomy_counts"] for sample in samples]),
            dtype=torch.int16,
        )

    if include_meta:
        batch["meta"] = [sample["meta"] for sample in samples]
    return batch


def unpack_img_mask_batch(mask: torch.Tensor, image_shape: Sequence[int]) -> torch.Tensor:
    """Return a dense boolean batch mask from bit-packed mask tensors."""
    image_shape = tuple(int(dim) for dim in image_shape)
    mask_numel = math.prod(image_shape)
    packed_numel = math.ceil(mask_numel / 8)
    if mask.dtype != torch.uint8:
        raise ValueError(f"packed img_mask must have dtype uint8, got {mask.dtype}")
    if mask.ndim != 2 or mask.shape[1] != packed_numel:
        raise ValueError(f"expected packed img_mask shape (B, {packed_numel}), got {tuple(mask.shape)}")

    shifts = torch.arange(7, -1, -1, device=mask.device, dtype=torch.uint8)
    bits = (mask.unsqueeze(-1).bitwise_right_shift(shifts) & 1).reshape(mask.shape[0], -1)
    return bits[:, :mask_numel].reshape((mask.shape[0], *image_shape)).bool()


def densify_sparse_image_batch(
    image_values: torch.Tensor,
    packed_img_mask: torch.Tensor,
    image_shape: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense images from concatenated brain-voxel values and packed masks."""
    image_shape = tuple(int(dim) for dim in image_shape)
    dtype = dtype or image_values.dtype
    masks = unpack_img_mask_batch(packed_img_mask, image_shape)
    batch_size = masks.shape[0]

    images = torch.zeros(
        (batch_size, *image_shape),
        device=image_values.device,
        dtype=dtype,
    )
    images[masks] = image_values.to(dtype=dtype)
    return images, masks


def expand_urls(urls: str | list[str]) -> list[str]:
    """
    Expand wds urls:

    - expand glob patterns
    - expand brace expressions
    - filter files that don't exist

    Adapted from `webdataset.shardlists.expand_urls`.
    """
    if isinstance(urls, str):
        urls = [urls]
    results = []
    for url in urls:
        chars = set(url)
        if chars.intersection("[*?"):
            result = sorted(glob(url))
        elif "{" in chars:
            result = braceexpand.braceexpand(url)
        else:
            result = [url]
        results.extend(result)
    return results



def warn_and_continue(exn):
    print(f"WARNING {repr(exn)}")
    return True


def extract_sparse_wds_sample(sample: dict, *, include_anat: bool = False) -> dict:
    extracted = {
        "image_values": np.asarray(sample["image_values.npy"], dtype=np.float16),
        "img_mask": np.asarray(sample["img_mask.npy"], dtype=np.uint8),
        "meta": sample["meta.json"],
    }
    if include_anat:
        extracted["anatomy_counts"] = sample["anatomy.npz"]["counts"]
    return extracted


def make_sparse_wds_dataset(
    url: str | list[str],
    *,
    shuffle: bool,
    buffer_size: int,
    include_anat: bool = False,
) -> wds.WebDataset:
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    dataset = dataset.decode().map(
        partial(extract_sparse_wds_sample, include_anat=include_anat),
        handler=warn_and_continue,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset


def subject_from_key(key: str) -> str:
    subject, separator, _ = key.partition("_ses-")
    if not separator:
        raise ValueError(f"sample key has no BIDS session: {key!r}")
    return subject


def _shard_signature(shards: Sequence[Path]) -> list[list[str | int]]:
    return [
        [str(shard.resolve()), shard.stat().st_size, shard.stat().st_mtime_ns]
        for shard in shards
    ]


def subject_index_is_current(
    index_path: str | Path,
    shards: Sequence[str | Path],
    *,
    include_anat: bool = False,
) -> bool:
    try:
        with Path(index_path).open() as stream:
            index = json.load(stream)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return (
        index.get("version") == SUBJECT_INDEX_VERSION
        and index.get("include_anat", False) == include_anat
        and index.get("signature") == _shard_signature([Path(shard) for shard in shards])
    )


def build_subject_index(
    index_path: str | Path,
    shards: Sequence[str | Path],
    *,
    include_anat: bool = False,
) -> Path:
    index_path = Path(index_path)
    shards = [Path(shard) for shard in shards]
    suffixes = [
        (".image_values.npy", "image"),
        (".img_mask.npy", "mask"),
    ]
    if include_anat:
        suffixes.append((".anatomy.npz", "anatomy"))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_name(f".{index_path.name}.tmp-{os.getpid()}")
    temporary_path.unlink(missing_ok=True)
    subjects: dict[str, list[tuple]] = defaultdict(list)
    try:
        for shard_id, shard in enumerate(shards):
            samples: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
            with tarfile.open(shard, mode="r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    for suffix, component in suffixes:
                        if member.name.endswith(suffix):
                            key = member.name.removeprefix("./")[: -len(suffix)]
                            samples[key][component] = (member.offset_data, member.size)
                            break

            required = {component for _, component in suffixes}
            for key, components in samples.items():
                missing = required.difference(components)
                if missing:
                    raise ValueError(f"sample {key!r} is missing {sorted(missing)} in {shard}")
                image_offset, image_size = components["image"]
                mask_offset, mask_size = components["mask"]
                reference = [
                    shard_id,
                    image_offset,
                    image_size,
                    mask_offset,
                    mask_size,
                ]
                if include_anat:
                    anatomy_offset, anatomy_size = components["anatomy"]
                    reference.extend((anatomy_offset, anatomy_size))
                reference.append(key)
                subjects[subject_from_key(key)].append(tuple(reference))

        with temporary_path.open("w") as stream:
            json.dump(
                {
                    "version": SUBJECT_INDEX_VERSION,
                    "include_anat": include_anat,
                    "signature": _shard_signature(shards),
                    "shards": [str(shard.resolve()) for shard in shards],
                    "subjects": subjects,
                },
                stream,
            )
        os.replace(temporary_path, index_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return index_path


def ensure_subject_index(
    index_path: str | Path,
    shards: Sequence[str | Path],
    *,
    include_anat: bool = False,
) -> Path:
    if not subject_index_is_current(index_path, shards, include_anat=include_anat):
        build_subject_index(index_path, shards, include_anat=include_anat)
    return Path(index_path)


class SubjectBalancedSparseDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        index_path: str | Path,
        *,
        seed: int,
        sampling: str = "subject",
        rank: int = 0,
        read_ahead: int = 256,
        include_anat: bool = False,
    ):
        with Path(index_path).open() as stream:
            index = json.load(stream)
        if include_anat and not index.get("include_anat", False):
            raise ValueError(f"subject index does not contain anatomy offsets: {index_path}")
        if read_ahead <= 0:
            raise ValueError("read_ahead must be positive")
        sampling = str(sampling).casefold()
        if sampling not in {"subject", "scan"}:
            raise ValueError("sampling must be one of: subject, scan")

        self.shards = index["shards"]
        self.samples = index["subjects"]
        if not self.samples:
            raise ValueError(f"subject index contains no samples: {index_path}")
        self.subjects = tuple(self.samples)
        self.scan_references = tuple(
            reference
            for subject in self.subjects
            for reference in self.samples[subject]
        )
        self.num_subjects = len(self.samples)
        self.num_samples = len(self.scan_references)
        self.seed = seed
        self.sampling = sampling
        self.rank = rank
        self.read_ahead = read_ahead
        self.include_anat = include_anat
        self.epoch = multiprocessing.Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.value = epoch

    def _draw_reference(self, rng: random.Random) -> Sequence:
        if self.sampling == "scan":
            return rng.choice(self.scan_references)
        subject = rng.choice(self.subjects)
        return rng.choice(self.samples[subject])

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(
            self.seed
            + 1_000_003 * self.rank
            + 10_007 * worker_id
            + 1_000_000_007 * self.epoch.value
        )
        descriptor = None
        open_shard = None
        try:
            while True:
                draws = [self._draw_reference(rng) for _ in range(self.read_ahead)]
                draws.sort(key=lambda reference: reference[0])
                for reference in draws:
                    shard_id, image_offset, image_size, mask_offset, mask_size = reference[:5]
                    shard = self.shards[shard_id]
                    if shard != open_shard:
                        if descriptor is not None:
                            os.close(descriptor)
                        descriptor = os.open(shard, os.O_RDONLY)
                        open_shard = shard
                    image = np.load(
                        io.BytesIO(os.pread(descriptor, image_size, image_offset)),
                        allow_pickle=False,
                    )
                    mask = np.load(
                        io.BytesIO(os.pread(descriptor, mask_size, mask_offset)),
                        allow_pickle=False,
                    )
                    sample = {
                        "image_values": np.asarray(image, dtype=np.float16),
                        "img_mask": np.asarray(mask, dtype=np.uint8),
                    }
                    if self.include_anat:
                        anatomy_offset, anatomy_size = reference[5:7]
                        with np.load(
                            io.BytesIO(os.pread(descriptor, anatomy_size, anatomy_offset)),
                            allow_pickle=False,
                        ) as anatomy:
                            sample["anatomy_counts"] = np.asarray(anatomy["counts"])
                    yield sample
        finally:
            if descriptor is not None:
                os.close(descriptor)
