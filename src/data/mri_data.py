import math
from functools import partial
from glob import glob
from typing import Sequence

import braceexpand
import numpy as np
import torch
import webdataset as wds
from torch import Tensor


def collate(
    samples: list[dict],
    *,
    include_meta: bool = True,
) -> dict[str, Tensor]:
    masks = [torch.as_tensor(sample["img_mask"]) for sample in samples]
    batch = {"img_mask": torch.stack(masks)}
    image_values = [
        torch.as_tensor(sample["image_values"], dtype=torch.float16) for sample in samples
    ]
    batch["image_values"] = torch.cat(image_values)

    has_anatomy = ["anatomy_counts" in sample for sample in samples]
    if any(has_anatomy) and not all(has_anatomy):
        raise ValueError("anatomy_counts must be present for every sample in a batch")
    if all(has_anatomy):
        anatomy_counts = [np.asarray(sample["anatomy_counts"]) for sample in samples]
        shapes = {counts.shape for counts in anatomy_counts}
        if len(shapes) != 1:
            raise ValueError(f"anatomy_counts shapes differ within a batch: {sorted(shapes)}")
        max_count = max(int(counts.max(initial=0)) for counts in anatomy_counts)
        if max_count > torch.iinfo(torch.int16).max:
            raise ValueError(f"anatomy patch count {max_count} does not fit in int16")
        batch["anatomy_counts"] = torch.stack(
            [torch.as_tensor(counts.copy(), dtype=torch.int16) for counts in anatomy_counts]
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


def extract_sparse_wds_sample(
    sample: dict,
    *,
    anatomy_key: str | None = None,
    expected_anatomy_label_values: Sequence[int] | None = None,
) -> dict:
    extracted = {
        "image_values": np.asarray(sample["image_values.npy"], dtype=np.float16),
        "img_mask": np.asarray(sample["img_mask.npy"], dtype=np.uint8),
        "meta": sample["meta.json"],
    }
    if anatomy_key is not None:
        anatomy = sample[anatomy_key]
        embedded_label_values = None
        if isinstance(anatomy, dict):
            embedded_label_values = anatomy.get("label_values")
            try:
                anatomy = anatomy["counts"]
            except KeyError as exc:
                raise KeyError(f"{anatomy_key!r} must contain a 'counts' array") from exc
        anatomy = np.asarray(anatomy)
        if anatomy.ndim != 2:
            raise ValueError(
                f"{anatomy_key!r} must decode to a [num_patches, num_classes] array, "
                f"got shape {anatomy.shape}"
            )
        if expected_anatomy_label_values is not None:
            if embedded_label_values is None:
                raise KeyError(
                    f"{anatomy_key!r} must embed label_values when a vocabulary is configured"
                )
            expected = np.asarray(expected_anatomy_label_values, dtype=np.int64)
            embedded = np.asarray(embedded_label_values, dtype=np.int64)
            if not np.array_equal(embedded, expected):
                raise ValueError(
                    f"{anatomy_key!r} label_values do not match the configured vocabulary"
                )
        extracted["anatomy_counts"] = anatomy
    return extracted


def make_sparse_wds_dataset(
    url: str | list[str],
    *,
    shuffle: bool,
    buffer_size: int,
    anatomy_key: str | None = None,
    expected_anatomy_label_values: Sequence[int] | None = None,
) -> wds.WebDataset:
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    extractor = partial(
        extract_sparse_wds_sample,
        anatomy_key=anatomy_key,
        expected_anatomy_label_values=expected_anatomy_label_values,
    )
    dataset = dataset.decode().map(extractor, handler=warn_and_continue)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset
