import json
import math
from functools import partial
from glob import glob
from typing import Any, Sequence

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
    masks = [torch.as_tensor(sample["img_mask"].copy()) for sample in samples]
    batch = {"img_mask": torch.stack(masks)}
    image_values = [torch.as_tensor(sample["image_values"].copy()) for sample in samples]
    image_dtypes = {values.dtype for values in image_values}
    if len(image_dtypes) != 1:
        raise ValueError(f"mixed image_values dtypes in batch: {sorted(map(str, image_dtypes))}")
    batch["image_values"] = torch.cat(image_values)

    quantizations = [sample.get("image_quantization") for sample in samples]
    if any(quantization is not None for quantization in quantizations):
        if not all(quantization is not None for quantization in quantizations):
            raise ValueError("cannot mix quantized and unquantized image values in one batch")
        batch["image_quantization"] = {
            "scale": torch.tensor(
                [quantization["scale"] for quantization in quantizations], dtype=torch.float32
            ),
            "value_min": torch.tensor(
                [quantization["value_min"] for quantization in quantizations],
                dtype=torch.float32,
            ),
            "qmin": torch.tensor(
                [quantization["qmin"] for quantization in quantizations], dtype=torch.int16
            ),
            "value_counts": torch.tensor(
                [values.numel() for values in image_values], dtype=torch.int64
            ),
        }

    if include_meta:
        batch["meta"] = [make_collatable(sample["meta"]) for sample in samples]
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
    image_quantization: dict[str, torch.Tensor] | None = None,
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
    if image_quantization is not None:
        counts = image_quantization["value_counts"]
        scales = torch.repeat_interleave(image_quantization["scale"], counts)
        value_mins = torch.repeat_interleave(image_quantization["value_min"], counts)
        qmins = torch.repeat_interleave(image_quantization["qmin"], counts)
        dense_values = (image_values.float() - qmins.float()) * scales + value_mins
    else:
        dense_values = image_values
    images[masks] = dense_values.to(dtype=dtype)
    return images, masks


def make_collatable(value: Any) -> Any:
    """Replace JSON null values in metadata dictionaries for PyTorch collation.

    Lists and tuples are stringified because variable-length sequences in
    metadata break `torch.utils.data.default_collate`.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return {key: make_collatable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value


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


def image_quantization_from_meta(meta: dict) -> dict[str, float | int] | None:
    quantization = meta.get("int8_quantization")
    if not quantization:
        return None

    if quantization.get("storage_dtype") != "int8":
        raise ValueError(f"unsupported image_values storage dtype: {quantization.get('storage_dtype')}")
    scheme = quantization.get("scheme")
    if scheme != "affine_per_sample_minmax":
        raise ValueError(f"unsupported image_values quantization: {scheme}")

    domain = quantization.get("domain", "normalized")
    if domain != "normalized":
        raise ValueError(f"unsupported image_values quantization domain: {domain}")

    return {
        "scale": float(quantization["scale"]),
        "value_min": float(quantization["value_min"]),
        "qmin": int(quantization["qmin"]),
    }


def extract_sparse_wds_sample(sample: dict) -> dict:
    meta = sample["meta.json"]
    quantization = image_quantization_from_meta(meta)
    values_dtype = np.int8 if quantization is not None else np.float16
    return {
        "image_values": np.asarray(sample["image_values.npy"], dtype=values_dtype),
        "image_quantization": quantization,
        "img_mask": np.asarray(sample["img_mask.npy"], dtype=np.uint8),
        "meta": meta,
    }


def make_sparse_wds_dataset(
    url: str | list[str],
    *,
    shuffle: bool,
    buffer_size: int,
) -> wds.WebDataset:
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    dataset = dataset.decode().map(extract_sparse_wds_sample, handler=warn_and_continue)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset
