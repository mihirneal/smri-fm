import inspect
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

import braceexpand
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from cloudpathlib import CloudPath
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "/tmp/datasets")

disable_progress_bars()


def make_mri_wds_dataset(
    url: str | list[str],
    shuffle: bool = True,
    buffer_size: int = 1000,
    img_size: Sequence[int] | None = (208, 240, 208),
) -> wds.WebDataset:
    """Make a structural MRI volume WebDataset.

    Expected sample members:
    - image.npy: float volume shaped [D, H, W]
    - mask.npy: binary brain mask shaped [D, H, W]
    - meta.json: scan metadata

    Returned samples contain:
    - image: float tensor shaped [1, D, H, W]
    - img_mask: float tensor shaped [D, H, W]
    - meta: collatable metadata dictionary
    """
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    dataset = dataset.decode().map(extract_vol_sample, handler=warn_and_continue)

    dataset = dataset.map(
        lambda sample: vol_transform(
            sample,
            img_size=img_size,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset


def expand_urls(urls: str | list[str]) -> list[str]:
    """
    Expand wds urls:

    - expand glob patterns
    - expand brace expressions
    - download/cache hf:// and s3:// dataset folders

    Adapted from `webdataset.shardlists.expand_urls`.
    """
    if isinstance(urls, str):
        urls = [urls]
    results = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme in {"hf", "s3"}:
            results.extend(sorted(str(path) for path in maybe_download(url).glob("*.tar")))
            continue

        chars = set(url)
        if chars.intersection("[*?"):
            result = sorted(glob(url))
        elif "{" in chars:
            result = braceexpand.braceexpand(url)
        else:
            result = [url]
        results.extend(result)
    return results


def maybe_download(url: str, cache_dir: str | Path | None = None) -> Path:
    cache_dir = Path(cache_dir or DATA_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)

    parsed = urlparse(url)
    if parsed.scheme == "hf":
        path = Path(parsed.path)
        repo_id = f"{parsed.netloc}{path.parents[-2]}"
        subfolder = path.relative_to(path.parents[-2])
        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/**",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        local_path = Path(local_path)
    elif parsed.scheme == "s3":
        path = CloudPath(url)
        local_path = Path(cache_dir) / path.name
        subprocess.run(
            ["aws", "s3", "sync", "--quiet", str(path), str(local_path)],
            check=True,
        )
    else:
        assert not parsed.scheme, f"unsupported url scheme {parsed.scheme}"
        local_path = Path(url)
    return local_path

def warn_and_continue(exn: Exception) -> bool:
    # modified wds warn and continue handler to send warning to stdout log.
    # but note, this won't propagate to the wandb console log since it will
    # originate in a child data loader worker process.
    print(f"WARNING {repr(exn)}")
    return True


def extract_vol_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract raw arrays and metadata from a volume WebDataset sample."""
    image = np.asarray(sample["image.npy"], dtype=np.float32)
    mask = np.asarray(sample["mask.npy"] > 0, dtype=np.uint8)
    meta = sample["meta.json"]

    if image.ndim != 3:
        raise ValueError(f"expected image.npy to be 3D, got shape {image.shape}")
    if mask.shape != image.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {image.shape}")

    return {
        "image": image,
        "img_mask": mask,
        "meta": meta,
    }


def vol_transform(
    sample: dict[str, Any],
    img_size: Sequence[int] | None = (208, 240, 208),
) -> dict[str, Any]:
    """Transform one structural MRI sample into model-ready tensors.

    Each individual volume is z-scored using only voxels inside that sample's
    brain mask.
    """
    image = torch.as_tensor(sample["image"]).float()
    mask = torch.as_tensor(sample["img_mask"] > 0).float()

    if img_size is not None:
        image = pad_to_size_3d(image, tuple(img_size))
        mask = pad_to_size_3d(mask, tuple(img_size))

    mask = (mask > 0).float()
    image = image * mask

    image = apply_masked_zscore(image, mask)

    meta = make_collatable(sample["meta"])
    sample_ = {
        "image": image[None],
        "img_mask": mask,
        "meta": meta,
    }
    return sample_


def apply_masked_zscore(
    image: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Z-score one volume using only voxels inside its brain mask."""
    values = image[mask > 0]
    if values.numel() == 0:
        raise ValueError("empty img_mask; cannot normalize volume")
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(eps)
    return ((image - mean) / std) * mask


def pad_to_size_3d(img: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Pad a [D, H, W] tensor around the edges to a fixed 3D size."""
    if img.ndim != 3:
        raise ValueError(f"expected [D, H, W] tensor, got shape {tuple(img.shape)}")

    d_new, h_new, w_new = size
    d, h, w = img.shape
    if d > d_new or h > h_new or w > w_new:
        raise ValueError(f"target size {size} is smaller than image shape {tuple(img.shape)}")

    pad_d = max(d_new - d, 0)
    pad_h = max(h_new - h, 0)
    pad_w = max(w_new - w, 0)
    if pad_d == pad_h == pad_w == 0:
        return img

    padding = (
        pad_w // 2,
        pad_w - pad_w // 2,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_d // 2,
        pad_d - pad_d // 2,
    )
    return F.pad(img, padding)


def make_collatable(value: Any) -> Any:
    """Replace JSON null values in metadata dictionaries for PyTorch collation."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return {key: make_collatable(item) for key, item in value.items()}
    return value
