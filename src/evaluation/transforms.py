from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from evaluation.core import DatasetBundle


SampleTransform = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class PadCenterCrop:
    size: Sequence[int]
    key: str = "image"
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        if len(self.size) not in {2, 3}:
            raise ValueError("size must contain 2 or 3 spatial dimensions")
        if any(int(dim) <= 0 for dim in self.size):
            raise ValueError("size dimensions must be positive")

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        value = sample[self.key]
        if not isinstance(value, Tensor):
            raise TypeError(f"{self.key!r} must be a torch.Tensor")
        transformed = dict(sample)
        transformed[self.key] = self._pad_center_crop(value)
        return transformed

    def _pad_center_crop(self, tensor: Tensor) -> Tensor:
        spatial_size = tuple(int(dim) for dim in self.size)
        if tensor.ndim < len(spatial_size):
            raise ValueError(
                f"{self.key!r} has shape {tuple(tensor.shape)}, "
                f"but size has {len(spatial_size)} spatial dimensions"
            )

        leading_shape = tensor.shape[: tensor.ndim - len(spatial_size)]
        spatial_shape = tensor.shape[-len(spatial_size) :]
        if not leading_shape:
            raise ValueError(f"{self.key!r} must include at least one leading dimension")

        pad_args: list[int] = []
        crop_slices: list[slice] = []
        for current, target in zip(
            reversed(spatial_shape), reversed(spatial_size), strict=True
        ):
            pad_total = max(0, target - current)
            pad_before = (pad_total + 1) // 2
            pad_after = pad_total - pad_before
            pad_args.extend([pad_before, pad_after])

        padded = F.pad(tensor, pad_args, mode="constant", value=float(self.pad_value))
        padded_spatial = padded.shape[-len(spatial_size) :]
        for current, target in zip(padded_spatial, spatial_size, strict=True):
            start = max(0, (current - target) // 2)
            crop_slices.append(slice(start, start + target))

        return padded[(Ellipsis, *crop_slices)]


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: SampleTransform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.dataset[index])


def build_transform(cfg: Mapping[str, Any] | None) -> SampleTransform | None:
    if not cfg:
        return None
    name = cfg.get("name")
    if name == "pad_center_crop":
        return PadCenterCrop(
            size=cfg["size"],
            key=str(cfg.get("key", "image")),
            pad_value=float(cfg.get("pad_value", 0.0)),
        )
    raise ValueError("unknown transform {!r}. available transforms: pad_center_crop".format(name))


def apply_transforms(bundle: DatasetBundle, cfg: Mapping[str, Any] | None) -> DatasetBundle:
    transform = build_transform(cfg)
    if transform is None:
        return bundle
    return DatasetBundle(
        train=TransformDataset(bundle.train, transform),
        val=TransformDataset(bundle.val, transform),
        test=TransformDataset(bundle.test, transform),
    )
