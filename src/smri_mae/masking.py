# Copyright (c) Sophont, Inc
# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/data.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import default_collate
from jaxtyping import Float, Int
from .modules import Patchify3D
from .utils import filter_kwargs


class PatchMasking(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        img_size: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = _to_3d_tuple(img_size, "img_size")
        self.patch_size = _to_3d_tuple(patch_size, "patch_size")

        self.patchify = Patchify3D(self.img_size, self.patch_size, in_chans=1)
        self.grid_size = self.patchify.grid_size

    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}"

    def _prepare_img_mask(
        self,
        img_mask: Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[Tensor, bool]:
        single_sample = True
        if img_mask is None:
            img_mask = torch.ones((1, 1, *self.img_size), device=device)
        else:
            img_mask = img_mask.to(device=device) if device is not None else img_mask
            spatial_ndim = len(self.img_size)
            single_sample = img_mask.ndim == spatial_ndim
            if img_mask.ndim == spatial_ndim:
                img_mask = img_mask.reshape(1, 1, *img_mask.shape)
            elif img_mask.ndim == spatial_ndim + 1:
                img_mask = img_mask.unsqueeze(1)
            elif img_mask.ndim != spatial_ndim + 2:
                raise ValueError(
                    f"expected mask with {spatial_ndim}, {spatial_ndim + 1}, "
                    f"or {spatial_ndim + 2} dims, got shape {tuple(img_mask.shape)}"
                )
            img_mask = img_mask.expand((img_mask.shape[0], 1, *self.img_size))
        return img_mask, single_sample

    def _patch_mask_from_img_mask(self, img_mask: Tensor) -> tuple[Tensor, Tensor]:
        mask_patches = self.patchify(img_mask)
        patch_mask = mask_patches.any(dim=-1)
        return patch_mask, mask_patches

    def _num_keep(self, valid_patch_mask: Tensor) -> int:
        min_count = valid_patch_mask.sum(dim=1).min()
        return int((1 - self.mask_ratio) * min_count.item())

    def _unpatchify_patch_mask(
        self,
        patch_mask: Tensor,
        mask_patches: Tensor,
        single_sample: bool,
    ) -> Tensor:
        mask_patches = patch_mask.to(mask_patches.dtype).unsqueeze(-1).expand_as(mask_patches)
        mask = self.patchify.unpatchify(mask_patches)
        mask = mask[:, 0]  # [B, D, H, W]
        if single_sample:
            mask = mask[0]
        return mask


class RandomMasking(PatchMasking):
    def forward(
        self,
        img_mask: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        img_mask, single_sample = self._prepare_img_mask(img_mask, device=device)
        patch_mask, mask_patches = self._patch_mask_from_img_mask(img_mask)
        patch_mask, _ = trim_patch_mask(
            patch_mask.float(), mask_ratio=self.mask_ratio, shuffle=True
        )
        return self._unpatchify_patch_mask(patch_mask, mask_patches, single_sample)


class BlockMasking(PatchMasking):
    def __init__(
        self,
        mask_ratio: float,
        img_size: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        block_size: int | tuple[int, int, int] = (2, 2, 2),
        max_block_attempts: int = 1_000,
    ):
        super().__init__(mask_ratio, img_size, patch_size)
        self.block_size = _to_3d_tuple(block_size, "block_size")
        if any(size <= 0 for size in self.block_size):
            raise ValueError(f"block_size entries must be positive, got {self.block_size}.")
        self.max_block_attempts = max_block_attempts

    def extra_repr(self):
        return (
            f"mask_ratio={self.mask_ratio}, block_size={self.block_size}, "
            f"max_block_attempts={self.max_block_attempts}"
        )

    def forward(
        self,
        img_mask: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        img_mask, single_sample = self._prepare_img_mask(img_mask, device=device)
        valid_patch_mask, mask_patches = self._patch_mask_from_img_mask(img_mask)
        visible_patch_mask = self._visible_patch_mask(valid_patch_mask)
        return self._unpatchify_patch_mask(visible_patch_mask, mask_patches, single_sample)

    def _visible_patch_mask(self, valid_patch_mask: Tensor) -> Tensor:
        len_keep = self._num_keep(valid_patch_mask)
        masked_patch_mask = torch.zeros_like(valid_patch_mask, dtype=torch.bool)
        valid_counts = valid_patch_mask.sum(dim=1)

        for batch_idx in range(valid_patch_mask.shape[0]):
            num_masked = int(valid_counts[batch_idx].item()) - len_keep
            valid_grid = valid_patch_mask[batch_idx].reshape(self.grid_size)
            masked_grid = self._sample_block_mask(valid_grid, num_masked)
            masked_patch_mask[batch_idx] = masked_grid.flatten()

        return valid_patch_mask & ~masked_patch_mask

    def _sample_block_mask(
        self,
        valid_grid: Tensor,
        num_masked: int,
        initial_masked_grid: Tensor | None = None,
    ) -> Tensor:
        if initial_masked_grid is None:
            masked_grid = torch.zeros_like(valid_grid, dtype=torch.bool)
        else:
            masked_grid = initial_masked_grid.clone()

        if num_masked <= 0:
            return masked_grid

        block_size = self.block_size
        max_starts = tuple(dim - size + 1 for dim, size in zip(self.grid_size, block_size))
        masked_count = int((masked_grid & valid_grid).sum().item())
        target_count = masked_count + num_masked

        if all(max_start > 0 for max_start in max_starts):
            for _ in range(self.max_block_attempts):
                remaining = target_count - masked_count
                if remaining <= 0:
                    break

                starts = [
                    int(torch.randint(max_start, (1,), device=valid_grid.device).item())
                    for max_start in max_starts
                ]
                slices = tuple(
                    slice(start, start + size) for start, size in zip(starts, block_size)
                )
                block_available = valid_grid[slices] & ~masked_grid[slices]
                num_new = int(block_available.sum().item())
                if num_new == 0 or num_new > remaining:
                    continue

                masked_grid[slices] |= block_available
                masked_count += num_new

        remaining = target_count - masked_count
        if remaining > 0:
            masked_grid = _random_fill_mask(valid_grid, masked_grid, remaining)
        return masked_grid


class HybridMasking(BlockMasking):
    def __init__(
        self,
        mask_ratio: float,
        img_size: int | tuple[int, ...],
        patch_size: int | tuple[int, ...],
        block_size: int | tuple[int, int, int] = (2, 2, 2),
        random_fraction: float = 0.5,
        block_fraction: float = 0.5,
        max_block_attempts: int = 1_000,
    ):
        super().__init__(
            mask_ratio,
            img_size,
            patch_size,
            block_size,
            max_block_attempts,
        )
        if random_fraction < 0 or block_fraction < 0:
            raise ValueError("random_fraction and block_fraction must be non-negative.")
        if random_fraction + block_fraction <= 0:
            raise ValueError("random_fraction and block_fraction cannot both be zero.")
        self.random_fraction = random_fraction
        self.block_fraction = block_fraction

    def extra_repr(self):
        return (
            f"mask_ratio={self.mask_ratio}, block_size={self.block_size}, "
            f"random_fraction={self.random_fraction}, block_fraction={self.block_fraction}, "
            f"max_block_attempts={self.max_block_attempts}"
        )

    def _visible_patch_mask(self, valid_patch_mask: Tensor) -> Tensor:
        len_keep = self._num_keep(valid_patch_mask)
        masked_patch_mask = torch.zeros_like(valid_patch_mask, dtype=torch.bool)
        valid_counts = valid_patch_mask.sum(dim=1)
        fraction_sum = self.random_fraction + self.block_fraction

        for batch_idx in range(valid_patch_mask.shape[0]):
            total_masked = int(valid_counts[batch_idx].item()) - len_keep
            num_random_masked = round(total_masked * self.random_fraction / fraction_sum)
            num_block_masked = total_masked - num_random_masked

            valid_grid = valid_patch_mask[batch_idx].reshape(self.grid_size)
            masked_grid = self._sample_block_mask(valid_grid, num_block_masked)
            masked_grid = _random_fill_mask(valid_grid, masked_grid, num_random_masked)
            masked_patch_mask[batch_idx] = masked_grid.flatten()

        return valid_patch_mask & ~masked_patch_mask


def _random_fill_mask(valid_grid: Tensor, masked_grid: Tensor, num_masked: int) -> Tensor:
    if num_masked <= 0:
        return masked_grid

    available = valid_grid & ~masked_grid
    available_ids = available.flatten().nonzero(as_tuple=False).flatten()
    if num_masked > available_ids.numel():
        raise ValueError(
            f"cannot mask {num_masked} patches; only {available_ids.numel()} valid patches remain"
        )
    order = torch.randperm(available_ids.numel(), device=valid_grid.device)
    fill_ids = available_ids[order[:num_masked]]
    flat_masked = masked_grid.flatten()
    flat_masked[fill_ids] = True
    return flat_masked.reshape_as(masked_grid)


def _to_3d_tuple(value: int | tuple[int, ...], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly 3 spatial dimensions, got {tuple(value)}")
    return tuple(int(item) for item in value)


MASKING_DICT = {
    "block": BlockMasking,
    "hybrid": HybridMasking,
    "random": RandomMasking,
}


def create_masking(name: str, **kwargs) -> PatchMasking:
    if name not in MASKING_DICT:
        valid = ", ".join(sorted(MASKING_DICT))
        raise ValueError(f"Unknown masking mode {name!r}. Available modes: {valid}.")
    cls = MASKING_DICT[name]
    kwargs = filter_kwargs(cls, kwargs)
    mask_fn = cls(**kwargs)
    return mask_fn


def mask_collate(
    samples: list[dict[str, Tensor]], *, mask_fn: PatchMasking | None = None
) -> dict[str, Tensor]:
    """
    Generates a visible mask for each sample, and pads the shape with singleton
    dimensions for batching.
    """
    for sample in samples:
        image = sample["image"]
        img_mask = sample.get("img_mask")
        if img_mask is not None:
            sample["img_mask"] = _unsqueeze_as(img_mask, image)
    metas = [sample.pop("meta", None) for sample in samples]
    batch = default_collate(samples)
    if mask_fn is not None:
        img_mask = batch.get("img_mask")
        if img_mask is None:
            img_mask = torch.ones(
                (batch["image"].shape[0], *mask_fn.img_size),
                dtype=batch["image"].dtype,
                device=batch["image"].device,
            )
        visible_mask = mask_fn(img_mask)
        batch["visible_mask"] = _unsqueeze_batch_mask_as(visible_mask, batch["image"])
    if any(meta is not None for meta in metas):
        batch["meta"] = metas
    return batch


def _unsqueeze_as(x: Tensor, other: Tensor) -> Tensor:
    assert other.ndim >= x.ndim
    x = x.reshape((1,) * (other.ndim - x.ndim) + x.shape)
    return x


def _unsqueeze_batch_mask_as(x: Tensor, other: Tensor) -> Tensor:
    spatial_ndim = other.ndim - 2
    if x.ndim == spatial_ndim:
        x = x.reshape(1, 1, *x.shape)
    elif x.ndim == spatial_ndim + 1:
        x = x.unsqueeze(1)
    elif x.ndim != spatial_ndim + 2:
        raise ValueError(f"cannot broadcast mask shape {tuple(x.shape)} to {tuple(other.shape)}")
    return x


def trim_patch_mask(
    patch_mask: Float[Tensor, "B N"],
    mask_ratio: float | None = None,
    len_keep: int | None = None,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "B N"], Int[Tensor, "B L"]]:
    """
    Trim a batch of patch masks to the same number of patches.
    Kept patches are selected randomly (shuffle=True) or sequentially (shuffle=False).
    """
    assert not (mask_ratio and len_keep), "can't set both mask_ratio and len_keep"
    B, N = patch_mask.shape
    device = patch_mask.device

    # shuffle patches for each sample
    if shuffle:
        noise = torch.rand(B, N, generator=generator, device=device)
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)
        patch_mask = patch_mask.gather(1, shuffle_ids)

    # all masks trimmed to have the same size, no bigger than the smallest mask
    min_count = patch_mask.sum(dim=1).min()
    if mask_ratio is not None:
        len_keep = int((1 - mask_ratio) * min_count.item())
    else:
        len_keep = min_count if len_keep is None else min_count.clamp(max=len_keep)

    # discard extra patches
    patch_mask = patch_mask * (patch_mask.cumsum(dim=1) <= len_keep)

    # shuffle patches back to original order
    if shuffle:
        patch_mask = patch_mask.gather(1, restore_ids)

    mask_ids = patch_mask.nonzero(as_tuple=False)[:, 1].reshape(B, -1)
    return patch_mask, mask_ids


def pad_image_mask(mask: Float[Tensor, "..."], pad: int = 1):
    """
    Dilate ("pad") an image or volume mask by a few voxels.
    """
    if pad <= 0:
        return mask

    dtype = mask.dtype
    device = mask.device
    kernel_size = 2 * pad + 1

    if mask.ndim < 5:
        raise ValueError(f"expected a batched 3D mask, got shape {tuple(mask.shape)}")

    *shape, D, H, W = mask.shape
    mask = mask.reshape(-1, 1, D, H, W)
    weight = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device, dtype=dtype)
    out_mask = F.conv3d(mask, weight, padding="same")
    out_mask = out_mask.reshape((*shape, D, H, W))

    out_mask = (out_mask > 0).to(dtype)
    return out_mask
