import math
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_resolution_plan(
    batch_size: int,
    generator: torch.Generator,
    *,
    identity_probability: float = 0.5,
    isotropic_probability: float = 0.25,
    isotropic_factors: Sequence[float] = (1.25, 1.5, 1.75, 2.0),
    anisotropic_factors: Sequence[float] = (2.0, 3.0, 4.0),
    anisotropic_axis_probabilities: Sequence[float] = (0.1, 0.1, 0.8),
    max_factors: Tensor | None = None,
) -> Tensor:
    """Draw CPU [B, XYZ] factors from discrete severity banks for grouped execution.

    Ineligible draws become identity; their probabilities are not redistributed.
    The supplied CPU generator is independent of model and masking randomness.
    Axis probabilities follow stored XYZ: L–R, A–P, S–I in these RAS volumes.
    """
    options = [[1.0, 1.0, 1.0]] + [[factor] * 3 for factor in isotropic_factors]
    probabilities = [identity_probability] + [
        isotropic_probability / len(isotropic_factors)
    ] * len(isotropic_factors)
    anisotropic_probability = 1 - identity_probability - isotropic_probability
    for axis, probability in enumerate(anisotropic_axis_probabilities):
        for factor in anisotropic_factors:
            option = [1.0, 1.0, 1.0]
            option[axis] = factor
            options.append(option)
            probabilities.append(anisotropic_probability * probability / len(anisotropic_factors))
    choices = torch.multinomial(
        torch.tensor(probabilities, device="cpu"), batch_size, replacement=True, generator=generator
    )
    plan = torch.tensor(options, dtype=torch.float64, device="cpu")[choices]
    if max_factors is not None:
        assert max_factors.shape == plan.shape
        plan[~(plan <= max_factors).all(dim=1)] = 1
    return plan


@lru_cache(maxsize=128)
def resolution_axis_parameters(size: int, coarse_size: int, device: torch.device) -> tuple:
    factor = size / coarse_size
    sigma = math.sqrt(factor**2 - 1) / math.sqrt(8 * math.log(2))
    radius = int(4 * sigma + 0.5)
    positions = torch.arange(-radius, radius + 1, dtype=torch.float64, device="cpu")
    kernel = torch.exp(-0.5 * (positions / sigma).square())
    kernel = kernel / kernel.sum()
    stride = size // coarse_size if size % coarse_size == 0 else 1
    padding = radius
    if stride > 1:
        padding -= stride // 2
        if stride % 2 == 0:
            # Even strides sample halfway between Gaussian-filtered voxels.
            kernel = 0.5 * (F.pad(kernel, (1, 0)) + F.pad(kernel, (0, 1)))
            padding += 1
    kernel = kernel.to(device=device, dtype=torch.float32)
    resampling = []
    for source_size, target_size in ((size, coarse_size), (coarse_size, size)):
        centers = torch.arange(target_size, dtype=torch.float64, device="cpu")
        coordinates = (centers + 0.5) * (source_size / target_size) - 0.5
        lower = coordinates.floor().long()
        upper = lower + 1
        fraction = coordinates - lower
        lower_weight = (1 - fraction) * ((lower >= 0) & (lower < source_size))
        upper_weight = fraction * ((upper >= 0) & (upper < source_size))
        resampling.append(
            (
                lower.clamp(0, source_size - 1).to(device),
                upper.clamp(0, source_size - 1).to(device),
                lower_weight.to(device=device, dtype=torch.float32),
                upper_weight.to(device=device, dtype=torch.float32),
            )
        )
    return kernel, resampling, stride, padding


def resample_resolution_axis(fields: Tensor, axis: int, parameters: tuple) -> Tensor:
    lower, upper, lower_weight, upper_weight = parameters
    weight_shape = [1] * fields.ndim
    weight_shape[axis] = lower.numel()
    result = fields.index_select(axis, lower)
    result.mul_(lower_weight.reshape(weight_shape))
    result.addcmul_(fields.index_select(axis, upper), upper_weight.reshape(weight_shape))
    return result


@torch.no_grad()
def apply_resolution_augmentation(
    reference: Tensor,
    brain_mask: Tensor,
    factors: Tensor,
    *,
    output_dtype: torch.dtype,
    chunk_size: int = 4,
) -> Tensor:
    """Degrade [B, 1, X, Y, Z] inputs while preserving reference and brain support.

    Matches preview_v1: normalized masked Gaussian filtering, centered linear
    down/up sampling, and zero background. Only transform chunks use float32.
    """
    batch_size, channels, *shape = reference.shape
    assert channels == 1 and len(shape) == 3
    assert brain_mask.shape == reference.shape and factors.shape == (batch_size, 3)
    groups = {}
    for index, requested in enumerate(factors.tolist()):
        coarse_shape = tuple(round(size / factor) for size, factor in zip(shape, requested))
        if coarse_shape != tuple(shape):
            groups.setdefault(coarse_shape, []).append(index)
    output = reference.to(dtype=output_dtype, copy=True)
    with torch.autocast(device_type=reference.device.type, enabled=False):
        for coarse_shape, indices in groups.items():
            axes = [axis for axis in range(3) if coarse_shape[axis] != shape[axis]]
            parameters = {
                axis: resolution_axis_parameters(shape[axis], coarse_shape[axis], reference.device)
                for axis in axes
            }
            for start in range(0, len(indices), chunk_size):
                selection = torch.tensor(
                    indices[start : start + chunk_size], device=reference.device
                )
                mask_chunk = brain_mask.index_select(0, selection)
                weights = mask_chunk.float()
                image_chunk = reference.index_select(0, selection).float()
                image_chunk.mul_(weights)
                fields = torch.cat((image_chunk, weights), dim=0)
                del image_chunk, weights
                for axis in axes:
                    kernel, resampling, axis_stride, axis_padding = parameters[axis]
                    kernel_shape = [1, 1, 1, 1, 1]
                    kernel_shape[axis + 2] = kernel.numel()
                    padding = [0, 0, 0]
                    padding[axis] = axis_padding
                    stride = [1, 1, 1]
                    stride[axis] = axis_stride
                    fields = F.conv3d(
                        fields, kernel.reshape(kernel_shape), padding=padding, stride=stride
                    )
                    if axis_stride == 1:
                        fields = resample_resolution_axis(fields, axis + 2, resampling[0])
                for axis in axes:
                    fields = resample_resolution_axis(fields, axis + 2, parameters[axis][1][1])
                numerator, denominator = fields.chunk(2, dim=0)
                denominator.masked_fill_(~mask_chunk, 1)
                numerator.div_(denominator).masked_fill_(~mask_chunk, 0)
                output.index_copy_(0, selection, numerator.to(dtype=output_dtype))
                del fields, numerator, denominator, mask_chunk
    return output
