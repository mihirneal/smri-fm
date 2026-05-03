from io import BytesIO
import os

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor


def fig2pil(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def plot_mask_pred(
    target: Tensor,
    pred: Tensor,
    visible_mask: Tensor | None = None,
    pred_mask: Tensor | None = None,
    img_mask: Tensor | None = None,
    sample_idx: int = 0,
    channel_idx: int = 0,
    slice_idx: int | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (12, 3),
):
    panels = {
        "target": target,
        "pred": pred,
        "visible": visible_mask,
        "predict": pred_mask,
        "brain": img_mask,
    }
    panels = {name: value for name, value in panels.items() if value is not None}

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    axes = axes[0]
    for ax, (name, value) in zip(axes, panels.items()):
        image = _central_slice(value, sample_idx=sample_idx, channel_idx=channel_idx, slice_idx=slice_idx)
        ax.imshow(image, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    return fig


def _central_slice(
    x: Tensor,
    sample_idx: int = 0,
    channel_idx: int = 0,
    slice_idx: int | None = None,
) -> Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 5:
        depth = x.shape[2]
        slice_idx = depth // 2 if slice_idx is None else slice_idx
        return x[sample_idx, channel_idx, slice_idx]
    if x.ndim == 4:
        return x[sample_idx, channel_idx]
    if x.ndim == 3:
        depth = x.shape[0]
        slice_idx = depth // 2 if slice_idx is None else slice_idx
        return x[slice_idx]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected a 2D image or 3D volume tensor, got shape {tuple(x.shape)}")
