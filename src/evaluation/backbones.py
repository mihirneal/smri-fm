from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from smri_mae.model_mae import MaskedViT


class SmriMaeBackbone(nn.Module):
    def __init__(
        self,
        *,
        use_input_mask: bool = False,
        calculate_mask: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        if calculate_mask not in {None, "mean"}:
            raise ValueError(
                f"calculate_mask must be one of None or 'mean', got {calculate_mask!r}"
            )
        if use_input_mask and calculate_mask is not None:
            raise ValueError("use_input_mask and calculate_mask cannot both be enabled")
        self.use_input_mask = bool(use_input_mask)
        self.calculate_mask = calculate_mask
        self.model = MaskedViT(**kwargs)
        self.embed_dim = self.model.patch_embed.out_features

    def _resolve_mask(self, images: Tensor, mask: Tensor | None) -> Tensor | None:
        """Return the mask to pass into the MAE embedding forward pass.

        The mask can either be supplied by the caller, derived from the image
        intensities, or omitted entirely depending on the backbone configuration.
        """
        # Use the caller-provided mask when the evaluation config requires it.
        if self.use_input_mask:
            if mask is None:
                raise ValueError("use_input_mask=True requires a mask input")
            return mask
        # Calculate a simple intensity mask only when mask inference is enabled.
        # This is suboptimal but fine for now.
        if self.calculate_mask == "mean":
            dims = tuple(range(1, images.ndim))
            return images > images.mean(dim=dims, keepdim=True)
        return None

    def forward(self, images: Tensor, mask: Tensor | None = None) -> dict[str, Tensor | None]:
        """Encode images and return MAE class, register, and patch embeddings.

        The optional input mask is resolved according to the backbone settings
        before delegating to the underlying ``MaskedViT`` embedding path.
        """
        # Check if a mask is necessary before extracting MAE embeddings.
        mask = self._resolve_mask(images, mask)
        cls, reg, patch = self.model.forward_embedding(images, mask=mask)
        return {"cls": cls, "reg": reg, "patch": patch}


def load_smri_mae_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    
    # Use only encoder weights
    prefix = "encoder."
    state_dict = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")


def _build_smri_mae_backbone(cfg: Mapping[str, Any]) -> nn.Module:
    kwargs = {
        "img_size": cfg["img_size"],
        "patch_size": cfg["patch_size"],
        "in_chans": cfg.get("in_chans", 1),
        "use_input_mask": bool(cfg.get("use_input_mask", False)),
        "calculate_mask": cfg.get("calculate_mask"),
        **dict(cfg.get("model_kwargs") or {}),
    }
    backbone = SmriMaeBackbone(**kwargs)
    if cfg.get("checkpoint_path"):
        load_smri_mae_checkpoint(backbone.model, cfg["checkpoint_path"])
    return backbone


_BACKBONE_BUILDERS: dict[str, Callable[[Mapping[str, Any]], nn.Module]] = {
    "smri_mae": _build_smri_mae_backbone,
}


def list_backbones() -> list[str]:
    return sorted(_BACKBONE_BUILDERS)


def build_backbone(cfg: Mapping[str, Any]):
    name = cfg.get("name")
    try:
        builder = _BACKBONE_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_backbones())
        raise ValueError(f"unknown backbone {name!r}. available backbones: {available}") from None
    return builder(cfg)
