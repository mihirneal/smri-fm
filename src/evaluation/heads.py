from collections.abc import Callable, Mapping
from typing import Any

import torch.nn as nn
from torch import Tensor

from evaluation.core import TargetSpec


def pool_tokens(tokens: Tensor, pooling: str) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"expected token sequence shaped [B, T, D], got {tokens.shape}")
    if pooling == "first":
        return tokens[:, 0]
    if pooling == "mean":
        return tokens.mean(dim=1)
    raise ValueError(f"unknown pooling: {pooling}")


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, pooling: str):
        super().__init__()
        if pooling not in {"first", "mean"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.linear(pool_tokens(tokens, self.pooling))


def _build_linear_head(
    cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int
) -> nn.Module:
    return LinearHead(input_dim=input_dim, output_dim=target_spec.dim, pooling=cfg["pooling"])


_HEAD_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "linear": _build_linear_head,
}


def list_heads() -> list[str]:
    return sorted(_HEAD_BUILDERS)


def build_head(cfg: Mapping[str, Any], *, target_spec: TargetSpec, input_dim: int):
    name = cfg.get("name")
    try:
        builder = _HEAD_BUILDERS[name]
    except KeyError:
        available = ", ".join(list_heads())
        raise ValueError(f"unknown head {name!r}. available heads: {available}") from None
    return builder(cfg, target_spec=target_spec, input_dim=input_dim)
