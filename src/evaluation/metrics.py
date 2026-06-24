import torch
from torch import Tensor


def regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = predictions.detach().float().reshape(targets.shape)
    targets = targets.detach().float()
    residuals = predictions - targets
    mae = residuals.abs().mean()
    mse = residuals.square().mean()
    rmse = mse.sqrt()
    bias = residuals.mean()
    total = ((targets - targets.mean()).square()).sum()
    residual = residuals.square().sum()
    r2 = 1.0 - residual / total if total > 0 else torch.tensor(float("nan"))
    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "bias": float(bias.item()),
        "r2": float(r2.item()),
    }


def is_better(value: float, best: float | None, selection_mode: str) -> bool:
    if selection_mode == "min":
        return best is None or value < best
    if selection_mode == "max":
        return best is None or value > best
    raise ValueError(f"selection_mode must be 'min' or 'max', got {selection_mode!r}")
