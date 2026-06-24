from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TargetSpec:
    kind: Literal["regression", "classification"]
    dim: int
    loss: str


@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset


class EvaluationTask(Protocol):
    """Protocol that tasks should support"""
    name: str

    def prepare(self, overwrite_data: bool = False) -> None: ...

    def target_spec(self) -> TargetSpec: ...

    def datasets(self) -> DatasetBundle: ...

    def collate_fn(self) -> Callable | None: ...

    def metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, float]: ...


def validate_batch(batch: dict) -> None:
    missing = [key for key in ("image", "target") if key not in batch]
    if missing:
        raise ValueError(
            "evaluation batches must contain required keys: "
            f"{', '.join(missing)} missing"
        )
