import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from evaluation.core import DatasetBundle, TargetSpec
from evaluation.metrics import regression_metrics


DEFAULT_DATA_ROOT = Path("data/asparagus/data/REGR002_FOMO26_BrainAge")
TASK_NAME = "REGR002_FOMO26_BrainAge"


def _load_json(path: Path):
    return json.loads(path.read_text())


@dataclass(frozen=True)
class FomoBrainAgeGapDataset(Dataset):
    paths: list[Path]
    data_root: Path

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, list | tuple) or len(data) < 2:
            raise ValueError(f"expected asparagus cls/reg tensor pair in {path}")
        return {
            "image": data[0],
            "target": data[1],
            "id": "_".join(path.with_suffix("").parts[-3:]),
            "meta": {"path": str(path)},
        }


@dataclass
class FomoBrainAgeGapTask:
    data_root: str | Path = DEFAULT_DATA_ROOT
    split: str = "split_80_10_10"
    test_split: str = "TEST_80_10_10"
    fold: int = 0
    name: str = "fomo_brain_age_gap"

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)

    def prepare(self, overwrite_data: bool = False) -> None:
        del overwrite_data
        required = [
            self.data_root,
            self.data_root / "dataset.json",
            self.data_root / f"{self.split}.json",
            self.data_root / f"{self.test_split}.json",
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            missing_list = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"{TASK_NAME} asparagus data is incomplete under {self.data_root}: "
                f"missing {missing_list}"
            )

    def target_spec(self) -> TargetSpec:
        return TargetSpec(kind="regression", dim=1, loss="mse")

    def datasets(self) -> DatasetBundle:
        self.prepare()
        split_data = _load_json(self.data_root / f"{self.split}.json")
        try:
            fold_split = split_data[int(self.fold)]
        except IndexError:
            raise ValueError(
                f"fold {self.fold} is unavailable in {self.data_root / f'{self.split}.json'}"
            ) from None
        test_paths = _load_json(self.data_root / f"{self.test_split}.json")
        return DatasetBundle(
            train=FomoBrainAgeGapDataset(
                self._resolve_paths(fold_split["train"]),
                self.data_root,
            ),
            val=FomoBrainAgeGapDataset(
                self._resolve_paths(fold_split["val"]),
                self.data_root,
            ),
            test=FomoBrainAgeGapDataset(
                self._resolve_paths(test_paths),
                self.data_root,
            ),
        )

    def collate_fn(self):
        return None

    def metrics(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        return regression_metrics(predictions, targets)

    def _resolve_paths(self, paths: list[str]) -> list[Path]:
        return [self._resolve_path(Path(path)) for path in paths]

    def _resolve_path(self, path: Path) -> Path:
        if path.exists():
            return path
        parts = path.parts
        if TASK_NAME in parts:
            relative = Path(*parts[parts.index(TASK_NAME) + 1 :])
            candidate = self.data_root / relative
            if candidate.exists():
                return candidate
        return path
