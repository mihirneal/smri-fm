import os
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import auroc, bacc, r2
from evaluation.tasks.registry import register_task

PPMI_EVAL_REPO_ID = "medarc/ppmi-mini"
IMAGE_COLUMN = "nifti"


def load_ppmi_eval() -> Dataset:
    return load_dataset(PPMI_EVAL_REPO_ID)["test"]


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


@register_task
def ppmi_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="ppmi_age",
        kind="regression",
        data=load_ppmi_eval(),
        image_column=IMAGE_COLUMN,
        target_column="age",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2,),
    )


@register_task
def ppmi_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    data = load_ppmi_eval()
    return ColumnTask(
        name="ppmi_sex",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="sex",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=data.features["sex"].names.index("Male"),
    )


@register_task
def ppmi_pd_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary PD-vs-control diagnosis classification."""
    data = _filter_diagnoses(load_ppmi_eval(), {"CN", "PD"})
    return ColumnTask(
        name="ppmi_pd_cn",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc),
        positive_label=data.features["diagnosis"].names.index("PD"),
    )


@register_task
def ppmi_diagnosis(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Multiclass PPMI cohort diagnosis classification."""
    return ColumnTask(
        name="ppmi_diagnosis",
        kind="classification",
        data=load_ppmi_eval(),
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc,),
    )
