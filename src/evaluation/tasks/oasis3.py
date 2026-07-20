from datasets import Dataset, load_dataset

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.metrics import (
    auprc,
    auroc,
    bacc,
    pearson_r,
    r2,
    spearman_r,
)
from evaluation.tasks.registry import register_task

OASIS3_EVAL_REPO_ID = "medarc/oasis-3-v1"
IMAGE_COLUMN = "nifti"


def load_oasis3_eval() -> Dataset:
    return load_dataset(
        OASIS3_EVAL_REPO_ID,
        split="eval",
    )


def _filter_diagnoses(data: Dataset, labels: set[str]) -> Dataset:
    names = data.features["diagnosis"].names
    keep = {names.index(label) for label in labels}
    return data.filter(lambda dx: dx in keep, input_columns="diagnosis")


@register_task
def oasis3_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="oasis3_age",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="age",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2, pearson_r),
    )


@register_task
def oasis3_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Sanity classification (expect near-saturated AUROC)."""
    data = load_oasis3_eval()
    return ColumnTask(
        name="oasis3_sex",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="sex",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["sex"].names.index("Male"),
    )


@register_task
def oasis3_ad_cn(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary AD-vs-CN diagnosis classification (MCI dropped)."""
    data = _filter_diagnoses(load_oasis3_eval(), {"CN", "AD"})
    return ColumnTask(
        name="oasis3_ad_cn",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["diagnosis"].names.index("AD"),
    )


@register_task
def oasis3_cn_mci_ad(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """3-way diagnosis classification over CN / MCI / AD (staging)."""
    data = load_oasis3_eval()
    return ColumnTask(
        name="oasis3_cn_mci_ad",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="diagnosis",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=data.features["diagnosis"].names.index("AD"),
    )


@register_task
def oasis3_synthseg_volumes(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="oasis3_synthseg_volumes",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="synthseg_volumes",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(r2,),
    )


@register_task
def oasis3_ad_cn_bag() -> ColumnTask:
    """Brain-age gap association (AD cases vs CN-trained age residual)."""
    dataset = load_oasis3_eval()
    diagnosis_names = dataset.features["diagnosis"].names
    return BrainAgeGapTask(
        name="oasis3_ad_cn_bag",
        data=dataset,
        age_column="age",
        dx_column="diagnosis",
        control_label=diagnosis_names.index("CN"),
        case_label=diagnosis_names.index("AD"),
        image_column=IMAGE_COLUMN,
    )


# ---------------------------------------------------------------------------
# Primary clinical path / conversion
# ---------------------------------------------------------------------------


@register_task
def oasis3_mci_conversion_3y(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Binary MCI→AD within 36 months (known-at-horizon only)."""
    data = _filter_diagnoses(load_oasis3_eval(), {"MCI"})
    return ColumnTask(
        name="oasis3_mci_conversion_3y",
        kind="classification",
        data=data,
        image_column=IMAGE_COLUMN,
        target_column="conversion_3y",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(bacc, auroc, auprc),
        positive_label=1.0,
    )


# ---------------------------------------------------------------------------
# Primary cognition (continuous annualized slopes, 48m)
# ---------------------------------------------------------------------------


@register_task
def oasis3_srt_free_recall_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    """Annualized Selective Reminding Test free-recall slope."""
    return ColumnTask(
        name="oasis3_srt_free_recall_slope_48m",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="srt_free_recall_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def oasis3_logical_memory_delayed_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    """Annualized Logical Memory delayed-recall slope."""
    return ColumnTask(
        name="oasis3_logical_memory_delayed_slope_48m",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="logical_memory_delayed_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def oasis3_digit_symbol_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized Digit Symbol processing-speed slope."""
    return ColumnTask(
        name="oasis3_digit_symbol_slope_48m",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="digit_symbol_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def oasis3_animal_fluency_slope_48m(
    n_splits: int = 5, seed: int = 0
) -> ColumnTask:
    """Annualized animal semantic-fluency slope."""
    return ColumnTask(
        name="oasis3_animal_fluency_slope_48m",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="animal_fluency_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


@register_task
def oasis3_mmse_slope_48m(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Annualized MMSE global-cognition slope."""
    return ColumnTask(
        name="oasis3_mmse_slope_48m",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="mmse_slope_48m",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, spearman_r, r2),
    )


# ---------------------------------------------------------------------------
# Secondary molecular association
# ---------------------------------------------------------------------------


@register_task
def oasis3_amyloid_centiloid(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: PET amyloid burden (centiloids), all diagnoses."""
    return ColumnTask(
        name="oasis3_amyloid_centiloid",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="amyloid_centiloid",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )


@register_task
def oasis3_tau_suvr(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    """Secondary: AV1451 Tauopathy PET burden, all diagnoses."""
    return ColumnTask(
        name="oasis3_tau_suvr",
        kind="regression",
        data=load_oasis3_eval(),
        image_column=IMAGE_COLUMN,
        target_column="tau_suvr",
        n_splits=n_splits,
        seed=seed,
        metric_fns=(pearson_r, r2),
    )
