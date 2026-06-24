import inspect

from evaluation.core import EvaluationTask
from evaluation.tasks.fomo_brain_age_gap import FomoBrainAgeGapTask

_TASK_REGISTRY: dict[str, type[EvaluationTask]] = {
    "fomo_brain_age_gap": FomoBrainAgeGapTask,
}


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)


def build_task(cfg):
    name = cfg.get("name")
    try:
        task_cls = _TASK_REGISTRY[name]
    except KeyError:
        available = ", ".join(list_tasks())
        raise ValueError(f"unknown task {name!r}. available tasks: {available}") from None
    parameters = inspect.signature(task_cls).parameters
    task_kwargs = {
        key: value
        for key, value in cfg.items()
        if key not in {"name", "overwrite_data"} and key in parameters
    }
    return task_cls(**task_kwargs)
