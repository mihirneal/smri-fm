import numpy as np
from scipy import stats
from sklearn import metrics as skm
from sklearn.pipeline import Pipeline


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def bacc(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return balanced_accuracy(y_true, y_pred)


def auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    positive_label=None,
    **_,
) -> float:
    return float(skm.roc_auc_score(np.asarray(y_true) == positive_label, y_score))


def auprc(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    y_score: np.ndarray | None = None,
    positive_label=None,
    **_,
) -> float:
    return float(skm.average_precision_score(np.asarray(y_true) == positive_label, y_score))


def classification_score(estimator: Pipeline, X: np.ndarray, positive_label) -> np.ndarray | None:
    proba = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    return proba[:, np.flatnonzero(classes == positive_label)[0]]


def r2(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    return float(skm.r2_score(y_true, y_pred, multioutput="uniform_average"))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray, **_) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1)).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"r2": r2(y_true, y_pred)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
    }
