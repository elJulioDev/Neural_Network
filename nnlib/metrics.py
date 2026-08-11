"""Metrics with serialization support."""
from typing import Any, Dict, Union

import numpy as np


class Metric:
    """Base class for metrics."""

    name: str = "metric"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the metric value."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Returns a JSON-serializable config dict."""
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Metric":
        """Reconstructs a metric from a config dict."""
        return cls(**config)


class BinaryAccuracy(Metric):
    """Accuracy for binary predictions (threshold-based)."""

    name = "binary_accuracy"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns fraction of correct binary predictions."""
        preds = (y_pred >= self.threshold).astype(int)
        return float(np.mean(preds == y_true.astype(int)))

    def get_config(self) -> Dict[str, Any]:
        """Returns BinaryAccuracy config."""
        return {"class_name": "BinaryAccuracy", "config": {"threshold": self.threshold}}


class CategoricalAccuracy(Metric):
    """Accuracy for one-hot encoded multiclass predictions."""

    name = "categorical_accuracy"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns fraction of correct multiclass predictions."""
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))


class SparseCategoricalAccuracy(Metric):
    """Accuracy for integer-label multiclass predictions."""

    name = "sparse_categorical_accuracy"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns fraction of correct sparse predictions."""
        return float(np.mean(np.argmax(y_pred, axis=1) == y_true.flatten().astype(int)))


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric."""

    name = "mae"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns MAE."""
        return float(np.mean(np.abs(y_pred - y_true)))


class RootMeanSquaredError(Metric):
    """Root Mean Squared Error metric."""

    name = "rmse"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns RMSE."""
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


class R2Score(Metric):
    """R-squared (coefficient of determination) metric."""

    name = "r2"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns R2 score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))


_METRICS = {
    "binary_accuracy": BinaryAccuracy,
    "accuracy": BinaryAccuracy,
    "categorical_accuracy": CategoricalAccuracy,
    "sparse_categorical_accuracy": SparseCategoricalAccuracy,
    "mae": MeanAbsoluteError,
    "rmse": RootMeanSquaredError,
    "r2": R2Score,
}

_METRIC_CLASSES = {
    "BinaryAccuracy": BinaryAccuracy,
    "CategoricalAccuracy": CategoricalAccuracy,
    "SparseCategoricalAccuracy": SparseCategoricalAccuracy,
    "MeanAbsoluteError": MeanAbsoluteError,
    "RootMeanSquaredError": RootMeanSquaredError,
    "R2Score": R2Score,
}


def get_metric(metric: Union[str, Dict[str, Any], Metric]) -> Metric:
    """Resolves a metric from string, dict, or instance."""
    if isinstance(metric, Metric):
        return metric
    if isinstance(metric, str):
        key = metric.lower()
        if key not in _METRICS:
            raise ValueError(f"Unknown metric: {metric}. Options: {list(_METRICS.keys())}")
        return _METRICS[key]()
    if isinstance(metric, dict):
        cls = _METRIC_CLASSES[metric["class_name"]]
        return cls.from_config(metric.get("config", {}))
    raise TypeError(f"Unsupported type: {type(metric)}")
