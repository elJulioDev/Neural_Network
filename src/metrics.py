"""Métricas con soporte de serialización."""
from typing import Dict, Any
import numpy as np


class Metric:
    name: str = "metric"

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        return {"class_name": type(self).__name__, "config": {}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Metric":
        return cls(**config)


class BinaryAccuracy(Metric):
    name = "binary_accuracy"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        preds = (y_pred >= self.threshold).astype(int)
        return float(np.mean(preds == y_true.astype(int)))

    def get_config(self):
        return {"class_name": "BinaryAccuracy", "config": {"threshold": self.threshold}}


class CategoricalAccuracy(Metric):
    name = "categorical_accuracy"

    def __call__(self, y_pred, y_true):
        return float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)))


class SparseCategoricalAccuracy(Metric):
    name = "sparse_categorical_accuracy"

    def __call__(self, y_pred, y_true):
        return float(np.mean(np.argmax(y_pred, axis=1) == y_true.flatten().astype(int)))


class MeanAbsoluteError(Metric):
    name = "mae"

    def __call__(self, y_pred, y_true):
        return float(np.mean(np.abs(y_pred - y_true)))


class RootMeanSquaredError(Metric):
    name = "rmse"

    def __call__(self, y_pred, y_true):
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


class R2Score(Metric):
    name = "r2"

    def __call__(self, y_pred, y_true):
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


def get_metric(metric) -> Metric:
    if isinstance(metric, Metric):
        return metric
    if isinstance(metric, str):
        key = metric.lower()
        if key not in _METRICS:
            raise ValueError(f"Métrica desconocida: {metric}. Opciones: {list(_METRICS.keys())}")
        return _METRICS[key]()
    if isinstance(metric, dict):
        cls = _METRIC_CLASSES[metric["class_name"]]
        return cls.from_config(metric.get("config", {}))
    raise TypeError(f"Tipo no soportado: {type(metric)}")