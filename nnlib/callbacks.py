"""
Keras-style callback system.

Allows hooking into the training cycle without modifying the network
code. Useful for: early stopping, checkpoints, lr adjustment, custom
logs, external integrations (TensorBoard, MLflow, etc).

Available hooks:
- on_train_begin / on_train_end
- on_epoch_begin / on_epoch_end
- on_batch_begin / on_batch_end
"""
import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("neural_network.callbacks")


def _detect_mode(monitor: str) -> str:
    """Auto-detect whether higher or lower is better for a metric."""
    return "max" if "acc" in monitor or "r2" in monitor else "min"


class Callback:
    """Inherit from here and override the hooks you need."""

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None): pass
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None): pass
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None): pass
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None): pass
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None): pass


class History(Callback):
    """
    Records all metrics of each epoch in a dict of lists.
    Automatically added to fit() and returned at the end.
    """

    def __init__(self):
        super().__init__()
        self.history: Dict[str, list] = {}

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


class EarlyStopping(Callback):
    """
    Stops training if a metric stops improving.

    Args:
        monitor: name of the metric to observe (e.g. 'val_loss', 'loss').
        patience: epochs to wait without improvement before stopping.
        min_delta: minimum change to consider improvement.
        mode: 'min' or 'max'. Auto-detected if contains 'loss'.
        restore_best_weights: if True, restores weights of the best epoch.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "auto",
        restore_best_weights: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        if mode == "auto":
            mode = _detect_mode(monitor)
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min', 'max' or 'auto'.")
        self.mode = mode

        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf if self.mode == "min" else -np.inf
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_better(current):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0:
            logger.info("EarlyStopping: stopped at epoch %d", self.stopped_epoch)


class ModelCheckpoint(Callback):
    """
    Saves weights when a metric improves.

    Args:
        filepath: destination path (.npz).
        monitor: metric to observe.
        save_best_only: if True, only saves on improvements.
        mode: 'min' or 'max'. Auto-detected if contains 'loss'.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "auto",
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only

        if mode == "auto":
            mode = _detect_mode(monitor)
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not self.save_best_only:
            self.model.save_weights(self.filepath)
            return
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (
            current < self.best if self.mode == "min" else current > self.best
        )
        if improved:
            self.best = current
            self.model.save_weights(self.filepath)


class ReduceLROnPlateau(Callback):
    """
    Reduces the learning rate when a metric plateaus.

    Args:
        monitor: metric to observe.
        factor: learning rate multiplier (e.g. 0.5 halves it).
        patience: epochs without improvement before reducing.
        min_lr: lower limit.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        mode: str = "auto",
    ):
        super().__init__()
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must be in (0, 1).")
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        if mode == "auto":
            mode = _detect_mode(monitor)
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (
            current < self.best if self.mode == "min" else current > self.best
        )
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.model.optimizer.lr
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    self.model.optimizer.lr = new_lr
                    logger.info("ReduceLROnPlateau: lr %.2e -> %.2e", old_lr, new_lr)
                self.wait = 0
