"""
Sistema de callbacks estilo Keras.

Permiten engancharse al ciclo de entrenamiento sin modificar el código
de la red. Útil para: early stopping, checkpoints, ajuste de lr, logs
personalizados, integraciones externas (TensorBoard, MLflow, etc).

Hooks disponibles:
- on_train_begin / on_train_end
- on_epoch_begin / on_epoch_end
- on_batch_begin / on_batch_end
"""
from typing import Optional, Dict, Any
import numpy as np


class Callback:
    """Hereda de aquí y sobreescribe los hooks que necesites."""

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
    Registra todas las métricas de cada epoch en un dict de listas.
    Se añade automáticamente al fit() y se devuelve al finalizar.
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
    Detiene el entrenamiento si una métrica deja de mejorar.

    Args:
        monitor: nombre de la métrica a observar (ej. 'val_loss', 'loss').
        patience: epochs a esperar sin mejora antes de detener.
        min_delta: cambio mínimo para considerar mejora.
        mode: 'min' o 'max'. Auto-detectado si contiene 'loss'.
        restore_best_weights: si True, restaura los pesos del mejor epoch.
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
            mode = "max" if "acc" in monitor or "r2" in monitor else "min"
        if mode not in ("min", "max"):
            raise ValueError("mode debe ser 'min', 'max' o 'auto'.")
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
            print(f"EarlyStopping: detenido en epoch {self.stopped_epoch}")


class ModelCheckpoint(Callback):
    """
    Guarda los pesos cuando una métrica mejora.

    Args:
        filepath: ruta destino (.npz).
        monitor: métrica a observar.
        save_best_only: si True, sólo guarda en mejoras.
        mode: 'min' o 'max'. Auto-detectado si contiene 'loss'.
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
            mode = "max" if "acc" in monitor or "r2" in monitor else "min"
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
    Reduce el learning rate cuando una métrica se estanca.

    Args:
        monitor: métrica a observar.
        factor: multiplicador del lr (ej. 0.5 lo reduce a la mitad).
        patience: epochs sin mejora antes de reducir.
        min_lr: límite inferior.
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
            raise ValueError("factor debe estar en (0, 1).")
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        if mode == "auto":
            mode = "max" if "acc" in monitor or "r2" in monitor else "min"
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
                    print(f"ReduceLROnPlateau: lr {old_lr:.2e} -> {new_lr:.2e}")
                self.wait = 0