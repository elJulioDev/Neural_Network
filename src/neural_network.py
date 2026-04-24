"""
Red neuronal principal (modelo Sequential).

Características:
- API tipo Keras: add(), compile(), fit(), evaluate(), predict().
- Soporte completo de callbacks (EarlyStopping, ModelCheckpoint, etc).
- Métricas configurables y registro de history por epoch.
- validation_split o validation_data para monitorear generalización.
- Gradient clipping en optimizers.
- save_weights / load_weights en formato .npz (portable, no usa pickle).
- save / load del modelo completo (pickle, conserva arquitectura).
- Logger estructurado con verbose 0/1/2.
"""
from typing import List, Optional, Tuple, Union, Dict, Any
import time
import json
import pickle
import logging
import numpy as np

from .layer import BaseLayer, Layer, Dense, Dropout, BatchNormalization
from .activations import Sigmoid, get_activation
from .losses import Loss, MSE, get_loss
from .optimizers import Optimizer, SGD, get_optimizer
from .metrics import Metric, get_metric
from .callbacks import Callback, History
from .utils import shuffle_arrays, batch_iterator, train_test_split

logger = logging.getLogger("neural_network")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class NeuralNetwork:
    """
    Modelo secuencial de redes neuronales.

    Uso típico:
        model = NeuralNetwork()
        model.add(Dense(64, input_size=10, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='bce', metrics=['accuracy'])
        history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
    """

    def __init__(
        self,
        loss_function: Optional[Loss] = None,
        optimizer: Optional[Optimizer] = None,
    ):
        self.layers: List[BaseLayer] = []
        self.loss_function: Optional[Loss] = (
            get_loss(loss_function) if loss_function is not None else None
        )
        self.optimizer: Optional[Optimizer] = (
            get_optimizer(optimizer) if optimizer is not None else None
        )
        self.metrics: List[Metric] = []
        self.stop_training: bool = False
        self._compiled: bool = False

    # ------------------------------------------------------------------
    # Construcción del modelo
    # ------------------------------------------------------------------

    def add(self, layer: BaseLayer) -> "NeuralNetwork":
        """Añade una capa ya construida (estilo Keras). Si la capa Dense
        no recibió input_size, se infiere desde la capa anterior."""
        if not isinstance(layer, BaseLayer):
            raise TypeError(f"Se esperaba BaseLayer, recibido {type(layer)}")

        # Inferencia automática de input_size para Dense sin construir
        if isinstance(layer, Layer) and layer.weights is None:
            if not self.layers:
                raise ValueError(
                    "La primera capa Dense necesita input_size definido."
                )
            layer.build(self._last_output_size())

        self.layers.append(layer)
        return self

    def add_layer(
        self,
        num_neurons: int,
        input_size: Optional[int] = None,
        activation=None,
        **kwargs,
    ) -> "NeuralNetwork":
        """API legacy. Crea y añade una capa Dense."""
        if activation is None:
            activation = Sigmoid()
        if not self.layers:
            if input_size is None:
                raise ValueError("input_size requerido para la primera capa.")
            self.layers.append(Layer(num_neurons, input_size, activation, **kwargs))
        else:
            last_size = self._last_output_size()
            self.layers.append(Layer(num_neurons, last_size, activation, **kwargs))
        return self

    def add_dropout(self, rate: float) -> "NeuralNetwork":
        self.layers.append(Dropout(rate))
        return self

    def add_batch_norm(self) -> "NeuralNetwork":
        size = self._last_output_size()
        self.layers.append(BatchNormalization(size))
        return self

    def _last_output_size(self) -> int:
        for layer in reversed(self.layers):
            if hasattr(layer, "n_neurons") and layer.n_neurons > 0:
                return layer.n_neurons
        raise ValueError("No hay capa anterior con tamaño definido.")

    # ------------------------------------------------------------------
    # Compilación
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer: Union[str, Optimizer] = "adam",
        loss: Union[str, Loss] = "mse",
        metrics: Optional[List[Union[str, Metric]]] = None,
    ) -> None:
        """Configura optimizer, loss y métricas. Llamar antes de fit()."""
        self.optimizer = get_optimizer(optimizer)
        self.loss_function = get_loss(loss)
        self.metrics = [get_metric(m) for m in (metrics or [])]
        self._compiled = True

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs, training=training)
        return inputs

    def backward(self, loss_gradient: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)

    def _regularization_loss(self) -> float:
        return sum(layer.regularization_loss() for layer in self.layers)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.0,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1,
        shuffle: bool = True,
    ) -> Dict[str, list]:
        """
        Entrena el modelo.

        verbose: 0 = silencioso, 1 = una línea por epoch, 2 = una línea
        por batch (útil para depuración).
        """
        self._ensure_ready()
        self._validate_input(x, y)

        # Validation split
        x_val, y_val = None, None
        if validation_data is not None:
            x_val, y_val = validation_data
        elif validation_split > 0.0:
            x, x_val, y, y_val = train_test_split(
                x, y, test_size=validation_split, shuffle=True
            )

        # Setup callbacks
        callbacks = list(callbacks or [])
        history = History()
        callbacks.append(history)
        for cb in callbacks:
            cb.set_model(self)

        self.stop_training = False
        for cb in callbacks:
            cb.on_train_begin()

        steps = max(1, (len(x) + batch_size - 1) // batch_size)

        for epoch in range(epochs):
            if self.stop_training:
                break

            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            t0 = time.time()
            x_shuf, y_shuf = (
                shuffle_arrays(x, y) if shuffle else (x, y)
            )

            epoch_loss = 0.0
            metric_sums = {m.name: 0.0 for m in self.metrics}

            for batch_idx, (x_batch, y_batch) in enumerate(
                batch_iterator(x_shuf, y_shuf, batch_size)
            ):
                for cb in callbacks:
                    cb.on_batch_begin(batch_idx)

                output = self.forward(x_batch, training=True)
                loss = self.loss_function.calculate(output, y_batch)
                loss += self._regularization_loss() / steps  # promediado por step
                epoch_loss += loss

                for m in self.metrics:
                    metric_sums[m.name] += m(output, y_batch)

                grad = self.loss_function.derivative(output, y_batch)
                self.backward(grad)
                self.optimizer.update(self.layers)

                for cb in callbacks:
                    cb.on_batch_end(batch_idx, {"loss": loss})

                if verbose >= 2:
                    print(f"  batch {batch_idx + 1}/{steps} - loss: {loss:.6f}")

            logs: Dict[str, Any] = {"loss": epoch_loss / steps}
            for name, total in metric_sums.items():
                logs[name] = total / steps

            # Validación
            if x_val is not None:
                val_logs = self.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
                for k, v in val_logs.items():
                    logs[f"val_{k}"] = v

            elapsed = time.time() - t0
            if verbose >= 1:
                msg = f"Epoch {epoch + 1}/{epochs} - {elapsed:.2f}s - " + " - ".join(
                    f"{k}: {v:.6f}" for k, v in logs.items()
                )
                print(msg)

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

        for cb in callbacks:
            cb.on_train_end()

        return history.history

    # API legacy compatible con la versión anterior
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, list]:
        if not self._compiled and self.loss_function is not None and self.optimizer is not None:
            self._compiled = True
        return self.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # ------------------------------------------------------------------
    # Evaluación e inferencia
    # ------------------------------------------------------------------

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, float]:
        """Calcula loss y métricas sobre un conjunto de datos."""
        self._ensure_ready()
        steps = max(1, (len(x) + batch_size - 1) // batch_size)
        total_loss = 0.0
        metric_sums = {m.name: 0.0 for m in self.metrics}

        for x_batch, y_batch in batch_iterator(x, y, batch_size):
            output = self.forward(x_batch, training=False)
            total_loss += self.loss_function.calculate(output, y_batch)
            for m in self.metrics:
                metric_sums[m.name] += m(output, y_batch)

        result = {"loss": total_loss / steps}
        for name, total in metric_sums.items():
            result[name] = total / steps

        if verbose >= 1:
            print(" - ".join(f"{k}: {v:.6f}" for k, v in result.items()))
        return result

    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Inferencia. Acepta entradas en mini-batches para no saturar memoria."""
        if batch_size is None or x.shape[0] <= batch_size:
            return self.forward(x, training=False)

        outputs = []
        for start in range(0, x.shape[0], batch_size):
            outputs.append(self.forward(x[start:start + batch_size], training=False))
        return np.concatenate(outputs, axis=0)

    def predict_classes(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Para clasificación: devuelve etiquetas, no probabilidades."""
        probs = self.predict(x)
        if probs.shape[1] == 1:
            return (probs >= threshold).astype(int)
        return np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # Persistencia: pesos (.npz portable) y modelo completo (.pkl)
    # ------------------------------------------------------------------

    def get_weights(self) -> List[np.ndarray]:
        """Devuelve todos los parámetros entrenables como lista de arrays."""
        weights = []
        for layer in self.layers:
            weights.extend([w.copy() for w in layer.get_params()])
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Asigna los pesos en el mismo orden que get_weights()."""
        idx = 0
        for layer in self.layers:
            params = layer.get_params()
            n = len(params)
            if n == 0:
                continue
            layer.set_params(weights[idx:idx + n])
            idx += n

    def save_weights(self, filepath: str) -> None:
        """Guarda los pesos en formato .npz (portable, sin pickle)."""
        weights = self.get_weights()
        named = {f"w_{i}": w for i, w in enumerate(weights)}
        np.savez(filepath, **named)

    def load_weights(self, filepath: str) -> None:
        """Carga pesos desde .npz. La arquitectura debe coincidir."""
        data = np.load(filepath)
        keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
        weights = [data[k] for k in keys]
        self.set_weights(weights)

    def save_model(self, filepath: str) -> None:
        """Guarda el modelo completo (arquitectura + pesos)."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath: str) -> "NeuralNetwork":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Imprime un resumen tipo Keras: capa, output shape, parámetros."""
        print("=" * 70)
        print(f"{'Layer':<25}{'Output Shape':<20}{'Params':>15}")
        print("=" * 70)
        total = 0
        for i, layer in enumerate(self.layers):
            name = f"{type(layer).__name__}_{i}"
            params = sum(p.size for p in layer.get_params())
            total += params
            out_shape = (
                f"(None, {layer.n_neurons})" if layer.n_neurons > 0 else "(same)"
            )
            print(f"{name:<25}{out_shape:<20}{params:>15,}")
        print("=" * 70)
        print(f"Total params: {total:,}")
        print("=" * 70)

    def _ensure_ready(self) -> None:
        if self.loss_function is None:
            raise RuntimeError("Loss no configurada. Llama compile() primero.")
        if self.optimizer is None:
            raise RuntimeError("Optimizer no configurado. Llama compile() primero.")
        if not self.layers:
            raise RuntimeError("Modelo sin capas. Añade capas con add().")

    def _validate_input(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"X debe ser 2D (batch, features). Recibido: {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Número de muestras de X e y no coincide.")