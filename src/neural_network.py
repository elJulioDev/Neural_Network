"""
NeuralNetwork v0.4 — modelo secuencial con:

1. Gestión externa de caches (ningún layer muta `self.inputs`/`self.z`).
   Esto permite reutilizar una misma capa en múltiples inputs
   (siamesas, triplet loss) sin romper el backprop.
2. Propagación de shapes en `build()`: detecta incompatibilidades
   dimensionales ANTES de entrenar (fail-fast).
3. Serialización portable: `save_topology_json()` + `save_weights_npz()`
   (o `save()` que combina ambos). Sin pickle en el camino crítico.
4. Optimizador genérico: `_collect_grads_and_params()` arma tuplas
   (layer_id, param_name, param, grad) sin asumir 'weights'/'biases'.

API principal (estilo Keras):
    model = NeuralNetwork()
    model.add(Dense(64, input_size=10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.build()                 # valida shapes (opcional, fit() lo hace)
    history = model.fit(X, y, epochs=50, validation_split=0.2)
    model.save('my_model/')       # my_model/topology.json + weights.npz
    loaded = NeuralNetwork.load('my_model/')
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import json
import pickle
import logging
import numpy as np

from .layer import BaseLayer, Layer, Dense, Dropout, BatchNormalization, layer_from_config
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

    VERSION = "0.4.0"

    def __init__(
        self,
        loss_function: Optional[Union[str, Loss]] = None,
        optimizer: Optional[Union[str, Optimizer]] = None,
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
        self._built: bool = False
        self._input_shape: Optional[Tuple[Optional[int], int]] = None

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    def add(self, layer: BaseLayer) -> "NeuralNetwork":
        if not isinstance(layer, BaseLayer):
            raise TypeError(f"Se esperaba BaseLayer, recibido {type(layer)}")
        self.layers.append(layer)
        # Si ya conocemos shape acumulado, vamos construyendo inmediato
        # para fail-fast al añadir capas en secuencia.
        if self.layers and self._accumulated_output_shape() is not None:
            shape = self._accumulated_output_shape()
            if not getattr(layer, "_built", False):
                try:
                    layer.build(shape)
                except ValueError as e:
                    raise ValueError(
                        f"Error al construir capa {len(self.layers)-1} "
                        f"({type(layer).__name__}): {e}"
                    )
        return self

    def _accumulated_output_shape(self) -> Optional[Tuple[Optional[int], int]]:
        """Devuelve el shape de salida del último layer construido."""
        for layer in reversed(self.layers):
            if getattr(layer, "output_shape", None) is not None:
                return layer.output_shape
        return None

    def add_layer(
        self,
        num_neurons: int,
        input_size: Optional[int] = None,
        activation=None,
        **kwargs,
    ) -> "NeuralNetwork":
        """API legacy compatible con v0.2/v0.3."""
        if activation is None:
            activation = "sigmoid"
        if not self.layers and input_size is None:
            raise ValueError("input_size requerido para la primera capa.")
        self.add(Layer(num_neurons, input_size=input_size, activation=activation, **kwargs))
        return self

    def add_dropout(self, rate: float) -> "NeuralNetwork":
        self.add(Dropout(rate))
        return self

    def add_batch_norm(self) -> "NeuralNetwork":
        self.add(BatchNormalization())
        return self

    def build(self, input_shape: Optional[Tuple] = None) -> None:
        """
        Propaga shapes a través de toda la red y valida compatibilidad.

        Fail-fast: si hay un mismatch dimensional, falla AQUÍ antes de
        cargar datos o callbacks.
        """
        if not self.layers:
            raise RuntimeError("No hay capas que construir.")

        if input_shape is None:
            first = self.layers[0]
            if getattr(first, "input_shape", None) is not None:
                input_shape = first.input_shape
            else:
                raise ValueError(
                    "No se puede construir: la primera capa no declara input_size. "
                    "Pasa `input_shape=(None, n_features)` a build()."
                )

        shape = input_shape
        self._input_shape = shape
        for i, layer in enumerate(self.layers):
            try:
                shape = layer.build(shape)
            except Exception as e:
                raise ValueError(
                    f"Error construyendo capa {i} ({type(layer).__name__}): {e}"
                )
        self._built = True

    # ------------------------------------------------------------------
    # Compilación
    # ------------------------------------------------------------------

    def compile(
        self,
        optimizer: Union[str, Optimizer] = "adam",
        loss: Union[str, Loss] = "mse",
        metrics: Optional[List[Union[str, Metric]]] = None,
    ) -> None:
        self.optimizer = get_optimizer(optimizer)
        self.loss_function = get_loss(loss)
        self.metrics = [get_metric(m) for m in (metrics or [])]
        self._compiled = True

        # Si tenemos shape conocido, corremos build() aquí para fail-fast
        if not self._built and self._accumulated_output_shape() is not None:
            try:
                self.build()
            except Exception as e:
                logger.warning(f"build() diferido: {e}")

    # ------------------------------------------------------------------
    # Forward / Backward — con gestión EXTERNA de caches
    # ------------------------------------------------------------------

    def _forward(self, inputs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """Forward pass. Retorna (output, lista_de_caches) — NADA se guarda en self.layers."""
        caches = []
        x = inputs
        for layer in self.layers:
            x, cache = layer.forward(x, training=training)
            caches.append(cache)
        return x, caches

    def _backward(self, loss_gradient: np.ndarray, caches: List[Dict]) -> List[Tuple[BaseLayer, Dict[str, np.ndarray]]]:
        """
        Backward pass. Devuelve lista de (layer, grads_dict) en orden
        de adelante hacia atrás. El gradiente fluye usando el cache
        específico de cada paso forward.
        """
        grads_per_layer = [None] * len(self.layers)
        grad = loss_gradient
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            cache = caches[i]
            grad, layer_grads = layer.backward(grad, cache)
            grads_per_layer[i] = (layer, layer_grads)
        return grads_per_layer

    def _collect_grads_and_params(
        self, grads_per_layer
    ) -> List[Tuple[int, str, np.ndarray, np.ndarray]]:
        """
        Reúne tuplas (layer_id, param_name, param_array, grad_array)
        a partir del dict de parámetros de cada capa y el dict de
        gradientes devueltos por backward. Indepedente de los nombres
        de parámetros concretos.
        """
        collected = []
        for layer, grads in grads_per_layer:
            if not grads:
                continue
            params = layer.parameters()
            for name, param in params.items():
                if name not in grads:
                    raise RuntimeError(
                        f"Capa {type(layer).__name__}: parámetro '{name}' "
                        f"sin gradiente correspondiente."
                    )
                collected.append((id(layer), name, param, grads[name]))
        return collected

    def _regularization_loss(self) -> float:
        return sum(layer.regularization_loss() for layer in self.layers)

    # ------------------------------------------------------------------
    # Inferencia (alias público)
    # ------------------------------------------------------------------

    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """API pública: devuelve sólo el output (descarta caches)."""
        output, _ = self._forward(inputs, training=training)
        return output

    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        if batch_size is None or x.shape[0] <= batch_size:
            return self.forward(x, training=False)
        outputs = []
        for start in range(0, x.shape[0], batch_size):
            outputs.append(self.forward(x[start:start + batch_size], training=False))
        return np.concatenate(outputs, axis=0)

    def predict_classes(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict(x)
        if probs.shape[1] == 1:
            return (probs >= threshold).astype(int)
        return np.argmax(probs, axis=1)

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
        self._ensure_ready()
        self._validate_input(x, y)

        # Fail-fast: asegura shapes consistentes antes del training loop
        if not self._built:
            self.build((None, x.shape[1]))
        elif self._input_shape is not None and x.shape[1] != self._input_shape[-1]:
            raise ValueError(
                f"X tiene {x.shape[1]} features, el modelo espera {self._input_shape[-1]}."
            )

        # Validation setup
        x_val, y_val = None, None
        if validation_data is not None:
            x_val, y_val = validation_data
        elif validation_split > 0.0:
            x, x_val, y, y_val = train_test_split(x, y, test_size=validation_split, shuffle=True)

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
            x_shuf, y_shuf = shuffle_arrays(x, y) if shuffle else (x, y)
            epoch_loss = 0.0
            metric_sums = {m.name: 0.0 for m in self.metrics}

            for batch_idx, (x_batch, y_batch) in enumerate(batch_iterator(x_shuf, y_shuf, batch_size)):
                for cb in callbacks:
                    cb.on_batch_begin(batch_idx)

                # Forward con caches externos
                output, caches = self._forward(x_batch, training=True)
                loss = self.loss_function.calculate(output, y_batch)
                loss += self._regularization_loss() / steps
                epoch_loss += loss

                for m in self.metrics:
                    metric_sums[m.name] += m(output, y_batch)

                # Backward con caches explícitos
                loss_grad = self.loss_function.derivative(output, y_batch)
                grads_per_layer = self._backward(loss_grad, caches)

                # Optimizer: interfaz genérica por (id, name, param, grad)
                grads_and_params = self._collect_grads_and_params(grads_per_layer)
                self.optimizer.apply_gradients(grads_and_params)

                for cb in callbacks:
                    cb.on_batch_end(batch_idx, {"loss": loss})

                if verbose >= 2:
                    print(f"  batch {batch_idx + 1}/{steps} - loss: {loss:.6f}")

            logs: Dict[str, Any] = {"loss": epoch_loss / steps}
            for name, total in metric_sums.items():
                logs[name] = total / steps

            if x_val is not None:
                val_logs = self.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
                for k, v in val_logs.items():
                    logs[f"val_{k}"] = v

            if verbose >= 1:
                elapsed = time.time() - t0
                msg = f"Epoch {epoch + 1}/{epochs} - {elapsed:.2f}s - " + " - ".join(
                    f"{k}: {v:.6f}" for k, v in logs.items()
                )
                print(msg)

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)

        for cb in callbacks:
            cb.on_train_end()

        return history.history

    def train(self, x, y, epochs=1000, batch_size=32, verbose=1):
        """API legacy."""
        if not self._compiled and self.loss_function is not None and self.optimizer is not None:
            self._compiled = True
        return self.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def evaluate(self, x, y, batch_size: int = 32, verbose: int = 1) -> Dict[str, float]:
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

    # ------------------------------------------------------------------
    # Serialización: topología JSON + pesos NPZ
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "class_name": "NeuralNetwork",
            "version": self.VERSION,
            "config": {
                "layers": [layer.get_config() for layer in self.layers],
                "optimizer": self.optimizer.get_config() if self.optimizer else None,
                "loss": self.loss_function.get_config() if self.loss_function else None,
                "metrics": [m.get_config() for m in self.metrics],
                "input_shape": list(self._input_shape) if self._input_shape else None,
            },
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NeuralNetwork":
        version = config.get("version", "unknown")
        cfg = config["config"]
        model = cls()
        for layer_cfg in cfg["layers"]:
            model.add(layer_from_config(layer_cfg))
        if cfg.get("optimizer"):
            model.optimizer = get_optimizer(cfg["optimizer"])
        if cfg.get("loss"):
            model.loss_function = get_loss(cfg["loss"])
        if cfg.get("metrics"):
            model.metrics = [get_metric(m) for m in cfg["metrics"]]
        if cfg.get("optimizer") and cfg.get("loss"):
            model._compiled = True
        if cfg.get("input_shape"):
            model.build(tuple(cfg["input_shape"]))
        return model

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.get_config(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "NeuralNetwork":
        return cls.from_config(json.loads(json_str))

    def _all_state_arrays(self) -> Dict[str, np.ndarray]:
        """Reúne TODOS los arrays (entrenables + no entrenables) con
        claves únicas 'layer_{i}/{param_name}'."""
        state = {}
        for i, layer in enumerate(self.layers):
            for name, arr in layer.parameters().items():
                state[f"layer_{i}/{name}"] = arr
            for name, arr in layer.non_trainable_state().items():
                state[f"layer_{i}/_{name}"] = arr
        return state

    def save_weights(self, filepath: str) -> None:
        """Guarda parámetros + estado no entrenable en NPZ."""
        np.savez(filepath, **self._all_state_arrays())

    def load_weights(self, filepath: str) -> None:
        """Carga pesos. Requiere misma arquitectura."""
        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            # Trainable params
            params = layer.parameters()
            new_params = {}
            for name in params:
                key = f"layer_{i}/{name}"
                if key not in data.files:
                    raise KeyError(f"No se encontró '{key}' en {filepath}.")
                new_params[name] = data[key]
            # Asignar in-place mediante set_parameters si existe, si no
            # modificamos las referencias directamente.
            for name, arr in new_params.items():
                params[name][...] = arr  # in-place para mantener refs

            # Non-trainable state
            for name, arr in layer.non_trainable_state().items():
                key = f"layer_{i}/_{name}"
                if key in data.files:
                    arr[...] = data[key]

    def save(self, directory: str) -> None:
        """
        Guarda en formato portable: `topology.json` + `weights.npz`.
        Sin pickle: sobrevive a refactors y es seguro compartir.
        """
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "topology.json"), "w", encoding="utf-8") as f:
            f.write(self.to_json())
        self.save_weights(os.path.join(directory, "weights.npz"))

    @classmethod
    def load(cls, directory: str) -> "NeuralNetwork":
        with open(os.path.join(directory, "topology.json"), "r", encoding="utf-8") as f:
            model = cls.from_json(f.read())
        model.load_weights(os.path.join(directory, "weights.npz"))
        return model

    # API legacy pickle (mantenida por compatibilidad, desaconsejada)
    def save_model(self, filepath: str) -> None:
        """Serialización legacy via pickle. Prefer `save(directory)`."""
        logger.warning(
            "save_model() usa pickle y es frágil. Usa save('dir/') para "
            "persistencia portable (JSON + NPZ)."
        )
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath: str) -> "NeuralNetwork":
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # Alias para retrocompatibilidad con v0.3
    def get_weights(self) -> List[np.ndarray]:
        """Lista plana de parámetros entrenables (mismo orden que set_weights)."""
        result = []
        for layer in self.layers:
            for _, arr in layer.parameters().items():
                result.append(arr.copy())
            for _, arr in layer.non_trainable_state().items():
                result.append(arr.copy())
        return result

    def set_weights(self, weights: List[np.ndarray]) -> None:
        idx = 0
        for layer in self.layers:
            for _, arr in layer.parameters().items():
                arr[...] = weights[idx]
                idx += 1
            for _, arr in layer.non_trainable_state().items():
                arr[...] = weights[idx]
                idx += 1

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def summary(self) -> None:
        print("=" * 70)
        print(f"{'Layer':<25}{'Output Shape':<22}{'Params':>15}")
        print("=" * 70)
        total = 0
        for i, layer in enumerate(self.layers):
            name = f"{type(layer).__name__}_{i}"
            params_count = sum(p.size for p in layer.parameters().values())
            total += params_count
            out_shape = (
                str(layer.output_shape) if layer.output_shape else "(unbuilt)"
            )
            print(f"{name:<25}{out_shape:<22}{params_count:>15,}")
        print("=" * 70)
        print(f"Total params: {total:,}")
        print("=" * 70)

    def _ensure_ready(self) -> None:
        if self.loss_function is None:
            raise RuntimeError("Loss no configurada. Llama compile() primero.")
        if self.optimizer is None:
            raise RuntimeError("Optimizer no configurado. Llama compile() primero.")
        if not self.layers:
            raise RuntimeError("Modelo sin capas.")

    def _validate_input(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"X debe ser 2D (batch, features). Recibido: {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("X e y tienen distinto número de muestras.")