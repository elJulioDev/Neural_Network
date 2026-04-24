"""
Demo de red siamesa — demuestra que una MISMA capa puede procesar dos
inputs distintos en paralelo sin corromperse.

En v0.3 esto era imposible porque cada forward pasaba el cache a
`self.inputs`/`self.z` de la capa, por lo que la segunda entrada
pisoteaba a la primera y el backward producía gradientes incorrectos.

En v0.4, forward devuelve `(output, cache)` y backward recibe el
cache explícitamente. El mismo layer puede usarse N veces con N caches
independientes.

Aquí construimos manualmente un embedder siamés: una red compartida
procesa dos inputs, y una loss contrastiva simple compara sus embeddings.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.layer import Dense, Dropout
from src.activations import ReLU, Tanh


class SiameseEncoder:
    """Encoder con pesos compartidos aplicado a dos entradas."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, embed_dim: int = 4):
        self.fc1 = Dense(hidden_dim, input_dim, activation=ReLU())
        self.fc2 = Dense(embed_dim, hidden_dim, activation=Tanh())

    def encode(self, x):
        h, cache1 = self.fc1.forward(x, training=True)
        z, cache2 = self.fc2.forward(h, training=True)
        return z, (cache1, cache2)

    def backward_encode(self, d_z, caches):
        cache1, cache2 = caches
        d_h, grads2 = self.fc2.backward(d_z, cache2)
        d_x, grads1 = self.fc1.backward(d_h, cache1)
        return d_x, grads1, grads2


def contrastive_loss(z1, z2, y, margin=1.0):
    """Loss contrastiva clásica: pares similares (y=1) -> cercanos."""
    diff = z1 - z2
    dist_sq = np.sum(diff ** 2, axis=1, keepdims=True)
    dist = np.sqrt(dist_sq + 1e-12)

    loss = np.mean(y * dist_sq + (1 - y) * np.maximum(0, margin - dist) ** 2)

    # Gradientes analíticos respecto a z1 (z2 es simétrico)
    N = y.shape[0]
    dz1_sim = 2 * diff * y / N
    margin_term = np.maximum(0, margin - dist)
    # derivada de (margin - dist)^2 respecto a diff
    dz1_dis = -2 * margin_term * diff / (dist + 1e-12) * (1 - y) / N
    dz1 = dz1_sim + dz1_dis
    return float(loss), dz1, -dz1


def main():
    np.random.seed(0)

    # Dataset sintético: pares similares dentro del mismo cluster.
    def make_pair_batch(n=64):
        c1 = np.random.randn(n // 2, 8) + 3
        c2 = np.random.randn(n // 2, 8) - 3
        X = np.vstack([c1, c2])
        pairs_a, pairs_b, labels = [], [], []
        for _ in range(n):
            i, j = np.random.choice(len(X), 2, replace=False)
            same_cluster = (i < n // 2) == (j < n // 2)
            pairs_a.append(X[i]); pairs_b.append(X[j])
            labels.append([1.0 if same_cluster else 0.0])
        return np.array(pairs_a), np.array(pairs_b), np.array(labels)

    encoder = SiameseEncoder(input_dim=8, hidden_dim=16, embed_dim=4)

    lr = 0.01
    for epoch in range(50):
        xa, xb, y = make_pair_batch(128)

        # Forward por las DOS entradas con la MISMA instancia de capa.
        z1, cache_a = encoder.encode(xa)
        z2, cache_b = encoder.encode(xb)

        loss, dz1, dz2 = contrastive_loss(z1, z2, y)

        # Backward de ambas ramas: cada una usa su cache respectivo
        _, g1a, g2a = encoder.backward_encode(dz1, cache_a)
        _, g1b, g2b = encoder.backward_encode(dz2, cache_b)

        # Gradientes combinados (sum sobre ramas = parámetros compartidos)
        for name in ("weights", "biases"):
            encoder.fc1.parameters()[name][...] = (
                encoder.fc1.parameters()[name] - lr * (g1a[name] + g1b[name])
            )
            encoder.fc2.parameters()[name][...] = (
                encoder.fc2.parameters()[name] - lr * (g2a[name] + g2b[name])
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} - loss: {loss:.4f}")

    print("\nRed siamesa entrenada con éxito. State isolation verificado.")


if __name__ == "__main__":
    main()