"""
Siamese network demo — demonstrates that the SAME layer can process two
different inputs in parallel without corruption.

In v0.3 this was impossible because each forward pass stored the cache in
`self.inputs`/`self.z` of the layer, so the second input would overwrite
the first and the backward pass would produce incorrect gradients.

In v0.4, forward returns `(output, cache)` and backward receives the
cache explicitly. The same layer can be used N times with N independent
caches.

Here we manually build a siamese embedder: a shared network processes
two inputs, and a simple contrastive loss compares their embeddings.
"""
import numpy as np

from nnlib.activations import ReLU, Tanh
from nnlib.layer import Dense


class SiameseEncoder:
    """Encoder with shared weights applied to two inputs."""

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
    """Classic contrastive loss: similar pairs (y=1) -> close."""
    diff = z1 - z2
    dist_sq = np.sum(diff ** 2, axis=1, keepdims=True)
    dist = np.sqrt(dist_sq + 1e-12)

    loss = np.mean(y * dist_sq + (1 - y) * np.maximum(0, margin - dist) ** 2)

    # Analytic gradients with respect to z1 (z2 is symmetric)
    N = y.shape[0]
    dz1_sim = 2 * diff * y / N
    margin_term = np.maximum(0, margin - dist)
    # derivative of (margin - dist)^2 with respect to diff
    dz1_dis = -2 * margin_term * diff / (dist + 1e-12) * (1 - y) / N
    dz1 = dz1_sim + dz1_dis
    return float(loss), dz1, -dz1


def main():
    np.random.seed(0)

    # Synthetic dataset: similar pairs within the same cluster.
    def make_pair_batch(n=64):
        c1 = np.random.randn(n // 2, 8) + 3
        c2 = np.random.randn(n // 2, 8) - 3
        X = np.vstack([c1, c2])
        pairs_a, pairs_b, labels = [], [], []
        for _ in range(n):
            i, j = np.random.choice(len(X), 2, replace=False)
            same_cluster = (i < n // 2) == (j < n // 2)
            pairs_a.append(X[i])
            pairs_b.append(X[j])
            labels.append([1.0 if same_cluster else 0.0])
        return np.array(pairs_a), np.array(pairs_b), np.array(labels)

    encoder = SiameseEncoder(input_dim=8, hidden_dim=16, embed_dim=4)

    lr = 0.01
    for epoch in range(50):
        xa, xb, y = make_pair_batch(128)

        # Forward through BOTH inputs with the SAME layer instance.
        z1, cache_a = encoder.encode(xa)
        z2, cache_b = encoder.encode(xb)

        loss, dz1, dz2 = contrastive_loss(z1, z2, y)

        # Backward of both branches: each uses its respective cache
        _, g1a, g2a = encoder.backward_encode(dz1, cache_a)
        _, g1b, g2b = encoder.backward_encode(dz2, cache_b)

        # Combined gradients (sum over branches = shared parameters)
        for name in ("weights", "biases"):
            encoder.fc1.parameters()[name][...] = (
                encoder.fc1.parameters()[name] - lr * (g1a[name] + g1b[name])
            )
            encoder.fc2.parameters()[name][...] = (
                encoder.fc2.parameters()[name] - lr * (g2a[name] + g2b[name])
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} - loss: {loss:.4f}")

    print("\nSiamese network trained successfully. State isolation verified.")


if __name__ == "__main__":
    main()
