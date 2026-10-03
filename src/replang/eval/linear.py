"""Classificador linear multiclasse com *backend* selecionável.

* ``sklearn`` — ``LogisticRegression`` (lbfgs): rápido em CPU para dezenas de milhares de exemplos.
* ``jax`` — regressão softmax com Adam em mini-lotes, compilada. Medido no POS tagging do
  Mac-Morpho (60k tokens × 503 traços × 26 classes, CPU 4 núcleos): 28 s → 3 s, com acurácia
  0,905 → 0,898. Em GPU (RTX 4060/5050) o mesmo treino leva frações de segundo. ``backend="auto"``
  escolhe JAX sempre que instalado (ver :mod:`replang.accel`).
"""

from __future__ import annotations

import numpy as np

from replang.accel import detect


class LinearClassifier:
    def __init__(
        self,
        backend: str = "auto",
        max_iter: int = 300,
        C: float = 1.0,
        epochs: int = 12,
        batch_size: int = 1024,
        lr: float = 5e-3,
        seed: int = 0,
    ):
        if backend == "auto":
            backend = "jax" if detect().use_jax_for("linear") else "sklearn"
        self.backend, self.max_iter, self.C, self.epochs, self.batch_size, self.lr, self.seed = (
            backend,
            max_iter,
            C,
            epochs,
            batch_size,
            lr,
            seed,
        )

    def fit(self, X: np.ndarray, y) -> LinearClassifier:
        y = np.asarray(y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        if self.backend == "sklearn":
            from sklearn.linear_model import LogisticRegression

            self._clf = LogisticRegression(max_iter=self.max_iter, C=self.C).fit(X, yi)
            return self
        import jax
        import jax.numpy as jnp

        X = np.asarray(X, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xn = jnp.asarray((X - self.mu_) / self.sd_)
        yj = jnp.asarray(yi)
        n, dim = X.shape
        C = len(self.classes_)
        l2 = 1.0 / (self.C * n)
        rng = np.random.default_rng(self.seed)
        params = (jnp.zeros((dim, C)), jnp.zeros(C))
        opt = tuple(jnp.zeros_like(p) for p in params) * 2

        def loss_fn(params, xb, yb):
            W, b = params
            lp = jax.nn.log_softmax(xb @ W + b)
            return -jnp.take_along_axis(lp, yb[:, None], 1).mean() + l2 * (W**2).sum()

        lr = self.lr

        @jax.jit
        def upd(params, opt, xb, yb, t):
            loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
            m = tuple(0.9 * a + 0.1 * g for a, g in zip(opt[:2], grads, strict=True))
            v = tuple(0.999 * a + 0.001 * g * g for a, g in zip(opt[2:], grads, strict=True))
            new = tuple(
                p - lr * (mi / (1 - 0.9**t)) / (jnp.sqrt(vi / (1 - 0.999**t)) + 1e-8)
                for p, mi, vi in zip(params, m, v, strict=True)
            )
            return new, m + v, loss

        # conjuntos pequenos: lotes menores e passos suficientes (≥ 400) para convergir
        bs = int(min(self.batch_size, max(32, n // 8)))
        steps_per_epoch = max(1, n // bs)
        epochs = max(self.epochs, -(-400 // steps_per_epoch))
        t = 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for b0 in range(0, n - bs + 1, bs):
                sl = jnp.asarray(order[b0 : b0 + bs])
                t += 1
                params, opt, _ = upd(params, opt, Xn[sl], yj[sl], t)
        self.W_, self.b_ = np.asarray(params[0]), np.asarray(params[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "sklearn":
            return self.classes_[self._clf.predict(X)]
        logits = ((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_) @ self.W_ + self.b_
        return self.classes_[logits.argmax(1)]

    def score(self, X, y) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())
