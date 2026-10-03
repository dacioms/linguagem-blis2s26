"""GloVe com JAX: o mesmo objetivo de :mod:`replang.models.glove_np`, com a época inteira
compilada (``jit``) e o *scatter-add* (``.at[].add``) no lugar do ``np.add.at``.

Em CPU já é ~8× mais rápido que a versão numpy; em GPU (RTX 4060/5050) o lote de 2^14–2^16
pares roda inteiro na placa. A API é idêntica à de :class:`~replang.models.glove_np.GloVe`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from replang.embedding import WordVectors
from replang.models.cooccurrence import build_vocab, cooccurrence_matrix


class GloVeJax:
    def __init__(
        self,
        dim: int = 50,
        window: int = 5,
        x_max: float = 100.0,
        alpha: float = 0.75,
        lr: float = 0.05,
        epochs: int = 20,
        min_count: int = 1,
        batch_size: int = 16384,
        seed: int = 42,
        max_vocab: int | None = None,
    ):
        import jax  # noqa: F401 – falha cedo se ausente

        self.dim, self.window, self.x_max, self.alpha, self.lr = dim, window, x_max, alpha, lr
        self.epochs, self.min_count, self.batch_size, self.max_vocab, self.seed = (
            epochs,
            min_count,
            batch_size,
            max_vocab,
            seed,
        )
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def fit(
        self, sentences: list[list[str]], callback: Callable[[dict], None] | None = None
    ) -> GloVeJax:
        import jax
        import jax.numpy as jnp

        self.words, self.index, self.freq = build_vocab(sentences, self.min_count, self.max_vocab)
        X = cooccurrence_matrix(
            sentences, self.index, window=self.window, weighting="harmonic"
        ).tocoo()
        self.X_ = X
        i, j, x = X.row.astype(np.int32), X.col.astype(np.int32), X.data.astype(np.float32)
        logx = np.log(x)
        fx = np.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0).astype(np.float32)
        V, d, n = len(self.words), self.dim, len(x)
        scale = 0.5 / d
        W = jnp.asarray((self.rng.random((V, d), dtype=np.float32) - 0.5) * 2 * scale)
        Wc = jnp.asarray((self.rng.random((V, d), dtype=np.float32) - 0.5) * 2 * scale)
        b = jnp.zeros(V, jnp.float32)
        bc = jnp.zeros(V, jnp.float32)
        params = (W, Wc, b, bc)
        acc = tuple(jnp.ones_like(p) for p in params)
        ij, jj, lx, f = (jnp.asarray(a) for a in (i, j, logx, fx))
        lr = self.lr
        bs = min(self.batch_size, n)
        nb = max(1, n // bs)

        @jax.jit
        def epoch(params, acc, order):
            def body(k, carry):
                params, acc, tot = carry
                sl = jax.lax.dynamic_slice_in_dim(order, k * bs, bs)
                ii, jj_ = ij[sl], jj[sl]
                W, Wc, b, bc = params
                gW, gWc, gb, gbc = acc
                wi, wj = W[ii], Wc[jj_]
                diff = (wi * wj).sum(1) + b[ii] + bc[jj_] - lx[sl]
                wd = f[sl] * diff
                loss = (wd * diff).sum()
                g = 2 * wd
                gwi = g[:, None] * wj
                gwj = g[:, None] * wi
                W = W.at[ii].add(-lr * gwi / jnp.sqrt(gW[ii]))
                Wc = Wc.at[jj_].add(-lr * gwj / jnp.sqrt(gWc[jj_]))
                b = b.at[ii].add(-lr * g / jnp.sqrt(gb[ii]))
                bc = bc.at[jj_].add(-lr * g / jnp.sqrt(gbc[jj_]))
                gW = gW.at[ii].add(gwi**2)
                gWc = gWc.at[jj_].add(gwj**2)
                gb = gb.at[ii].add(g**2)
                gbc = gbc.at[jj_].add(g**2)
                return (W, Wc, b, bc), (gW, gWc, gb, gbc), tot + loss

            return jax.lax.fori_loop(0, nb, body, (params, acc, jnp.float32(0.0)))

        for ep in range(self.epochs):
            order = jnp.asarray(self.rng.permutation(n)[: nb * bs].astype(np.int32))
            params, acc, tot = epoch(params, acc, order)
            rec = {"epoch": ep, "loss": float(tot) / (nb * bs)}
            self.history.append(rec)
            if callback:
                callback(rec)
        self.W, self.Wc = np.asarray(params[0]), np.asarray(params[1])
        self.b, self.bc = np.asarray(params[2]), np.asarray(params[3])
        return self

    def to_wordvectors(self, name: str = "GloVe (JAX)") -> WordVectors:
        return WordVectors(self.words, self.W + self.Wc, name, self.freq)
