"""GloVe *from scratch* (Pennington, Socher & Manning, 2014).

Objetivo (eq. 8 do artigo): ``J = Σ_ij f(X_ij) (w_iᵀ w̃_j + b_i + b̃_j − log X_ij)²`` com
``f(x) = (x/x_max)^α`` se ``x < x_max`` senão 1. É um método **contagem-baseado** (parte da matriz
global de coocorrência ``X``, ponderada por ``1/d``) treinado como um problema de mínimos
quadrados ponderados com AdaGrad — o "meio-termo" entre LSA/HAL e word2vec. Hartmann et al.
(2017) observam que o GloVe é o melhor em analogias **semânticas** em PT, mas o pior nas tarefas
extrínsecas (POS, similaridade de sentenças).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from replang.embedding import WordVectors
from replang.models.cooccurrence import build_vocab, cooccurrence_matrix


class GloVe:
    def __init__(
        self,
        dim: int = 50,
        window: int = 5,
        x_max: float = 100.0,
        alpha: float = 0.75,
        lr: float = 0.05,
        epochs: int = 20,
        min_count: int = 1,
        batch_size: int = 4096,
        seed: int = 42,
        max_vocab: int | None = None,
    ):
        self.dim, self.window, self.x_max, self.alpha, self.lr = dim, window, x_max, alpha, lr
        self.epochs, self.min_count, self.batch_size, self.max_vocab = (
            epochs,
            min_count,
            batch_size,
            max_vocab,
        )
        self.rng = np.random.default_rng(seed)
        self.history: list[dict] = []

    def fit(
        self, sentences: list[list[str]], callback: Callable[[dict], None] | None = None
    ) -> GloVe:
        self.words, self.index, self.freq = build_vocab(sentences, self.min_count, self.max_vocab)
        X = cooccurrence_matrix(
            sentences, self.index, window=self.window, weighting="harmonic"
        ).tocoo()
        self.X_ = X
        i, j, x = X.row, X.col, X.data.astype(np.float32)
        logx = np.log(x)
        fx = np.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0).astype(np.float32)
        V = len(self.words)
        scale = 0.5 / self.dim
        self.W = (self.rng.random((V, self.dim), dtype=np.float32) - 0.5) * scale * 2
        self.Wc = (self.rng.random((V, self.dim), dtype=np.float32) - 0.5) * scale * 2
        self.b = np.zeros(V, dtype=np.float32)
        self.bc = np.zeros(V, dtype=np.float32)
        # acumuladores do AdaGrad
        gW, gWc = np.ones_like(self.W), np.ones_like(self.Wc)
        gb, gbc = np.ones_like(self.b), np.ones_like(self.bc)
        n = len(x)
        for epoch in range(self.epochs):
            order = self.rng.permutation(n)
            total = 0.0
            for b0 in range(0, n, self.batch_size):
                sl = order[b0 : b0 + self.batch_size]
                ii, jj = i[sl], j[sl]
                wi, wj = self.W[ii], self.Wc[jj]
                diff = (wi * wj).sum(1) + self.b[ii] + self.bc[jj] - logx[sl]
                wdiff = fx[sl] * diff
                total += float((wdiff * diff).sum())
                g = 2 * wdiff  # dJ/d(pred)
                gw_i = g[:, None] * wj
                gw_j = g[:, None] * wi
                # AdaGrad: atualização ∝ g / sqrt(acumulado)
                np.add.at(self.W, ii, -self.lr * gw_i / np.sqrt(gW[ii]))
                np.add.at(self.Wc, jj, -self.lr * gw_j / np.sqrt(gWc[jj]))
                np.add.at(self.b, ii, -self.lr * g / np.sqrt(gb[ii]))
                np.add.at(self.bc, jj, -self.lr * g / np.sqrt(gbc[jj]))
                np.add.at(gW, ii, gw_i**2)
                np.add.at(gWc, jj, gw_j**2)
                np.add.at(gb, ii, g**2)
                np.add.at(gbc, jj, g**2)
            rec = {"epoch": epoch, "loss": total / n}
            self.history.append(rec)
            if callback:
                callback(rec)
        return self

    def to_wordvectors(self, name: str = "GloVe (numpy)") -> WordVectors:
        # O artigo soma W + W̃ (seção 4.2): reduz ruído e melhora ligeiramente os resultados.
        return WordVectors(self.words, self.W + self.Wc, name, self.freq)

    def cooccurrence_df(self):
        import pandas as pd

        return pd.DataFrame(self.X_.toarray(), index=self.words, columns=self.words)


def glove_probability_ratios(
    counts: sp.spmatrix, words: list[str], target_a: str, target_b: str, probes: list[str]
):
    """Tabela 1 do artigo GloVe: razões ``P(k|a)/P(k|b)`` que motivam o modelo (ice/steam)."""
    import pandas as pd

    idx = {w: i for i, w in enumerate(words)}
    m = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    rows = {}
    for t in (target_a, target_b):
        row = m[idx[t]]
        p = row / max(row.sum(), 1)
        rows[f"P(k|{t})"] = [p[idx[k]] for k in probes]
    df = pd.DataFrame(rows, index=probes)
    df[f"P(k|{target_a})/P(k|{target_b})"] = df.iloc[:, 0] / df.iloc[:, 1].replace(0, np.nan)
    return df
