"""Modelos **contagem-baseados**: a premissa basal de todas as representações distribucionais.

Hipótese distribucional (Harris, 1954
Firth, 1957): *"you shall know a word by the company it
keeps"*. A versão mais simples é a matriz de coocorrência palavra × contexto, seguida de uma
reponderação (PPMI) e de uma redução de dimensionalidade (SVD truncada — LSA/HAL). Levy & Goldberg
(2014) mostraram que o Skip-gram com amostragem negativa fatora implicitamente uma matriz
PMI deslocada
Pennington et al. (2014, GloVe) partem explicitamente das contagens. Este módulo
torna esse caminho visível passo a passo.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp

from replang.embedding import WordVectors


def build_vocab(
    sentences: Iterable[list[str]], min_count: int = 1, max_size: int | None = None
) -> tuple[list[str], dict[str, int], np.ndarray]:
    counts = Counter(t for s in sentences for t in s)
    words = [w for w, c in counts.most_common(max_size) if c >= min_count]
    index = {w: i for i, w in enumerate(words)}
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    return words, index, freq


def cooccurrence_matrix(
    sentences: Iterable[list[str]],
    index: dict[str, int],
    *,
    window: int = 2,
    weighting: str = "flat",
    symmetric: bool = True,
) -> sp.csr_matrix:
    """Matriz |V|×|V| de coocorrências numa janela de ``window`` palavras.

    ``weighting``:
    * ``"flat"`` – cada vizinho conta 1 (HAL/LSA clássico)
    * ``"harmonic"`` – vizinho à distância *d* conta ``1/d`` (como no GloVe)
    * ``"linear"`` – ``(window - d + 1)/window`` (como no word2vec, que sorteia a janela efetiva).
    """
    rows, cols, vals = [], [], []
    for sent in sentences:
        ids = [index.get(t, -1) for t in sent]
        n = len(ids)
        for i, wi in enumerate(ids):
            if wi < 0:
                continue
            for d in range(1, window + 1):
                j = i + d
                if j >= n:
                    break
                wj = ids[j]
                if wj < 0:
                    continue
                w = (
                    1.0
                    if weighting == "flat"
                    else (1.0 / d if weighting == "harmonic" else (window - d + 1) / window)
                )
                rows.append(wi)
                cols.append(wj)
                vals.append(w)
                if symmetric:
                    rows.append(wj)
                    cols.append(wi)
                    vals.append(w)
    n = len(index)
    m = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    return m.tocsr()


def ppmi(m: sp.csr_matrix, *, alpha: float = 1.0, shift: float = 0.0) -> sp.csr_matrix:
    """PMI positivo: ``max(0, log P(w,c)/(P(w)P(c)^α) − log k)``.

    * ``alpha=0.75`` é o *context distribution smoothing* de Levy, Goldberg & Dagan (2015), análogo
      ao expoente 3/4 da distribuição de ruído do word2vec
    * ``shift=log(k)`` reproduz o deslocamento implícito da amostragem negativa com *k* negativos.
    """
    m = m.tocoo()
    total = m.sum()
    row_sum = np.asarray(m.sum(axis=1)).ravel()
    col_sum = np.asarray(m.sum(axis=0)).ravel() ** alpha
    col_sum = col_sum / col_sum.sum()
    pw = row_sum / total
    with np.errstate(divide="ignore"):
        pmi = np.log(m.data / total) - np.log(pw[m.row]) - np.log(col_sum[m.col]) - shift
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    out = sp.coo_matrix((pmi, (m.row, m.col)), shape=m.shape).tocsr()
    out.eliminate_zeros()
    return out


def svd_embeddings(
    m: sp.spmatrix | np.ndarray, k: int = 50, *, eig_weight: float = 0.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """SVD truncada ``M ≈ U Σ Vᵀ``
    retorna ``(U Σ^p, Σ)``. ``eig_weight=0.5`` é a escolha
    recomendada por Levy et al. (2015) (``p=1`` é a LSA clássica, ``p=0`` só U)."""
    from scipy.sparse.linalg import svds

    k = min(k, min(m.shape) - 1)
    if sp.issparse(m):
        u, s, _ = svds(m.asfptype(), k=k, random_state=seed)
    else:
        u, s, _ = np.linalg.svd(np.asarray(m, dtype=np.float64), full_matrices=False)
        u, s = u[:, :k], s[:k]
    order = np.argsort(-s)
    u, s = u[:, order], s[order]
    return u * (s**eig_weight), s


class CountModel:
    """Pipeline completo contagem → PPMI → SVD, exposto como ``WordVectors``."""

    def __init__(
        self,
        window: int = 2,
        dim: int = 50,
        min_count: int = 1,
        weighting: str = "flat",
        alpha: float = 0.75,
        shift: float = 0.0,
        eig_weight: float = 0.5,
        max_vocab: int | None = None,
    ):
        self.window, self.dim, self.min_count, self.weighting = window, dim, min_count, weighting
        self.alpha, self.shift, self.eig_weight, self.max_vocab = (
            alpha,
            shift,
            eig_weight,
            max_vocab,
        )
        self.words: list[str] = []
        self.index: dict[str, int] = {}
        self.freq: np.ndarray | None = None
        self.counts_: sp.csr_matrix | None = None
        self.ppmi_: sp.csr_matrix | None = None
        self.sigma_: np.ndarray | None = None
        self.vectors_: np.ndarray | None = None

    def fit(self, sentences: list[list[str]]) -> CountModel:
        self.words, self.index, self.freq = build_vocab(sentences, self.min_count, self.max_vocab)
        self.counts_ = cooccurrence_matrix(
            sentences, self.index, window=self.window, weighting=self.weighting
        )
        self.ppmi_ = ppmi(self.counts_, alpha=self.alpha, shift=self.shift)
        self.vectors_, self.sigma_ = svd_embeddings(
            self.ppmi_, self.dim, eig_weight=self.eig_weight
        )
        return self

    def to_wordvectors(self, name: str = "contagem→PPMI→SVD") -> WordVectors:
        assert self.vectors_ is not None, "chame fit() primeiro"
        return WordVectors(self.words, self.vectors_, name, self.freq)

    def dense_counts(self):
        import pandas as pd

        return pd.DataFrame(self.counts_.toarray(), index=self.words, columns=self.words)

    def dense_ppmi(self):
        import pandas as pd

        return pd.DataFrame(self.ppmi_.toarray(), index=self.words, columns=self.words)


def one_hot(words: list[str], word: str) -> np.ndarray:
    """Vetor *one-hot*: a representação "sem significado" que motiva os embeddings."""
    v = np.zeros(len(words))
    v[words.index(word)] = 1.0
    return v


def bag_of_words(sentences: list[list[str]], words: list[str]) -> np.ndarray:
    """Matriz documento × termo (contagens), base do TF-IDF usado como baseline por Hartmann (2016)."""
    idx = {w: i for i, w in enumerate(words)}
    m = np.zeros((len(sentences), len(words)))
    for i, s in enumerate(sentences):
        for t in s:
            if t in idx:
                m[i, idx[t]] += 1
    return m


def tfidf(bow: np.ndarray) -> np.ndarray:
    df = (bow > 0).sum(axis=0)
    idf = np.log((1 + bow.shape[0]) / (1 + df)) + 1
    tf = bow / np.maximum(bow.sum(axis=1, keepdims=True), 1)
    return tf * idf
