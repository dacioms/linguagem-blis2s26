"""Avaliação **extrínseca**: embeddings como *features* de tarefas reais (Hartmann et al., 2017, §4.2).

* **POS tagging** (Mac-Morpho): classificador linear sobre a concatenação dos vetores de uma
  janela ``[w_{i−k} … w_{i+k}]`` (versão simplificada do *tagger* nlpnet com janela, sem camada
  oculta, para ser rápida em CPU). O que importa é a **comparação relativa** entre embeddings.
* **Similaridade de sentenças** (estilo ASSIN): vetor da sentença = soma (ou média) dos vetores
  das palavras; similaridade = cosseno; regressão linear → escala 1–5; métricas Pearson e MSE
  exatamente como Hartmann (2016) e Hartmann et al. (2017, Tabela 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression

from replang.embedding import WordVectors
from replang.utils.text import tokenize


def _window_features(wv: WordVectors, tokens: list[str], k: int, oov_fn=None) -> np.ndarray:
    d = wv.dim
    vecs = []
    for t in tokens:
        tl = t.lower()
        v = wv.get(tl)
        if v is None and oov_fn is not None:
            v = oov_fn(tl)
        vecs.append(v if v is not None else np.zeros(d, dtype=np.float32))
    pad = np.zeros(d, dtype=np.float32)
    padded = [pad] * k + vecs + [pad] * k
    feats = np.stack([np.concatenate(padded[i : i + 2 * k + 1]) for i in range(len(tokens))])
    # traços ortográficos simples (como no nlpnet): capitalização e sufixo (hash)
    caps = np.array(
        [[t[:1].isupper(), t.isupper(), any(ch.isdigit() for ch in t)] for t in tokens],
        dtype=np.float32,
    )
    return np.hstack([feats, caps])


def pos_tagging_eval(
    wv: WordVectors,
    train_sents: list[list[tuple[str, str]]],
    test_sents: list[list[tuple[str, str]]],
    *,
    window: int = 2,
    max_train_tokens: int = 60000,
    oov_fn=None,
    C: float = 1.0,
) -> dict:
    """Acurácia de POS tagging com regressão logística multinomial sobre janelas de embeddings."""
    X, y, n = [], [], 0
    for s in train_sents:
        toks = [w for w, _ in s]
        X.append(_window_features(wv, toks, window, oov_fn))
        y.extend(t for _, t in s)
        n += len(s)
        if n >= max_train_tokens:
            break
    X = np.vstack(X)
    clf = LogisticRegression(max_iter=300, C=C, n_jobs=-1)
    clf.fit(X, y)
    Xt, yt = [], []
    for s in test_sents:
        toks = [w for w, _ in s]
        Xt.append(_window_features(wv, toks, window, oov_fn))
        yt.extend(t for _, t in s)
    Xt = np.vstack(Xt)
    pred = clf.predict(Xt)
    yt = np.array(yt)
    oov = np.mean([w.lower() not in wv for s in test_sents for w, _ in s])
    acc = float((pred == yt).mean())
    acc_oov_mask = np.array([w.lower() not in wv for s in test_sents for w, _ in s])
    return {
        "acurácia": acc,
        "acurácia_oov": float((pred[acc_oov_mask] == yt[acc_oov_mask]).mean())
        if acc_oov_mask.any()
        else np.nan,
        "taxa_oov": float(oov),
        "n_treino": len(y),
        "n_teste": len(yt),
        "clf": clf,
    }


def sentence_vector(
    wv: WordVectors, sentence: str | list[str], *, mode: str = "sum", oov_fn=None
) -> np.ndarray:
    toks = tokenize(sentence) if isinstance(sentence, str) else sentence
    vecs = []
    for t in toks:
        v = wv.get(t)
        if v is None and oov_fn is not None:
            v = oov_fn(t)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return np.zeros(wv.dim, dtype=np.float32)
    m = np.stack(vecs)
    return m.sum(0) if mode == "sum" else m.mean(0)


def sentence_similarity_eval(
    wv: WordVectors,
    pairs: list[tuple[str, str, float]],
    *,
    mode: str = "sum",
    oov_fn=None,
    folds: int = 5,
    seed: int = 0,
) -> dict:
    """Pearson ρ e MSE (validação cruzada) de um regressor linear ``cos → nota``."""
    cos = np.array(
        [
            WordVectors.cosine(
                sentence_vector(wv, a, mode=mode, oov_fn=oov_fn),
                sentence_vector(wv, b, mode=mode, oov_fn=oov_fn),
            )
            for a, b, _ in pairs
        ]
    )
    gold = np.array([s for _, _, s in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    preds = np.zeros_like(gold)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        reg = LinearRegression().fit(cos[train, None], gold[train])
        preds[test] = reg.predict(cos[test, None])
    rho = pearsonr(gold, preds)[0] if len(gold) > 2 else np.nan
    mse = float(np.mean((gold - preds) ** 2))
    df = pd.DataFrame(
        {
            "s1": [p[0] for p in pairs],
            "s2": [p[1] for p in pairs],
            "ouro": gold,
            "cosseno": cos,
            "previsto": preds,
        }
    )
    return {
        "pearson": float(rho),
        "mse": mse,
        "pearson_cosseno": float(pearsonr(gold, cos)[0]),
        "tabela": df,
    }
