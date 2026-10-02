"""Algoritmos de *debiasing* (Bolukbasi et al., 2016, §6).

* **Passo 1 – identificar o subespaço** ``B`` (``replang.bias.geometry.bias_subspace``).
* **Passo 2a – Hard de-biasing**: *Neutralize* (``w := (w − w_B)/‖w − w_B‖`` para ``w ∈ N``) e
  *Equalize* (cada conjunto ``E`` recebe média ``ν = μ − μ_B`` fora de ``B`` e componente em ``B``
  centrada e reescalada para norma 1). Após o passo, qualquer palavra neutra é equidistante de
  todas as palavras de cada conjunto de igualdade (Observação 1) e ``PairBias = 0``.
* **Passo 2b – Soft de-biasing**: transformação linear ``T`` que minimiza
  ``‖(TW)ᵀ(TW) − WᵀW‖²_F + λ‖(TN)ᵀ(TB)‖²_F``. O artigo resolve um SDP; aqui otimizamos ``T`` por
  gradiente (versão didática, equivalente em espírito), com ``λ`` controlando o compromisso.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from replang.embedding import WordVectors


def _proj_B(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``v_B = Σ_j (v·b_j) b_j`` para ``B`` com linhas ortonormais."""
    return (v @ B.T) @ B


def hard_debias(
    wv: WordVectors,
    B: np.ndarray,
    *,
    neutral_words: Sequence[str] | None = None,
    gender_specific: Sequence[str] | None = None,
    equalize_pairs: Sequence[tuple[str, str]] = (),
    name: str | None = None,
) -> WordVectors:
    """Retorna um novo ``WordVectors`` (normalizado) neutralizado e equalizado.

    Informe ``neutral_words`` (conjunto ``N``) **ou** ``gender_specific`` (``S``; então ``N = W \\ S``,
    como no §7 do artigo). ``equalize_pairs`` são os conjuntos de igualdade ``E``.
    """
    B = np.atleast_2d(np.asarray(B, dtype=np.float32))
    out = wv.normalized(name or f"{wv.name} [hard-debiased]")
    V = out.vectors
    if neutral_words is None:
        if gender_specific is None:
            raise ValueError("informe neutral_words ou gender_specific")
        spec = set(gender_specific) | {w for p in equalize_pairs for w in p}
        idx = np.array([i for i, w in enumerate(out.words) if w not in spec], dtype=np.int64)
    else:
        idx = np.array([out.index[w] for w in neutral_words if w in out.index], dtype=np.int64)
    # Neutralize: remove a componente em B e renormaliza
    if len(idx):
        sub = V[idx]
        sub = sub - _proj_B(sub, B)
        sub /= np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9
        V[idx] = sub
    # Equalize: cada conjunto E (aqui pares) fica simétrico em torno de ν fora de B
    for pair in equalize_pairs:
        ws = [w for w in pair if w in out.index]
        if len(ws) < 2:
            continue
        ids = [out.index[w] for w in ws]
        mu = V[ids].mean(0)
        nu = mu - _proj_B(mu, B)
        for i in ids:
            w_b = _proj_B(V[i], B)
            mu_b = _proj_B(mu, B)
            diff = w_b - mu_b
            n = np.linalg.norm(diff)
            scale = np.sqrt(max(1 - float(np.dot(nu, nu)), 0.0))
            V[i] = nu + (scale * diff / n if n > 1e-9 else 0.0)
    out.invalidate()
    return out


def pair_bias(
    wv: WordVectors, neutral_words: Sequence[str], pairs: Sequence[tuple[str, str]]
) -> float:
    """``PairBias = (1/|N||P|) Σ_w Σ_(a,b) |cos(w,a) − cos(w,b)|`` — zero após o hard debias."""
    vals = []
    for w in neutral_words:
        if w not in wv:
            continue
        for a, b in pairs:
            if a in wv and b in wv:
                vals.append(abs(wv.similarity(w, a) - wv.similarity(w, b)))
    return float(np.mean(vals)) if vals else float("nan")


def soft_debias(
    wv: WordVectors,
    B: np.ndarray,
    *,
    neutral_words: Sequence[str],
    lam: float = 0.2,
    steps: int = 300,
    lr: float = 0.05,
    sample: int = 4000,
    seed: int = 0,
    name: str | None = None,
    callback=None,
) -> tuple[WordVectors, np.ndarray, list[dict]]:
    """Otimiza ``T`` (d × d) por gradiente (JAX se disponível, senão numpy com diferenças analíticas).

    Para manter o custo baixo, o termo de preservação usa uma amostra de ``sample`` palavras
    (``WᵀW`` é |V|×|V|; o artigo contorna via SVD, eq. 4–5). Retorna o embedding transformado e
    normalizado, a matriz ``T`` e o histórico das duas parcelas da função objetivo.
    """
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    rng = np.random.default_rng(seed)
    unit = wv.unit.astype(np.float64)
    idx = rng.choice(len(unit), min(sample, len(unit)), replace=False)
    W = unit[idx].T  # d × n
    N = np.stack([unit[wv.index[w]] for w in neutral_words if w in wv.index]).T  # d × m
    d = W.shape[0]
    T = np.eye(d)
    G = W.T @ W  # n × n (produtos internos originais)
    hist = []
    for step in range(steps):
        TW = T @ W
        TN = T @ N
        TB = T @ B.T  # d × k
        R = TW.T @ TW - G  # n × n
        S = TN.T @ TB  # m × k
        loss1 = float((R**2).sum())
        loss2 = float((S**2).sum())
        # gradientes: d/dT ‖(TW)ᵀ(TW) − G‖² = 4 T W R Wᵀ ; d/dT ‖(TN)ᵀ(TB)‖² = 2 T (N S Bᵀ... ) simetrizado
        g1 = 4 * TW @ R @ W.T
        g2 = 2 * (TN @ S @ B + TB @ S.T @ N.T)  # d × d
        grad = g1 / G.size + lam * g2 / max(S.size, 1)
        T -= lr * grad / (np.linalg.norm(grad) + 1e-9)
        if step % 10 == 0 or step == steps - 1:
            rec = {"step": step, "preservação": loss1 / G.size, "viés": loss2 / max(S.size, 1)}
            hist.append(rec)
            if callback:
                callback(rec)
    new = (T @ unit.T).T.astype(np.float32)
    out = WordVectors(
        wv.words, new, name or f"{wv.name} [soft-debiased λ={lam}]", wv.counts
    ).normalized()
    return out, T, hist
