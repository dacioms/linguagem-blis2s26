"""Geometria do viés de gênero em embeddings (Bolukbasi et al., 2016, §4–§5 e §7).

Notação do artigo: vetores unitários ``w ∈ R^d``
conjunto de pares definicionais
``P = {(she, he), (woman, man), …}``
direção de gênero ``g`` = 1ª componente principal das
diferenças centradas dos pares (§5.1, Fig. 6)
projeção ``w_B = (w·g) g``
``DirectBias_c = (1/|N|) Σ_{w∈N} |cos(w, g)|^c`` (§5.2)
``β(w, v) = (w·v − w_⊥·v_⊥ / (‖w_⊥‖‖v_⊥‖)) / (w·v)`` (§5.3, viés indireto).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from replang.embedding import WordVectors


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def pca_pairs(wv: WordVectors, pairs: Sequence[tuple[str, str]], *, k: int | None = None):
    """Passo 1 (§6): centra cada par pela média e faz PCA das diferenças. Retorna
    ``(componentes (k×d), variância explicada (k,), pares usados)``."""
    from sklearn.decomposition import PCA

    used, rows = [], []
    for a, b in pairs:
        if a in wv and b in wv:
            va, vb = _unit(wv[a]), _unit(wv[b])
            mu = (va + vb) / 2
            rows += [va - mu, vb - mu]
            used.append((a, b))
    if len(rows) < 4:
        raise ValueError("menos de 2 pares definicionais presentes no vocabulário")
    X = np.stack(rows)
    k = k or min(10, len(rows))
    pca = PCA(n_components=min(k, len(rows) - 1, X.shape[1])).fit(X)
    return pca.components_, pca.explained_variance_ratio_, used


def gender_direction(wv: WordVectors, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
    """``g`` = 1ª componente principal (vetor unitário). Orientada para que ``g·fem > 0``."""
    comps, _, used = pca_pairs(wv, pairs)
    g = _unit(comps[0])
    fem = np.mean([_unit(wv[a]) for a, _ in used], axis=0)
    masc = np.mean([_unit(wv[b]) for _, b in used], axis=0)
    if np.dot(g, fem - masc) < 0:
        g = -g
    return g.astype(np.float32)


def bias_subspace(wv: WordVectors, pairs: Sequence[tuple[str, str]], k: int = 1) -> np.ndarray:
    """Subespaço ``B`` (k × d) = primeiras k componentes (k=1 recupera ``g``)."""
    comps, _, _ = pca_pairs(wv, pairs, k=max(k, 2))
    B = comps[:k]
    if k == 1:
        return gender_direction(wv, pairs)[None, :]
    return B.astype(np.float32)


def random_pca_baseline(
    dim: int, n_pairs: int = 10, trials: int = 200, seed: int = 0
) -> np.ndarray:
    """Fig. 6 (direita): variância explicada média para pares de vetores unitários **aleatórios**."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    acc = np.zeros(min(10, 2 * n_pairs - 1))
    for _ in range(trials):
        rows = []
        for _ in range(n_pairs):
            a, b = rng.normal(size=dim), rng.normal(size=dim)
            a, b = _unit(a), _unit(b)
            mu = (a + b) / 2
            rows += [a - mu, b - mu]
        X = np.stack(rows)
        pca = PCA(n_components=len(acc)).fit(X)
        acc += pca.explained_variance_ratio_
    return acc / trials


def project(wv: WordVectors, words: Sequence[str], g: np.ndarray) -> pd.Series:
    """Projeção escalar ``cos(w, g)`` (vetores normalizados) — eixo *she–he* das Figs. 1, 4, 7."""
    g = _unit(np.asarray(g, dtype=np.float32))
    vals = {w: float(np.dot(_unit(wv[w]), g)) for w in words if w in wv}
    return pd.Series(vals).sort_values()


def extremes(
    wv: WordVectors, g: np.ndarray, words: Sequence[str], n: int = 12
) -> tuple[list[str], list[str]]:
    """``n`` palavras mais extremas em cada ponta da direção ``g`` (Fig. 1 e Fig. 3)."""
    p = project(wv, words, g)
    return list(p.index[-n:][::-1]), list(p.index[:n])


def direct_bias(
    wv: WordVectors, neutral_words: Sequence[str], g: np.ndarray, c: float = 1.0
) -> float:
    """``DirectBias_c = (1/|N|) Σ |cos(w, g)|^c`` (§5.2). No artigo, 327 profissões → 0,08."""
    g = _unit(np.asarray(g, dtype=np.float32))
    vals = [abs(float(np.dot(_unit(wv[w]), g))) ** c for w in neutral_words if w in wv]
    return float(np.mean(vals)) if vals else float("nan")


def indirect_bias(wv: WordVectors, w: str, v: str, g: np.ndarray) -> float:
    """``β(w, v)`` (§5.3): fração da similaridade ``w·v`` explicada pela componente de gênero.

    ``w = w_g + w_⊥``
    ``β = (w·v − (w_⊥·v_⊥)/(‖w_⊥‖‖v_⊥‖)) / (w·v)``.
    """
    g = _unit(np.asarray(g, dtype=np.float32))
    a, b = _unit(wv[w]), _unit(wv[v])
    a_perp = a - np.dot(a, g) * g
    b_perp = b - np.dot(b, g) * g
    dot = float(np.dot(a, b))
    if dot == 0:
        return 0.0
    perp = float(np.dot(a_perp, b_perp) / (np.linalg.norm(a_perp) * np.linalg.norm(b_perp)))
    return (dot - perp) / dot


def generate_analogies(
    wv: WordVectors,
    a: str,
    b: str,
    *,
    candidates: Sequence[str] | None = None,
    delta: float = 1.0,
    topn: int = 30,
    exclude: Sequence[str] = (),
    one_per_word: bool = True,
    restrict: int | None = 30000,
) -> pd.DataFrame:
    """Geração de analogias ``a : x :: b : y`` (eq. 1, §4): maximiza ``cos(a − b, x − y)`` sujeito a
    ``‖x − y‖ ≤ δ`` (δ = 1 ⇒ ângulo ≤ π/3, isto é, x e y semanticamente próximos).

    Versão vetorizada: para cada ``x`` candidato, calcula o escore de todos os ``y`` e mantém o
    melhor
    retorna os ``topn`` pares de maior escore, sem repetir palavras.
    """
    seed_dir = _unit(_unit(wv[a]) - _unit(wv[b]))
    unit = wv.unit if restrict is None else wv.unit[:restrict]
    n = len(unit)
    if candidates is None:
        cand_idx = np.arange(n)
    else:
        cand_idx = np.array([wv.index[w] for w in candidates if w in wv and wv.index[w] < n])
    excl = {a, b, *exclude}
    excl_idx = np.array(
        [wv.index[w] for w in excl if w in wv.index and wv.index[w] < n], dtype=np.int64
    )
    rows = []
    sq = (unit**2).sum(1)
    diff_dot = unit @ seed_dir  # y·s para todo y
    for i in cand_idx:
        if wv.words[i] in excl:
            continue
        x = unit[i]
        # cos(s, x − y) = (x·s − y·s)/‖x − y‖
        dist2 = sq[i] + sq - 2 * (unit @ x)
        dist = np.sqrt(np.maximum(dist2, 1e-12))
        score = (float(x @ seed_dir) - diff_dot) / dist
        score[dist > delta] = -np.inf
        score[i] = -np.inf
        if len(excl_idx):
            score[excl_idx] = -np.inf
        j = int(score.argmax())
        if np.isfinite(score[j]):
            rows.append((wv.words[i], wv.words[j], float(score[j]), float(dist[j])))
    df = pd.DataFrame(rows, columns=["x", "y", "score", "dist"]).sort_values(
        "score", ascending=False
    )
    if one_per_word:
        seen, keep = set(), []
        for r in df.itertuples(index=False):
            if r.x in seen or r.y in seen:
                continue
            seen.update((r.x, r.y))
            keep.append(r)
        df = pd.DataFrame(keep, columns=["x", "y", "score", "dist"])
    return df.head(topn).reset_index(drop=True)


def gender_classifier(
    wv: WordVectors,
    gender_specific: Sequence[str],
    neutral: Sequence[str],
    *,
    C: float = 1.0,
    seed: int = 0,
):
    """§7: SVM linear que separa palavras *específicas de gênero* das *neutras* (Fig. 7, eixo y).

    Retorna ``(clf, acurácia balanceada em validação cruzada)``.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import LinearSVC

    X, y = [], []
    for w in gender_specific:
        if w in wv:
            X.append(_unit(wv[w]))
            y.append(1)
    for w in neutral:
        if w in wv:
            X.append(_unit(wv[w]))
            y.append(0)
    X, y = np.stack(X), np.array(y)
    clf = LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=5000)
    score = cross_val_score(
        clf, X, y, cv=min(5, max(2, np.bincount(y).min())), scoring="balanced_accuracy"
    ).mean()
    clf.fit(X, y)
    return clf, float(score)


def profile(
    wv: WordVectors, words: Sequence[str], g: np.ndarray, labels: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Tabela com projeção em ``g`` e rótulo opcional por grupo (para gráficos)."""
    p = project(wv, words, g)
    df = pd.DataFrame({"palavra": p.index, "proj_g": p.values})
    df["grupo"] = "outras"
    if labels:
        for lab, ws in labels.items():
            df.loc[df.palavra.isin(ws), "grupo"] = lab
    return df


def is_morphological_pair(x: str, y: str) -> bool:
    """Heurística para o português: ``x``/``y`` diferem só pela marca de gênero (``-a/-o``,
    ``-a/-e``, ``-ora/-or``, ``-esa/-ês``, ``-triz/-tor`` …) ou pelo plural. Serve para separar
    analogias **gramaticais** (*professora:professor*) de **estereótipos** (*costura:carpintaria*),
    a questão levantada no §9 de Bolukbasi et al. para línguas com gênero gramatical."""
    from replang.utils.text import strip_accents

    a, b = strip_accents(x.lower()), strip_accents(y.lower())
    if a == b:
        return True
    for s in ("s",):
        if a.endswith(s) and a[:-1] == b or b.endswith(s) and b[:-1] == a:
            return True
    pairs = [
        ("a", "o"),
        ("a", "e"),
        ("ora", "or"),
        ("esa", "es"),
        ("essa", "e"),
        ("triz", "tor"),
        ("eira", "eiro"),
        ("ina", "ino"),
        ("isa", "e"),
        ("ona", "ao"),
        ("as", "os"),
        ("oras", "ores"),
        ("ia", "io"),
    ]
    for fa, fb in pairs:
        if a.endswith(fa) and b.endswith(fb) and a[: -len(fa)] == b[: -len(fb)]:
            return True
        if b.endswith(fa) and a.endswith(fb) and b[: -len(fa)] == a[: -len(fb)]:
            return True
    return False


def split_analogies_pt(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a saída de :func:`generate_analogies` em (gramaticais, potenciais estereótipos)."""
    mask = (
        df.apply(lambda r: is_morphological_pair(r.x, r.y), axis=1)
        if len(df)
        else pd.Series([], dtype=bool)
    )
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def expand_gender_specific(
    wv: WordVectors,
    seed: Sequence[str],
    neutral_pool: Sequence[str] | None = None,
    *,
    candidates: Sequence[str] | None = None,
    threshold: float = 0.5,
    C: float = 1.0,
) -> tuple[list[str], float]:
    """§7 do artigo: generaliza a lista-semente de palavras específicas de gênero a todo o vocabulário
    com um SVM linear. Retorna ``(lista expandida, acurácia balanceada do SVM)``. ``neutral_pool``
    (padrão: 3000 palavras frequentes fora da semente) são os negativos; ``candidates`` (padrão:
    20k mais frequentes) são classificadas; ``threshold`` é o corte na função de decisão."""
    seed_set = {w for w in seed if w in wv}
    if neutral_pool is None:
        neutral_pool = [w for w in wv.words[:8000] if w not in seed_set][:3000]
    clf, acc = gender_classifier(wv, sorted(seed_set), neutral_pool, C=C)
    if candidates is None:
        candidates = [w for w in wv.words[:20000] if w not in seed_set]
    cands = [w for w in candidates if w in wv]
    scores = clf.decision_function(wv.matrix(cands))
    extra = [w for w, s in zip(cands, scores, strict=True) if s > threshold]
    return sorted(seed_set | set(extra)), acc


def top_indirect_bias_pairs(
    wv: WordVectors, words: Sequence[str], g: np.ndarray, *, topn: int = 15, min_cos: float = 0.3
) -> pd.DataFrame:
    """Pares de palavras neutras cuja similaridade mais depende da direção de gênero (maior ``β``),
    entre pares com ``cos ≥ min_cos`` — versão orientada a dados da Fig. 3 do artigo."""
    ws = [w for w in words if w in wv]
    g = _unit(np.asarray(g, dtype=np.float32))
    U = np.stack([_unit(wv[w]) for w in ws])
    proj = U @ g
    perp = U - np.outer(proj, g)
    perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9
    cos = U @ U.T
    cos_perp = perp @ perp.T
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (cos - cos_perp) / cos
    rows = []
    n = len(ws)
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] >= min_cos and np.isfinite(beta[i, j]):
                rows.append(
                    (
                        ws[i],
                        ws[j],
                        float(cos[i, j]),
                        float(beta[i, j]),
                        float(proj[i]),
                        float(proj[j]),
                    )
                )
    df = pd.DataFrame(rows, columns=["w", "v", "cos", "β", "proj_g(w)", "proj_g(v)"]).sort_values(
        "β", ascending=False
    )
    return df.head(topn).reset_index(drop=True)
