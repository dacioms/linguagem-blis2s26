"""Avaliação **intrínseca** por analogias (Mikolov et al., 2013a; Hartmann et al., 2017, Tabela 2).

Para cada questão ``a : b :: c : d`` calculamos ``argmax_y cos(y, b − a + c)`` (3CosAdd), excluindo
``a, b, c`` da busca, e contamos acerto se ``y == d``. Questões com palavras fora do vocabulário
são descartadas (como no ``compute-accuracy.c``); reportamos a cobertura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from replang.data.analogies import AnalogySet
from replang.embedding import WordVectors


def solve_analogy(
    wv: WordVectors,
    a: str,
    b: str,
    c: str,
    *,
    method: str = "add",
    topn: int = 5,
    restrict: int | None = None,
):
    return wv.analogy(a, b, c, method=method, topn=topn, restrict=restrict)


def evaluate_analogies(
    wv: WordVectors,
    aset: AnalogySet,
    *,
    method: str = "add",
    restrict: int | None = None,
    max_per_category: int | None = None,
    batch: int = 1024,
    seed: int = 0,
) -> pd.DataFrame:
    """Acurácia por categoria. Colunas: categoria, tipo, n, cobertas (no vocabulário), avaliadas
    (após a amostragem ``max_per_category``), acertos, acurácia (= acertos/avaliadas)."""
    unit = wv.unit if restrict is None else wv.unit[:restrict]
    n_search = len(unit)
    rng = np.random.default_rng(seed)
    rows = []
    for cat, qs in aset.categories.items():
        qs_in = [q for q in qs if all(w in wv.index and wv.index[w] < n_search for w in q)]
        n_covered = len(qs_in)
        if max_per_category and len(qs_in) > max_per_category:
            idx = rng.choice(len(qs_in), max_per_category, replace=False)
            qs_in = [qs_in[i] for i in sorted(idx)]
        correct = 0
        for b0 in range(0, len(qs_in), batch):
            chunk = qs_in[b0 : b0 + batch]
            ia = np.array([wv.index[q[0]] for q in chunk])
            ib = np.array([wv.index[q[1]] for q in chunk])
            ic = np.array([wv.index[q[2]] for q in chunk])
            idd = np.array([wv.index[q[3]] for q in chunk])
            if method == "add":
                q = unit[ib] - unit[ia] + unit[ic]
                q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
                sims = q @ unit.T
            else:  # 3CosMul
                sa, sb, sc = (
                    (unit[ia] @ unit.T + 1) / 2,
                    (unit[ib] @ unit.T + 1) / 2,
                    (unit[ic] @ unit.T + 1) / 2,
                )
                sims = sb * sc / (sa + 1e-3)
            r = np.arange(len(chunk))
            sims[r, ia] = sims[r, ib] = sims[r, ic] = -np.inf
            pred = sims.argmax(1)
            correct += int((pred == idd).sum())
        rows.append(
            {
                "categoria": cat,
                "tipo": "semântica" if aset.is_semantic(cat) else "sintática",
                "n": len(qs),
                "cobertas": n_covered,
                "avaliadas": len(qs_in),
                "acertos": correct,
                "acurácia": correct / len(qs_in) if qs_in else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    tot = {
        "categoria": "TOTAL",
        "tipo": "",
        "n": df.n.sum(),
        "cobertas": df.cobertas.sum(),
        "acertos": df.acertos.sum(),
    }
    tot["acurácia"] = tot["acertos"] / tot["cobertas"] if tot["cobertas"] else np.nan
    for tipo in ("semântica", "sintática"):
        sub = df[df.tipo == tipo]
        rows.append(
            {
                "categoria": f"TOTAL {tipo}",
                "tipo": tipo,
                "n": sub.n.sum(),
                "cobertas": sub.cobertas.sum(),
                "acertos": sub.acertos.sum(),
                "acurácia": sub.acertos.sum() / sub.cobertas.sum()
                if sub.cobertas.sum()
                else np.nan,
            }
        )
    rows.append(tot)
    return pd.DataFrame(rows)


def compare_models(models: dict[str, WordVectors], aset: AnalogySet, **kw) -> pd.DataFrame:
    """Tabela modelo × {semântica, sintática, total} (formato da Tabela 2 de Hartmann et al.)."""
    out = []
    for name, wv in models.items():
        df = evaluate_analogies(wv, aset, **kw).set_index("categoria")
        out.append(
            {
                "modelo": name,
                "dim": wv.dim,
                "vocab": len(wv),
                "sintática": df.loc["TOTAL sintática", "acurácia"],
                "semântica": df.loc["TOTAL semântica", "acurácia"],
                "total": df.loc["TOTAL", "acurácia"],
                "cobertura": df.loc["TOTAL", "cobertas"] / df.loc["TOTAL", "n"],
            }
        )
    return pd.DataFrame(out)
