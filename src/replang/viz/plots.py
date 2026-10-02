"""Figuras Plotly reutilizadas pelos notebooks e pela interface Streamlit.

Convenções: fundo neutro, paleta categórica fixa (``PALETTE``), rótulos em português, e cada
função devolve um ``plotly.graph_objects.Figure`` (chame ``.show()`` no notebook ou
``st.plotly_chart`` no app).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from replang.embedding import WordVectors

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#EECA3B",
]
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)


def projection_2d(
    wv: WordVectors,
    words: Sequence[str],
    *,
    method: str = "pca",
    seed: int = 0,
    perplexity: float = 15.0,
) -> pd.DataFrame:
    """Projeta ``words`` em 2D por PCA ou t-SNE; retorna DataFrame ``palavra, x, y``."""
    words = [w for w in words if w in wv]
    X = wv.matrix(words)
    if method == "pca":
        from sklearn.decomposition import PCA

        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        Z = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=min(perplexity, max(2, len(words) - 1)),
            init="pca",
        ).fit_transform(X)
    else:
        raise ValueError(method)
    return pd.DataFrame({"palavra": words, "x": Z[:, 0], "y": Z[:, 1]})


def scatter_words(
    wv: WordVectors,
    words: Sequence[str] | dict[str, Sequence[str]],
    *,
    method: str = "pca",
    title: str = "",
    arrows: Sequence[tuple[str, str]] = (),
    seed: int = 0,
    size: int = 10,
) -> go.Figure:
    """Dispersão 2D de palavras, opcionalmente agrupadas (``{grupo: [palavras]}``) e com setas
    entre pares (ex.: país → capital, como na Fig. 2 de Mikolov et al., 2013b)."""
    if isinstance(words, dict):
        groups = {g: [w for w in ws if w in wv] for g, ws in words.items()}
        flat = [w for ws in groups.values() for w in ws]
    else:
        groups = {"palavras": [w for w in words if w in wv]}
        flat = groups["palavras"]
    df = projection_2d(wv, flat, method=method, seed=seed)
    df["grupo"] = [next(g for g, ws in groups.items() if w in ws) for w in df.palavra]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="palavra",
        color="grupo",
        color_discrete_sequence=PALETTE,
        title=title,
    )
    fig.update_traces(textposition="top center", marker=dict(size=size))
    pos = {r.palavra: (r.x, r.y) for r in df.itertuples()}
    for a, b in arrows:
        if a in pos and b in pos:
            fig.add_annotation(
                x=pos[b][0],
                y=pos[b][1],
                ax=pos[a][0],
                ay=pos[a][1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor="rgba(80,80,80,0.5)",
                arrowwidth=1.2,
            )
    fig.update_layout(
        **_LAYOUT,
        xaxis_title=f"{method.upper()} 1",
        yaxis_title=f"{method.upper()} 2",
        showlegend=len(groups) > 1,
    )
    return fig


def analogy_figure(
    wv: WordVectors,
    a: str,
    b: str,
    c: str,
    d: str | None = None,
    *,
    extra: Sequence[str] = (),
    title: str | None = None,
) -> go.Figure:
    """Paralelogramo ``a → b`` e ``c → d`` em 2D (PCA das 4 palavras + extras)."""
    d = d or wv.analogy(a, b, c, topn=1)[0][0]
    words = [a, b, c, d, *extra]
    fig = scatter_words(
        wv, words, arrows=[(a, b), (c, d)], title=title or f"{a} : {b} :: {c} : {d}"
    )
    return fig


def neighbors_bar(
    pairs: Sequence[tuple[str, float]], title: str = "", color: str = PALETTE[0]
) -> go.Figure:
    words = [p[0] for p in pairs][::-1]
    sims = [p[1] for p in pairs][::-1]
    fig = go.Figure(
        go.Bar(
            x=sims,
            y=words,
            orientation="h",
            marker_color=color,
            text=[f"{s:.2f}" for s in sims],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_LAYOUT, title=title, xaxis_title="cosseno", height=max(300, 28 * len(words) + 80)
    )
    return fig


def cooccurrence_heatmap(
    df: pd.DataFrame, title: str = "Matriz de coocorrência", zmax: float | None = None
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            text=np.round(df.values, 1),
            texttemplate="%{text}" if df.shape[0] <= 25 else None,
        )
    )
    fig.update_layout(
        **_LAYOUT,
        title=title,
        height=max(400, 22 * df.shape[0] + 100),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def training_curve(
    history: Sequence[dict],
    x: str = "epoch",
    y: str = "loss",
    title: str = "Curva de treino",
    color: str = PALETTE[0],
) -> go.Figure:
    df = pd.DataFrame(history)
    fig = px.line(df, x=x, y=y, markers=True, title=title, color_discrete_sequence=[color])
    fig.update_layout(**_LAYOUT)
    return fig


def pca_variance_figure(
    explained: Sequence[float],
    baseline: Sequence[float] | None = None,
    title: str = "Variância explicada (PCA das diferenças dos pares)",
) -> go.Figure:
    """Reprodução da Fig. 6 de Bolukbasi et al.: componente dominante vs. baseline aleatório."""
    fig = go.Figure()
    k = np.arange(1, len(explained) + 1)
    fig.add_bar(x=k, y=explained, name="pares de gênero", marker_color=PALETTE[0])
    if baseline is not None:
        fig.add_bar(
            x=np.arange(1, len(baseline) + 1),
            y=baseline,
            name="pares aleatórios",
            marker_color=PALETTE[8],
        )
    fig.update_layout(
        **_LAYOUT,
        title=title,
        xaxis_title="componente principal",
        yaxis_title="fração da variância",
        barmode="group",
    )
    return fig


def bias_axis_figure(
    df: pd.DataFrame,
    *,
    title: str = "Projeção na direção de gênero",
    axis_label: str = "← masculino (ele)      feminino (ela) →",
    color_col: str = "grupo",
    height: int | None = None,
) -> go.Figure:
    """Strip plot: cada palavra como ponto no eixo ``proj_g`` (Figs. 1/3 do artigo)."""
    df = df.sort_values("proj_g").reset_index(drop=True)
    df["_y"] = np.arange(len(df))
    fig = px.scatter(
        df,
        x="proj_g",
        y="_y",
        text="palavra",
        color=color_col if color_col in df else None,
        color_discrete_sequence=PALETTE,
        title=title,
        hover_data=["proj_g"],
    )
    fig.update_traces(textposition="middle right", marker=dict(size=9))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        **_LAYOUT,
        xaxis_title=axis_label,
        yaxis=dict(visible=False),
        height=height or max(400, 18 * len(df) + 120),
    )
    return fig


def bias_scatter_two_embeddings(p1: pd.Series, p2: pd.Series, name1: str, name2: str) -> go.Figure:
    """Fig. 4 do artigo: viés das mesmas profissões em dois embeddings diferentes."""
    common = p1.index.intersection(p2.index)
    df = pd.DataFrame({"palavra": common, name1: p1[common].values, name2: p2[common].values})
    from scipy.stats import spearmanr

    rho = spearmanr(df[name1], df[name2]).correlation
    fig = px.scatter(
        df,
        x=name1,
        y=name2,
        hover_name="palavra",
        title=f"Viés de gênero por palavra em dois embeddings (Spearman ρ = {rho:.2f})",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(**_LAYOUT)
    return fig


def ngram_bar(
    pairs: Sequence[tuple[str, float]], title: str = "Importância dos n-gramas (cosseno ao remover)"
) -> go.Figure:
    grams = [p[0] for p in pairs]
    vals = [1 - p[1] for p in pairs]
    fig = go.Figure(go.Bar(x=grams, y=vals, marker_color=PALETTE[1]))
    fig.update_layout(
        **_LAYOUT, title=title, xaxis_title="n-grama removido", yaxis_title="1 − cos(w, w \\ g)"
    )
    return fig


def layer_weights_figure(
    weights: dict[str, Sequence[float]],
    layer_names: Sequence[str] = ("camada 0 (tokens)", "camada 1", "camada 2"),
    title: str = "Pesos softmax das camadas do biLM por tarefa (Fig. 2 de Peters et al.)",
) -> go.Figure:
    fig = go.Figure()
    for i, ln in enumerate(layer_names):
        fig.add_bar(
            name=ln,
            x=list(weights.keys()),
            y=[w[i] for w in weights.values()],
            marker_color=PALETTE[i],
        )
    fig.update_layout(**_LAYOUT, barmode="stack", title=title, yaxis_title="peso normalizado")
    return fig
