import numpy as np
import pandas as pd
import streamlit as st

from replang.data.corpora import toy_corpus
from replang.models.cooccurrence import CountModel, one_hot
from replang.viz import cooccurrence_heatmap, scatter_words


def render():
    st.title("1 · Da coocorrência ao vetor: contagem → PPMI → SVD")
    st.markdown(
        "**Premissa basal** (hipótese distribucional): palavras que ocorrem em contextos parecidos têm significados "
        "parecidos. A representação mais simples é contar com que vizinhos cada palavra aparece."
    )
    default = "\n".join(" ".join(s) for s in toy_corpus())
    text = st.text_area("Corpus (uma sentença por linha)", default, height=180)
    sents = [line.split() for line in text.splitlines() if line.strip()]
    c1, c2, c3 = st.columns(3)
    window = c1.slider("janela (palavras de cada lado)", 1, 5, 2)
    weighting = c2.selectbox(
        "ponderação por distância",
        ["flat", "harmonic", "linear"],
        help="flat: 1; harmonic: 1/d (GloVe); linear: (janela−d+1)/janela (word2vec)",
    )
    dim = c3.slider("dimensões após SVD", 2, 20, 5)
    alpha = st.slider(
        "α do PPMI (context distribution smoothing; 1 = PMI clássico)", 0.5, 1.0, 0.75, 0.05
    )
    cm = CountModel(window=window, dim=dim, weighting=weighting, alpha=alpha).fit(sents)
    st.markdown(f"Vocabulário: **{len(cm.words)}** tipos · tokens: **{sum(map(len, sents))}**")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["one-hot vs. distribuição", "matriz de coocorrência", "PPMI", "SVD → vetores densos"]
    )
    with tab1:
        w = st.selectbox(
            "palavra", cm.words, index=cm.words.index("rei") if "rei" in cm.words else 0
        )
        oh = one_hot(cm.words, w)
        st.markdown(
            f"**one-hot** de *{w}* (|V| = {len(cm.words)}): todos os vetores são ortogonais — a similaridade entre quaisquer duas palavras é 0."
        )
        st.code(" ".join(str(int(x)) for x in oh))
        row = cm.counts_[cm.index[w]].toarray().ravel()
        st.markdown(
            "**linha da matriz de coocorrência** (quantas vezes cada palavra aparece na janela):"
        )
        st.dataframe(
            pd.DataFrame({"contexto": cm.words, "contagem": row})
            .query("contagem > 0")
            .sort_values("contagem", ascending=False)
            .T
        )
    with tab2:
        st.plotly_chart(
            cooccurrence_heatmap(
                cm.dense_counts(), "Contagens de coocorrência (janela = %d)" % window
            ),
            width="stretch",
        )
    with tab3:
        st.markdown(
            r"$\mathrm{PPMI}(w,c)=\max\!\left(0,\ \log\frac{P(w,c)}{P(w)\,P(c)^{\alpha}}\right)$ — realça associações *acima do acaso* e zera o resto."
        )
        st.plotly_chart(cooccurrence_heatmap(cm.dense_ppmi(), "PPMI"), width="stretch")
    with tab4:
        st.markdown(
            r"SVD truncada $M \approx U\Sigma V^\top$: usamos $U\Sigma^{1/2}$ como vetores (Levy et al., 2015)."
        )
        st.plotly_chart(
            scatter_words(cm.to_wordvectors(), cm.words, title="Vetores densos (PCA 2D)"),
            width="stretch",
        )
        wv = cm.to_wordvectors()
        q = st.selectbox(
            "vizinhos de",
            cm.words,
            index=cm.words.index("rei") if "rei" in cm.words else 0,
            key="q2",
        )
        st.table(pd.DataFrame(wv.most_similar(q, topn=5), columns=["palavra", "cosseno"]))
        st.markdown(
            f"Valores singulares (energia por dimensão): `{np.round(cm.sigma_, 2).tolist()}`"
        )
    with st.expander("O que observar"):
        st.markdown(
            """
* Com *one-hot*, **rei** e **rainha** são tão diferentes quanto **rei** e **banana**. A contagem de contextos já aproxima palavras com vizinhos parecidos.
* O PPMI corrige o viés das palavras frequentes (*o*, *a*, *e*) que coocorrem com tudo — o mesmo problema que motiva a **subamostragem** no word2vec.
* A SVD comprime |V| dimensões esparsas em poucas dimensões densas: é a **LSA** (Deerwester et al., 1990) e, por Levy & Goldberg (2014), o Skip-gram com amostragem negativa fatora implicitamente uma matriz PMI deslocada.
"""
        )
