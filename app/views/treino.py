import numpy as np
import pandas as pd
import streamlit as st

from replang.data.corpora import synthetic_gender_corpus, toy_corpus
from replang.models import HuffmanTree, Word2Vec
from replang.viz import scatter_words, training_curve


def render():
    st.title("2 · Treinar word2vec ao vivo (implementação numpy)")
    st.markdown(
        "Skip-gram e CBOW de Mikolov et al. (2013a) com **amostragem negativa** ou **softmax hierárquico** "
        "(2013b), implementados em `replang.models.word2vec_np`. Ajuste os hiperparâmetros e treine."
    )
    corpus_name = st.sidebar.radio(
        "corpus", ["toy (18 frases × 20)", "sintético de gênero (4000 frases)"]
    )
    sents = (
        toy_corpus() * 20
        if corpus_name.startswith("toy")
        else synthetic_gender_corpus(4000, bias=0.85)
    )
    c1, c2, c3, c4 = st.columns(4)
    arch = c1.selectbox("arquitetura", ["Skip-gram", "CBOW"])
    obj = c2.selectbox("objetivo", ["amostragem negativa", "softmax hierárquico"])
    dim = c3.slider("dimensões", 5, 100, 30)
    window = c4.slider("janela", 1, 5, 2)
    c5, c6, c7, c8 = st.columns(4)
    negative = c5.slider("k negativos", 1, 20, 5)
    epochs = c6.slider("épocas", 1, 40, 15)
    alpha = c7.select_slider("taxa de aprendizado", [0.005, 0.01, 0.025, 0.05], value=0.025)
    sample = c8.select_slider("subamostragem t", [0, 1e-5, 1e-4, 1e-3, 1e-2], value=0)
    if st.button("Treinar", type="primary"):
        m = Word2Vec(
            dim=dim,
            window=window,
            sg=int(arch == "Skip-gram"),
            hs=int(obj.startswith("softmax")),
            negative=negative,
            epochs=epochs,
            alpha=alpha,
            sample=sample,
            batch_size=64,
        )
        bar = st.progress(0.0, "treinando…")
        log = []

        def cb(r):
            if r.get("end_epoch"):
                log.append(r)
                bar.progress(
                    (r["epoch"] + 1) / epochs,
                    f"época {r['epoch'] + 1}/{epochs} — perda {r['loss']:.3f}",
                )

        m.train(sents, callback=cb)
        st.session_state["w2v_model"] = m
    m = st.session_state.get("w2v_model")
    if m is None:
        return
    wv = m.to_wordvectors()
    st.plotly_chart(training_curve(m.history, title="perda média por época"), width="stretch")
    st.markdown(f"Vocabulário {len(wv)} · pares de treino/época ≈ {m.history[-1]['pairs']:,}")
    col1, col2 = st.columns(2)
    with col1:
        q = st.selectbox("vizinhos de", wv.words, index=wv.words.index("rei") if "rei" in wv else 0)
        st.table(pd.DataFrame(wv.most_similar(q, topn=8), columns=["palavra", "cosseno"]))
    with col2:
        words = wv.words[:60]
        st.plotly_chart(
            scatter_words(wv, words, title="PCA dos vetores aprendidos"), width="stretch"
        )
    if m.hs:
        st.markdown("#### Árvore de Huffman (softmax hierárquico)")
        tree: HuffmanTree = m.tree
        df = pd.DataFrame(
            {
                "palavra": wv.words,
                "freq": m.vocab.counts.astype(int),
                "código": ["".join(map(str, c)) for c in tree.codes],
            }
        )
        st.dataframe(df.head(20), width="stretch")
        st.markdown(
            f"comprimento médio de código ponderado por frequência: **{tree.average_code_length(m.vocab.counts):.2f}** bits vs. log₂|V| = {np.log2(len(wv)):.2f}"
        )
    else:
        st.markdown("#### Distribuição de ruído $U(w)^{3/4}$ usada nos negativos")
        st.bar_chart(
            pd.DataFrame({"palavra": wv.words[:25], "P_n(w)": m.noise[:25]}).set_index("palavra")
        )
    with st.expander("O que observar"):
        st.markdown(
            """
* **Skip-gram** gera um par por palavra de contexto (mais pares, mais lento, melhor para palavras raras); **CBOW** faz a média do contexto (mais rápido, suaviza).
* Com **amostragem negativa**, a saída `W_out` tem um vetor por palavra; com **softmax hierárquico**, um vetor por nó interno da árvore de Huffman (|V|−1 nós) e o custo cai de |V| para ~log₂|V|.
* Aumentar *t* da subamostragem descarta mais ocorrências de palavras frequentes (*o*, *a*, *e*): acelera e melhora palavras raras (Mikolov 2013b, §2.3).
"""
        )
