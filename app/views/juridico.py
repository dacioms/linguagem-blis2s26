import pandas as pd
import streamlit as st

from common import load_model, local_models
from replang.data.legal import LEGAL, neighborhood_overlap


@st.cache_resource(show_spinner="carregando modelos…")
def _models():
    pt = load_model("Wikipedia2Vec PT 100d (Wikipédia PT)")
    legal_label = next((k for k in local_models() if k.startswith("legal_sg100")), None)
    leg = load_model(legal_label) if legal_label else None
    return pt, leg


def render():
    st.title("10 · Contexto jurídico: geral × jurídico")
    st.markdown(
        "Embeddings treinados em decisões do STF (RulingBR) comparados ao Wikipedia2Vec PT. "
        "Veja [`docs/juridico.md`](https://github.com/dacioms/linguagem-blis2s26/blob/main/docs/juridico.md) para o debate completo."
    )
    pt, leg = _models()
    if leg is None:
        st.info(
            "Modelo jurídico não encontrado — execute `uv run replang prepare && uv run replang train legal`."
        )
        return
    pt.name, leg.name = "Wikipédia", "STF"
    tabs = st.tabs(
        [
            "vizinhos geral × jurídico",
            "deslocamento de domínio",
            "analogias jurídicas",
            "glossário polissêmico",
        ]
    )
    with tabs[0]:
        w = st.text_input("palavra", "sentença").strip().lower()
        c1, c2 = st.columns(2)
        for col, wv in ((c1, pt), (c2, leg)):
            with col:
                st.markdown(f"**{wv.name}**")
                if w in wv:
                    st.table(
                        pd.DataFrame(wv.most_similar(w, topn=12), columns=["vizinho", "cosseno"])
                    )
                else:
                    st.warning("fora do vocabulário")
    with tabs[1]:
        words = st.multiselect("palavras", LEGAL.probe_words, default=LEGAL.probe_words[:12])
        if words:
            df = neighborhood_overlap(pt, leg, words, k=10)
            st.dataframe(df.style.format({"jaccard@k": "{:.2f}"}), width="stretch")
            st.bar_chart(df.set_index("palavra")["jaccard@k"])
            st.caption(
                "Jaccard entre os 10 vizinhos em cada embedding: ≈ 0 significa que a palavra muda de sentido ao entrar no Direito."
            )
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        a = c1.text_input("a", "autor").lower().strip()
        b = c2.text_input("b", "réu").lower().strip()
        c = c3.text_input("c", "apelante").lower().strip()
        for wv in (pt, leg):
            st.markdown(
                f"**{wv.name}**: "
                + (
                    ", ".join(f"{x} ({s:.2f})" for x, s in wv.analogy(a, b, c, topn=5))
                    if wv.has(a, b, c)
                    else "OOV"
                )
            )
        st.markdown(
            "Sugestões do léxico: "
            + "; ".join(f"`{a} : {b} :: {c} : {d}`" for a, b, c, d in LEGAL.analogies[:8])
        )
    with tabs[3]:
        st.dataframe(
            pd.DataFrame(
                LEGAL.polysemous, columns=["palavra", "sentido geral", "sentido jurídico"]
            ),
            width="stretch",
            height=500,
        )
    with st.expander("O que observar"):
        st.markdown(
            """
* Palavras técnicas (*sentença, título, pena, trânsito, remédio*) têm vizinhanças **disjuntas** nos dois embeddings: o vetor geral "não sabe Direito".
* Analogias de papéis (*autor : réu :: apelante : apelado*) só funcionam no modelo de domínio.
* Isso é o argumento de Hartmann et al. levado ao limite: escolha e avalie o embedding **na tarefa e no domínio**.
"""
        )
