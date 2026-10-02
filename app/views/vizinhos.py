import pandas as pd
import streamlit as st

from common import load_model, model_options
from replang.viz import neighbors_bar


def render():
    st.title("3 · Vizinhos mais próximos e similaridade")
    st.markdown(
        "Similaridade = **cosseno** entre vetores (Bolukbasi et al., §3: com vetores unitários, cosseno = produto interno). "
        "Compare como modelos diferentes (corpus, algoritmo, dimensão) organizam a vizinhança de uma palavra."
    )
    opts = model_options()
    sel = st.multiselect(
        "modelos a comparar", opts, default=opts[:1] + [o for o in opts if "sg100 " in o][:1]
    )
    word = st.text_input("palavra", "rei").strip().lower()
    topn = st.slider("vizinhos", 5, 30, 10)
    cols = st.columns(max(1, len(sel)))
    for col, label in zip(cols, sel, strict=False):
        wv = load_model(label)
        with col:
            st.markdown(f"**{label}**")
            if word not in wv:
                st.warning("fora do vocabulário")
                continue
            st.plotly_chart(
                neighbors_bar(wv.most_similar(word, topn=topn), title=""), width="stretch"
            )
    st.markdown("#### Similaridade entre duas palavras")
    c1, c2 = st.columns(2)
    a = c1.text_input("palavra A", "rei").lower().strip()
    b = c2.text_input("palavra B", "rainha").lower().strip()
    rows = []
    for label in sel:
        wv = load_model(label)
        rows.append(
            {
                "modelo": label,
                "cos(A,B)": wv.similarity(a, b) if a in wv and b in wv else float("nan"),
            }
        )
    st.table(pd.DataFrame(rows))
    with st.expander("O que observar"):
        st.markdown(
            """
* Em corpora diferentes a vizinhança muda: na **Wikipédia** *rei* ≈ *monarca, soberano, trono*; em **Machado de Assis** aparecem personagens e vocabulário oitocentista.
* Vizinhos misturam relações **paradigmáticas** (sinônimos, co-hipônimos: *gato–cachorro*) e **sintagmáticas** (coocorrentes: *café–xícara*). O cosseno não distingue os dois.
* A similaridade é *relativa* ao modelo: não compare valores absolutos entre modelos, compare **rankings**.
"""
        )
