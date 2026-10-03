"""Interface interativa — Representações de Linguagem (BLIS 2S26).

Execute: ``uv run replang app`` ou ``uv run streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

st.set_page_config(page_title="Representações de Linguagem", page_icon="🧭", layout="wide")

from views import (  # noqa: E402
    analogias,
    avaliacao,
    contextual,
    coocorrencia,
    inicio,
    juridico,
    projecoes,
    subpalavras,
    treino,
    vies,
    vizinhos,
)

# `url_path` explícito: todas as páginas usam a função `render`, e o Streamlit inferiria o mesmo
# pathname ("render") para todas — o que dispara "Multiple Pages specified with URL pathname render".
pages = [
    st.Page(inicio.render, title="Início", icon=":material/home:", url_path="inicio", default=True),
    st.Page(coocorrencia.render, title="1 · Coocorrência → PPMI → SVD", icon=":material/grid_on:", url_path="coocorrencia"),
    st.Page(treino.render, title="2 · Treinar word2vec ao vivo", icon=":material/model_training:", url_path="treino"),
    st.Page(vizinhos.render, title="3 · Vizinhos e similaridade", icon=":material/search:", url_path="vizinhos"),
    st.Page(analogias.render, title="4 · Analogias", icon=":material/extension:", url_path="analogias"),
    st.Page(projecoes.render, title="5 · Projeções 2D", icon=":material/map:", url_path="projecoes"),
    st.Page(subpalavras.render, title="6 · Subpalavras (fastText)", icon=":material/abc:", url_path="subpalavras"),
    st.Page(avaliacao.render, title="7 · Avaliação intrínseca × extrínseca", icon=":material/straighten:", url_path="avaliacao"),
    st.Page(vies.render, title="8 · Viés de gênero e debias", icon=":material/balance:", url_path="vies"),
    st.Page(contextual.render, title="9 · Representações contextuais (ELMo)", icon=":material/hub:", url_path="contextual"),
    st.Page(juridico.render, title="10 · Contexto jurídico", icon=":material/gavel:", url_path="juridico"),
]
nav = st.navigation(pages)
nav.run()
