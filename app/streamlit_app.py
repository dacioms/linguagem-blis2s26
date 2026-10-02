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
    projecoes,
    subpalavras,
    treino,
    vies,
    vizinhos,
)

pages = [
    st.Page(inicio.render, title="Início", icon="🏠", default=True),
    st.Page(coocorrencia.render, title="1 · Coocorrência → PPMI → SVD", icon="🧮"),
    st.Page(treino.render, title="2 · Treinar word2vec ao vivo", icon="⚙️"),
    st.Page(vizinhos.render, title="3 · Vizinhos e similaridade", icon="🔍"),
    st.Page(analogias.render, title="4 · Analogias", icon="🧩"),
    st.Page(projecoes.render, title="5 · Projeções 2D", icon="🗺️"),
    st.Page(subpalavras.render, title="6 · Subpalavras (fastText)", icon="🔤"),
    st.Page(avaliacao.render, title="7 · Avaliação intrínseca × extrínseca", icon="📏"),
    st.Page(vies.render, title="8 · Viés de gênero e debias", icon="⚖️"),
    st.Page(contextual.render, title="9 · Representações contextuais (ELMo)", icon="🌀"),
]
nav = st.navigation(pages)
nav.run()
