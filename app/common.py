"""Utilidades compartilhadas pelas páginas do app (carregadores com cache)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from replang.config import PATHS  # noqa: E402
from replang.embedding import WordVectors  # noqa: E402

PRETRAINED = {
    "Wikipedia2Vec PT 100d (Wikipédia PT)": ("pt", None),
    "GloVe 6B 50d (inglês)": ("en", None),
}


def local_models() -> dict[str, Path]:
    out = {}
    for p in sorted(PATHS.models.glob("*.npz")):
        meta = p.with_suffix(".json")
        label = p.stem
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                label = f"{p.stem} ({m.get('algo', '?')}, {m.get('dim', '?')}d, Machado)"
            except Exception:  # noqa: BLE001
                pass
        out[label] = p
    return out


def model_options(langs: tuple[str, ...] = ("pt", "en")) -> list[str]:
    opts = [k for k, (lang, _) in PRETRAINED.items() if lang in langs]
    if "pt" in langs:
        opts += list(local_models().keys())
    return opts


@st.cache_resource(show_spinner="carregando embeddings…")
def load_model(label: str, top_n: int = 50000) -> WordVectors:
    if label in PRETRAINED:
        from replang.data.embeddings import load_glove_en, load_ptwiki

        lang = PRETRAINED[label][0]
        return load_ptwiki(top_n) if lang == "pt" else load_glove_en(top_n)
    path = local_models()[label]
    wv = WordVectors.load(path)
    wv.name = label
    return wv


def lang_of(label: str) -> str:
    return PRETRAINED[label][0] if label in PRETRAINED else "pt"


@st.cache_resource(show_spinner="carregando fastText…")
def load_fasttext_model():
    from replang.models.trainers import load_fasttext_model

    return load_fasttext_model("machado_ft100")


@st.cache_resource(show_spinner="carregando biLM…")
def load_bilm():
    from replang.models.bilm_jax import HAS_JAX, BiLM

    p = PATHS.models / "machado_bilm.pkl"
    if not HAS_JAX or not p.exists():
        return None, None
    return BiLM.load(p)


@st.cache_resource
def lexicon(lang: str):
    from replang.data.lexicons import EN, PT

    return PT.get() if lang == "pt" else EN.get()


def paper_ref(key: str) -> str:
    refs = {
        "mikolov13a": "Mikolov et al. (2013a) — *Efficient Estimation of Word Representations in Vector Space*",
        "mikolov13b": "Mikolov et al. (2013b) — *Distributed Representations of Words and Phrases and their Compositionality*",
        "glove": "Pennington, Socher & Manning (2014) — *GloVe*",
        "fasttext": "Bojanowski et al. (2017) — *Enriching Word Vectors with Subword Information*",
        "hartmann": "Hartmann et al. (2017) — *Portuguese Word Embeddings*",
        "bolukbasi": "Bolukbasi et al. (2016) — *Man is to Computer Programmer as Woman is to Homemaker?*",
        "elmo": "Peters et al. (2018) — *Deep contextualized word representations*",
        "doc2vec": "Le & Mikolov (2014) — *Distributed Representations of Sentences and Documents*",
    }
    return refs[key]


def sidebar_model(key: str = "model", langs=("pt", "en"), default_index: int = 0) -> WordVectors:
    label = st.sidebar.selectbox("Embedding", model_options(langs), index=default_index, key=key)
    wv = load_model(label)
    st.sidebar.caption(f"{len(wv):,} palavras × {wv.dim} dimensões")
    return wv
