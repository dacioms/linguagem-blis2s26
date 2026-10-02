import pandas as pd
import streamlit as st

from common import load_model, model_options
from replang.data.analogies import load_analogies
from replang.data.corpora import load_macmorpho
from replang.eval import MINI_STS_PT, evaluate_analogies, pos_tagging_eval, sentence_similarity_eval


@st.cache_data(show_spinner="avaliando…")
def _evaluate(label: str, n_train: int, n_test: int, max_per_cat: int) -> dict:
    wv = load_model(label)
    aset = load_analogies("pt-br")
    an = evaluate_analogies(wv, aset, restrict=30000, max_per_category=max_per_cat).set_index(
        "categoria"
    )
    train = load_macmorpho(split="train")
    test = load_macmorpho(split="test", limit_sentences=n_test)
    pos = pos_tagging_eval(wv, train, test, max_train_tokens=n_train)
    sts = sentence_similarity_eval(wv, MINI_STS_PT)
    return {
        "modelo": label,
        "analogias sintáticas": an.loc["TOTAL sintática", "acurácia"],
        "analogias semânticas": an.loc["TOTAL semântica", "acurácia"],
        "analogias total": an.loc["TOTAL", "acurácia"],
        "cobertura analogias": an.loc["TOTAL", "cobertas"] / an.loc["TOTAL", "n"],
        "POS acurácia": pos["acurácia"],
        "POS taxa OOV": pos["taxa_oov"],
        "STS Pearson (mini)": sts["pearson"],
        "STS MSE (mini)": sts["mse"],
    }


def render():
    st.title("7 · Avaliação intrínseca × extrínseca (Hartmann et al., 2017)")
    st.markdown(
        "Hartmann et al. treinaram 31 modelos (word2vec, wang2vec, GloVe, fastText; 50–1000d) e compararam **analogias** "
        "(intrínseca) com **POS tagging** no Mac-Morpho e **similaridade de sentenças** no ASSIN (extrínsecas). "
        "Conclusão: os rankings não concordam — analogias não são bom proxy. Reproduza a comparação com os modelos disponíveis."
    )
    opts = [o for o in model_options(("pt",))]
    sel = st.multiselect("modelos", opts, default=opts[:2])
    c1, c2, c3 = st.columns(3)
    n_train = c1.select_slider("tokens de treino (POS)", [10000, 30000, 60000, 120000], value=30000)
    n_test = c2.select_slider("sentenças de teste (POS)", [200, 500, 1000], value=500)
    max_per_cat = c3.select_slider("questões por categoria", [100, 300, 1000], value=300)
    if st.button("Avaliar modelos selecionados", type="primary") and sel:
        rows = [_evaluate(label, n_train, n_test, max_per_cat) for label in sel]
        df = pd.DataFrame(rows).set_index("modelo")
        st.dataframe(df.style.format("{:.3f}"), width="stretch")
        ranks = df[["analogias total", "POS acurácia", "STS Pearson (mini)"]].rank(ascending=False)
        st.markdown("**Ranking por métrica (1 = melhor)** — observe se as colunas concordam:")
        st.dataframe(ranks.astype(int), width="stretch")
    with st.expander("O que observar"):
        st.markdown(
            """
* O **POS tagging** depende de morfossintaxe e da taxa de OOV: fastText deveria ajudar, mas Hartmann et al. observaram o contrário (Tab. 3) — e sugerem o tokenizador (clíticos) como causa.
* O mini-STS aqui é **ilustrativo** (30 pares autorais); o ASSIN tem milhares de pares anotados. Veja o notebook 07 para o conector do ASSIN.
* Rankings discordantes entre colunas é exatamente o achado central do artigo: *escolha o embedding pela tarefa*.
"""
        )
