import pandas as pd
import streamlit as st

from common import lang_of, lexicon, load_model, model_options
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
from replang.viz import analogy_figure


@st.cache_data(show_spinner="avaliando analogias…")
def _eval(label: str, lang: str, method: str, restrict: int, max_per_cat: int) -> pd.DataFrame:
    wv = load_model(label)
    aset = load_analogies("en" if lang == "en" else "pt-br")
    return evaluate_analogies(
        wv, aset, method=method, restrict=restrict, max_per_category=max_per_cat
    )


def render():
    st.title("4 · Analogias: *a* está para *b* assim como *c* está para ?")
    st.markdown(
        r"3CosAdd: $\arg\max_y \cos(y,\ b - a + c)$ (Mikolov et al., 2013a). 3CosMul (Levy & Goldberg, 2014): "
        r"$\arg\max_y \frac{\cos(y,b)\cos(y,c)}{\cos(y,a)+\epsilon}$ — penaliza candidatos próximos de *a*."
    )
    label = st.sidebar.selectbox("Embedding", model_options())
    wv = load_model(label)
    lang = lang_of(label)
    lex = lexicon(lang)
    defaults = ("man", "king", "woman") if lang == "en" else ("homem", "rei", "mulher")
    c1, c2, c3 = st.columns(3)
    a = c1.text_input("a", defaults[0]).lower().strip()
    b = c2.text_input("b", defaults[1]).lower().strip()
    c = c3.text_input("c", defaults[2]).lower().strip()
    method = st.radio("método", ["add", "mul"], horizontal=True)
    excl = st.checkbox(
        "excluir palavras específicas de gênero dos candidatos (útil para profissões)", value=False
    )
    if all(w in wv for w in (a, b, c)):
        res = wv.analogy(
            a, b, c, method=method, topn=8, exclude_words=lex.gender_specific if excl else ()
        )
        st.table(pd.DataFrame(res, columns=["candidato d", "pontuação"]))
        st.plotly_chart(analogy_figure(wv, a, b, c, res[0][0]), width="stretch")
    else:
        st.warning("alguma palavra está fora do vocabulário")

    st.markdown(
        "#### Benchmark por categoria (Mikolov 2013a / LX-4WAnalogies PT-BR, usado por Hartmann et al. 2017)"
    )
    cc1, cc2, cc3 = st.columns(3)
    restrict = cc1.select_slider(
        "restringir busca às N palavras mais frequentes", [10000, 20000, 30000, 50000], value=30000
    )
    max_per_cat = cc2.select_slider(
        "questões por categoria (amostra)", [100, 300, 1000, 5000], value=300
    )
    if cc3.button("Avaliar", type="primary"):
        df = _eval(label, lang, method, restrict, max_per_cat)
        st.dataframe(df.style.format({"acurácia": "{:.1%}"}), width="stretch")
        tot = df[df.categoria.str.startswith("TOTAL")]
        st.bar_chart(tot.set_index("categoria")["acurácia"])
    with st.expander("O que observar"):
        st.markdown(
            """
* A famosa `rei − homem + mulher ≈ rainha` funciona, mas a **cobertura** importa: questões com palavras fora do vocabulário são descartadas (como no `compute-accuracy.c`).
* Categorias **semânticas** (capitais, moedas, família) × **sintáticas** (plural, comparativo, tempo verbal): modelos com subpalavras (fastText) e ordem (wang2vec) vão melhor nas sintáticas; GloVe e Skip-gram nas semânticas (Hartmann et al., Tabela 2).
* Hartmann et al. concluem que analogias **não** predizem o desempenho em tarefas reais — veja a página 7.
"""
        )
