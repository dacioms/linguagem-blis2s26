import streamlit as st

from common import lang_of, load_model, model_options
from replang.viz import scatter_words

PRESETS_PT = {
    "países → capitais (Fig. 2 de Mikolov 2013b)": {
        "países": [
            "brasil",
            "portugal",
            "frança",
            "itália",
            "espanha",
            "alemanha",
            "japão",
            "china",
            "argentina",
            "méxico",
        ],
        "capitais": [
            "brasília",
            "lisboa",
            "paris",
            "roma",
            "madri",
            "berlim",
            "tóquio",
            "pequim",
            "buenos",
            "cidade",
        ],
    },
    "família": {
        "feminino": ["mãe", "filha", "irmã", "avó", "tia", "esposa", "rainha"],
        "masculino": ["pai", "filho", "irmão", "avô", "tio", "marido", "rei"],
    },
    "verbos: infinitivo → particípio": {
        "infinitivo": ["comer", "falar", "andar", "dormir", "escrever", "ler"],
        "particípio": ["comido", "falado", "andado", "dormido", "escrito", "lido"],
    },
    "profissões e áreas": {
        "áreas": [
            "medicina",
            "engenharia",
            "direito",
            "música",
            "futebol",
            "cozinha",
            "enfermagem",
            "física",
        ],
        "profissões": [
            "médico",
            "engenheiro",
            "advogado",
            "músico",
            "jogador",
            "cozinheiro",
            "enfermeira",
            "físico",
        ],
    },
}
PRESETS_EN = {
    "countries → capitals": {
        "countries": [
            "france",
            "italy",
            "spain",
            "germany",
            "japan",
            "china",
            "russia",
            "portugal",
            "greece",
            "turkey",
        ],
        "capitals": [
            "paris",
            "rome",
            "madrid",
            "berlin",
            "tokyo",
            "beijing",
            "moscow",
            "lisbon",
            "athens",
            "ankara",
        ],
    },
    "family": {
        "female": ["mother", "daughter", "sister", "queen", "aunt", "wife"],
        "male": ["father", "son", "brother", "king", "uncle", "husband"],
    },
}


def render():
    st.title("5 · Projeções 2D: a estrutura linear dos embeddings")
    st.markdown(
        "PCA ou t-SNE de grupos de palavras. As setas ligam pares (ex.: país → capital): se forem aproximadamente paralelas, a relação é um **deslocamento vetorial** quase constante."
    )
    label = st.sidebar.selectbox("Embedding", model_options())
    wv = load_model(label)
    presets = PRESETS_EN if lang_of(label) == "en" else PRESETS_PT
    name = st.selectbox("conjunto", list(presets))
    groups = presets[name]
    method = st.radio("método", ["pca", "tsne"], horizontal=True)
    custom = st.text_input("ou digite palavras separadas por vírgula (sobrescreve o conjunto)", "")
    if custom.strip():
        groups = {"palavras": [w.strip().lower() for w in custom.split(",") if w.strip()]}
    keys = list(groups)
    arrows = list(zip(groups[keys[0]], groups[keys[1]], strict=False)) if len(keys) == 2 else []
    st.plotly_chart(
        scatter_words(wv, groups, method=method, arrows=arrows, title=f"{name} — {label}"),
        width="stretch",
    )
    with st.expander("O que observar"):
        st.markdown(
            """
* A PCA é linear e preserva a estrutura global (setas paralelas = relação linear); o **t-SNE** preserva vizinhanças locais e distorce distâncias globais — ótimo para *clusters*, péssimo para setas.
* Mikolov et al. (2013b, Fig. 2) usam exatamente esta figura para argumentar que o Skip-gram aprende relações *não supervisionadas* (país ↔ capital) como deslocamentos.
* Em 2D perdemos quase toda a informação de 100 dimensões: use as projeções como **ilustração**, não como evidência.
"""
        )
