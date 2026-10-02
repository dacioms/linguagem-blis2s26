import streamlit as st

from common import PATHS, local_models


def render():
    st.title("Representações de Linguagem")
    st.subheader(
        "Como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto"
    )
    st.markdown(
        """
Esta interface acompanha a apresentação (4h) e os notebooks do repositório. Cada página corresponde a
um bloco do roteiro e a um ou mais artigos:

| Página | Tema | Artigo(s) |
|---|---|---|
| 1 · Coocorrência → PPMI → SVD | a premissa distribucional e os modelos de contagem | Harris (1954), LSA/HAL, Levy & Goldberg (2014) |
| 2 · Treinar word2vec ao vivo | Skip-gram/CBOW, amostragem negativa, softmax hierárquico | Mikolov et al. (2013a, 2013b) |
| 3 · Vizinhos e similaridade | cosseno, vizinhos mais próximos, comparação entre modelos | Mikolov (2013a), Hartmann (2017) |
| 4 · Analogias | 3CosAdd / 3CosMul, benchmark PT-BR e EN por categoria | Mikolov (2013a), Hartmann (2017) |
| 5 · Projeções 2D | PCA / t-SNE de grupos de palavras (país → capital) | Mikolov (2013b) |
| 6 · Subpalavras | n-gramas de caracteres, OOV, morfologia | Bojanowski et al. (2017) |
| 7 · Avaliação | intrínseca (analogias) × extrínseca (POS tagging, similaridade de sentenças) | Hartmann et al. (2017) |
| 8 · Viés de gênero | direção de gênero, DirectBias, viés indireto, analogias geradas, hard/soft debias | Bolukbasi et al. (2016) |
| 9 · Contextual | biLM, uma palavra ↦ vários vetores, pesos por camada | Peters et al. (2018) |
| 10 · Contexto jurídico | embeddings gerais × treinados em decisões do STF: vizinhos, deslocamento de domínio, analogias jurídicas | `docs/juridico.md` |
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Embeddings disponíveis")
        st.markdown(
            "- **Wikipedia2Vec PT 100d** — pré-treinado na Wikipédia em português (50k palavras mais frequentes)"
        )
        st.markdown("- **GloVe 6B 50d** — inglês, para reproduzir Bolukbasi et al.")
        for name in local_models():
            st.markdown(f"- **{name}** — treinado localmente no corpus Machado de Assis")
        if not local_models():
            st.info(
                "Nenhum modelo local: execute `uv run replang train` para treinar sobre o corpus Machado de Assis."
            )
    with c2:
        st.markdown("#### Como usar na apresentação")
        st.markdown(
            """
1. Siga a ordem das páginas (ela espelha a ordem dos notebooks `notebooks/01…13`).
2. Em cada página há um painel *"O que observar"* ligando a demonstração ao artigo.
3. Os parâmetros na barra lateral (modelo, janela, k negativos, δ, λ…) permitem explorar
   ao vivo as perguntas da plateia.
"""
        )
    st.caption(f"Dados em `{PATHS.data}`. Código em `src/replang`. Notebooks em `notebooks/`.")
