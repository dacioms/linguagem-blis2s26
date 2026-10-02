import numpy as np
import pandas as pd
import streamlit as st

from common import load_bilm, load_model, local_models
from replang.models.bilm_jax import elmo_mix
from replang.utils.text import tokenize

DEFAULT = """ele sentou no banco da praça para descansar
ela foi ao banco sacar o dinheiro da conta
o banco de madeira ficava debaixo da árvore
o gerente do banco negou o empréstimo
a manga da camisa estava rasgada
comi uma manga madura e doce"""


def render():
    st.title("9 · Representações contextuais: uma palavra, vários vetores (ELMo-lite)")
    st.markdown(
        "Peters et al. (2018): o vetor de um *token* é função de **toda a sentença**, calculado por um modelo de linguagem "
        "bidirecional (biLM) de 2 camadas; a tarefa aprende uma mistura $\\gamma \\sum_j s_j h_j$ das camadas."
    )
    model, vocab = load_bilm()
    if model is None:
        st.info(
            "biLM não treinado ou JAX ausente. Execute `uv sync --extra contextual && uv run replang train bilm`."
        )
        return
    text = st.text_area("sentenças (uma por linha)", DEFAULT, height=160)
    target = st.text_input("palavra-alvo", "banco").strip().lower()
    sents = [tokenize(line) for line in text.splitlines() if target in tokenize(line)]
    if len(sents) < 2:
        st.warning("informe ao menos duas sentenças contendo a palavra-alvo")
        return
    c1, c2 = st.columns(2)
    layer_mode = c1.radio(
        "representação",
        ["camada 0 (token)", "camada 1", "camada 2", "mistura ELMo"],
        index=2,
        horizontal=True,
    )
    s = (
        [c2.slider(f"s_{j}", 0.0, 1.0, v, 0.05) for j, v in enumerate((0.2, 0.4, 0.4))]
        if layer_mode.startswith("mistura")
        else None
    )
    vecs = []
    for sent in sents:
        reps = model.representations(vocab.encode(sent))
        pos = sent.index(target) + 1
        if layer_mode.startswith("mistura"):
            vecs.append(elmo_mix(reps, np.log(np.array(s) + 1e-6))[pos])
        else:
            vecs.append(reps[int(layer_mode.split()[1])][pos])
    V = np.stack(vecs)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    sim = V @ V.T
    labels = [" ".join(x)[:45] for x in sents]
    st.markdown(f"**Cosseno entre os vetores de *{target}* em cada sentença** ({layer_mode})")
    st.dataframe(
        pd.DataFrame(sim, index=labels, columns=[f"s{i + 1}" for i in range(len(sents))])
        .style.format("{:.2f}")
        .background_gradient(cmap="Blues", vmin=0, vmax=1),
        width="stretch",
    )
    static = [k for k in local_models() if "sg100 " in k]
    if static:
        wv = load_model(static[0])
        st.markdown(
            f"Com um embedding **estático** ({static[0].split(' (')[0]}), *{target}* tem **um único vetor** → cosseno 1,00 em todas as células. "
            f"Vizinhos estáticos: {', '.join(w for w, _ in wv.most_similar(target, topn=6)) if target in wv else 'OOV'}."
        )
    st.markdown(
        "#### Vizinhos contextuais: tokens de outras sentenças mais parecidos com o alvo em cada sentença"
    )
    pool = [tokenize(line) for line in text.splitlines() if line.strip()]
    tok_vecs, tok_labels = [], []
    for sent in pool:
        reps = model.representations(vocab.encode(sent))
        layer = 2 if layer_mode.startswith("mistura") else int(layer_mode.split()[1])
        for i, w in enumerate(sent):
            tok_vecs.append(reps[layer][i + 1])
            tok_labels.append(f"{w} ⟨{' '.join(sent)[:30]}…⟩")
    T = np.stack(tok_vecs)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    for k, sent in enumerate(sents[:3]):
        reps = model.representations(vocab.encode(sent))
        layer = 2 if layer_mode.startswith("mistura") else int(layer_mode.split()[1])
        q = reps[layer][sent.index(target) + 1]
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = T @ q
        order = np.argsort(-sims)[1:6]
        st.markdown(
            f"**{' '.join(sent)}** → "
            + "; ".join(f"{tok_labels[i]} ({sims[i]:.2f})" for i in order)
        )
    hist = model.history
    if hist:
        st.line_chart(pd.DataFrame(hist).set_index("step")["ppl"], y_label="perplexidade (treino)")
    with st.expander("O que observar"):
        st.markdown(
            """
* Na **camada 0** (embedding do token) todos os *banco* são idênticos (cosseno 1). Nas camadas LSTM os sentidos *assento* e *instituição financeira* se separam.
* Peters et al. (§5.3): camadas baixas capturam sintaxe (POS), camadas altas capturam sentido (WSD). Nosso modelo é minúsculo (treinado em Machado de Assis por minutos), por isso o efeito é parcial.
* Este é o passo que leva de word2vec a BERT: o vetor deixa de ser uma entrada de tabela e passa a ser uma **função do contexto**.
"""
        )
