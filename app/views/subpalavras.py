import pandas as pd
import streamlit as st

from common import load_fasttext_model, load_model, local_models
from replang.models.fasttext_np import char_ngrams, fnv1a


def render():
    st.title("6 · Subpalavras: fastText e palavras fora do vocabulário")
    st.markdown(
        "Bojanowski et al. (2017): cada palavra é uma *bag* de n-gramas de caracteres (3 ≤ n ≤ 6) com marcadores `<` e `>`; "
        "o vetor da palavra é a soma dos vetores dos n-gramas, mapeados em *buckets* por *hashing* FNV-1a."
    )
    word = st.text_input("palavra", "coraçãozinho").strip().lower()
    c1, c2 = st.columns(2)
    min_n = c1.slider("n mínimo", 2, 4, 3)
    max_n = c2.slider("n máximo", 3, 8, 6)
    grams = char_ngrams(word, min_n, max_n)
    st.markdown(f"**{len(grams)} n-gramas** de `<{word}>`:")
    st.code(" ".join(grams))
    st.dataframe(
        pd.DataFrame(
            {
                "n-grama": grams[:12],
                "hash FNV-1a mod 2·10⁶": [fnv1a(g) % 2_000_000 for g in grams[:12]],
            }
        ),
        width="stretch",
    )

    model = load_fasttext_model()
    if model is None:
        st.info("Modelo fastText local não encontrado — execute `uv run replang train fasttext`.")
        return
    st.markdown("#### Vetores OOV e vizinhos (fastText treinado em Machado de Assis, 100d)")
    in_vocab = word in model.wv.key_to_index
    st.markdown(
        f"*{word}* está no vocabulário de treino? **{'sim' if in_vocab else 'não — vetor composto só de n-gramas'}**"
    )
    try:
        nn = model.wv.most_similar(word, topn=10)
        st.table(pd.DataFrame(nn, columns=["vizinho", "cosseno"]))
    except KeyError:
        st.warning("palavra sem n-gramas suficientes")
    w2v_labels = [k for k in local_models() if "sg100 " in k]
    if w2v_labels:
        w2v = load_model(w2v_labels[0])
        st.markdown(
            f"Comparação com Skip-gram sem subpalavras: *{word}* {'está' if word in w2v else '**não está**'} no vocabulário → "
            + (
                "vizinhos: " + ", ".join(w for w, _ in w2v.most_similar(word, topn=6))
                if word in w2v
                else "sem vetor (OOV)."
            )
        )
    st.markdown("#### Pares morfológicos: similaridade fastText × word2vec")
    pairs = [
        ("menino", "menininho"),
        ("casa", "casinha"),
        ("amor", "amoroso"),
        ("escrever", "escrevia"),
        ("rainha", "rei"),
        ("triste", "tristeza"),
    ]
    rows = []
    for a, b in pairs:
        r = {
            "par": f"{a} – {b}",
            "fastText": model.wv.similarity(a, b) if a in model.wv.key_to_index or True else None,
        }
        if w2v_labels and a in w2v and b in w2v:
            r["Skip-gram"] = w2v.similarity(a, b)
        rows.append(r)
    st.table(pd.DataFrame(rows))
    with st.expander("O que observar"):
        st.markdown(
            """
* Palavras **raras ou inéditas** (diminutivos, erros de digitação, neologismos) ganham vetores razoáveis pela morfologia compartilhada — o Skip-gram clássico simplesmente não tem vetor.
* O ganho é maior em línguas morfologicamente ricas (alemão, tcheco, russo e, em parte, o português) e em analogias **sintáticas**; para analogias semânticas pode até piorar (Bojanowski, Tab. 2; Hartmann, Tab. 2).
* `<` e `>` distinguem prefixos e sufixos: `her>` (final de *where*) ≠ `<her` (início de *her*).
"""
        )
