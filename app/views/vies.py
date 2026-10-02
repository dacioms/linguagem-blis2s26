import numpy as np
import pandas as pd
import streamlit as st

from common import lang_of, lexicon, load_model, model_options
from replang.bias import (
    direct_bias,
    gender_direction,
    generate_analogies,
    hard_debias,
    indirect_bias,
    pair_bias,
    pca_pairs,
    random_pca_baseline,
    soft_debias,
    split_analogies_pt,
)
from replang.bias.geometry import profile
from replang.viz import bias_axis_figure, pca_variance_figure


@st.cache_resource(show_spinner="calculando debias…")
def _debiased(label: str, kind: str, lam: float):
    wv = load_model(label)
    lex = lexicon(lang_of(label))
    g = gender_direction(wv, lex.definitional_pairs)
    neutral = [w for w in lex.professions + lex.stereotype_neutral if w in wv]
    if kind == "hard":
        return hard_debias(
            wv, g[None, :], gender_specific=lex.gender_specific, equalize_pairs=lex.equalize_pairs
        )
    out, _, _ = soft_debias(wv, g[None, :], neutral_words=neutral, lam=lam, steps=150, sample=2000)
    return out


def render():
    st.title("8 · Viés de gênero na geometria e algoritmos de *debias*")
    st.markdown(
        "Bolukbasi et al. (2016): o embedding codifica estereótipos como **direção**; é possível medi-los "
        "(DirectBias, viés indireto β) e removê-los das palavras neutras (Neutralize + Equalize ou Soft)."
    )
    label = st.sidebar.selectbox("Embedding", model_options())
    wv = load_model(label)
    lang = lang_of(label)
    lex = lexicon(lang)
    st.sidebar.markdown("**Versão**")
    version = st.sidebar.radio(
        "versão", ["original", "hard-debiased", "soft-debiased"], label_visibility="collapsed"
    )
    lam = st.sidebar.slider("λ (soft)", 0.05, 2.0, 0.2, 0.05)
    g = gender_direction(wv, lex.definitional_pairs)
    comps, ev, used = pca_pairs(wv, lex.definitional_pairs)
    cur = (
        wv
        if version == "original"
        else _debiased(label, "hard" if version.startswith("hard") else "soft", lam)
    )

    tabs = st.tabs(
        [
            "1 · direção de gênero",
            "2 · profissões no eixo",
            "3 · DirectBias / viés indireto",
            "4 · analogias geradas",
            "5 · antes × depois",
        ]
    )
    with tabs[0]:
        st.markdown(
            f"Pares definicionais usados ({len(used)}): " + ", ".join(f"*{a}–{b}*" for a, b in used)
        )
        st.plotly_chart(
            pca_variance_figure(ev, random_pca_baseline(wv.dim, len(used))), width="stretch"
        )
        st.markdown(
            f"A 1ª componente explica **{ev[0]:.0%}** da variância das diferenças (aleatório: ~{random_pca_baseline(wv.dim, len(used))[0]:.0%}) → há uma direção *g* dominante."
        )
        she, he = lex.she_he
        st.markdown(
            f"cos(*{she}*, g) = {float(np.dot(wv.unit[wv.index[she]], g)):+.2f} · cos(*{he}*, g) = {float(np.dot(wv.unit[wv.index[he]], g)):+.2f}"
        )
    with tabs[1]:
        words = [w for w in lex.professions + lex.stereotype_neutral if w in cur]
        extra = st.text_input("adicione palavras (vírgula)", "")
        words += [w.strip().lower() for w in extra.split(",") if w.strip().lower() in cur]
        n = st.slider("mostrar N mais extremas de cada lado", 5, 40, 15)
        df = profile(cur, words, g, lex.extra.get("labels"))
        df = pd.concat([df.nsmallest(n, "proj_g"), df.nlargest(n, "proj_g")]).drop_duplicates()
        axis = f"← {lex.she_he[1]}      {lex.she_he[0]} →"
        st.plotly_chart(
            bias_axis_figure(df, title=f"Projeção em g — {version}", axis_label=axis),
            width="stretch",
        )
    with tabs[2]:
        neutral = [w for w in lex.professions + lex.stereotype_neutral if w in cur]
        c = st.slider("c (rigor)", 0.5, 2.0, 1.0, 0.5)
        st.metric(
            f"DirectBias_c sobre {len(neutral)} palavras neutras ({version})",
            f"{direct_bias(cur, neutral, g, c):.4f}",
            delta=f"{direct_bias(cur, neutral, g, c) - direct_bias(wv, neutral, g, c):+.4f} vs original"
            if version != "original"
            else None,
        )
        st.markdown(
            r"$\mathrm{DirectBias}_c = \frac{1}{|N|}\sum_{w\in N} |\cos(\vec w, g)|^c$ — no artigo, 327 profissões em w2vNEWS: 0,08."
        )
        st.markdown(
            "**Viés indireto** β(w, v): fração da similaridade entre duas palavras *neutras* explicada pela componente de gênero."
        )
        d1 = ("softball", "receptionist") if lang == "en" else ("cozinha", "enfermagem")
        c1, c2 = st.columns(2)
        w1 = c1.text_input("w", d1[0]).lower().strip()
        w2 = c2.text_input("v", d1[1]).lower().strip()
        if w1 in cur and w2 in cur:
            st.markdown(
                f"cos(w, v) = **{cur.similarity(w1, w2):.3f}** · β(w, v) = **{indirect_bias(cur, w1, w2, g):.1%}** da similaridade vem de g"
            )
        st.markdown(
            f"PairBias (neutras × pares definicionais): **{pair_bias(cur, neutral[:80], lex.definitional_pairs):.4f}**"
        )
    with tabs[3]:
        she, he = lex.she_he
        c1, c2, c3 = st.columns(3)
        a = c1.text_input("a (semente)", she).lower().strip()
        b = c2.text_input("b (semente)", he).lower().strip()
        delta = c3.slider("δ (distância máxima entre x e y)", 0.5, 1.5, 1.0, 0.05)
        if st.button("Gerar analogias a:x :: b:y", type="primary"):
            df = generate_analogies(
                cur, a, b, delta=delta, topn=40, restrict=25000, exclude=lex.all_gender_words()
            )
            if lang == "pt":
                gram, stereo = split_analogies_pt(df)
                cc1, cc2 = st.columns(2)
                cc1.markdown("**Pares gramaticais** (gênero morfológico — esperados em PT)")
                cc1.dataframe(gram, width="stretch")
                cc2.markdown("**Candidatos a estereótipo** (sem relação morfológica)")
                cc2.dataframe(stereo, width="stretch")
            else:
                st.dataframe(df, width="stretch")
        st.caption(
            "Eq. (1) do artigo: maximiza cos(a − b, x − y) com ‖x − y‖ ≤ δ. No artigo, a plateia (Mechanical Turk) julgou cada par como estereótipo ou apropriado."
        )
    with tabs[4]:
        neutral = [w for w in lex.professions + lex.stereotype_neutral if w in wv]
        hard = _debiased(label, "hard", lam)
        rows = [
            {
                "versão": "original",
                "DirectBias": direct_bias(wv, neutral, g),
                "PairBias": pair_bias(wv, neutral[:80], lex.definitional_pairs),
            },
            {
                "versão": "hard-debiased",
                "DirectBias": direct_bias(hard, neutral, g),
                "PairBias": pair_bias(hard, neutral[:80], lex.definitional_pairs),
            },
        ]
        if version == "soft-debiased":
            rows.append(
                {
                    "versão": f"soft (λ={lam})",
                    "DirectBias": direct_bias(cur, neutral, g),
                    "PairBias": pair_bias(cur, neutral[:80], lex.definitional_pairs),
                }
            )
        st.table(pd.DataFrame(rows).set_index("versão").style.format("{:.4f}"))
        q = ("he", "doctor", "she") if lang == "en" else ("ele", "médico", "ela")
        c1, c2, c3 = st.columns(3)
        a = c1.text_input("a", q[0], key="ba").lower()
        b = c2.text_input("b", q[1], key="bb").lower()
        c = c3.text_input("c", q[2], key="bc").lower()
        if all(w in wv for w in (a, b, c)):
            cols = st.columns(2)
            cols[0].markdown("**original**")
            cols[0].table(
                pd.DataFrame(
                    wv.analogy(a, b, c, topn=6, exclude_words=lex.gender_specific),
                    columns=["d", "score"],
                )
            )
            cols[1].markdown("**hard-debiased**")
            cols[1].table(
                pd.DataFrame(
                    hard.analogy(a, b, c, topn=6, exclude_words=lex.gender_specific),
                    columns=["d", "score"],
                )
            )
        st.markdown(
            "Vizinhança preservada? Compare os vizinhos de uma palavra neutra antes/depois:"
        )
        w = st.text_input("palavra", "medicina" if lang == "pt" else "nurse").lower().strip()
        if w in wv:
            cols = st.columns(2)
            cols[0].write([x for x, _ in wv.most_similar(w, topn=8)])
            cols[1].write([x for x, _ in hard.most_similar(w, topn=8)])
    with st.expander("O que observar"):
        st.markdown(
            """
* A direção *g* emerge **sem supervisão**: só pela PCA de 10 pares. Em PT a 1ª componente costuma explicar >50% — o gênero gramatical reforça o sinal.
* Profissões *epicenas* (dentista, gerente, cientista) e áreas (cozinha, engenharia) não têm gênero ortográfico; sua posição no eixo é **estereótipo** aprendido do corpus.
* O **hard debias** zera DirectBias e PairBias para as neutras e mantém pares definicionais (rei/rainha). O **soft** troca redução de viés por preservação de produtos internos (λ).
* Pergunta de §9 do artigo: e em línguas com gênero gramatical? O painel 4 separa pares morfológicos de estereótipos — uma resposta parcial.
"""
        )
