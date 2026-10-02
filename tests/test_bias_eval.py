import numpy as np

from replang.bias import (
    direct_bias,
    gender_direction,
    generate_analogies,
    hard_debias,
    indirect_bias,
    is_morphological_pair,
    pair_bias,
    pca_pairs,
    project,
    soft_debias,
)
from replang.data.analogies import AnalogySet
from replang.data.corpora import SYNTHETIC_PROFESSIONS
from replang.eval import evaluate_analogies, sentence_similarity_eval, word_similarity_eval
from replang.models import Word2Vec

PAIRS = [("ela", "ele"), ("mulher", "homem"), ("rainha", "rei")]


def test_gender_direction_orientation(small_wv):
    g = gender_direction(small_wv, PAIRS)
    assert np.isclose(np.linalg.norm(g), 1.0)
    p = project(small_wv, small_wv.words, g)
    assert p["ela"] > 0 > p["ele"] and p["enfermagem"] > p["engenharia"]
    _, ev, used = pca_pairs(small_wv, PAIRS)
    assert ev[0] > 0.5 and len(used) == 3


def test_direct_and_indirect_bias_and_hard_debias(small_wv):
    g = gender_direction(small_wv, PAIRS)
    neutral = ["enfermagem", "engenharia", "casa", "futebol", "música"]
    before = direct_bias(small_wv, neutral, g)
    deb = hard_debias(small_wv, g[None, :], neutral_words=neutral, equalize_pairs=PAIRS)
    after = direct_bias(deb, neutral, g)
    assert after < 1e-5 < before
    assert pair_bias(deb, neutral, PAIRS) < 1e-5
    # equalize: pares equidistantes das neutras e com norma 1
    assert np.isclose(np.linalg.norm(deb["ela"]), 1.0, atol=1e-5)
    assert abs(deb.similarity("casa", "ela") - deb.similarity("casa", "ele")) < 1e-5
    b = indirect_bias(small_wv, "enfermagem", "casa", g)
    assert np.isfinite(b)


def test_soft_debias_reduces_bias(small_wv):
    g = gender_direction(small_wv, PAIRS)
    neutral = ["enfermagem", "engenharia", "casa", "futebol"]
    soft, T, hist = soft_debias(
        small_wv, g[None, :], neutral_words=neutral, steps=60, lr=0.1, sample=11
    )
    assert direct_bias(soft, neutral, g) < direct_bias(small_wv, neutral, g)
    assert T.shape == (20, 20) and hist[-1]["viés"] <= hist[0]["viés"]


def test_generate_analogies_and_morph_filter(small_wv):
    df = generate_analogies(small_wv, "ela", "ele", delta=5.0, topn=5, restrict=None)
    assert {"x", "y", "score"} <= set(df.columns) and len(df) > 0
    assert is_morphological_pair("professora", "professor") and not is_morphological_pair(
        "costura", "carpintaria"
    )


def test_synthetic_corpus_bias_emerges(synthetic):
    wv = (
        Word2Vec(dim=30, window=3, epochs=4, negative=5, batch_size=128, seed=0)
        .train(synthetic)
        .to_wordvectors()
    )
    g = gender_direction(
        wv, [("ela", "ele"), ("mulher", "homem"), ("mãe", "pai"), ("filha", "filho")]
    )
    p = project(
        wv,
        SYNTHETIC_PROFESSIONS["estereotipo_feminino"]
        + SYNTHETIC_PROFESSIONS["estereotipo_masculino"],
        g,
    )
    fem = p[SYNTHETIC_PROFESSIONS["estereotipo_feminino"]].mean()
    masc = p[SYNTHETIC_PROFESSIONS["estereotipo_masculino"]].mean()
    assert fem > masc


def test_evaluate_analogies(small_wv):
    aset = AnalogySet(
        "t",
        {
            "family": [("ele", "ela", "rei", "rainha"), ("ela", "ele", "rainha", "rei")],
            "gram1-x": [("ele", "ela", "homem", "mulher")],
        },
    )
    df = evaluate_analogies(small_wv, aset)
    assert df.iloc[-1]["categoria"] == "TOTAL" and df.iloc[-1]["cobertas"] == 3


def test_similarity_evals(small_wv):
    r = word_similarity_eval(
        small_wv, [("rei", "rainha", 8.0), ("rei", "futebol", 2.0), ("casa", "música", 3.0)]
    )
    assert "spearman" in r
    s = sentence_similarity_eval(
        small_wv,
        [
            ("ela rainha", "mulher rainha", 5.0),
            ("ela rainha", "futebol", 1.0),
            ("casa", "casa música", 4.0),
            ("rei", "ele rei", 4.5),
            ("homem", "mulher", 3.0),
        ],
        folds=2,
    )
    assert np.isfinite(s["mse"])
