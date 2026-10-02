import numpy as np
import pytest

from replang.embedding import WordVectors


def test_basic_ops(tmp_path):
    wv = WordVectors(["a", "b", "c"], np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    assert len(wv) == 3 and "a" in wv and wv.dim == 2
    assert wv.similarity("a", "b") == pytest.approx(0.0)
    assert wv.similarity("a", "c") == pytest.approx(1 / np.sqrt(2))
    assert wv.most_similar("a", topn=1)[0][0] == "c"
    p = wv.save(tmp_path / "x.npz")
    wv2 = WordVectors.load(p)
    assert wv2.words == wv.words and np.allclose(wv2.vectors, wv.vectors)


def test_analogy_add_and_mul():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 8))
    # rei - homem + mulher ≈ rainha por construção
    words = ["homem", "mulher", "rei", "rainha", "x", "y"]
    base[3] = base[2] - base[0] + base[1]
    wv = WordVectors(words, base)
    assert wv.analogy("homem", "rei", "mulher", topn=1)[0][0] == "rainha"
    assert wv.analogy("homem", "rei", "mulher", method="mul", topn=1)[0][0] == "rainha"
    assert "rainha" not in [
        w for w, _ in wv.analogy("homem", "rei", "mulher", topn=3, exclude_words=["rainha"])
    ]


def test_from_word2vec_text(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("3 2\nENTITY/x 9 9\na 1 0\nb 0 1\nc 1 1\n", encoding="utf-8")
    wv = WordVectors.from_word2vec_text(p, top_n=2, skip_prefix="ENTITY/")
    assert wv.words == ["a", "b"]
    p2 = tmp_path / "glove.txt"
    p2.write_text("a 1 0\nb 0 1\n", encoding="utf-8")
    assert WordVectors.from_word2vec_text(p2).words == ["a", "b"]


def test_gensim_roundtrip():
    wv = WordVectors(["a", "b"], np.eye(2))
    kv = wv.to_gensim()
    back = WordVectors.from_gensim(kv)
    assert back.words == ["a", "b"] and np.allclose(back.vectors, np.eye(2))
