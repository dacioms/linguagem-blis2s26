import numpy as np

from replang.models import (
    CountModel,
    FastText,
    GloVe,
    HuffmanTree,
    Word2Vec,
    char_ngrams,
    fnv1a,
    learn_phrases,
    ppmi,
)
from replang.models.cooccurrence import cooccurrence_matrix


def test_cooccurrence_symmetric_and_ppmi():
    sents = [["a", "b", "c"], ["a", "b"]]
    idx = {"a": 0, "b": 1, "c": 2}
    m = cooccurrence_matrix(sents, idx, window=1)
    assert m[0, 1] == 2 and m[1, 0] == 2 and m[0, 2] == 0
    p = ppmi(m)
    assert (p.toarray() >= 0).all()


def test_count_model_neighbors(toy):
    wv = CountModel(window=2, dim=8).fit(toy).to_wordvectors()
    top = [w for w, _ in wv.most_similar("rei", topn=5)]
    assert "rainha" in top or "príncipe" in top or "reino" in top


def test_word2vec_sg_neg_learns(toy):
    m = Word2Vec(dim=20, window=2, epochs=15, negative=5, sample=0, batch_size=64, seed=1).train(
        toy
    )
    assert m.history[0]["loss"] > m.history[-1]["loss"]
    wv = m.to_wordvectors()
    assert np.isfinite(wv.vectors).all()
    top = [w for w, _ in wv.most_similar("rei", topn=6)]
    assert any(w in top for w in ("rainha", "reino", "príncipe", "governa"))


def test_word2vec_hs_and_cbow(toy):
    hs = Word2Vec(dim=16, window=2, epochs=8, hs=1, sample=0, batch_size=64).train(toy)
    assert hs.tree is not None and np.isfinite(hs.history[-1]["loss"])
    cb = Word2Vec(dim=16, window=2, epochs=8, sg=0, sample=0, batch_size=64).train(toy)
    assert cb.history[0]["loss"] > cb.history[-1]["loss"]


def test_subsampling_keep_prob(toy):
    m = Word2Vec(sample=1e-2).build_vocab(toy)
    keep = m.vocab.keep_prob(1e-2)
    assert keep.max() <= 1.0 and keep[0] < keep[-1]  # a palavra mais frequente é mais descartada


def test_huffman_prefix_free():
    tree = HuffmanTree(np.array([40, 30, 20, 10]))
    codes = ["".join(map(str, c)) for c in tree.codes]
    for i, a in enumerate(codes):
        for j, b in enumerate(codes):
            assert i == j or not b.startswith(a)
    assert len(tree.codes[0]) <= len(tree.codes[-1])


def test_glove_learns(toy):
    g = GloVe(dim=16, window=2, epochs=30, x_max=5).fit(toy)
    assert g.history[0]["loss"] > g.history[-1]["loss"]
    assert np.isfinite(g.to_wordvectors().vectors).all()


def test_fasttext_ngrams_hash_and_oov(toy):
    assert char_ngrams("onde", 3, 3) == ["<on", "ond", "nde", "de>"]
    assert fnv1a("a") == fnv1a("a") and fnv1a("a") != fnv1a("b")
    f = FastText(dim=16, window=2, epochs=8, sample=0, bucket=1000, batch_size=64).train(toy)
    v = f.word_vector("reinado")  # OOV
    assert v.shape == (16,) and np.isfinite(v).all()
    imp = f.ngram_importance("rainha")
    assert len(imp) > 0


def test_learn_phrases():
    sents = [["nova", "york", "é", "grande"]] * 20 + [["nova", "casa"], ["york", "fica"]] * 3
    merged, scores = learn_phrases(sents, min_count=5, threshold=1)
    assert "nova_york" in scores and merged[0][0] == "nova_york"
