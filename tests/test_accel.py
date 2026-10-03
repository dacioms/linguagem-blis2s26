import numpy as np
import pytest

from replang.accel import detect
from replang.eval.linear import LinearClassifier
from replang.eval.parallel import parallel_map

jax = pytest.importorskip("jax")


def test_detect_reports():
    a = detect()
    assert a.has_jax and a.backend in ("jax", "numpy") and a.n_jobs >= 1
    assert "backend=" in a.summary()


def test_glove_jax_matches_numpy(toy):
    from replang.models.glove_jax import GloVeJax
    from replang.models.glove_np import GloVe

    gn = GloVe(dim=12, window=2, epochs=25, x_max=5, seed=1).fit(toy)
    gj = GloVeJax(dim=12, window=2, epochs=25, x_max=5, seed=1, batch_size=256).fit(toy)
    assert gj.history[0]["loss"] > gj.history[-1]["loss"]
    assert gn.history[0]["loss"] > gn.history[-1]["loss"]
    assert abs(gn.history[-1]["loss"] - gj.history[-1]["loss"]) < 0.2  # ordens de lote diferentes
    top = [w for w, _ in gj.to_wordvectors().most_similar("rei", topn=5)]
    assert any(w in top for w in ("rainha", "reino", "príncipe", "menino"))


def test_skipgram_jax_learns(toy):
    from replang.models.word2vec_jax import SkipGramJax

    m = SkipGramJax(dim=16, window=2, negative=5, sample=0, epochs=10, batch_size=64, seed=1).train(
        toy
    )
    assert m.history[0]["loss"] > m.history[-1]["loss"]
    wv = m.to_wordvectors()
    assert np.isfinite(wv.vectors).all()
    sub = SkipGramJax(
        dim=16,
        window=2,
        negative=5,
        sample=0,
        epochs=3,
        batch_size=64,
        seed=1,
        subword=True,
        bucket=500,
    ).train(toy)
    assert sub.word_vector("reinado").shape == (16,)


def test_linear_classifier_backends():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 20)).astype(np.float32)
    y = np.array(["a", "b", "c"])[(X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5).astype(int)]
    for be in ("sklearn", "jax"):
        clf = LinearClassifier(backend=be, epochs=15).fit(X, y)
        assert clf.score(X, y) > 0.75, be
        assert set(clf.predict(X[:5])) <= {"a", "b", "c"}


def test_parallel_map():
    assert parallel_map(lambda x: x * 2, [1, 2, 3], n_jobs=2) == [2, 4, 6]
    assert parallel_map(lambda x: x * 2, [5], n_jobs=4) == [10]
