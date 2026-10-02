import numpy as np
import pytest

from replang.data.corpora import synthetic_gender_corpus, toy_corpus
from replang.embedding import WordVectors


@pytest.fixture(scope="session")
def toy():
    return toy_corpus() * 10


@pytest.fixture(scope="session")
def synthetic():
    return synthetic_gender_corpus(4000, bias=0.9, seed=1)


@pytest.fixture(scope="session")
def small_wv():
    rng = np.random.default_rng(0)
    words = [
        "ela",
        "ele",
        "mulher",
        "homem",
        "rainha",
        "rei",
        "enfermagem",
        "engenharia",
        "casa",
        "futebol",
        "música",
    ]
    base = rng.normal(size=(len(words), 20)).astype(np.float32)
    g = rng.normal(size=20).astype(np.float32)
    g /= np.linalg.norm(g)
    sign = np.array([1, -1, 1, -1, 1, -1, 0.6, -0.6, 0.3, -0.3, 0.0], dtype=np.float32)
    base = base * 0.3 + sign[:, None] * g[None, :] * 2
    return WordVectors(words, base, "sintético")
