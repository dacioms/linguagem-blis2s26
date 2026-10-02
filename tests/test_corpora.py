from replang.data.corpora import synthetic_gender_corpus, toy_corpus
from replang.data.lexicons import PT


def test_toy_and_synthetic():
    assert len(toy_corpus()) > 10
    s = synthetic_gender_corpus(100, seed=3)
    assert len(s) == 100 and all(isinstance(x, list) for x in s)


def test_pt_lexicon_consistency():
    lex = PT.get()
    assert ("mulher", "homem") in lex.definitional_pairs
    assert "dentista" in lex.professions and "ela" in lex.gender_specific
    assert not set(lex.professions) & set(lex.gender_specific)
