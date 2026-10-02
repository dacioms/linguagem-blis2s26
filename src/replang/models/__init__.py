from replang.models.cooccurrence import CountModel, cooccurrence_matrix, ppmi, svd_embeddings
from replang.models.fasttext_np import FastText, char_ngrams, fnv1a
from replang.models.glove_np import GloVe
from replang.models.word2vec_np import HuffmanTree, Vocab, Word2Vec, learn_phrases


def __getattr__(name):  # carregamento preguiçoso dos backends JAX (opcionais)
    if name == "GloVeJax":
        from replang.models.glove_jax import GloVeJax

        return GloVeJax
    if name == "SkipGramJax":
        from replang.models.word2vec_jax import SkipGramJax

        return SkipGramJax
    raise AttributeError(name)


__all__ = [
    "CountModel",
    "FastText",
    "GloVe",
    "HuffmanTree",
    "Vocab",
    "Word2Vec",
    "char_ngrams",
    "cooccurrence_matrix",
    "fnv1a",
    "learn_phrases",
    "ppmi",
    "svd_embeddings",
]
