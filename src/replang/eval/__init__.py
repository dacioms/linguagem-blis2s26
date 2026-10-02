from replang.eval.analogies import evaluate_analogies, solve_analogy
from replang.eval.extrinsic import pos_tagging_eval, sentence_similarity_eval, sentence_vector
from replang.eval.similarity import MINI_STS_PT, MINI_WORDSIM_PT, word_similarity_eval

__all__ = [
    "MINI_STS_PT",
    "MINI_WORDSIM_PT",
    "evaluate_analogies",
    "pos_tagging_eval",
    "sentence_similarity_eval",
    "sentence_vector",
    "solve_analogy",
    "word_similarity_eval",
]
