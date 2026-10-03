from replang.eval.analogies import evaluate_analogies, solve_analogy
from replang.eval.extrinsic import pos_tagging_eval, sentence_similarity_eval, sentence_vector
from replang.eval.linear import LinearClassifier
from replang.eval.parallel import parallel_map
from replang.eval.similarity import MINI_STS_PT, MINI_WORDSIM_PT, word_similarity_eval

__all__ = [
    "MINI_STS_PT",
    "LinearClassifier",
    "MINI_WORDSIM_PT",
    "evaluate_analogies",
    "parallel_map",
    "pos_tagging_eval",
    "sentence_similarity_eval",
    "sentence_vector",
    "solve_analogy",
    "word_similarity_eval",
]
