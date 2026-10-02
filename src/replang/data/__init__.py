from replang.data.analogies import AnalogySet, load_analogies
from replang.data.corpora import (
    load_machado_sentences,
    load_macmorpho,
    synthetic_gender_corpus,
    toy_corpus,
)
from replang.data.embeddings import load_glove_en, load_pretrained, load_ptwiki
from replang.data.lexicons import EN, PT, Lexicon

__all__ = [
    "EN",
    "PT",
    "AnalogySet",
    "Lexicon",
    "load_analogies",
    "load_glove_en",
    "load_machado_sentences",
    "load_macmorpho",
    "load_pretrained",
    "load_ptwiki",
    "synthetic_gender_corpus",
    "toy_corpus",
]
