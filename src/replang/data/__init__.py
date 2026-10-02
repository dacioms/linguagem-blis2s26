from replang.data.analogies import AnalogySet, load_analogies
from replang.data.corpora import (
    load_machado_sentences,
    load_macmorpho,
    synthetic_gender_corpus,
    toy_corpus,
)
from replang.data.embeddings import load_glove_en, load_pretrained, load_ptwiki
from replang.data.legal import (
    LEGAL,
    LegalLexicon,
    load_legal_sentences,
    load_legal_sts,
    load_lener,
    load_rulingbr_ementas,
)
from replang.data.lexicons import EN, PT, Lexicon

__all__ = [
    "EN",
    "LEGAL",
    "LegalLexicon",
    "PT",
    "AnalogySet",
    "Lexicon",
    "load_analogies",
    "load_glove_en",
    "load_legal_sentences",
    "load_legal_sts",
    "load_lener",
    "load_machado_sentences",
    "load_macmorpho",
    "load_pretrained",
    "load_ptwiki",
    "load_rulingbr_ementas",
    "synthetic_gender_corpus",
    "toy_corpus",
]
