"""Normalização e tokenização de texto em português.

Segue as decisões de pré-processamento de Hartmann et al. (2017, §2.1):

* minúsculas;
* numerais normalizados para ``0`` (``1.234,56`` → ``0.000,00``);
* URLs → ``URL`` e e-mails → ``EMAIL``;
* tokenização por espaços e pontuação, **mantendo** clíticos hifenizados (``machucou-se``);
* sentenças com menos de ``min_tokens`` tokens são descartadas;
* tipos com frequência inferior a ``min_count`` são substituídos por ``UNKNOWN`` (feito nos
  modelos, via ``min_count``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DIGIT_RE = re.compile(r"\d")
# palavra (com acentos), opcionalmente com hífens internos (clíticos: "deu-se", "dir-se-ia")
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*|0[0.,:/]*0?|[^\s\w]", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n{2,}")


def normalize_text(text: str, *, lowercase: bool = True) -> str:
    """Aplica as normalizações léxicas de Hartmann et al. (2017)."""
    text = _URL_RE.sub(" URL ", text)
    text = _EMAIL_RE.sub(" EMAIL ", text)
    text = _DIGIT_RE.sub("0", text)
    if lowercase:
        text = text.lower()
    return text


def tokenize(text: str, *, keep_punct: bool = False, lowercase: bool = True) -> list[str]:
    """Tokeniza uma string normalizada. Por padrão descarta pontuação."""
    text = normalize_text(text, lowercase=lowercase)
    toks = _TOKEN_RE.findall(text)
    if not keep_punct:
        toks = [t for t in toks if t[0].isalnum()]
    return toks


def iter_sentences(
    text: str, *, min_tokens: int = 5, keep_punct: bool = False
) -> Iterator[list[str]]:
    """Divide ``text`` em sentenças tokenizadas (descarta as curtas)."""
    for sent in _SENT_SPLIT_RE.split(text):
        toks = tokenize(sent, keep_punct=keep_punct)
        if len(toks) >= min_tokens:
            yield toks


def sentences_to_tokens(sentences: Iterable[list[str]]) -> list[str]:
    return [t for s in sentences for t in s]


def build_vocab(
    sentences: Iterable[list[str]],
    *,
    min_count: int = 5,
    max_size: int | None = None,
    unk: str = "UNKNOWN",
) -> tuple[list[str], Counter]:
    """Vocabulário ordenado por frequência decrescente (índice 0 = ``unk`` se usado)."""
    counts: Counter = Counter()
    for s in sentences:
        counts.update(s)
    words = [w for w, c in counts.most_common(max_size) if c >= min_count]
    if unk and unk not in words:
        words = [unk, *words]
    return words, counts


def strip_accents(text: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
