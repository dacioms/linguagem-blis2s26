"""Conjuntos de analogias ``a : b :: c : d`` no formato do word2vec (``questions-words.txt``).

* Inglês: 19.544 questões em 14 categorias (5 semânticas, 9 sintáticas) – Mikolov et al. (2013a).
* Português: tradução/adaptação LX-4WAnalogies (Rodrigues et al., 2016), BR e EU – usada por
  Hartmann et al. (2017) na avaliação intrínseca (Tabela 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from replang.config import PATHS, SOURCES
from replang.utils.download import download

SEMANTIC = {"capital-common-countries", "capital-world", "currency", "city-in-state", "family"}


@dataclass
class AnalogySet:
    name: str
    categories: dict[str, list[tuple[str, str, str, str]]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return sum(len(v) for v in self.categories.values())

    def is_semantic(self, cat: str) -> bool:
        return cat in SEMANTIC or not cat.startswith("gram")

    def questions(self, cats: list[str] | None = None):
        for c, qs in self.categories.items():
            if cats is None or c in cats:
                for q in qs:
                    yield c, q

    def summary(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "categoria": c,
                    "tipo": "semântica" if self.is_semantic(c) else "sintática",
                    "questões": len(q),
                    "exemplo": " : ".join(q[0][:2]) + " :: " + " : ".join(q[0][2:]),
                }
                for c, q in self.categories.items()
            ]
        )


def parse_questions(path: Path | str, *, lowercase: bool = True, name: str = "") -> AnalogySet:
    aset = AnalogySet(name or Path(path).stem)
    cat = "default"
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cat = line[1:].strip()
                aset.categories.setdefault(cat, [])
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            if lowercase:
                parts = [p.lower() for p in parts]
            aset.categories.setdefault(cat, []).append(tuple(parts))
    return aset


def load_analogies(lang: str = "pt-br") -> AnalogySet:
    """``"en"`` (questions-words), ``"pt-br"`` ou ``"pt-eu"`` (LX-4WAnalogies)."""
    lang = lang.lower()
    key, fname = {
        "en": ("questions_words", "questions-words.txt"),
        "pt-br": ("lx_analogies_br", "LX-4WAnalogiesBr.txt"),
        "pt": ("lx_analogies_br", "LX-4WAnalogiesBr.txt"),
        "pt-eu": ("lx_analogies_eu", "LX-4WAnalogies.txt"),
    }[lang]
    for cand in (PATHS.samples / "analogies" / fname, PATHS.raw / fname):
        if cand.exists():
            return parse_questions(cand, name=lang)
    path = download(SOURCES[key]["url"], PATHS.raw / fname)
    return parse_questions(path, name=lang)
