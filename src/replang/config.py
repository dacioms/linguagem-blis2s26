"""Configuração central: caminhos, sementes e flags de execução.

Todos os caminhos derivam da raiz do repositório (ou de ``REPLANG_DATA`` se definida),
de modo que notebooks, app Streamlit, scripts e testes compartilhem o mesmo *layout*::

    data/
      raw/        downloads brutos (ignorado pelo git)
      samples/    artefatos pequenos versionados (embeddings truncados, léxicos, analogias)
      processed/  corpora normalizados (gerados por ``replang prepare``)
      models/     modelos treinados localmente (gerados por ``replang train``)
      cache/      caches diversos
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


ROOT: Path = _find_root()
DATA_ROOT: Path = Path(os.environ.get("REPLANG_DATA", ROOT / "data"))


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = DATA_ROOT
    raw: Path = DATA_ROOT / "raw"
    samples: Path = DATA_ROOT / "samples"
    processed: Path = DATA_ROOT / "processed"
    models: Path = DATA_ROOT / "models"
    cache: Path = DATA_ROOT / "cache"
    notebooks: Path = ROOT / "notebooks"
    docs: Path = ROOT / "docs"
    configs: Path = ROOT / "configs"

    def ensure(self) -> Paths:
        for p in (self.raw, self.samples, self.processed, self.models, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


@dataclass
class Settings:
    """Flags globais (lidas de variáveis de ambiente)."""

    seed: int = int(os.environ.get("REPLANG_SEED", "42"))
    #: Quando ``True`` os notebooks e scripts usam épocas/tamanhos reduzidos (CI, demonstração rápida).
    fast: bool = os.environ.get("REPLANG_FAST", "0") == "1"
    #: Número de palavras mantidas ao truncar embeddings pré-treinados.
    top_n_pretrained: int = int(os.environ.get("REPLANG_TOP_N", "50000"))
    workers: int = int(os.environ.get("REPLANG_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))
    extra: dict = field(default_factory=dict)


settings = Settings()

# Fontes públicas de dados utilizadas pelo projeto (todas com licença permissiva ou domínio público).
SOURCES: dict[str, dict[str, str]] = {
    "machado": {
        "url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/machado.zip",
        "desc": "Obra completa de Machado de Assis (domínio público) – corpus PT-BR literário",
        "license": "Domínio público (machado.mec.gov.br)",
    },
    "mac_morpho": {
        "url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/mac_morpho.zip",
        "desc": "Mac-Morpho: corpus PT-BR anotado com classes gramaticais (POS)",
        "license": "CC BY 4.0 (NILC/USP)",
    },
    "floresta": {
        "url": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/floresta.zip",
        "desc": "Floresta Sintá(c)tica – treebank PT",
        "license": "Linguateca",
    },
    "ptwiki2vec_100d": {
        "url": "https://wikipedia2vec.s3.amazonaws.com/models/pt/2018-04-20/ptwiki_20180420_100d.txt.bz2",
        "desc": "Wikipedia2Vec PT (skip-gram, 100d, Wikipédia 2018) – embeddings pré-treinados",
        "license": "Apache 2.0 (Wikipedia2Vec) / CC BY-SA (texto da Wikipédia)",
    },
    "glove_en_50d": {
        "url": "https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz",
        "desc": "GloVe 6B 50d (Wikipedia+Gigaword) empacotado pelo gensim-data",
        "license": "PDDL (Stanford GloVe)",
    },
    "questions_words": {
        "url": "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt",
        "desc": "Conjunto de analogias do word2vec (Mikolov et al., 2013a)",
        "license": "Apache 2.0",
    },
    "lx_analogies_br": {
        "url": "https://raw.githubusercontent.com/nlx-group/lx-dsemvectors/master/testsets/LX-4WAnalogiesBr.txt",
        "desc": "Analogias traduzidas para PT-BR (Rodrigues et al., 2016) – usadas por Hartmann et al. (2017)",
        "license": "LX-DSemVectors (NLX, Univ. Lisboa)",
    },
    "lx_analogies_eu": {
        "url": "https://raw.githubusercontent.com/nlx-group/lx-dsemvectors/master/testsets/LX-4WAnalogies.txt",
        "desc": "Analogias traduzidas para PT-EU (Rodrigues et al., 2016)",
        "license": "LX-DSemVectors (NLX, Univ. Lisboa)",
    },
    "debiaswe": {
        "url": "https://raw.githubusercontent.com/tolga-b/debiaswe/master/data/",
        "desc": "Listas de pares definicionais, pares de equalização, palavras específicas de gênero e profissões (Bolukbasi et al., 2016)",
        "license": "MIT",
    },
    # Conector opcional (não acessível a partir de todas as redes): embeddings NILC de Hartmann et al. (2017)
    "nilc": {
        "url": "http://nilc.icmc.usp.br/embeddings",
        "desc": "Repositório de embeddings PT do NILC (Hartmann et al., 2017): word2vec, GloVe, wang2vec, fastText",
        "license": "Uso acadêmico (NILC/USP)",
    },
}
