"""Carregadores de embeddings pré-treinados (com truncamento e cache em ``data/samples``).

* :func:`load_ptwiki` – Wikipedia2Vec PT 100d (skip-gram sobre a Wikipédia em português).
* :func:`load_glove_en` – GloVe 6B 50d (inglês) via gensim-data; usado para reproduzir
  Bolukbasi et al. (2016) em inglês.
* :func:`load_nilc` – conector para os modelos de Hartmann et al. (2017) (``nilc.icmc.usp.br``),
  que exigem download manual em algumas redes.
"""

from __future__ import annotations

from pathlib import Path

from replang.config import PATHS, SOURCES, settings
from replang.embedding import WordVectors
from replang.utils.download import download


def _cached(name: str, top_n: int) -> Path:
    return PATHS.samples / f"{name}_top{top_n // 1000}k.npz"


def load_ptwiki(top_n: int | None = None, *, force: bool = False) -> WordVectors:
    """Wikipedia2Vec PT (100d). Mantém as ``top_n`` palavras mais frequentes (ignora entidades)."""
    top_n = top_n or settings.top_n_pretrained
    cache = _cached("ptwiki100d", top_n)
    if cache.exists() and not force:
        wv = WordVectors.load(cache)
        wv.name = "Wikipedia2Vec PT 100d"
        return wv
    # tenta reaproveitar um cache maior
    for bigger in sorted(PATHS.samples.glob("ptwiki100d_top*k.npz"), reverse=True):
        wv = WordVectors.load(bigger)
        if len(wv) >= top_n:
            return wv.head(top_n, "Wikipedia2Vec PT 100d")
    raw = download(SOURCES["ptwiki2vec_100d"]["url"], PATHS.raw / "ptwiki_20180420_100d.txt.bz2")
    wv = WordVectors.from_word2vec_text(
        raw, top_n=top_n, skip_prefix="ENTITY/", name="Wikipedia2Vec PT 100d"
    )
    wv.save(cache)
    return wv


def load_glove_en(top_n: int | None = None, *, force: bool = False) -> WordVectors:
    """GloVe 6B 50d (Wikipedia + Gigaword), 400k palavras ordenadas por frequência."""
    top_n = top_n or settings.top_n_pretrained
    cache = _cached("glove50d", top_n)
    if cache.exists() and not force:
        wv = WordVectors.load(cache)
        wv.name = "GloVe 6B 50d (EN)"
        return wv
    for bigger in sorted(PATHS.samples.glob("glove50d_top*k.npz"), reverse=True):
        wv = WordVectors.load(bigger)
        if len(wv) >= top_n:
            return wv.head(top_n, "GloVe 6B 50d (EN)")
    raw = download(SOURCES["glove_en_50d"]["url"], PATHS.raw / "glove-wiki-gigaword-50.gz")
    wv = WordVectors.from_word2vec_text(raw, top_n=top_n, name="GloVe 6B 50d (EN)")
    wv.save(cache)
    return wv


def load_nilc(
    path: Path | str, *, top_n: int | None = None, name: str | None = None
) -> WordVectors:
    """Carrega um modelo do repositório NILC (ex.: ``skip_s100.txt`` ou o ``.zip``).

    Os arquivos estão em formato texto do word2vec. Baixe manualmente em
    http://nilc.icmc.usp.br/embeddings e aponte ``path`` para o arquivo.
    """
    import zipfile

    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(".txt"))
            target = PATHS.raw / inner
            if not target.exists():
                zf.extract(inner, PATHS.raw)
        path = target
    wv = WordVectors.from_word2vec_text(path, top_n=top_n, name=name or f"NILC {path.stem}")
    return wv


def load_pretrained(which: str = "pt", top_n: int | None = None) -> WordVectors:
    """Atalho: ``"pt"`` → Wikipedia2Vec PT; ``"en"`` → GloVe EN."""
    if which.lower().startswith("pt"):
        return load_ptwiki(top_n)
    if which.lower().startswith("en"):
        return load_glove_en(top_n)
    raise ValueError(which)


def load_local_model(name: str) -> WordVectors:
    """Carrega um modelo treinado localmente por ``replang train`` (``data/models/<name>.npz``)."""
    p = PATHS.models / f"{name}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"modelo {name!r} não encontrado em {PATHS.models}; execute `uv run replang train`"
        )
    return WordVectors.load(p)
