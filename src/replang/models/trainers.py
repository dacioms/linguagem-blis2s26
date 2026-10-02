"""Treinadores baseados no **gensim** para corpora reais (Machado de Assis), com cache em ``data/models``.

Produzem ``WordVectors`` compatíveis com todo o restante do pacote. São os "modelos de produção"
que acompanham as implementações didáticas *from scratch* de :mod:`replang.models`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from replang.config import PATHS, settings
from replang.embedding import WordVectors


def _meta(path: Path, **kw) -> None:
    path.with_suffix(".json").write_text(json.dumps(kw, ensure_ascii=False, indent=2))


def train_word2vec(
    sentences,
    *,
    name: str,
    dim: int = 100,
    window: int = 5,
    sg: int = 1,
    hs: int = 0,
    negative: int = 10,
    min_count: int = 5,
    epochs: int = 5,
    sample: float = 1e-4,
    seed: int | None = None,
    force: bool = False,
    workers: int | None = None,
) -> WordVectors:
    """Word2Vec (Skip-gram ou CBOW) do gensim. ``name`` identifica o cache ``data/models/<name>.npz``."""
    from gensim.models import Word2Vec

    out = PATHS.models / f"{name}.npz"
    if out.exists() and not force:
        return WordVectors.load(out)
    t = time.time()
    model = Word2Vec(
        sentences,
        vector_size=dim,
        window=window,
        sg=sg,
        hs=hs,
        negative=negative,
        min_count=min_count,
        epochs=epochs,
        sample=sample,
        seed=seed or settings.seed,
        workers=workers or settings.workers,
    )
    wv = WordVectors.from_gensim(model.wv, name)
    PATHS.models.mkdir(parents=True, exist_ok=True)
    wv.save(out)
    _meta(
        out,
        algo="word2vec",
        sg=sg,
        hs=hs,
        dim=dim,
        window=window,
        negative=negative,
        min_count=min_count,
        epochs=epochs,
        sample=sample,
        seconds=round(time.time() - t, 1),
        vocab=len(wv),
    )
    return wv


def train_fasttext(
    sentences,
    *,
    name: str,
    dim: int = 100,
    window: int = 5,
    sg: int = 1,
    min_count: int = 5,
    epochs: int = 5,
    min_n: int = 3,
    max_n: int = 6,
    negative: int = 10,
    sample: float = 1e-4,
    seed: int | None = None,
    force: bool = False,
    workers: int | None = None,
    keep_model: bool = True,
    bucket: int = 200_000,
):
    """fastText do gensim. Além do ``.npz`` salva o modelo completo (``.model``) para vetores OOV."""
    from gensim.models import FastText

    out = PATHS.models / f"{name}.npz"
    model_path = PATHS.models / f"{name}.model"
    if out.exists() and not force:
        wv = WordVectors.load(out)
        model = FastText.load(str(model_path)) if model_path.exists() else None
        return wv, model
    t = time.time()
    model = FastText(
        sentences,
        vector_size=dim,
        window=window,
        sg=sg,
        min_count=min_count,
        epochs=epochs,
        min_n=min_n,
        max_n=max_n,
        negative=negative,
        sample=sample,
        bucket=bucket,
        seed=seed or settings.seed,
        workers=workers or settings.workers,
    )
    wv = WordVectors.from_gensim(model.wv, name)
    PATHS.models.mkdir(parents=True, exist_ok=True)
    wv.save(out)
    if keep_model:
        model.save(str(model_path))
    _meta(
        out,
        algo="fasttext",
        bucket=bucket,
        sg=sg,
        dim=dim,
        window=window,
        min_n=min_n,
        max_n=max_n,
        min_count=min_count,
        epochs=epochs,
        seconds=round(time.time() - t, 1),
        vocab=len(wv),
    )
    return wv, model


def load_fasttext_model(name: str):
    from gensim.models import FastText

    p = PATHS.models / f"{name}.model"
    return FastText.load(str(p)) if p.exists() else None


def train_doc2vec(
    documents: list[list[str]],
    *,
    name: str,
    dim: int = 100,
    window: int = 5,
    dm: int = 1,
    min_count: int = 5,
    epochs: int = 20,
    seed: int | None = None,
    force: bool = False,
    workers: int | None = None,
):
    """Doc2Vec (Le & Mikolov, 2014): ``dm=1`` PV-DM, ``dm=0`` PV-DBOW. Retorna o modelo gensim."""
    from gensim.models import Doc2Vec
    from gensim.models.doc2vec import TaggedDocument

    model_path = PATHS.models / f"{name}.d2v"
    if model_path.exists() and not force:
        return Doc2Vec.load(str(model_path))
    t = time.time()
    tagged = [TaggedDocument(words=d, tags=[i]) for i, d in enumerate(documents)]
    model = Doc2Vec(
        tagged,
        vector_size=dim,
        window=window,
        dm=dm,
        min_count=min_count,
        epochs=epochs,
        seed=seed or settings.seed,
        workers=workers or settings.workers,
    )
    PATHS.models.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    _meta(
        model_path.with_suffix(".npz"),
        algo="doc2vec",
        dm=dm,
        dim=dim,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seconds=round(time.time() - t, 1),
        docs=len(documents),
    )
    return model


def list_models() -> list[dict]:
    out = []
    for p in sorted(PATHS.models.glob("*.json")):
        meta = json.loads(p.read_text())
        if not isinstance(meta, dict) or "algo" not in meta:
            continue
        meta["name"] = p.stem
        meta.pop("history", None)
        out.append(meta)
    return out
