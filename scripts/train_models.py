"""Treina os modelos locais usados por notebooks e app (cache em ``data/models``).

Uso: ``uv run replang train [all|w2v|fasttext|glove|doc2vec|bilm] [--fast] [--force]``
ou ``uv run python scripts/train_models.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replang.config import PATHS, settings  # noqa: E402
from replang.data.corpora import load_machado_sentences  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def train_gensim(sentences, fast: bool, force: bool):
    from replang.models.trainers import train_fasttext, train_word2vec

    ep = 3 if fast else 5
    log("word2vec skip-gram 100d …")
    wv = train_word2vec(
        sentences, name="machado_sg100", dim=100, sg=1, negative=10, epochs=ep, force=force
    )
    log("  ", wv, wv.most_similar("capitu", topn=5))
    log("word2vec CBOW 100d …")
    wv = train_word2vec(
        sentences, name="machado_cbow100", dim=100, sg=0, negative=10, epochs=ep, force=force
    )
    log("  ", wv, wv.most_similar("capitu", topn=5))
    log("word2vec skip-gram HS 100d …")
    wv = train_word2vec(
        sentences, name="machado_sg100_hs", dim=100, sg=1, hs=1, negative=0, epochs=ep, force=force
    )
    log("  ", wv)
    log("word2vec skip-gram 50d (sem subamostragem) …")
    wv = train_word2vec(
        sentences,
        name="machado_sg50_nosub",
        dim=50,
        sg=1,
        negative=5,
        epochs=ep,
        sample=0,
        force=force,
    )
    log("  ", wv)
    log("fastText skip-gram 100d …")
    wv, model = train_fasttext(
        sentences, name="machado_ft100", dim=100, sg=1, epochs=ep, force=force
    )
    log("  ", wv, wv.most_similar("capitu", topn=5))


def train_glove(sentences, fast: bool, force: bool):
    from replang.accel import detect

    out = PATHS.models / "machado_glove100.npz"
    if out.exists() and not force:
        log("glove: cache existente")
        return
    if detect().use_jax_for("glove"):
        from replang.models.glove_jax import GloVeJax as GloVe

        log(f"GloVe (JAX, {detect().device_kind}) 100d …")
    else:
        from replang.models.glove_np import GloVe

        log("GloVe (numpy) 100d …")
    t = time.time()
    g = GloVe(dim=100, window=5, epochs=8 if fast else 15, min_count=5, x_max=50, batch_size=16384)
    g.fit(sentences, callback=lambda r: log("  glove", r))
    wv = g.to_wordvectors("GloVe 100d (numpy, Machado)")
    wv.save(out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "algo": "glove-jax" if detect().use_jax_for("glove") else "glove-numpy",
                "dim": 100,
                "window": 5,
                "epochs": g.epochs,
                "seconds": round(time.time() - t, 1),
                "vocab": len(wv),
                "history": g.history,
            }
        )
    )
    log("  ", wv, wv.most_similar("capitu", topn=5))


def train_doc2vec(fast: bool, force: bool):
    from replang.data.corpora import _strip_machado_header, iter_machado_raw
    from replang.models.trainers import train_doc2vec
    from replang.utils.text import iter_sentences

    docs, meta = [], []
    for fname, text in iter_machado_raw():
        sents = list(iter_sentences(_strip_machado_header(text)))
        # documentos = blocos de ~60 sentenças (capítulos aproximados)
        for i in range(0, len(sents), 60):
            chunk = [t for s in sents[i : i + 60] for t in s]
            if len(chunk) > 100:
                docs.append(chunk)
                meta.append({"arquivo": fname, "genero": fname.split("/")[1], "bloco": i // 60})
    (PATHS.models / "machado_docs_meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    log(f"doc2vec PV-DM sobre {len(docs)} blocos …")
    train_doc2vec(
        docs, name="machado_pvdm100", dim=100, dm=1, epochs=10 if fast else 20, force=force
    )
    log("doc2vec PV-DBOW …")
    train_doc2vec(
        docs, name="machado_pvdbow100", dim=100, dm=0, epochs=10 if fast else 20, force=force
    )


def train_bilm(sentences, fast: bool, force: bool):
    from replang.models.bilm_jax import HAS_JAX, BiLM, BiLMVocab

    out = PATHS.models / "machado_bilm.pkl"
    if out.exists() and not force:
        log("bilm: cache existente")
        return
    if not HAS_JAX:
        log("bilm: JAX não instalado (uv sync --extra contextual); pulando")
        return
    log("biLM (ELMo-lite, JAX) …")
    vocab = BiLMVocab(sentences, max_size=8000, min_count=3)
    enc = [vocab.encode(s) for s in sentences]
    model = BiLM(len(vocab), emb=64, hid=128, n_layers=2)
    steps = 400 if fast else 2500
    model.fit(
        enc,
        batch_size=32,
        seq_len=24,
        steps=steps,
        lr=3e-3,
        callback=lambda r: log("  bilm", r),
        log_every=50,
    )
    model.save(out, vocab)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "algo": "bilm-jax",
                "emb": 64,
                "hid": 128,
                "layers": 2,
                "steps": steps,
                "vocab": len(vocab),
                "history": model.history,
            }
        )
    )
    log("  salvo", out)


def train_legal(fast: bool, force: bool):
    """Modelos do eixo jurídico: Skip-gram, CBOW e fastText sobre decisões do STF (RulingBR)."""
    from replang.data.legal import load_legal_sentences
    from replang.models.trainers import train_fasttext, train_word2vec

    sentences = load_legal_sentences()
    log(
        f"corpus jurídico (RulingBR/STF): {len(sentences)} sentenças, {sum(map(len, sentences))} tokens"
    )
    ep = 3 if fast else 5
    log("legal word2vec skip-gram 100d …")
    wv = train_word2vec(
        sentences, name="legal_sg100", dim=100, sg=1, negative=10, epochs=ep, force=force
    )
    log("  ", wv, wv.most_similar("sentença", topn=5))
    log("legal word2vec CBOW 100d …")
    wv = train_word2vec(
        sentences, name="legal_cbow100", dim=100, sg=0, negative=10, epochs=ep, force=force
    )
    log("  ", wv)
    log("legal fastText skip-gram 100d …")
    wv, _ = train_fasttext(sentences, name="legal_ft100", dim=100, sg=1, epochs=ep, force=force)
    log("  ", wv, wv.most_similar("inconstitucionalidade", topn=5))


def main(what: str = "all", fast: bool = False, force: bool = False):
    PATHS.ensure()
    sentences = load_machado_sentences()
    log(
        f"corpus Machado: {len(sentences)} sentenças, {sum(map(len, sentences))} tokens; fast={fast}"
    )
    if what in ("all", "w2v", "fasttext"):
        train_gensim(sentences, fast, force)
    if what in ("all", "glove"):
        train_glove(sentences, fast, force)
    if what in ("all", "doc2vec"):
        train_doc2vec(fast, force)
    if what in ("all", "bilm"):
        train_bilm(sentences, fast, force)
    if what in ("all", "legal"):
        train_legal(fast, force)
    log("concluído")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.what, a.fast or settings.fast, a.force)
