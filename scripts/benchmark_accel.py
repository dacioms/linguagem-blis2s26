"""Mede o ganho dos backends acelerados nesta máquina (numpy × JAX; série × paralelo).

Uso: ``uv run python scripts/benchmark_accel.py [--quick]`` → imprime uma tabela Markdown.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from replang.accel import report  # noqa: E402


def timed(fn):
    t = time.time()
    out = fn()
    return out, time.time() - t


def main(quick: bool = False):
    from replang.data.corpora import load_machado_sentences, load_macmorpho, synthetic_gender_corpus
    from replang.data.embeddings import load_ptwiki
    from replang.eval.extrinsic import pos_tagging_eval
    from replang.eval.parallel import parallel_map
    from replang.models.glove_jax import GloVeJax
    from replang.models.glove_np import GloVe
    from replang.models.word2vec_jax import SkipGramJax
    from replang.models.word2vec_np import Word2Vec

    print("Ambiente:", report())
    rows = []
    sents = load_machado_sentences(30000 if quick else 60000)
    ep = 2 if quick else 3
    _, tn = timed(
        lambda: GloVe(dim=100, window=5, epochs=ep, min_count=5, x_max=50, batch_size=16384).fit(
            sents
        )
    )
    _, tj = timed(
        lambda: GloVeJax(dim=100, window=5, epochs=ep, min_count=5, x_max=50, batch_size=16384).fit(
            sents
        )
    )
    rows.append((f"GloVe 100d, Machado ({len(sents)} sentenças, {ep} épocas)", tn, tj))
    corpus = synthetic_gender_corpus(8000 if quick else 15000, bias=0.9, seed=0)
    _, tn = timed(
        lambda: Word2Vec(
            dim=50,
            window=5,
            negative=10,
            epochs=4 if quick else 8,
            sample=0,
            batch_size=128,
            seed=0,
        ).train(corpus)
    )
    _, tj = timed(
        lambda: SkipGramJax(
            dim=50, window=5, negative=10, epochs=4 if quick else 8, sample=0, seed=0
        ).train(corpus)
    )
    rows.append((f"Skip-gram NEG-10 50d, corpus sintético ({len(corpus)} frases)", tn, tj))
    pt = load_ptwiki()
    tr = load_macmorpho(split="train")
    te = load_macmorpho(split="test", limit_sentences=300)
    ntok = 30000 if quick else 60000
    _, ts = timed(lambda: pos_tagging_eval(pt, tr, te, max_train_tokens=ntok, backend="sklearn"))
    _, tj = timed(lambda: pos_tagging_eval(pt, tr, te, max_train_tokens=ntok, backend="jax"))
    rows.append((f"Classificador POS ({ntok} tokens, janela ±2)", ts, tj))
    models = [pt.head(20000, f"m{i}") for i in range(4)]

    def fn(wv):
        return pos_tagging_eval(wv, tr, te, max_train_tokens=ntok // 2, backend="jax")["acurácia"]

        "acurácia"
    ]  # noqa: E731
    _, t1 = timed(lambda: [fn(m) for m in models])
    _, tp = timed(lambda: parallel_map(fn, models))
    rows.append(("4 avaliações POS: série × paralelo (joblib)", t1, tp))
    print("\n| Tarefa | base (s) | acelerado (s) | ganho |\n|---|---|---|---|")
    for name, a, b in rows:
        print(f"| {name} | {a:.1f} | {b:.1f} | {a / b:.1f}× |")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
