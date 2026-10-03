"""Skip-gram com amostragem negativa (e opcionalmente subpalavras, como o fastText) em JAX.

Mesmo algoritmo de :mod:`replang.models.word2vec_np` (eq. 4 de Mikolov 2013b), com o passo de
gradiente compilado e o *scatter-add* na GPU/CPU. Diferenças práticas:

* lotes maiores (padrão 512 em CPU, 4096 em GPU) com **recorte do gradiente agregado por linha**,
  para que atualizações repetidas da mesma palavra num lote não divirjam;
* a amostragem negativa e o embaralhamento são feitos com ``numpy`` fora do ``jit`` (barato).

Use quando precisar treinar muitos modelos pequenos (ex.: o experimento sintético do notebook 10)
ou corpora médios sem o gensim; para corpora grandes em CPU o gensim (C, multi-thread) continua
competitivo.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from replang.accel import detect
from replang.embedding import WordVectors
from replang.models.fasttext_np import char_ngrams, fnv1a
from replang.models.word2vec_np import Vocab


class SkipGramJax:
    def __init__(
        self,
        dim: int = 50,
        window: int = 5,
        negative: int = 5,
        sample: float = 1e-3,
        alpha: float = 0.025,
        min_alpha: float = 1e-4,
        epochs: int = 5,
        min_count: int = 1,
        batch_size: int | None = None,
        seed: int = 42,
        max_vocab: int | None = None,
        subword: bool = False,
        min_n: int = 3,
        max_n: int = 6,
        bucket: int = 200_000,
        clip: float = 20.0,
        dynamic_window: bool = True,
    ):
        import jax  # noqa: F401

        self.dim, self.window, self.negative, self.sample = dim, window, negative, sample
        self.alpha, self.min_alpha, self.epochs, self.min_count, self.max_vocab = (
            alpha,
            min_alpha,
            epochs,
            min_count,
            max_vocab,
        )
        self.batch_size = batch_size or (4096 if detect().gpu else 512)
        self.subword, self.min_n, self.max_n, self.bucket, self.clip = (
            subword,
            min_n,
            max_n,
            bucket,
            clip,
        )
        self.dynamic_window = dynamic_window
        self.rng = np.random.default_rng(seed)
        self.vocab: Vocab | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------- preparação
    def build_vocab(self, sentences):
        self.vocab = Vocab(sentences, self.min_count, self.max_vocab)
        V = len(self.vocab)
        self.rows = V + (self.bucket if self.subword else 0)
        self.keep = self.vocab.keep_prob(self.sample) if self.sample > 0 else None
        self._noise_cdf = np.cumsum(self.vocab.noise_distribution())
        if self.subword:
            ids = [self._ngram_ids(w, i) for i, w in enumerate(self.vocab.words)]
            L = max(len(g) for g in ids)
            self.ng_arr = np.zeros((V, L), dtype=np.int32)
            self.ng_mask = np.zeros((V, L), dtype=np.float32)
            for i, g in enumerate(ids):
                self.ng_arr[i, : len(g)] = g
                self.ng_mask[i, : len(g)] = 1.0
        return self

    def _ngram_ids(self, word: str, wid: int | None) -> np.ndarray:
        V = len(self.vocab)
        ids = ([wid] if wid is not None else []) + [
            V + (fnv1a(g) % self.bucket) for g in char_ngrams(word, self.min_n, self.max_n)
        ]
        return np.array(ids, dtype=np.int32)

    def _pairs(self, sentences) -> np.ndarray:
        out = []
        for s in sentences:
            ids = np.array(self.vocab.encode(s), dtype=np.int32)
            if self.keep is not None and len(ids):
                ids = ids[self.rng.random(len(ids)) < self.keep[ids]]
            n = len(ids)
            if n < 2:
                continue
            b = (
                self.rng.integers(1, self.window + 1, size=n)
                if self.dynamic_window
                else np.full(n, self.window)
            )
            for i in range(n):
                lo, hi = max(0, i - b[i]), min(n, i + b[i] + 1)
                for j in range(lo, hi):
                    if j != i:
                        out.append((ids[i], ids[j]))
        return np.array(out, dtype=np.int32) if out else np.zeros((0, 2), dtype=np.int32)

    # ------------------------------------------------------------- treino
    def train(
        self, sentences, callback: Callable[[dict], None] | None = None, log_every: int = 50
    ) -> SkipGramJax:
        import jax
        import jax.numpy as jnp

        if self.vocab is None:
            self.build_vocab(sentences)
        V, d, k, clip = len(self.vocab), self.dim, self.negative, self.clip
        W_in = jnp.asarray((self.rng.random((self.rows, d), dtype=np.float32) - 0.5) / d)
        W_out = jnp.zeros((V, d), jnp.float32)
        if self.subword:
            ng_arr, ng_mask = jnp.asarray(self.ng_arr), jnp.asarray(self.ng_mask)
            ng_len = ng_mask.sum(1)

        @jax.jit
        def step(W_in, W_out, centers, targets, neg, lr):
            idx = jnp.concatenate([targets[:, None], neg], 1)  # (B, 1+k)
            if self.subword:
                ng, m, ln = ng_arr[centers], ng_mask[centers], ng_len[centers]
                h = (W_in[ng] * m[:, :, None]).sum(1) / ln[:, None]
            else:
                h = W_in[centers]
            out = W_out[idx]
            labels = jnp.concatenate(
                [jnp.ones((centers.shape[0], 1)), jnp.zeros((centers.shape[0], k))], 1
            )
            p = jax.nn.sigmoid(jnp.einsum("bd,bkd->bk", h, out))
            loss = (
                -(jnp.log(p[:, 0] + 1e-9).sum() + jnp.log(1 - p[:, 1:] + 1e-9).sum())
                / centers.shape[0]
            )
            g = p - labels
            grad_h = jnp.einsum("bk,bkd->bd", g, out)
            grad_out = jnp.einsum("bk,bd->bkd", g, h).reshape(-1, d)
            # acumula por linha, recorta a norma do agregado (evita divergência com lotes grandes) e aplica
            upd_out = jnp.zeros_like(W_out).at[idx.reshape(-1)].add(grad_out)
            nrm = jnp.linalg.norm(upd_out, axis=1, keepdims=True)
            upd_out = upd_out * jnp.minimum(1.0, clip / (nrm + 1e-9))
            W_out = W_out - lr * upd_out
            if self.subword:
                gh = (grad_h / ln[:, None])[:, None, :] * m[:, :, None]
                upd_in = jnp.zeros_like(W_in).at[ng.reshape(-1)].add(gh.reshape(-1, d))
            else:
                upd_in = jnp.zeros_like(W_in).at[centers].add(grad_h)
            nrm = jnp.linalg.norm(upd_in, axis=1, keepdims=True)
            upd_in = upd_in * jnp.minimum(1.0, clip / (nrm + 1e-9))
            W_in = W_in - lr * upd_in
            return W_in, W_out, loss

        bs = self.batch_size
        total_steps, stepn = None, 0
        for ep in range(self.epochs):
            pairs = self._pairs(sentences)
            self.rng.shuffle(pairs)
            nb = max(1, math.ceil(len(pairs) / bs))
            if total_steps is None:
                total_steps = self.epochs * nb
            losses = []
            for b0 in range(0, len(pairs), bs):
                batch = pairs[b0 : b0 + bs]
                lr = max(self.min_alpha, self.alpha * (1 - stepn / total_steps))
                neg = (
                    np.searchsorted(self._noise_cdf, self.rng.random((len(batch), k)))
                    .clip(0, V - 1)
                    .astype(np.int32)
                )
                W_in, W_out, loss = step(
                    W_in,
                    W_out,
                    jnp.asarray(batch[:, 0]),
                    jnp.asarray(batch[:, 1]),
                    jnp.asarray(neg),
                    jnp.float32(lr),
                )
                losses.append(float(loss))
                stepn += 1
                if callback and stepn % log_every == 0:
                    callback(
                        {
                            "epoch": ep,
                            "step": stepn,
                            "loss": float(np.mean(losses[-log_every:])),
                            "lr": lr,
                        }
                    )
            rec = {
                "epoch": ep,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "pairs": len(pairs),
            }
            self.history.append(rec)
            if callback:
                callback({**rec, "step": stepn, "lr": lr, "end_epoch": True})
        self.W_in, self.W_out = np.asarray(W_in), np.asarray(W_out)
        return self

    # ------------------------------------------------------------- saída
    def to_wordvectors(self, name: str | None = None) -> WordVectors:
        name = name or ("fastText (JAX)" if self.subword else "Skip-gram NEG (JAX)")
        V = len(self.vocab)
        if self.subword:
            vec = (self.W_in[self.ng_arr] * self.ng_mask[:, :, None]).sum(1) / self.ng_mask.sum(1)[
                :, None
            ]
            return WordVectors(self.vocab.words, vec[:V], name, self.vocab.counts)
        return WordVectors(self.vocab.words, self.W_in[:V], name, self.vocab.counts)

    def word_vector(self, word: str) -> np.ndarray:
        if not self.subword:
            return self.W_in[self.vocab.index[word]]
        ids = self._ngram_ids(word, self.vocab.index.get(word))
        return self.W_in[ids].mean(0)
