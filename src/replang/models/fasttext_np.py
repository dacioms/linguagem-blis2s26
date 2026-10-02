"""fastText *from scratch* (Bojanowski et al., 2017): Skip-gram com informação de **subpalavras**.

Cada palavra ``w`` é uma *bag* de n-gramas de caracteres (3 ≤ n ≤ 6) delimitados por ``<`` e ``>``,
mais a própria palavra. Ex.: ``<onde>`` → ``<on, ond, nde, de>, <ond, onde, nde>, ... , <onde>``.
A pontuação passa a ser ``s(w, c) = Σ_{g ∈ G_w} z_gᵀ v_c`` (eq. da §3.2) e os n-gramas são
mapeados em ``K`` *buckets* por *hashing* FNV-1a (§3.2, K = 2·10⁶ no original). Consequências:

* compartilhamento de parâmetros entre palavras morfologicamente relacionadas;
* vetores para palavras **fora do vocabulário** (OOV) — soma dos n-gramas;
* melhor em línguas morfologicamente ricas (e nas analogias *sintáticas* de Hartmann et al., 2017).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from replang.embedding import WordVectors
from replang.models.word2vec_np import Word2Vec

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619


def fnv1a(s: str) -> int:
    """FNV-1a de 32 bits sobre os bytes UTF-8 (como no fastText original)."""
    h = FNV_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def char_ngrams(word: str, min_n: int = 3, max_n: int = 6, *, boundaries: bool = True) -> list[str]:
    """N-gramas de caracteres com delimitadores de palavra (``<`` e ``>``)."""
    w = f"<{word}>" if boundaries else word
    grams = []
    for n in range(min_n, max_n + 1):
        for i in range(0, len(w) - n + 1):
            grams.append(w[i : i + n])
    return grams


class FastText(Word2Vec):
    """Skip-gram com amostragem negativa onde o vetor de entrada é a média dos vetores
    dos n-gramas (+ palavra inteira). ``bucket`` controla o tamanho da tabela de *hash*."""

    def __init__(self, dim: int = 50, min_n: int = 3, max_n: int = 6, bucket: int = 200_000, **kw):
        kw.setdefault("sg", 1)
        kw["hs"] = 0
        super().__init__(dim=dim, **kw)
        self.min_n, self.max_n, self.bucket = min_n, max_n, bucket
        self.ngram_ids: list[np.ndarray] = []

    def _word_ngram_ids(self, word: str, word_id: int | None) -> np.ndarray:
        V = len(self.vocab)
        ids = [word_id] if word_id is not None else []
        ids += [V + (fnv1a(g) % self.bucket) for g in char_ngrams(word, self.min_n, self.max_n)]
        return np.array(ids, dtype=np.int64)

    def build_vocab(self, sentences):
        super().build_vocab(sentences)
        V = len(self.vocab)
        self.W_in = (
            self.rng.random((V + self.bucket, self.dim), dtype=np.float32) - 0.5
        ) / self.dim
        self.ngram_ids = [self._word_ngram_ids(w, i) for i, w in enumerate(self.vocab.words)]
        L = max(len(g) for g in self.ngram_ids)
        self.ng_arr = np.zeros((V, L), dtype=np.int64)
        self.ng_mask = np.zeros((V, L), dtype=np.float32)
        for i, g in enumerate(self.ngram_ids):
            self.ng_arr[i, : len(g)] = g
            self.ng_mask[i, : len(g)] = 1.0
        self.ng_len = self.ng_mask.sum(1)
        return self

    def train(self, sentences, callback: Callable[[dict], None] | None = None, log_every: int = 50):
        if self.vocab is None:
            self.build_vocab(sentences)
        import math

        total_steps = None
        step = 0
        for epoch in range(self.epochs):
            enc = self._encode_corpus(sentences)
            pairs = list(self._pairs(enc))
            self.rng.shuffle(pairs)
            if total_steps is None:
                total_steps = max(1, self.epochs * math.ceil(len(pairs) / self.batch_size))
            losses = []
            for b0 in range(0, len(pairs), self.batch_size):
                batch = pairs[b0 : b0 + self.batch_size]
                lr = max(self.min_alpha, self.alpha * (1 - step / total_steps))
                centers = np.array([p[0] for p in batch])
                targets = np.array([p[1] for p in batch])
                ng, m, ln = self.ng_arr[centers], self.ng_mask[centers], self.ng_len[centers]
                h = (self.W_in[ng] * m[:, :, None]).sum(1) / ln[:, None]
                loss, grad_h = self._step_neg(h, targets, lr)
                g = (grad_h / ln[:, None])[:, None, :] * m[:, :, None]
                np.add.at(self.W_in, ng.ravel(), -g.reshape(-1, self.dim))
                losses.append(loss)
                step += 1
                if callback and step % log_every == 0:
                    callback(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": float(np.mean(losses[-log_every:])),
                            "lr": lr,
                        }
                    )
            rec = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "pairs": len(pairs),
            }
            self.history.append(rec)
            if callback:
                callback({**rec, "step": step, "lr": lr, "end_epoch": True})
        return self

    # ------------------------------------------------------------- vetores
    def word_vector(self, word: str) -> np.ndarray:
        """Vetor de **qualquer** palavra (inclusive OOV): média dos n-gramas (+ palavra se conhecida)."""
        wid = self.vocab.index.get(word)
        ids = self._word_ngram_ids(word, wid)
        return self.W_in[ids].mean(0)

    def to_wordvectors(self, name: str = "fastText (numpy)", **_) -> WordVectors:
        V = len(self.vocab)
        vec = (self.W_in[self.ng_arr] * self.ng_mask[:, :, None]).sum(1) / self.ng_len[:, None]
        return WordVectors(self.vocab.words, vec[:V], name, self.vocab.counts)

    def ngram_importance(self, word: str) -> list[tuple[str, float]]:
        """§6.2 do artigo: importância de cada n-grama = queda do cosseno ao removê-lo."""
        grams = char_ngrams(word, self.min_n, self.max_n)
        wid = self.vocab.index.get(word)
        V = len(self.vocab)
        all_ids = ([wid] if wid is not None else []) + [V + (fnv1a(g) % self.bucket) for g in grams]
        full = self.W_in[all_ids].mean(0)
        out = []
        for k, g in enumerate(grams):
            keep = [i for n, i in enumerate(all_ids) if n != k + (1 if wid is not None else 0)]
            v = self.W_in[keep].mean(0)
            out.append((g, float(WordVectors.cosine(full, v))))
        return sorted(out, key=lambda t: t[1])
