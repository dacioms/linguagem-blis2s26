"""word2vec *from scratch* (numpy): Skip-gram e CBOW com amostragem negativa ou softmax hierárquico.

Implementa, de forma legível e vetorizada por mini-lotes, as ideias de:

* Mikolov et al. (2013a) – arquiteturas **CBOW** (contexto → palavra) e **Skip-gram** (palavra →
  contexto), modelos log-lineares sem camada oculta não-linear (complexidade ``N×D + D×log2(V)``);
* Mikolov et al. (2013b) – **amostragem negativa** (NEG, eq. 4), distribuição de ruído ``U(w)^{3/4}``,
  **subamostragem** de palavras frequentes (eq. 5) e **detecção de frases** (eq. 6);
* softmax hierárquico com árvore de **Huffman** (Morin & Bengio, 2005; Mikolov 2013b §2.1).

A implementação prioriza clareza sobre velocidade; para corpora grandes use
:mod:`replang.models.trainers` (gensim).
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np

from replang.embedding import WordVectors


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ----------------------------------------------------------------------- vocabulário
class Vocab:
    def __init__(
        self, sentences: Iterable[list[str]], min_count: int = 1, max_size: int | None = None
    ):
        counts = Counter(t for s in sentences for t in s)
        self.words = [w for w, c in counts.most_common(max_size) if c >= min_count]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = np.array([counts[w] for w in self.words], dtype=np.float64)
        self.total = float(self.counts.sum())

    def __len__(self) -> int:
        return len(self.words)

    def encode(self, sent: list[str]) -> list[int]:
        return [self.index[t] for t in sent if t in self.index]

    # ------------------------------------------------- subamostragem (Mikolov 2013b, eq. 5)
    def keep_prob(self, t: float = 1e-5) -> np.ndarray:
        """``P(manter w) = sqrt(t / f(w))`` (limitado a 1); ``f`` é a frequência relativa."""
        f = self.counts / self.total
        return np.minimum(1.0, np.sqrt(t / f))

    # ------------------------------------------------- distribuição de ruído U(w)^0.75
    def noise_distribution(self, power: float = 0.75) -> np.ndarray:
        p = self.counts**power
        return p / p.sum()


# ----------------------------------------------------------- softmax hierárquico (Huffman)
class HuffmanTree:
    """Codifica cada palavra como um caminho (nós internos + bits) na árvore de Huffman.

    Palavras frequentes ficam perto da raiz → códigos curtos → treino mais rápido.
    """

    def __init__(self, counts: np.ndarray):
        n = len(counts)
        heap = [(float(c), i, None) for i, c in enumerate(counts)]
        heapq.heapify(heap)
        parent = {}
        code = {}
        next_id = n
        while len(heap) > 1:
            c1, i1, _ = heapq.heappop(heap)
            c2, i2, _ = heapq.heappop(heap)
            parent[i1], code[i1] = next_id, 0
            parent[i2], code[i2] = next_id, 1
            heapq.heappush(heap, (c1 + c2, next_id, None))
            next_id += 1
        self.n_inner = next_id - n
        self.paths: list[list[int]] = []
        self.codes: list[list[int]] = []
        for w in range(n):
            nodes, bits = [], []
            cur = w
            while cur in parent:
                nodes.append(parent[cur] - n)
                bits.append(code[cur])
                cur = parent[cur]
            self.paths.append(nodes[::-1])
            self.codes.append(bits[::-1])
        self.max_len = max(len(p) for p in self.paths)
        # versões "preenchidas" para vetorização
        self.path_arr = np.zeros((n, self.max_len), dtype=np.int64)
        self.code_arr = np.zeros((n, self.max_len), dtype=np.float32)
        self.mask_arr = np.zeros((n, self.max_len), dtype=np.float32)
        for w in range(n):
            L = len(self.paths[w])
            self.path_arr[w, :L] = self.paths[w]
            self.code_arr[w, :L] = self.codes[w]
            self.mask_arr[w, :L] = 1.0

    def average_code_length(self, counts: np.ndarray) -> float:
        return float(
            sum(len(p) * c for p, c in zip(self.paths, counts, strict=True)) / counts.sum()
        )


# ------------------------------------------------------------------------- modelo
class Word2Vec:
    """Skip-gram/CBOW com NEG ou HS, treinado por SGD em mini-lotes.

    Parâmetros principais (nomes espelham o ``word2vec.c``/gensim):
    ``sg`` (1 = Skip-gram, 0 = CBOW), ``hs`` (1 = softmax hierárquico, 0 = amostragem negativa),
    ``negative`` (k negativos), ``sample`` (limiar t da subamostragem; 0 desliga), ``window``,
    ``alpha`` (taxa de aprendizado inicial, decai linearmente), ``dynamic_window`` (sorteia a
    janela efetiva em [1, window] como no código original).
    """

    def __init__(
        self,
        dim: int = 50,
        window: int = 5,
        sg: int = 1,
        hs: int = 0,
        negative: int = 5,
        sample: float = 1e-3,
        alpha: float = 0.025,
        min_alpha: float = 1e-4,
        epochs: int = 5,
        min_count: int = 1,
        batch_size: int = 128,
        dynamic_window: bool = True,
        seed: int = 42,
        max_vocab: int | None = None,
        cbow_mean: bool = True,
    ):
        self.dim, self.window, self.sg, self.hs, self.negative = dim, window, sg, hs, negative
        self.sample, self.alpha, self.min_alpha, self.epochs = sample, alpha, min_alpha, epochs
        self.min_count, self.batch_size, self.dynamic_window, self.cbow_mean = (
            min_count,
            batch_size,
            dynamic_window,
            cbow_mean,
        )
        self.max_vocab = max_vocab
        self.rng = np.random.default_rng(seed)
        self.vocab: Vocab | None = None
        self.W_in: np.ndarray | None = None  # vetores de entrada v_w
        self.W_out: np.ndarray | None = None  # vetores de saída v'_w (ou dos nós internos, no HS)
        self.tree: HuffmanTree | None = None
        self.history: list[dict] = []

    # ------------------------------------------------------------- preparação
    def build_vocab(self, sentences: list[list[str]]) -> Word2Vec:
        self.vocab = Vocab(sentences, self.min_count, self.max_vocab)
        V = len(self.vocab)
        self.W_in = (self.rng.random((V, self.dim), dtype=np.float32) - 0.5) / self.dim
        if self.hs:
            self.tree = HuffmanTree(self.vocab.counts)
            self.W_out = np.zeros((self.tree.n_inner, self.dim), dtype=np.float32)
        else:
            self.W_out = np.zeros((V, self.dim), dtype=np.float32)
            self.noise = self.vocab.noise_distribution()
            self._noise_cdf = np.cumsum(self.noise)
        self.keep = self.vocab.keep_prob(self.sample) if self.sample > 0 else None
        return self

    def _encode_corpus(self, sentences: list[list[str]]) -> list[np.ndarray]:
        out = []
        for s in sentences:
            ids = np.array(self.vocab.encode(s), dtype=np.int64)
            if self.keep is not None and len(ids):
                ids = ids[self.rng.random(len(ids)) < self.keep[ids]]
            if len(ids) > 1:
                out.append(ids)
        return out

    def _pairs(self, sentences: list[np.ndarray]):
        """Gera pares (centro, contexto) — Skip-gram — ou (contextos, centro) — CBOW."""
        for ids in sentences:
            n = len(ids)
            for i in range(n):
                b = self.rng.integers(1, self.window + 1) if self.dynamic_window else self.window
                lo, hi = max(0, i - b), min(n, i + b + 1)
                ctx = [ids[j] for j in range(lo, hi) if j != i]
                if not ctx:
                    continue
                if self.sg:
                    for c in ctx:
                        yield ids[i], c
                else:
                    yield ctx, ids[i]

    def _sample_negatives(self, size: tuple[int, ...]) -> np.ndarray:
        u = self.rng.random(size)
        return np.searchsorted(self._noise_cdf, u).clip(0, len(self.noise) - 1)

    # ------------------------------------------------------------- passos de gradiente
    def _step_neg(self, h: np.ndarray, targets: np.ndarray, lr: float) -> tuple[float, np.ndarray]:
        """Amostragem negativa (eq. 4 de Mikolov 2013b).

        ``h``: (B, d) representação de entrada (vetor da palavra central no SG; média do contexto no CBOW);
        ``targets``: (B,) palavra a prever. Retorna perda média e gradiente em relação a ``h``.
        """
        B = len(targets)
        neg = self._sample_negatives((B, self.negative))
        # (B, 1+k, d): positivo seguido de k negativos
        idx = np.concatenate([targets[:, None], neg], axis=1)
        out = self.W_out[idx]
        labels = np.zeros((B, 1 + self.negative), dtype=np.float32)
        labels[:, 0] = 1.0
        score = np.einsum("bd,bkd->bk", h, out)
        p = sigmoid(score)
        loss = -(np.log(p[:, 0] + 1e-9).sum() + np.log(1 - p[:, 1:] + 1e-9).sum()) / B
        g = (p - labels) * lr  # dL/dscore
        grad_h = np.einsum("bk,bkd->bd", g, out)
        grad_out = np.einsum("bk,bd->bkd", g, h)
        np.add.at(self.W_out, idx.ravel(), -grad_out.reshape(-1, self.dim))
        return float(loss), grad_h

    def _step_hs(self, h: np.ndarray, targets: np.ndarray, lr: float) -> tuple[float, np.ndarray]:
        """Softmax hierárquico (eq. 3): produto de sigmoides ao longo do caminho de Huffman."""
        nodes = self.tree.path_arr[targets]  # (B, L)
        codes = self.tree.code_arr[targets]
        mask = self.tree.mask_arr[targets]
        out = self.W_out[nodes]  # (B, L, d)
        score = np.einsum("bd,bld->bl", h, out)
        p = sigmoid(score)
        # bit 0 → ramo esquerdo com prob σ(score); bit 1 → direito com prob 1−σ(score)
        ll = (1 - codes) * np.log(p + 1e-9) + codes * np.log(1 - p + 1e-9)
        loss = -(ll * mask).sum() / len(targets)
        g = (p - (1 - codes)) * mask * lr
        grad_h = np.einsum("bl,bld->bd", g, out)
        grad_out = np.einsum("bl,bd->bld", g, h)
        np.add.at(self.W_out, nodes.ravel(), -grad_out.reshape(-1, self.dim))
        return float(loss), grad_h

    # ------------------------------------------------------------- treino
    def train(
        self,
        sentences: list[list[str]],
        callback: Callable[[dict], None] | None = None,
        log_every: int = 50,
    ) -> Word2Vec:
        if self.vocab is None:
            self.build_vocab(sentences)
        step_fn = self._step_hs if self.hs else self._step_neg
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
                if self.sg:
                    centers = np.array([p[0] for p in batch])
                    targets = np.array([p[1] for p in batch])
                    h = self.W_in[centers]
                    loss, grad_h = step_fn(h, targets, lr)
                    np.add.at(self.W_in, centers, -grad_h)
                else:
                    targets = np.array([p[1] for p in batch])
                    lens = np.array([len(p[0]) for p in batch])
                    L = lens.max()
                    ctx = np.zeros((len(batch), L), dtype=np.int64)
                    m = np.zeros((len(batch), L), dtype=np.float32)
                    for i, p in enumerate(batch):
                        ctx[i, : len(p[0])] = p[0]
                        m[i, : len(p[0])] = 1.0
                    denom = lens[:, None] if self.cbow_mean else 1.0
                    h = (self.W_in[ctx] * m[:, :, None]).sum(1) / denom
                    loss, grad_h = step_fn(h, targets, lr)
                    g = (
                        grad_h[:, None, :] / denom[:, :, None]
                        if self.cbow_mean
                        else grad_h[:, None, :]
                    ) * m[:, :, None]
                    np.add.at(self.W_in, ctx.ravel(), -g.reshape(-1, self.dim))
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

    # ------------------------------------------------------------- saída
    def to_wordvectors(self, name: str | None = None, *, combine: str = "in") -> WordVectors:
        """``combine``: ``"in"`` (vetores de entrada, padrão do word2vec), ``"out"`` ou ``"sum"``."""
        name = name or ("Skip-gram" if self.sg else "CBOW") + (
            "-HS" if self.hs else f"-NEG{self.negative}"
        )
        if combine == "in" or self.hs:
            vec = self.W_in
        elif combine == "out":
            vec = self.W_out
        else:
            vec = self.W_in + self.W_out
        return WordVectors(self.vocab.words, vec, name, self.vocab.counts)

    def predict_context(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        """``p(c | w)`` pelo softmax completo (eq. 2) – só viável para vocabulários pequenos."""
        v = self.W_in[self.vocab.index[word]]
        scores = self.W_out @ v if not self.hs else None
        if scores is None:
            raise ValueError(
                "predict_context exige modelo com amostragem negativa (W_out por palavra)"
            )
        p = np.exp(scores - scores.max())
        p /= p.sum()
        idx = np.argsort(-p)[:topn]
        return [(self.vocab.words[i], float(p[i])) for i in idx]


# ----------------------------------------------------------- frases (Mikolov 2013b, §4)
def learn_phrases(
    sentences: list[list[str]],
    *,
    min_count: int = 5,
    threshold: float = 10.0,
    delta: float | None = None,
    passes: int = 1,
    sep: str = "_",
) -> tuple[list[list[str]], dict[str, float]]:
    """Detecta bigramas com ``score = (count(ab) − δ) / (count(a)·count(b))`` acima de ``threshold``.

    O escore é multiplicado por ``N`` (total de tokens), como no ``word2phrase.c``, para que
    ``threshold`` tenha escala interpretável. ``δ`` desconta bigramas raros (padrão: ``min_count``).
    Retorna o corpus com as frases unidas por ``sep`` e o dicionário ``{frase: score}``.
    """
    delta = min_count if delta is None else delta
    for _ in range(passes):
        uni: Counter = Counter()
        bi: Counter = Counter()
        for s in sentences:
            uni.update(s)
            bi.update(zip(s, s[1:], strict=False))
        N = sum(uni.values())
        scores = {}
        for (a, b), c in bi.items():
            if c < min_count:
                continue
            sc = (c - delta) / (uni[a] * uni[b]) * N
            if sc > threshold:
                scores[a + sep + b] = sc
        new = []
        for s in sentences:
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i] + sep + s[i + 1]) in scores:
                    out.append(s[i] + sep + s[i + 1])
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            new.append(out)
        sentences = new
    return sentences, scores
