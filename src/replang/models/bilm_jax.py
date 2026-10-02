"""ELMo-lite: modelo de linguagem **bidirecional** (biLM) em JAX e representações contextuais.

Reproduz, em miniatura, Peters et al. (2018):

* um LM *forward* ``p(t_k | t_1..t_{k-1})`` e um *backward* ``p(t_k | t_{k+1}..t_N)`` com LSTMs de
  ``L=2`` camadas e parâmetros de embedding/softmax compartilhados entre as direções (§3.1);
* para cada token, ``2L+1`` representações: ``x_k`` (camada 0, duplicada) e ``[h→_{k,j}; h←_{k,j}]``
  para ``j=1..L`` (§3.2);
* ``ELMo_k = γ Σ_j s_j h_{k,j}`` com pesos ``s`` *softmax-normalizados* aprendidos pela tarefa (eq. 1).

Sem CNN de caracteres (usamos *lookup* de palavras) e com dimensões pequenas para treinar em CPU em
poucos minutos. O objetivo é **mostrar** que (i) a mesma palavra ganha vetores diferentes conforme
o contexto (polissemia: *banco*, *manga*) e (ii) camadas diferentes codificam informação
diferente (sintaxe na camada 1, semântica/sentido na camada 2 — §5.3).
"""

from __future__ import annotations

import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:  # jax é opcional (extra "contextual")
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except Exception:  # noqa: BLE001
    HAS_JAX = False

BOS, EOS, UNK, PAD = "<s>", "</s>", "<unk>", "<pad>"


class BiLMVocab:
    def __init__(self, sentences, max_size: int = 8000, min_count: int = 2):
        counts = Counter(t for s in sentences for t in s)
        words = [w for w, c in counts.most_common(max_size) if c >= min_count]
        self.words = [PAD, UNK, BOS, EOS, *words]
        self.index = {w: i for i, w in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def encode(self, sent: list[str]) -> list[int]:
        unk = self.index[UNK]
        return [self.index[BOS], *[self.index.get(t, unk) for t in sent], self.index[EOS]]


def _init_lstm(key, in_dim, hid):
    k1, k2 = jax.random.split(key)
    s = 1.0 / np.sqrt(in_dim + hid)
    return {
        "Wx": jax.random.uniform(k1, (in_dim, 4 * hid), minval=-s, maxval=s),
        "Wh": jax.random.uniform(k2, (hid, 4 * hid), minval=-s, maxval=s),
        "b": jnp.zeros(4 * hid).at[hid : 2 * hid].set(1.0),  # forget bias = 1
    }


def _lstm_scan(params, xs, hid):
    """xs: (T, B, in) → hs: (T, B, hid)."""

    def step(carry, x):
        h, c = carry
        z = x @ params["Wx"] + h @ params["Wh"] + params["b"]
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), h

    B = xs.shape[1]
    init = (jnp.zeros((B, hid)), jnp.zeros((B, hid)))
    _, hs = jax.lax.scan(step, init, xs)
    return hs


class BiLM:
    """Dois LMs (→ e ←) com ``n_layers`` LSTMs cada, embedding e softmax compartilhados."""

    def __init__(
        self, vocab_size: int, emb: int = 64, hid: int = 128, n_layers: int = 2, seed: int = 0
    ):
        assert HAS_JAX, "instale o extra: uv sync --extra contextual"
        self.V, self.emb, self.hid, self.L = vocab_size, emb, hid, n_layers
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2 * n_layers + 2)
        self.params = {
            "E": jax.random.normal(keys[0], (vocab_size, emb)) * 0.1,
            "proj_in": jax.random.normal(keys[1], (emb, hid)) * (1 / np.sqrt(emb)),
            "fwd": [_init_lstm(keys[2 + j], hid, hid) for j in range(n_layers)],
            "bwd": [_init_lstm(keys[2 + n_layers + j], hid, hid) for j in range(n_layers)],
            "out_W": jax.random.normal(keys[-1], (hid, emb)) * (1 / np.sqrt(hid)),
            "out_b": jnp.zeros(vocab_size),
        }
        self.history: list[dict] = []
        self._fwd_fn = jax.jit(self._forward_all)
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss))

    # ------------------------------------------------------------- modelo
    def _forward_all(self, params, tokens):
        """tokens: (B, T) → camadas (L+1, B, T, 2*hid) e logits fwd/bwd (B, T, V)."""
        x = params["E"][tokens]  # (B, T, emb)
        x = jnp.tanh(x @ params["proj_in"])  # (B, T, hid)
        xs = jnp.swapaxes(x, 0, 1)  # (T, B, hid)
        h_f, h_b = xs, xs[::-1]
        layers_f, layers_b = [], []
        for j in range(self.L):
            out_f = _lstm_scan(params["fwd"][j], h_f, self.hid)
            out_b = _lstm_scan(params["bwd"][j], h_b, self.hid)
            h_f = out_f + h_f if j > 0 else out_f  # conexão residual entre camadas
            h_b = out_b + h_b if j > 0 else out_b
            layers_f.append(h_f)
            layers_b.append(h_b[::-1])
        # softmax compartilhado: projeta hid → emb e usa a matriz de embedding (tied)
        logits_f = (jnp.swapaxes(layers_f[-1], 0, 1) @ params["out_W"]) @ params["E"].T + params[
            "out_b"
        ]
        logits_b = (jnp.swapaxes(layers_b[-1], 0, 1) @ params["out_W"]) @ params["E"].T + params[
            "out_b"
        ]
        reps = [jnp.concatenate([x, x], -1)] + [
            jnp.swapaxes(jnp.concatenate([layers_f[j], layers_b[j]], -1), 0, 1)
            for j in range(self.L)
        ]
        return jnp.stack(reps), logits_f, logits_b

    def _loss(self, params, tokens, mask):
        _, lf, lb = self._forward_all(params, tokens)
        # forward prevê t_{k+1} a partir da posição k; backward prevê t_{k-1} a partir de k
        tgt_f, tgt_b = tokens[:, 1:], tokens[:, :-1]
        lp_f = jax.nn.log_softmax(lf[:, :-1], -1)
        lp_b = jax.nn.log_softmax(lb[:, 1:], -1)
        nll_f = -jnp.take_along_axis(lp_f, tgt_f[..., None], -1)[..., 0] * mask[:, 1:]
        nll_b = -jnp.take_along_axis(lp_b, tgt_b[..., None], -1)[..., 0] * mask[:, :-1]
        n = mask[:, 1:].sum()
        return (nll_f.sum() + nll_b.sum()) / (2 * n)

    # ------------------------------------------------------------- treino (Adam manual)
    def train(
        self,
        batches,
        *,
        epochs: int = 1,
        lr: float = 2e-3,
        callback=None,
        log_every: int = 20,
        max_steps: int | None = None,
    ):
        import optax  # noqa: F401  # pragma: no cover

    def fit(
        self,
        encoded: list[list[int]],
        *,
        batch_size: int = 32,
        seq_len: int = 24,
        steps: int = 600,
        lr: float = 3e-3,
        callback=None,
        log_every: int = 25,
        seed: int = 0,
    ) -> BiLM:
        rng = np.random.default_rng(seed)
        # Adam
        m = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        v = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        b1, b2, eps = 0.9, 0.999, 1e-8

        @jax.jit
        def update(params, m, v, grads, t):
            m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
            v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, v, grads)
            mhat = jax.tree_util.tree_map(lambda a: a / (1 - b1**t), m)
            vhat = jax.tree_util.tree_map(lambda a: a / (1 - b2**t), v)
            params = jax.tree_util.tree_map(
                lambda p, a, b: p - lr * a / (jnp.sqrt(b) + eps), params, mhat, vhat
            )
            return params, m, v

        data = [s for s in encoded if len(s) >= 4]
        t0 = time.time()
        losses = []
        for step in range(1, steps + 1):
            idx = rng.integers(0, len(data), batch_size)
            tok = np.zeros((batch_size, seq_len), dtype=np.int32)
            mask = np.zeros((batch_size, seq_len), dtype=np.float32)
            for i, k in enumerate(idx):
                s = data[k][:seq_len]
                tok[i, : len(s)] = s
                mask[i, : len(s)] = 1
            loss, grads = self._loss_grad(self.params, jnp.asarray(tok), jnp.asarray(mask))
            self.params, m, v = update(self.params, m, v, grads, step)
            losses.append(float(loss))
            if step % log_every == 0:
                rec = {
                    "step": step,
                    "loss": float(np.mean(losses[-log_every:])),
                    "ppl": float(np.exp(np.mean(losses[-log_every:]))),
                    "seconds": round(time.time() - t0, 1),
                }
                self.history.append(rec)
                if callback:
                    callback(rec)
        return self

    # ------------------------------------------------------------- representações
    def representations(self, encoded: list[int], *, bucket: int = 16) -> np.ndarray:
        """``(L+1, T, 2*hid)`` para uma sentença codificada (inclui <s> e </s>).

        O comprimento é preenchido ao próximo múltiplo de ``bucket`` (com ``<pad>`` à direita, que
        não afeta as posições anteriores no LM *forward* mas afeta levemente o *backward*), para que o
        ``jax.jit`` compile poucas vezes em vez de uma por comprimento de sentença."""
        T = len(encoded)
        Tp = max(bucket, ((T + bucket - 1) // bucket) * bucket)
        arr = np.zeros((1, Tp), dtype=np.int32)
        arr[0, :T] = encoded
        reps, _, _ = self._fwd_fn(self.params, jnp.asarray(arr))
        return np.array(
            reps[:, 0, :T]
        )  # cópia gravável (np.asarray de um array JAX é somente leitura)

    def perplexity(self, encoded: list[list[int]], seq_len: int = 24) -> float:
        tot, n = 0.0, 0
        for s in encoded:
            s = s[:seq_len]
            tok = jnp.asarray(np.array(s, dtype=np.int32)[None, :])
            mask = jnp.ones((1, len(s)), dtype=jnp.float32)
            tot += float(self._loss(self.params, tok, mask)) * (len(s) - 1)
            n += len(s) - 1
        return float(np.exp(tot / max(n, 1)))

    def save(self, path: Path | str, vocab: BiLMVocab) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = jax.tree_util.tree_map(lambda a: np.asarray(a), self.params)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "params": params,
                    "words": vocab.words,
                    "emb": self.emb,
                    "hid": self.hid,
                    "L": self.L,
                    "history": self.history,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: Path | str) -> tuple[BiLM, BiLMVocab]:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        vocab = BiLMVocab.__new__(BiLMVocab)
        vocab.words = d["words"]
        vocab.index = {w: i for i, w in enumerate(vocab.words)}
        model = cls(len(vocab.words), d["emb"], d["hid"], d["L"])
        model.params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        model.history = d.get("history", [])
        return model, vocab


def elmo_mix(reps: np.ndarray, weights: np.ndarray | None = None, gamma: float = 1.0) -> np.ndarray:
    """Eq. (1): ``ELMo = γ Σ_j softmax(s)_j h_j``. ``reps``: (L+1, T, D)."""
    L1 = reps.shape[0]
    s = np.zeros(L1) if weights is None else np.asarray(weights, dtype=float)
    s = np.exp(s - s.max())
    s /= s.sum()
    return gamma * np.tensordot(s, reps, axes=(0, 0))


def contextual_vectors(
    model: BiLM,
    vocab: BiLMVocab,
    sentence: list[str],
    word: str,
    layer: int | None = None,
    weights=None,
) -> np.ndarray:
    """Vetor contextual de ``word`` numa sentença (camada ``layer`` ou mistura ELMo)."""
    enc = vocab.encode(sentence)
    reps = model.representations(enc)
    pos = sentence.index(word) + 1  # +1 por causa de <s>
    if layer is not None:
        return reps[layer, pos]
    return elmo_mix(reps, weights)[pos]
