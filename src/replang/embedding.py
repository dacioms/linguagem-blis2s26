"""``WordVectors``: contêiner leve e transparente para embeddings de palavras.

Serve a dois propósitos didáticos:

1. tornar explícita a matemática que bibliotecas como o gensim escondem (cosseno, 3CosAdd,
   3CosMul, normalização);
2. unificar o acesso a vetores vindos de fontes diferentes (treinados *from scratch* em numpy,
   gensim, arquivos ``word2vec`` texto, Wikipedia2Vec, GloVe, NILC).
"""

from __future__ import annotations

import bz2
import gzip
import io
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def _open_text(path: Path | str):
    path = Path(path)
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


class WordVectors:
    """Matriz ``vectors`` (n × d) alinhada à lista ``words``.

    >>> wv = WordVectors(["rei", "rainha"], np.eye(2))
    >>> wv["rei"]
    array([1., 0.], dtype=float32)
    """

    def __init__(
        self,
        words: Sequence[str],
        vectors: np.ndarray,
        name: str = "",
        counts: Sequence[int] | None = None,
    ):
        self.words: list[str] = list(words)
        self.vectors: np.ndarray = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.words):
            raise ValueError("vectors deve ter forma (len(words), d)")
        self.index: dict[str, int] = {w: i for i, w in enumerate(self.words)}
        self.name = name
        self.counts = np.asarray(counts) if counts is not None else None
        self._norm_cache: np.ndarray | None = None

    # ------------------------------------------------------------------ básicos
    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def __getitem__(self, word: str) -> np.ndarray:
        try:
            return self.vectors[self.index[word]]
        except KeyError as e:
            raise KeyError(f"palavra fora do vocabulário: {word!r}") from e

    def __repr__(self) -> str:
        return f"WordVectors(name={self.name!r}, n={len(self)}, dim={self.dim})"

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def get(self, word: str, default=None):
        i = self.index.get(word)
        return self.vectors[i] if i is not None else default

    def has(self, *words: str) -> bool:
        return all(w in self.index for w in words)

    def vector(self, word: str) -> np.ndarray:
        return self[word]

    # --------------------------------------------------------------- normalização
    @property
    def unit(self) -> np.ndarray:
        """Matriz com todas as linhas normalizadas (cache)."""
        if self._norm_cache is None or self._norm_cache.shape != self.vectors.shape:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._norm_cache = self.vectors / norms
        return self._norm_cache

    def normalized(self, name: str | None = None) -> WordVectors:
        """Nova instância com vetores unitários (como em Bolukbasi et al., §3)."""
        return WordVectors(self.words, self.unit.copy(), name or self.name, self.counts)

    def invalidate(self) -> None:
        self._norm_cache = None

    # ---------------------------------------------------------------- similaridade
    @staticmethod
    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    def similarity(self, a: str, b: str) -> float:
        return self.cosine(self[a], self[b])

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Cosseno entre ``query`` e todas as palavras."""
        q = np.asarray(query, dtype=np.float32)
        nq = np.linalg.norm(q)
        if nq == 0:
            return np.zeros(len(self), dtype=np.float32)
        return self.unit @ (q / nq)

    def most_similar(
        self,
        positive: Iterable[str] | str = (),
        negative: Iterable[str] | str = (),
        topn: int = 10,
        *,
        exclude: bool = True,
        restrict: int | None = None,
        exclude_words: Iterable[str] = (),
    ) -> list[tuple[str, float]]:
        """Vizinhos pelo método **3CosAdd** (Mikolov et al., 2013c): ``argmax cos(y, b − a + c)``.

        ``restrict`` limita a busca às ``restrict`` palavras mais frequentes (como em
        avaliações que usam só as 30k/300k primeiras palavras do vocabulário).
        """
        pos = [positive] if isinstance(positive, str) else list(positive)
        neg = [negative] if isinstance(negative, str) else list(negative)
        if not pos and not neg:
            raise ValueError("informe ao menos uma palavra")
        unit = self.unit
        q = np.zeros(self.dim, dtype=np.float32)
        for w in pos:
            q += unit[self.index[w]]
        for w in neg:
            q -= unit[self.index[w]]
        sims = self.similarities(q)
        if restrict:
            sims = sims[:restrict]
        if exclude:
            for w in pos + neg:
                i = self.index[w]
                if i < len(sims):
                    sims[i] = -np.inf
        for w in exclude_words:
            i = self.index.get(w)
            if i is not None and i < len(sims):
                sims[i] = -np.inf
        k = min(topn, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.words[i], float(sims[i])) for i in idx if np.isfinite(sims[i])]

    def most_similar_cosmul(
        self,
        positive: Iterable[str] | str = (),
        negative: Iterable[str] | str = (),
        topn: int = 10,
        *,
        eps: float = 1e-3,
        restrict: int | None = None,
    ) -> list[tuple[str, float]]:
        """Vizinhos pelo método **3CosMul** (Levy & Goldberg, 2014), citado por Bolukbasi (Apêndice A):

        ``argmax_y  Π_pos (1+cos(y,p))/2  /  (Π_neg (1+cos(y,n))/2 + ε)``.
        """
        pos = [positive] if isinstance(positive, str) else list(positive)
        neg = [negative] if isinstance(negative, str) else list(negative)
        unit = self.unit if restrict is None else self.unit[:restrict]
        num = np.ones(len(unit), dtype=np.float64)
        for w in pos:
            num *= (1.0 + unit @ self.unit[self.index[w]]) / 2.0
        den = np.ones(len(unit), dtype=np.float64)
        for w in neg:
            den *= (1.0 + unit @ self.unit[self.index[w]]) / 2.0
        score = num / (den + eps)
        for w in pos + neg:
            i = self.index[w]
            if i < len(score):
                score[i] = -np.inf
        k = min(topn, len(score))
        idx = np.argpartition(-score, k - 1)[:k]
        idx = idx[np.argsort(-score[idx])]
        return [(self.words[i], float(score[i])) for i in idx if np.isfinite(score[i])]

    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        *,
        method: str = "add",
        topn: int = 5,
        restrict: int | None = None,
        exclude_words: Iterable[str] = (),
    ):
        """Resolve ``a : b :: c : ?`` (ex.: ``homem : rei :: mulher : ?`` → *rainha*).

        ``exclude_words`` remove candidatos (ex.: palavras específicas de gênero, para ver o que o
        embedding associa a uma profissão *além* do próprio gênero)."""
        if method == "add":
            return self.most_similar(
                positive=[b, c],
                negative=[a],
                topn=topn,
                restrict=restrict,
                exclude_words=exclude_words,
            )
        if method == "mul":
            return self.most_similar_cosmul(
                positive=[b, c], negative=[a], topn=topn, restrict=restrict
            )
        raise ValueError("method deve ser 'add' ou 'mul'")

    # ---------------------------------------------------------------- utilidades
    def subset(
        self, words: Iterable[str], name: str | None = None, *, strict: bool = False
    ) -> WordVectors:
        keep = []
        for w in words:
            if w in self.index:
                keep.append(w)
            elif strict:
                raise KeyError(w)
        idx = [self.index[w] for w in keep]
        counts = self.counts[idx] if self.counts is not None else None
        return WordVectors(keep, self.vectors[idx], name or self.name, counts)

    def head(self, n: int, name: str | None = None) -> WordVectors:
        counts = self.counts[:n] if self.counts is not None else None
        return WordVectors(self.words[:n], self.vectors[:n], name or self.name, counts)

    def matrix(self, words: Iterable[str]) -> np.ndarray:
        return np.stack([self[w] for w in words])

    def filter_words(self, words: Iterable[str]) -> list[str]:
        return [w for w in words if w in self.index]

    def copy(self, name: str | None = None) -> WordVectors:
        return WordVectors(list(self.words), self.vectors.copy(), name or self.name, self.counts)

    def with_vectors(self, vectors: np.ndarray, name: str | None = None) -> WordVectors:
        return WordVectors(list(self.words), vectors, name or self.name, self.counts)

    # ----------------------------------------------------------------------- I/O
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            words=np.array(self.words, dtype=object),
            vectors=self.vectors.astype(
                np.float16 if self.vectors.shape[0] > 20000 else np.float32
            ),
            name=np.array(self.name),
            counts=self.counts if self.counts is not None else np.array([]),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> WordVectors:
        with np.load(Path(path), allow_pickle=True) as z:
            counts = z["counts"] if "counts" in z and z["counts"].size else None
            return cls(list(z["words"]), z["vectors"].astype(np.float32), str(z["name"]), counts)

    @classmethod
    def from_word2vec_text(
        cls,
        path: Path | str,
        *,
        top_n: int | None = None,
        skip_prefix: str | None = None,
        lowercase: bool = False,
        name: str = "",
        header: bool | None = None,
    ) -> WordVectors:
        """Lê o formato texto do word2vec/GloVe (``palavra v1 v2 ...``), opcionalmente truncado.

        ``skip_prefix`` ignora entradas (ex.: ``"ENTITY/"`` do Wikipedia2Vec). Assume que o arquivo
        está ordenado por frequência decrescente quando ``top_n`` é usado.
        """
        words: list[str] = []
        vecs: list[np.ndarray] = []
        seen: set[str] = set()
        with _open_text(path) as fh:
            first = fh.readline()
            parts = first.rstrip("\n").split(" ")
            is_header = (
                header
                if header is not None
                else (len(parts) == 2 and all(p.isdigit() for p in parts))
            )
            if not is_header:
                fh = io.StringIO(first + fh.read()) if top_n is None else _chain(first, fh)
            for line in fh:
                if not line.strip():
                    continue
                w, _, rest = line.rstrip("\n").partition(" ")
                if skip_prefix and w.startswith(skip_prefix):
                    continue
                if lowercase:
                    w = w.lower()
                if w in seen:
                    continue
                seen.add(w)
                words.append(w)
                vecs.append(np.fromstring(rest, sep=" ", dtype=np.float32))
                if top_n and len(words) >= top_n:
                    break
        return cls(words, np.stack(vecs), name or Path(path).name)

    def to_gensim(self):
        from gensim.models import KeyedVectors

        kv = KeyedVectors(self.dim)
        kv.add_vectors(self.words, self.vectors)
        return kv

    @classmethod
    def from_gensim(cls, kv, name: str = "") -> WordVectors:
        words = list(kv.index_to_key)
        counts = None
        try:
            counts = [kv.get_vecattr(w, "count") for w in words]
        except Exception:  # noqa: BLE001 – atributo ausente em alguns modelos
            counts = None
        return cls(words, kv.vectors, name, counts)


def _chain(first: str, fh):
    yield first
    yield from fh
