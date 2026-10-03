"""Paralelismo de dados simples com ``joblib`` (processos ``loky``).

Use para avaliar vários embeddings de uma vez (POS, STS, analogias): cada avaliação é
independente e dominada por Python/numpy de uma só *thread*, logo escala quase linearmente com os
núcleos. ``REPLANG_JOBS`` define o padrão.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from replang.accel import detect


def parallel_map(
    fn: Callable,
    items: Iterable,
    *,
    n_jobs: int | None = None,
    backend: str = "loky",
    verbose: int = 0,
) -> list:
    from joblib import Parallel, delayed

    items = list(items)
    n_jobs = n_jobs or detect().n_jobs
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    return Parallel(n_jobs=min(n_jobs, len(items)), backend=backend, verbose=verbose)(
        delayed(fn)(x) for x in items
    )
