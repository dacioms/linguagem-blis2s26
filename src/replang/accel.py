"""Aceleração: detecção de *backend* (numpy × JAX), dispositivo (CPU × GPU) e paralelismo.

Variáveis de ambiente:

* ``REPLANG_BACKEND`` = ``auto`` (padrão) | ``numpy`` | ``jax`` — ``auto`` usa JAX quando instalado:
  medido em CPU (4 núcleos), o GloVe fica ~5–8× mais rápido, o Skip-gram ~1,7× e o classificador
  linear (sondagens POS/NER) ~9× em relação ao ``LogisticRegression`` (lbfgs); em GPU os ganhos são
  maiores porque os lotes rodam inteiros na placa.
* ``REPLANG_DEVICE`` = ``auto`` | ``cpu`` | ``gpu`` — força o dispositivo do JAX.
* ``REPLANG_JOBS`` = processos para avaliações paralelas (padrão: ``cpus // 4``, 1 em máquinas de
  até 4 núcleos — cada processo já usa várias *threads*; medido em 4 núcleos, 4 processos foram
  0,5× mais lentos que a série).

GPU: instale o extra ``cuda`` (``uv sync --extra cuda``; Linux ou WSL2). RTX 4060 (Ada, sm_89) e
RTX 5050 (Blackwell, sm_120) são suportadas pelas *wheels* ``jax[cuda12]``; a 5050 exige CUDA ≥ 12.8
e *driver* recente. Ver ``docs/aceleracao.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Accel:
    has_jax: bool
    jax_version: str | None
    device_kind: str  # "gpu" | "cpu" | "none"
    device_name: str
    backend: str  # "jax" | "numpy"
    n_jobs: int

    @property
    def gpu(self) -> bool:
        return self.device_kind == "gpu"

    def use_jax_for(self, kernel: str) -> bool:
        """Decide por *kernel* (``glove``, ``sgns``, ``linear``): todos ganham com JAX já em CPU; a
        função existe para permitir exceções futuras (ex.: ``REPLANG_BACKEND=numpy``)."""
        return self.backend == "jax"

    def summary(self) -> str:
        dev = (
            f"{self.device_kind.upper()} ({self.device_name})"
            if self.has_jax
            else "— (JAX ausente)"
        )
        return f"backend={self.backend} · dispositivo={dev} · jobs={self.n_jobs}"


@lru_cache(maxsize=1)
def detect() -> Accel:
    want_backend = os.environ.get("REPLANG_BACKEND", "auto").lower()
    want_device = os.environ.get("REPLANG_DEVICE", "auto").lower()
    # processos para avaliações paralelas: cada um usa ~4 threads (BLAS/JAX); em 4 núcleos o
    # paralelismo de processos não compensa (medido: 0,5×), em 16 núcleos ~3 processos ajudam
    cpus = os.cpu_count() or 1
    n_jobs = int(os.environ.get("REPLANG_JOBS", str(max(1, cpus // 4) if cpus > 4 else 1)))
    has_jax, version, kind, name = False, None, "none", ""
    try:
        if want_device == "cpu":
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax

        has_jax, version = True, jax.__version__
        devs = jax.devices()
        gpus = [d for d in devs if d.platform in ("gpu", "cuda", "rocm")]
        if gpus and want_device != "cpu":
            kind, name = "gpu", getattr(gpus[0], "device_kind", str(gpus[0]))
        else:
            kind, name = "cpu", f"{os.cpu_count()} threads"
    except Exception:  # noqa: BLE001 – JAX ausente ou sem dispositivo
        has_jax = False
    if want_backend == "numpy" or not has_jax:
        backend = "numpy"
    else:
        backend = "jax"
    return Accel(has_jax, version, kind, name, backend, n_jobs)


def report() -> str:
    a = detect()
    return a.summary()
