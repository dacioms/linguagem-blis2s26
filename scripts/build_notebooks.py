"""Constrói os notebooks ``.ipynb`` a partir de fontes Python em ``notebooks/_src`` e os executa.

Cada fonte define ``TITLE`` e ``CELLS`` (lista criada com as funções ``md`` e ``code`` de
``nbsrc``). Vantagens: *diffs* legíveis no git, revisão textual fácil e geração reprodutível.

Uso::

    uv run replang notebooks build          # gera notebooks/*.ipynb (sem saídas)
    uv run replang notebooks run            # executa e grava as saídas
    uv run replang notebooks both --only 08 # filtra por substring
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"
sys.path.insert(0, str(SRC))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_one(path: Path) -> Path:
    mod = _load(path)
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3 (replang)",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["replang"] = {"source": path.name, "title": mod.TITLE}
    for kind, src in mod.CELLS:
        nb.cells.append(
            new_markdown_cell(src.strip("\n")) if kind == "md" else new_code_cell(src.strip("\n"))
        )
    out = OUT / (path.stem.replace("nb_", "") + ".ipynb")
    nbformat.write(nb, out)
    return out


def _match(name: str, only: str) -> bool:
    """``only`` numérico casa só com o prefixo (``"13"`` não deve casar ``mikolov2013b``)."""
    if not only:
        return True
    stem = name.removeprefix("nb_")
    return stem.startswith(only) if only.isdigit() else only in name


def sources(only: str = "") -> list[Path]:
    return sorted(p for p in SRC.glob("nb_*.py") if _match(p.name, only))


def build_all(only: str = "") -> list[Path]:
    outs = []
    for p in sources(only):
        outs.append(build_one(p))
        print("built", outs[-1].name)
    return outs


def run_one(path: Path, timeout: int = 3600) -> float:
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(OUT)}},
        allow_errors=False,
    )
    t = time.time()
    client.execute()
    nbformat.write(nb, path)
    return time.time() - t


def _run_report(args):
    path, timeout = args
    try:
        return path.name, run_one(path, timeout), None
    except Exception as e:  # noqa: BLE001
        return path.name, None, f"{type(e).__name__}: {str(e)[:800]}"


def run_all(only: str = "", timeout: int = 3600, jobs: int = 1) -> None:
    """Executa os notebooks; ``jobs > 1`` roda vários em paralelo (processos independentes).

    Os notebooks não dependem uns dos outros (só de ``data/`` e ``data/models``), então o
    paralelismo é seguro; em máquinas com 8+ núcleos, ``jobs=3`` ou ``4`` reduz o tempo total a
    menos da metade. Cada processo usa várias *threads* (BLAS, gensim, JAX), por isso não compensa
    passar de ``cpus // 2``."""
    paths = [p for p in sorted(OUT.glob("*.ipynb")) if _match(p.name, only)]
    if jobs <= 1 or len(paths) <= 1:
        for p in paths:
            print(f"running {p.name} …", flush=True)
            name, dt, err = _run_report((p, timeout))
            if err:
                print(f"  FALHOU: {err}", flush=True)
                raise RuntimeError(err)
            print(f"  ok em {dt:.0f}s", flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"executando {len(paths)} notebooks com {jobs} processos …", flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run_report, (p, timeout)): p for p in paths}
        for fut in as_completed(futs):
            name, dt, err = fut.result()
            if err:
                failures.append((name, err))
                print(f"{name}: FALHOU: {err}", flush=True)
            else:
                print(f"{name}: ok em {dt:.0f}s", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} notebook(s) falharam: {[f[0] for f in failures]}")


def clean_outputs(only: str = "") -> None:
    for p in sorted(OUT.glob("*.ipynb")):
        if not _match(p.name, only):
            continue
        nb = nbformat.read(p, as_version=4)
        for c in nb.cells:
            if c.cell_type == "code":
                c.outputs = []
                c.execution_count = None
        nbformat.write(nb, p)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="build", choices=["build", "run", "both", "clean"])
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.action in ("build", "both"):
        build_all(a.only)
    if a.action in ("run", "both"):
        run_all(a.only, a.timeout, a.jobs)
    if a.action == "clean":
        clean_outputs(a.only)
