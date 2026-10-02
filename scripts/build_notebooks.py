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


def run_all(only: str = "", timeout: int = 3600) -> None:
    for p in sorted(OUT.glob("*.ipynb")):
        if not _match(p.name, only):
            continue
        print(f"running {p.name} …", flush=True)
        try:
            dt = run_one(p, timeout)
            print(f"  ok em {dt:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  FALHOU: {type(e).__name__}: {str(e)[:800]}", flush=True)
            raise


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
    a = ap.parse_args()
    if a.action in ("build", "both"):
        build_all(a.only)
    if a.action in ("run", "both"):
        run_all(a.only, a.timeout)
    if a.action == "clean":
        clean_outputs(a.only)
