"""Garante que todas as fontes de notebooks são válidas e geram .ipynb bem formados (sem executar)."""

import importlib.util
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"


def _load(path: Path):
    sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sources_exist_and_are_ordered():
    srcs = sorted(SRC.glob("nb_*.py"))
    assert len(srcs) >= 14
    nums = [p.name[3:5] for p in srcs]
    assert nums == sorted(nums)


def test_each_source_builds_valid_notebook(tmp_path):
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    for path in sorted(SRC.glob("nb_*.py")):
        mod = _load(path)
        assert isinstance(mod.TITLE, str) and mod.TITLE
        assert mod.CELLS and mod.CELLS[0][0] == "md"
        nb = new_notebook()
        for kind, src in mod.CELLS:
            assert kind in ("md", "code") and src.strip()
            nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
            if kind == "code":
                compile(src, path.name, "exec")  # sintaxe válida
        nbformat.validate(nb)
        nbformat.write(nb, tmp_path / (path.stem + ".ipynb"))
