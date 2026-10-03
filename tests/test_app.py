"""Smoke test do app Streamlit completo: a navegação precisa montar (url_path únicos) e a página
inicial renderizar sem exceções. Roda em segundos e não depende de modelos locais."""

from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_navigation_builds_and_home_renders():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Representações de Linguagem" in t.value for t in at.title)


def test_page_url_paths_are_unique():
    src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    import re

    paths = re.findall(r'url_path="([^"]+)"', src)
    assert len(paths) >= 10 and len(paths) == len(set(paths))
