"""Download com cache e extração de arquivos.

Usa ``requests`` (que respeita ``HTTPS_PROXY``/``REQUESTS_CA_BUNDLE``) e grava em
``data/raw`` por padrão. Arquivos existentes não são baixados novamente.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from replang.config import PATHS


def download(
    url: str, dest: Path | str | None = None, *, force: bool = False, timeout: int = 60
) -> Path:
    """Baixa ``url`` para ``dest`` (padrão: ``data/raw/<nome>``) com barra de progresso.

    Retorna o caminho local. Lança ``requests.HTTPError`` em falhas HTTP.
    """
    if dest is None:
        dest = PATHS.raw / url.rsplit("/", 1)[-1].split("?")[0]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            open(tmp, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def extract_zip(path: Path | str, dest: Path | str | None = None) -> Path:
    """Extrai um ``.zip`` em ``dest`` (padrão: pasta com o mesmo nome ao lado do zip)."""
    path = Path(path)
    dest = Path(dest) if dest else path.with_suffix("")
    if not dest.exists():
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    return dest
