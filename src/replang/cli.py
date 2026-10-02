"""Linha de comando: ``uv run replang --help``."""

from __future__ import annotations

import subprocess
import sys

import typer
from rich import print as rprint
from rich.table import Table

from replang.config import PATHS, SOURCES, settings

app = typer.Typer(help="replang — Representações de Linguagem (BLIS 2S26)", no_args_is_help=True)


@app.command()
def info():
    """Mostra caminhos, fontes de dados e modelos disponíveis."""
    t = Table(title="Caminhos")
    t.add_column("nome")
    t.add_column("caminho")
    t.add_column("existe")
    for k in ("root", "raw", "samples", "processed", "models", "notebooks"):
        p = getattr(PATHS, k)
        t.add_row(k, str(p), "✓" if p.exists() else "✗")
    rprint(t)
    t = Table(title="Fontes públicas")
    t.add_column("chave")
    t.add_column("descrição")
    t.add_column("licença")
    for k, v in SOURCES.items():
        t.add_row(k, v["desc"], v["license"])
    rprint(t)
    from replang.models.trainers import list_models

    t = Table(title="Modelos treinados localmente (data/models)")
    t.add_column("nome")
    t.add_column("algoritmo")
    t.add_column("dim")
    t.add_column("vocab")
    t.add_column("segundos")
    for m in list_models():
        t.add_row(
            m["name"],
            m.get("algo", ""),
            str(m.get("dim", "")),
            str(m.get("vocab", m.get("docs", ""))),
            str(m.get("seconds", "")),
        )
    rprint(t)
    rprint(
        f"[dim]settings: fast={settings.fast} seed={settings.seed} top_n={settings.top_n_pretrained} workers={settings.workers}[/dim]"
    )


@app.command()
def download(all: bool = typer.Option(True, help="baixa corpora e embeddings pré-treinados")):
    """Baixa os dados públicos para ``data/raw`` (GitHub, Wikipedia2Vec)."""
    from replang.utils.download import download as dl

    for key in (
        "machado",
        "mac_morpho",
        "questions_words",
        "lx_analogies_br",
        "lx_analogies_eu",
        "glove_en_50d",
        "ptwiki2vec_100d",
    ):
        url = SOURCES[key]["url"]
        rprint(f"[cyan]{key}[/cyan] ← {url}")
        try:
            dl(url)
        except Exception as e:  # noqa: BLE001
            rprint(f"  [red]falhou:[/red] {e}")


@app.command()
def prepare(
    top_n: int = typer.Option(None, help="palavras mantidas dos embeddings pré-treinados"),
    legal: bool = typer.Option(
        True, help="também prepara os corpora jurídicos (RulingBR, LeNER-Br, JurisBERT)"
    ),
):
    """Gera corpora processados e caches truncados de embeddings (``data/processed``, ``data/samples``)."""
    from replang.data.corpora import prepare_machado, prepare_macmorpho
    from replang.data.embeddings import load_glove_en, load_ptwiki

    rprint("Machado →", prepare_machado())
    rprint("Mac-Morpho →", prepare_macmorpho())
    rprint("PT →", load_ptwiki(top_n))
    rprint("EN →", load_glove_en(top_n))
    if legal:
        from replang.data.legal import (
            prepare_legal_corpus,
            prepare_legal_sts,
            prepare_lener,
            prepare_rulingbr_ementas,
        )

        rprint("Jurídico (RulingBR) →", prepare_legal_corpus())
        rprint("LeNER-Br →", prepare_lener())
        rprint("JurisBERT STS →", prepare_legal_sts())
        rprint("Ementas com área →", prepare_rulingbr_ementas())


@app.command()
def train(
    what: str = typer.Argument("all", help="all | w2v | fasttext | glove | doc2vec | bilm | legal"),
    fast: bool = typer.Option(False, help="versões reduzidas (épocas/passos menores)"),
    force: bool = typer.Option(False, help="retreina mesmo se o cache existir"),
):
    """Treina os modelos locais sobre o corpus Machado de Assis (ver ``scripts/train_models.py``)."""
    sys.path.insert(0, str(PATHS.root / "scripts"))
    from train_models import main as train_main

    train_main(what=what, fast=fast or settings.fast, force=force)


@app.command()
def notebooks(
    action: str = typer.Argument("build", help="build | run | both | clean"),
    only: str = typer.Option("", help="substring para filtrar notebooks"),
    timeout: int = typer.Option(3600),
    jobs: int = typer.Option(1, help="notebooks executados em paralelo (processos)"),
):
    """Constrói (de ``notebooks/_src``) e/ou executa os notebooks ``.ipynb``."""
    sys.path.insert(0, str(PATHS.root / "scripts"))
    import build_notebooks

    if action in ("build", "both"):
        build_notebooks.build_all(only=only)
    if action in ("run", "both"):
        build_notebooks.run_all(only=only, timeout=timeout, jobs=jobs)
    if action == "clean":
        build_notebooks.clean_outputs(only=only)


@app.command(name="app")
def app_cmd(port: int = typer.Option(8501)):
    """Abre a interface Streamlit."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PATHS.root / "app" / "streamlit_app.py"),
            "--server.port",
            str(port),
        ],
        check=False,
    )


@app.command()
def test():
    """Roda a suíte de testes."""
    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "pytest", str(PATHS.root / "tests")], check=False
        ).returncode
    )


if __name__ == "__main__":
    app()
