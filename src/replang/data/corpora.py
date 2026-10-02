"""Corpora públicos usados nas demonstrações.

* **Machado de Assis** (domínio público, 246 textos, ≈2,5M tokens) – corpus PT-BR para treinar
  word2vec/GloVe/fastText ao vivo. Versão normalizada é gravada em ``data/processed``.
* **Mac-Morpho** (NILC/USP) – corpus anotado com POS, usado por Hartmann et al. (2017) na avaliação
  extrínseca.
* **Corpus sintético de gênero** – gerador controlado para ilustrar como regularidades
  estatísticas viram viés geométrico (Bolukbasi et al., 2016).
* **Toy corpus** – frases minúsculas para acompanhar cada passo das implementações *from scratch*.
"""

from __future__ import annotations

import gzip
import random
import zipfile
from collections.abc import Iterator
from pathlib import Path

from replang.config import PATHS, SOURCES
from replang.utils.download import download
from replang.utils.text import iter_sentences

MACHADO_PROCESSED = PATHS.processed / "machado_sentences.txt.gz"
MACMORPHO_PROCESSED = PATHS.processed / "macmorpho.conll.gz"


# ----------------------------------------------------------------------------- Machado
def _machado_zip() -> Path:
    local = PATHS.raw / "machado.zip"
    return local if local.exists() else download(SOURCES["machado"]["url"], local)


def iter_machado_raw(genres: tuple[str, ...] | None = None) -> Iterator[tuple[str, str]]:
    """Itera ``(nome_arquivo, texto)`` do zip original (codificação latin-1)."""
    with zipfile.ZipFile(_machado_zip()) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            parts = info.filename.split("/")
            if not info.filename.endswith(".txt") or len(parts) < 3:
                continue
            if genres and parts[1] not in genres:
                continue
            text = zf.read(info).decode("latin-1")
            yield info.filename, text


def prepare_machado(min_tokens: int = 5, force: bool = False) -> Path:
    """Gera ``machado_sentences.txt.gz`` (uma sentença tokenizada por linha)."""
    if MACHADO_PROCESSED.exists() and not force:
        return MACHADO_PROCESSED
    MACHADO_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(MACHADO_PROCESSED, "wt", encoding="utf-8") as out:
        for _, text in iter_machado_raw():
            body = _strip_machado_header(text)
            for sent in iter_sentences(body, min_tokens=min_tokens):
                out.write(" ".join(sent) + "\n")
                n += 1
    return MACHADO_PROCESSED


def _strip_machado_header(text: str) -> str:
    # Os arquivos começam com metadados ("Texto de referência: ... Nova Aguilar ..."); tentamos
    # pular até o primeiro bloco de capítulo ou após ~12 linhas.
    lines = text.splitlines()
    for i, line in enumerate(lines[:60]):
        if (
            line.strip()
            .upper()
            .startswith(("CAPÍTULO", "CAPITULO", "I\n", "PRÓLOGO", "ADVERTÊNCIA"))
            and i > 3
        ):
            return "\n".join(lines[i:])
    return "\n".join(lines[12:])


def load_machado_sentences(
    limit: int | None = None, *, path: Path | None = None
) -> list[list[str]]:
    """Carrega as sentenças tokenizadas (gera o arquivo processado se necessário)."""
    path = path or prepare_machado()
    sents: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            sents.append(line.split())
            if limit and len(sents) >= limit:
                break
    return sents


# -------------------------------------------------------------------------- Mac-Morpho
def _macmorpho_zip() -> Path:
    local = PATHS.raw / "mac_morpho.zip"
    return local if local.exists() else download(SOURCES["mac_morpho"]["url"], local)


def prepare_macmorpho(force: bool = False) -> Path:
    """Converte o Mac-Morpho (``palavra_TAG`` por linha) para CoNLL (``palavra\\tTAG``, linha vazia entre sentenças)."""
    if MACMORPHO_PROCESSED.exists() and not force:
        return MACMORPHO_PROCESSED
    MACMORPHO_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(_macmorpho_zip()) as zf,
        gzip.open(MACMORPHO_PROCESSED, "wt", encoding="utf-8") as out,
    ):
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not info.filename.endswith(".txt"):
                continue
            out.write(f"# doc {Path(info.filename).name}\n")
            for line in zf.read(info).decode("latin-1").splitlines():
                line = line.strip()
                if not line:
                    continue
                word, _, tag = line.rpartition("_")
                tag = tag.split("|")[0]  # "PREP|+" → "PREP"
                out.write(f"{word}\t{tag}\n")
                if tag == "." or word == ".":
                    out.write("\n")
    return MACMORPHO_PROCESSED


def load_macmorpho(
    limit_sentences: int | None = None,
    *,
    split: str = "all",
    test_fraction: float = 0.1,
    seed: int = 13,
) -> list[list[tuple[str, str]]]:
    """Sentenças ``[(palavra, tag), ...]`` do Mac-Morpho. ``split`` ∈ {"all", "train", "test"}."""
    path = prepare_macmorpho()
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            w, t = line.split("\t")
            cur.append((w, t))
    if cur:
        sents.append(cur)
    if split != "all":
        rng = random.Random(seed)
        idx = list(range(len(sents)))
        rng.shuffle(idx)
        n_test = int(len(sents) * test_fraction)
        chosen = idx[:n_test] if split == "test" else idx[n_test:]
        sents = [sents[i] for i in sorted(chosen)]
    if limit_sentences:
        sents = sents[:limit_sentences]
    return sents


# ------------------------------------------------------------------ corpora sintéticos
def toy_corpus() -> list[list[str]]:
    """Mini-corpus (PT) com estrutura distribucional clara: realeza, animais, comida."""
    raw = [
        "o rei governa o reino com a rainha",
        "a rainha governa o reino com o rei",
        "o rei é um homem e a rainha é uma mulher",
        "o príncipe é filho do rei e a princesa é filha da rainha",
        "o homem e a mulher caminham pela cidade",
        "o menino e a menina brincam no parque",
        "o gato come peixe e o cachorro come carne",
        "o cachorro late e o gato mia",
        "o gato dorme no sofá e o cachorro dorme no tapete",
        "a maçã e a banana são frutas doces",
        "o menino come a maçã e a menina come a banana",
        "a mulher cozinha o peixe e o homem cozinha a carne",
        "o rei come carne e a rainha come peixe",
        "o príncipe brinca com o cachorro e a princesa brinca com o gato",
        "paris é a capital da frança e lisboa é a capital de portugal",
        "roma é a capital da itália e madri é a capital da espanha",
        "a frança e a itália ficam na europa",
        "portugal e a espanha ficam na europa",
    ]
    return [s.split() for s in raw]


def synthetic_gender_corpus(
    n_sentences: int = 20000, *, bias: float = 0.8, seed: int = 0
) -> list[list[str]]:
    """Gera frases-molde onde profissões *estereotipadas* coocorrem mais com um gênero.

    ``bias`` ∈ [0.5, 1]: probabilidade de a profissão estereotipada aparecer com o gênero
    "esperado". Com ``bias=0.5`` não há viés; com ``bias=1`` o viés é total. Profissões
    *neutras* sempre têm 50%. Serve para mostrar que a **direção de gênero** emerge de
    coocorrências e que o viés é mensurável pela projeção nessa direção.
    """
    rng = random.Random(seed)
    fem = [
        "ela",
        "mulher",
        "mãe",
        "filha",
        "irmã",
        "menina",
        "esposa",
        "senhora",
        "rainha",
        "maria",
    ]
    masc = ["ele", "homem", "pai", "filho", "irmão", "menino", "marido", "senhor", "rei", "joão"]
    prof_f = [
        "enfermagem",
        "cozinha",
        "costura",
        "secretariado",
        "babá",
        "recepção",
        "moda",
        "dança",
    ]
    prof_m = [
        "engenharia",
        "futebol",
        "programação",
        "mecânica",
        "piloto",
        "guerra",
        "finanças",
        "física",
    ]
    prof_n = [
        "jornalismo",
        "medicina",
        "pesquisa",
        "música",
        "escola",
        "comércio",
        "política",
        "arte",
    ]
    templates = [
        "{g} trabalha com {p} todos os dias",
        "{g} gosta muito de {p} e fala sobre {p}",
        "a vida de {g} é dedicada a {p}",
        "{g} estudou {p} na universidade",
        "{g} disse que {p} é importante",
        "ontem {g} foi ao trabalho de {p}",
    ]
    fillers = [
        "o tempo passou e a cidade cresceu",
        "a casa fica perto do rio e da ponte",
        "o livro novo chegou na biblioteca hoje",
        "amanhã vai chover na serra e no litoral",
    ]
    out: list[list[str]] = []
    for _ in range(n_sentences):
        r = rng.random()
        if r < 0.15:
            out.append(rng.choice(fillers).split())
            continue
        kind = rng.choice(["f", "m", "n"])
        if kind == "f":
            p = rng.choice(prof_f)
            g = rng.choice(fem) if rng.random() < bias else rng.choice(masc)
        elif kind == "m":
            p = rng.choice(prof_m)
            g = rng.choice(masc) if rng.random() < bias else rng.choice(fem)
        else:
            p = rng.choice(prof_n)
            g = rng.choice(fem + masc)
        out.append(rng.choice(templates).format(g=g, p=p).split())
    return out


SYNTHETIC_PROFESSIONS = {
    "estereotipo_feminino": [
        "enfermagem",
        "cozinha",
        "costura",
        "secretariado",
        "babá",
        "recepção",
        "moda",
        "dança",
    ],
    "estereotipo_masculino": [
        "engenharia",
        "futebol",
        "programação",
        "mecânica",
        "piloto",
        "guerra",
        "finanças",
        "física",
    ],
    "neutras": [
        "jornalismo",
        "medicina",
        "pesquisa",
        "música",
        "escola",
        "comércio",
        "política",
        "arte",
    ],
}
