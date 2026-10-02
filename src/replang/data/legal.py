"""Corpora e léxicos **jurídicos** brasileiros (eixo "contexto jurídico" da apresentação).

Fontes públicas (todas no GitHub, portanto acessíveis sem credenciais):

* **RulingBR** (Feijó & Moreira, PROPOR 2018): 10.574 decisões do STF (2011–2018) com *ementa*,
  *acórdão*, *relatório*, *voto*, *área*, *classe* e *relator* — ≈ 165 M caracteres. É o corpus
  de treino dos embeddings jurídicos e a base das tarefas de classificação de área.
* **LeNER-Br** (Luz de Araujo et al., PROPOR 2018): NER em decisões judiciais (STF, STJ, TJs):
  PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA — avaliação extrínseca jurídica.
* **JurisBERT / brazilian-legal-text-dataset** (Viegas, Costa & Ishii, ICCSA 2023): pares de ementas
  (STJ, TJMS, PJERJ) com rótulo de similaridade (binário e escala 0–3) — o "ASSIN jurídico".
* **UlyssesNER-Br** (Albuquerque et al., PROPOR 2022): projetos de lei da Câmara anotados (NER).

Os arquivos grandes ficam em ``data/raw/legal`` (ignorados pelo git); amostras processadas e
pequenas são versionadas em ``data/samples/legal`` e ``data/processed`` para uso *offline*.
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import tarfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from replang.config import PATHS
from replang.utils.download import download
from replang.utils.text import iter_sentences

LEGAL_SOURCES: dict[str, dict[str, str]] = {
    "rulingbr": {
        "url": "https://raw.githubusercontent.com/diego-feijo/rulingbr/master/rulingbr-v1.2.tar.xz",
        "desc": "RulingBR: 10.574 decisões do STF (2011–2018) com ementa, relatório, voto, área, classe e relator",
        "license": "Dados públicos do STF; dataset acadêmico (Feijó & Moreira, 2018)",
    },
    "lener_br": {
        "url": "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/{split}/{split}.conll",
        "desc": "LeNER-Br: NER em textos jurídicos brasileiros (decisões)",
        "license": "Acadêmico (Luz de Araujo et al., 2018)",
    },
    "jurisbert_sts": {
        "url": "https://raw.githubusercontent.com/alfaneo-ai/brazilian-legal-text-dataset/main/resources/sts/benchmark/{court}.csv",
        "desc": "JurisBERT: pares de ementas com similaridade (STJ, TJMS, PJERJ)",
        "license": "Dados públicos dos tribunais; dataset acadêmico (Viegas et al., 2023)",
    },
    "ulysses_ner": {
        "url": "https://raw.githubusercontent.com/ulysses-camara/ulysses-ner-br/main/PL-corpus_v1/pl_corpus_categorias/{split}.txt",
        "desc": "UlyssesNER-Br: projetos de lei da Câmara dos Deputados anotados (NER)",
        "license": "Acadêmico (Albuquerque et al., 2022)",
    },
}

RAW = PATHS.raw / "legal"
SAMPLES = PATHS.samples / "legal"
RULINGBR_JSONL = RAW / "rulingbr-v1.2.jsonl"
LEGAL_SENTENCES = PATHS.processed / "legal_sentences.txt.gz"  # corpus de treino (não versionado)
LEGAL_SENTENCES_SAMPLE = (
    PATHS.processed / "legal_sentences_sample.txt.gz"
)  # versionado (~1,5 M tokens)
LENER_PROCESSED = PATHS.processed / "lener_br.conll.gz"
RULINGBR_EMENTAS = SAMPLES / "rulingbr_ementas_sample.jsonl.gz"
LEGAL_STS_SAMPLE = SAMPLES / "jurisbert_sts_sample.csv.gz"


# ------------------------------------------------------------------------- RulingBR
def download_rulingbr(force: bool = False) -> Path:
    if RULINGBR_JSONL.exists() and not force:
        return RULINGBR_JSONL
    RAW.mkdir(parents=True, exist_ok=True)
    tar = download(LEGAL_SOURCES["rulingbr"]["url"], RAW / "rulingbr-v1.2.tar.xz", force=force)
    with tarfile.open(tar, "r:xz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".jsonl"))
        member.name = RULINGBR_JSONL.name
        tf.extract(member, RAW)
    return RULINGBR_JSONL


AREA_MAP = {
    "penal": "direito penal",
    "administrativo": "direito administrativo",
    "constitucional": "direito constitucional",
    "processual civil": "direito processual civil",
    "processual penal": "direito processual penal",
    "civil": "direito civil",
    "tributário": "direito tributário",
    "previdenciário": "direito previdenciário",
    "trabalho": "direito do trabalho",
    "eleitoral": "direito eleitoral",
    "do trabalho": "direito do trabalho",
}


def normalize_area(area: str | None) -> str | None:
    if not area:
        return None
    a = area.strip().lower()
    return AREA_MAP.get(a, a)


def iter_rulingbr(limit: int | None = None) -> Iterator[dict]:
    """Itera os documentos (dicts) do RulingBR, baixando se necessário."""
    path = download_rulingbr()
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            o = json.loads(line)
            o["area"] = normalize_area(o.get("area"))
            yield o


def prepare_legal_corpus(
    *,
    max_tokens: int = 6_000_000,
    sample_tokens: int = 1_500_000,
    min_tokens: int = 5,
    seed: int = 7,
    force: bool = False,
) -> tuple[Path, Path]:
    """Gera o corpus de sentenças jurídicas (ementa + relatório + voto do RulingBR), normalizado com
    o mesmo pipeline de Hartmann et al. usado para Machado — para que as comparações sejam justas.
    Documentos são embaralhados; grava um corpus de treino (``max_tokens``) e uma amostra versionada."""
    if LEGAL_SENTENCES.exists() and LEGAL_SENTENCES_SAMPLE.exists() and not force:
        return LEGAL_SENTENCES, LEGAL_SENTENCES_SAMPLE
    docs = list(iter_rulingbr())
    rng = random.Random(seed)
    rng.shuffle(docs)
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    n_full = n_sample = 0
    with (
        gzip.open(LEGAL_SENTENCES, "wt", encoding="utf-8") as full,
        gzip.open(LEGAL_SENTENCES_SAMPLE, "wt", encoding="utf-8") as samp,
    ):
        for d in docs:
            text = "\n".join(d.get(k) or "" for k in ("ementa", "relatorio", "voto"))
            for sent in iter_sentences(text, min_tokens=min_tokens):
                line = " ".join(sent) + "\n"
                if n_full < max_tokens:
                    full.write(line)
                    n_full += len(sent)
                if n_sample < sample_tokens:
                    samp.write(line)
                    n_sample += len(sent)
            if n_full >= max_tokens and n_sample >= sample_tokens:
                break
    return LEGAL_SENTENCES, LEGAL_SENTENCES_SAMPLE


def load_legal_sentences(limit: int | None = None, *, prefer_full: bool = True) -> list[list[str]]:
    """Sentenças tokenizadas do corpus jurídico (treino completo se existir, senão a amostra versionada)."""
    path = LEGAL_SENTENCES if (prefer_full and LEGAL_SENTENCES.exists()) else LEGAL_SENTENCES_SAMPLE
    if not path.exists():
        prepare_legal_corpus()
        path = LEGAL_SENTENCES if prefer_full else LEGAL_SENTENCES_SAMPLE
    out: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(line.split())
            if limit and len(out) >= limit:
                break
    return out


def prepare_rulingbr_ementas(n: int = 4000, seed: int = 11, force: bool = False) -> Path:
    """Amostra versionada de ementas com área/classe/relator (classificação de área do Direito)."""
    if RULINGBR_EMENTAS.exists() and not force:
        return RULINGBR_EMENTAS
    SAMPLES.mkdir(parents=True, exist_ok=True)
    docs = [d for d in iter_rulingbr() if d.get("ementa") and d.get("area")]
    rng = random.Random(seed)
    rng.shuffle(docs)
    with gzip.open(RULINGBR_EMENTAS, "wt", encoding="utf-8") as fh:
        for d in docs[:n]:
            fh.write(
                json.dumps(
                    {
                        "ementa": d["ementa"][:3000],
                        "area": d["area"],
                        "classe": d.get("classe"),
                        "relator": d.get("relator"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return RULINGBR_EMENTAS


def load_rulingbr_ementas(n: int | None = None):
    import pandas as pd

    path = RULINGBR_EMENTAS if RULINGBR_EMENTAS.exists() else prepare_rulingbr_ementas()
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- LeNER-Br
def _parse_conll(lines, sep: str | None = None) -> list[list[tuple[str, str]]]:
    sents, cur = [], []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split(sep)
        if len(parts) < 2:
            continue
        cur.append((parts[0], parts[-1]))
    if cur:
        sents.append(cur)
    return sents


def prepare_lener(force: bool = False) -> Path:
    if LENER_PROCESSED.exists() and not force:
        return LENER_PROCESSED
    RAW.mkdir(parents=True, exist_ok=True)
    PATHS.processed.mkdir(parents=True, exist_ok=True)
    with gzip.open(LENER_PROCESSED, "wt", encoding="utf-8") as out:
        for split in ("train", "dev", "test"):
            local = RAW / f"lener_{split}.conll"
            if not local.exists():
                download(LEGAL_SOURCES["lener_br"]["url"].format(split=split), local)
            out.write(f"# split {split}\n")
            with open(local, encoding="utf-8") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if not line.strip():
                        out.write("\n")
                        continue
                    w, _, t = line.rpartition(" ")
                    out.write(f"{w}\t{t}\n")
            out.write("\n")
    return LENER_PROCESSED


def load_lener(
    split: str = "train", limit_sentences: int | None = None
) -> list[list[tuple[str, str]]]:
    """Sentenças ``[(token, tag BIO)]`` do LeNER-Br; ``split`` ∈ {train, dev, test, all}."""
    path = prepare_lener()
    sents, cur, current = [], [], None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("# split"):
                current = line.split()[-1]
                continue
            if not line.strip():
                if cur and (split == "all" or current == split):
                    sents.append(cur)
                cur = []
                continue
            w, t = line.split("\t")
            cur.append((w, t))
    if cur and (split == "all" or current == split):
        sents.append(cur)
    return sents[:limit_sentences] if limit_sentences else sents


# ------------------------------------------------------------ JurisBERT STS (ementas)
def prepare_legal_sts(n_per_court: int = 600, seed: int = 5, force: bool = False) -> Path:
    """Amostra balanceada (similar/não similar) dos *benchmarks* STJ e TJMS do JurisBERT, versionada."""
    if LEGAL_STS_SAMPLE.exists() and not force:
        return LEGAL_STS_SAMPLE
    SAMPLES.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10**9)
    rng = random.Random(seed)
    rows_out = []
    for court in ("STJ", "TJMS"):
        local = RAW / f"{court}.csv"
        if not local.exists():
            download(LEGAL_SOURCES["jurisbert_sts"]["url"].format(court=court), local)
        with open(local, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh, delimiter="|"))
        pos = [r for r in rows if r["similarity"].startswith("1")]
        neg = [r for r in rows if r["similarity"].startswith("0")]
        rng.shuffle(pos)
        rng.shuffle(neg)
        for r in pos[: n_per_court // 2] + neg[: n_per_court // 2]:
            rows_out.append(
                {
                    "tribunal": court,
                    "grupo": r["group"],
                    "ementa1": r["ementa1"][:1500],
                    "ementa2": r["ementa2"][:1500],
                    "similar": int(float(r["similarity"])),
                }
            )
    rng.shuffle(rows_out)
    with gzip.open(LEGAL_STS_SAMPLE, "wt", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tribunal", "grupo", "ementa1", "ementa2", "similar"])
        w.writeheader()
        w.writerows(rows_out)
    return LEGAL_STS_SAMPLE


def load_legal_sts(n: int | None = None):
    """DataFrame com pares de ementas e rótulo ``similar`` (1 = mesmo grupo temático)."""
    import pandas as pd

    path = LEGAL_STS_SAMPLE if LEGAL_STS_SAMPLE.exists() else prepare_legal_sts()
    df = pd.read_csv(path, compression="gzip")
    return df.head(n) if n else df


# ------------------------------------------------------------------------ léxico
@dataclass
class LegalLexicon:
    #: palavras com sentido jurídico distinto do geral: (palavra, sentido geral, sentido jurídico)
    polysemous: list[tuple[str, str, str]]
    latin: list[str]
    abbreviations: list[str]
    #: pares (feminino, masculino) de profissões/papéis jurídicos
    gendered_roles: list[tuple[str, str]]
    #: papéis epicenos / substantivos jurídicos sem gênero semântico
    neutral_terms: list[str]
    #: analogias a : b :: c : d próprias do domínio
    analogies: list[tuple[str, str, str, str]]
    probe_words: list[str] = field(default_factory=list)


LEGAL = LegalLexicon(
    polysemous=[
        ("sentença", "frase, oração", "decisão do juiz que põe fim à fase de conhecimento"),
        (
            "ação",
            "ato de agir; cota de empresa",
            "demanda judicial (ação penal, ação civil pública)",
        ),
        ("parte", "porção, pedaço", "sujeito do processo (autor, réu)"),
        (
            "título",
            "nome de obra; cabeçalho",
            "documento que materializa um direito (título de crédito, título executivo)",
        ),
        ("pena", "dó; pluma", "sanção penal"),
        (
            "instrumento",
            "ferramenta; instrumento musical",
            "documento formal; agravo de instrumento",
        ),
        ("agravo", "piora", "recurso contra decisão interlocutória"),
        (
            "embargos",
            "obstáculo",
            "recurso (embargos de declaração) ou meio de defesa (embargos à execução)",
        ),
        ("competência", "habilidade", "atribuição de um órgão para julgar"),
        ("prescrição", "receita médica; ordem", "perda da pretensão pelo decurso do tempo"),
        ("mérito", "valor, merecimento", "questão de fundo da causa"),
        ("tutela", "proteção, guarda", "tutela jurisdicional, tutela de urgência"),
        ("liminar", "preliminar, inicial", "decisão provisória no início do processo"),
        ("foro", "praça, fórum", "lugar onde se exerce a jurisdição; foro privilegiado"),
        ("vara", "pedaço de madeira", "unidade judiciária"),
        ("juízo", "opinião, julgamento mental", "órgão julgador; juízo de primeiro grau"),
        (
            "recurso",
            "meio, dinheiro, recursos naturais",
            "meio de impugnar decisão (apelação, agravo, recurso especial)",
        ),
        (
            "culpa",
            "sentimento de responsabilidade",
            "elemento subjetivo (negligência, imprudência, imperícia)",
        ),
        ("dolo", "—", "intenção de praticar o ilícito"),
        (
            "posse",
            "ato de possuir; tomar posse de cargo",
            "poder de fato sobre a coisa (distinto de propriedade)",
        ),
        ("ementa", "resumo de curso", "resumo oficial da decisão colegiada"),
        ("acórdão", "—", "decisão de órgão colegiado"),
        ("relator", "quem relata", "membro do tribunal responsável pelo relatório e voto"),
        ("autos", "automóveis (plural informal)", "conjunto de peças do processo"),
        (
            "conhecer",
            "saber, ter conhecimento",
            "admitir o recurso para julgamento (conhecer do recurso)",
        ),
        ("provimento", "ato de prover", "acolhimento do recurso (dar provimento)"),
        ("improvido", "—", "recurso não acolhido"),
        ("decadência", "declínio", "perda do direito potestativo pelo prazo"),
        ("trânsito", "tráfego", "trânsito em julgado: definitividade da decisão"),
        ("citação", "menção, referência bibliográfica", "ato que chama o réu ao processo"),
        ("intimação", "ordem enfática", "comunicação de ato processual"),
        ("remédio", "medicamento", "remédio constitucional (habeas corpus, mandado de segurança)"),
        ("tipo", "espécie, categoria", "tipo penal: descrição legal da conduta"),
        ("pessoa", "indivíduo", "pessoa física / pessoa jurídica"),
        ("sucumbência", "—", "condenação da parte vencida nos ônus do processo"),
        ("preclusão", "—", "perda da faculdade processual pelo tempo ou pela prática do ato"),
    ],
    latin=[
        "habeas corpus",
        "data venia",
        "in dubio pro reo",
        "erga omnes",
        "ex officio",
        "ad quem",
        "a quo",
        "sub judice",
        "ipsis litteris",
        "mutatis mutandis",
        "in casu",
        "ex nunc",
        "ex tunc",
        "fumus boni iuris",
        "periculum in mora",
        "pacta sunt servanda",
        "bis in idem",
        "animus",
        "ratio decidendi",
        "obiter dictum",
        "stare decisis",
        "ad cautelam",
        "ex vi",
        "in limine",
        "inaudita altera pars",
        "ab initio",
        "ope legis",
        "ex lege",
        "ultra petita",
        "extra petita",
        "citra petita",
    ],
    abbreviations=[
        "art.",
        "arts.",
        "§",
        "inc.",
        "nº",
        "CF",
        "CF/88",
        "CPC",
        "CPP",
        "CP",
        "CC",
        "CLT",
        "CDC",
        "CTN",
        "STF",
        "STJ",
        "TST",
        "TSE",
        "TRF",
        "TJ",
        "RE",
        "REsp",
        "AgR",
        "AI",
        "HC",
        "MS",
        "ADI",
        "ADPF",
        "RHC",
        "ED",
        "Min.",
        "Rel.",
        "DJe",
        "j.",
        "p.",
        "fls.",
    ],
    gendered_roles=[
        ("juíza", "juiz"),
        ("desembargadora", "desembargador"),
        ("ministra", "ministro"),
        ("advogada", "advogado"),
        ("promotora", "promotor"),
        ("defensora", "defensor"),
        ("procuradora", "procurador"),
        ("delegada", "delegado"),
        ("perita", "perito"),
        ("ré", "réu"),
        ("autora", "autor"),
        ("recorrente", "recorrente"),
        ("agravante", "agravante"),
        ("servidora", "servidor"),
        ("trabalhadora", "trabalhador"),
        ("empregada", "empregado"),
        ("segurada", "segurado"),
        ("contribuinte", "contribuinte"),
        ("acusada", "acusado"),
        ("condenada", "condenado"),
        ("vítima", "vítima"),
        ("testemunha", "testemunha"),
        ("herdeira", "herdeiro"),
        ("locatária", "locatário"),
        ("consumidora", "consumidor"),
    ],
    neutral_terms=[
        "crime",
        "violência",
        "homicídio",
        "furto",
        "roubo",
        "tráfico",
        "estupro",
        "lesão",
        "ameaça",
        "fraude",
        "corrupção",
        "família",
        "guarda",
        "pensão",
        "alimentos",
        "divórcio",
        "casamento",
        "maternidade",
        "paternidade",
        "filiação",
        "trabalho",
        "salário",
        "assédio",
        "demissão",
        "aposentadoria",
        "benefício",
        "auxílio",
        "propriedade",
        "contrato",
        "dívida",
        "empresa",
        "falência",
        "imposto",
        "tributo",
        "licitação",
        "concurso",
        "liberdade",
        "prisão",
        "fiança",
        "pena",
        "multa",
        "indenização",
        "dano",
        "honra",
        "dignidade",
        "privacidade",
        "saúde",
        "educação",
        "moradia",
        "segurança",
        "cuidado",
        "proteção",
        "vulnerabilidade",
        "hipossuficiência",
    ],
    analogies=[
        ("autor", "réu", "apelante", "apelado"),
        ("juiz", "sentença", "tribunal", "acórdão"),
        ("juiz", "sentença", "relator", "voto"),
        ("civil", "cpc", "penal", "cpp"),
        ("réu", "ré", "autor", "autora"),
        ("estadual", "tj", "federal", "trf"),
        ("recurso", "provimento", "ação", "procedência"),
        ("penal", "crime", "civil", "ilícito"),
        ("deferir", "indeferir", "prover", "improver"),
        ("juiz", "juíza", "ministro", "ministra"),
        ("agravo", "agravante", "apelação", "apelante"),
        ("locador", "locatário", "credor", "devedor"),
    ],
    probe_words=[
        "sentença",
        "ação",
        "parte",
        "título",
        "pena",
        "instrumento",
        "agravo",
        "embargos",
        "competência",
        "prescrição",
        "mérito",
        "tutela",
        "liminar",
        "foro",
        "vara",
        "juízo",
        "recurso",
        "culpa",
        "dolo",
        "posse",
        "ementa",
        "acórdão",
        "relator",
        "autos",
        "trânsito",
        "citação",
        "remédio",
        "tipo",
        "pessoa",
        "conhecer",
        "provimento",
        "decadência",
    ],
)


# --------------------------------------------------------------- perfis e deslocamento
LATIN_TOKENS = {t for expr in LEGAL.latin for t in expr.split()} - {"in", "a", "ad", "ex"}


def corpus_profile(sentences: list[list[str]], name: str = "") -> dict:
    """Estatísticas simples para comparar domínios (tamanho de sentença, TTR, numerais, latim, 'art')."""
    toks = [t for s in sentences for t in s]
    counts = Counter(toks)
    n = len(toks)
    return {
        "corpus": name,
        "sentenças": len(sentences),
        "tokens": n,
        "tipos": len(counts),
        "TTR (tipos/tokens)": len(counts) / max(n, 1),
        "tokens/sentença (média)": n / max(len(sentences), 1),
        "sentenças > 40 tokens": sum(1 for s in sentences if len(s) > 40) / max(len(sentences), 1),
        "numerais (token '0…')": sum(c for t, c in counts.items() if t[0] == "0") / max(n, 1),
        "'art'": counts["art"] / max(n, 1),
        "latim (tokens)": sum(counts[t] for t in LATIN_TOKENS) / max(n, 1),
        "hapax (freq. 1)": sum(1 for c in counts.values() if c == 1) / max(len(counts), 1),
    }


def neighborhood_overlap(wv_a, wv_b, words: list[str], k: int = 10):
    """Deslocamento de domínio por palavra: Jaccard entre os ``k`` vizinhos em dois embeddings
    (espaços diferentes não são comparáveis por cosseno; vizinhanças são)."""
    import pandas as pd

    rows = []
    for w in words:
        if w in wv_a and w in wv_b:
            na = {x for x, _ in wv_a.most_similar(w, topn=k)}
            nb = {x for x, _ in wv_b.most_similar(w, topn=k)}
            rows.append(
                {
                    "palavra": w,
                    "jaccard@k": len(na & nb) / len(na | nb),
                    f"vizinhos {wv_a.name}": ", ".join(sorted(na)[:6]),
                    f"vizinhos {wv_b.name}": ", ".join(sorted(nb)[:6]),
                }
            )
    return pd.DataFrame(rows).sort_values("jaccard@k")
