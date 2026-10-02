"""Léxicos usados na análise de viés (Bolukbasi et al., 2016) em inglês e em português.

Inglês: reutilizamos as listas públicas do repositório ``tolga-b/debiaswe`` (MIT):
``definitional_pairs`` (10 pares usados para a PCA da Fig. 6), ``equalize_pairs``,
``gender_specific_seed`` (218 palavras, Apêndice C) e ``professions`` (com notas da *crowd*).

Português: listas autorais inspiradas nas do artigo. Observação didática importante
(§9 do artigo): o português tem **gênero gramatical** — a maioria dos substantivos de
profissão carrega marca morfológica (*enfermeiro/enfermeira*). Por isso separamos:

* ``professions_epicene`` – formas comuns de dois gêneros (*dentista, gerente, estudante*),
  onde um viés geométrico não pode ser explicado pela ortografia;
* ``professions_gendered_pairs`` – pares morfológicos, úteis como *pares definicionais
  adicionais* e para discutir o que *deve* permanecer após o *debias*;
* ``stereotype_neutral`` – substantivos sem gênero semântico (*futebol, cozinha, ciência*).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from replang.config import PATHS, SOURCES
from replang.utils.download import download


@dataclass
class Lexicon:
    lang: str
    definitional_pairs: list[tuple[str, str]]  # (feminino, masculino)
    equalize_pairs: list[tuple[str, str]]
    gender_specific: list[str]
    professions: list[str]
    stereotype_neutral: list[str] = field(default_factory=list)
    professions_gendered_pairs: list[tuple[str, str]] = field(default_factory=list)
    she_he: tuple[str, str] = ("she", "he")
    extra: dict = field(default_factory=dict)

    def all_gender_words(self) -> set[str]:
        s = set(self.gender_specific)
        for a, b in self.definitional_pairs + self.equalize_pairs + self.professions_gendered_pairs:
            s.update((a, b))
        return s


# ----------------------------------------------------------------------------- inglês
def _debiaswe(name: str) -> Path:
    local = PATHS.raw / f"debiaswe_{name}.json"
    if local.exists():
        return local
    sample = PATHS.samples / "lexicons" / f"debiaswe_{name}.json"
    if sample.exists():
        return sample
    return download(SOURCES["debiaswe"]["url"] + f"{name}.json", local)


def load_en() -> Lexicon:
    # Tudo em minúsculas: os embeddings usados (GloVe 6B) são lowercase; o w2vNEWS original não era.
    defs = [tuple(w.lower() for w in p) for p in json.load(open(_debiaswe("definitional_pairs")))]
    eq = [tuple(w.lower() for w in p) for p in json.load(open(_debiaswe("equalize_pairs")))]
    seed = [w.lower() for w in json.load(open(_debiaswe("gender_specific_seed")))]
    profs = [[p[0].lower(), p[1], p[2]] for p in json.load(open(_debiaswe("professions")))]
    prof_words = [p[0] for p in profs]
    return Lexicon(
        lang="en",
        definitional_pairs=defs,
        equalize_pairs=eq,
        gender_specific=seed,
        professions=prof_words,
        stereotype_neutral=[
            "nurse",
            "receptionist",
            "homemaker",
            "librarian",
            "nanny",
            "hairdresser",
            "stylist",
            "housekeeper",
            "bookkeeper",
            "maestro",
            "skipper",
            "philosopher",
            "captain",
            "architect",
            "financier",
            "warrior",
            "broadcaster",
            "magician",
            "boss",
            "programmer",
            "engineer",
            "scientist",
            "doctor",
            "teacher",
            "lawyer",
            "pilot",
            "chef",
            "cook",
            "softball",
            "football",
            "cosmetics",
            "pharmaceuticals",
            "sewing",
            "carpentry",
            "giggle",
            "chuckle",
        ],
        she_he=("she", "he"),
        extra={"professions_meta": profs},
    )


# -------------------------------------------------------------------------- português
PT_DEFINITIONAL_PAIRS: list[tuple[str, str]] = [
    ("mulher", "homem"),
    ("ela", "ele"),
    ("menina", "menino"),
    ("mãe", "pai"),
    ("filha", "filho"),
    ("irmã", "irmão"),
    ("rainha", "rei"),
    ("esposa", "marido"),
    ("feminino", "masculino"),
    ("maria", "joão"),
]
PT_EQUALIZE_PAIRS: list[tuple[str, str]] = PT_DEFINITIONAL_PAIRS + [
    ("dela", "dele"),
    ("senhora", "senhor"),
    ("garota", "garoto"),
    ("moça", "moço"),
    ("avó", "avô"),
    ("tia", "tio"),
    ("atriz", "ator"),
    ("princesa", "príncipe"),
    ("madrinha", "padrinho"),
    ("sobrinha", "sobrinho"),
    ("neta", "neto"),
    ("namorada", "namorado"),
    ("viúva", "viúvo"),
    ("imperatriz", "imperador"),
    ("duquesa", "duque"),
    ("deusa", "deus"),
    ("fêmea", "macho"),
    ("mulheres", "homens"),
    ("meninas", "meninos"),
    ("filhas", "filhos"),
    ("irmãs", "irmãos"),
    ("elas", "eles"),
    ("delas", "deles"),
    ("senhoras", "senhores"),
    ("mães", "pais"),
]
PT_GENDER_SPECIFIC: list[str] = sorted(
    {
        *[w for p in PT_EQUALIZE_PAIRS for w in p],
        "feminina",
        "masculina",
        "femininas",
        "masculinas",
        "menininha",
        "menininho",
        "donzela",
        "dama",
        "cavalheiro",
        "freira",
        "monge",
        "padre",
        "madre",
        "noiva",
        "noivo",
        "sogra",
        "sogro",
        "nora",
        "genro",
        "cunhada",
        "cunhado",
        "madrasta",
        "padrasto",
        "enteada",
        "enteado",
        "patroa",
        "patrão",
        "rapaz",
        "rapariga",
        "moças",
        "moços",
        "rapazes",
        "garotas",
        "garotos",
        "senhorita",
        "mocinha",
        "mocinho",
        "avós",
        "tias",
        "tios",
        "esposas",
        "maridos",
        "viúvas",
        "viúvos",
        "rainhas",
        "reis",
        "princesas",
        "príncipes",
        "atrizes",
        "atores",
        "útero",
        "próstata",
        "gravidez",
        "grávida",
        "menstruação",
        "testosterona",
        "estrogênio",
        "lésbica",
        "gay",
        "feminismo",
        "feminista",
        "machismo",
        "machista",
        "maternidade",
        "paternidade",
        "materna",
        "paterno",
        "materno",
        "paterna",
    }
)
PT_PROFESSIONS_EPICENE: list[str] = [
    "dentista",
    "jornalista",
    "estudante",
    "gerente",
    "presidente",
    "cientista",
    "artista",
    "pianista",
    "motorista",
    "policial",
    "oficial",
    "atleta",
    "agente",
    "assistente",
    "docente",
    "intérprete",
    "economista",
    "analista",
    "contabilista",
    "taxista",
    "florista",
    "recepcionista",
    "telefonista",
    "estilista",
    "modelo",
    "chefe",
    "líder",
    "cliente",
    "paciente",
    "comerciante",
    "dirigente",
    "representante",
    "tenente",
    "sargento",
    "capitão",
    "general",
    "soldado",
    "juiz",
    "pintor",
    "escritor",
    "poeta",
    "cantor",
    "dançarino",
    "guarda",
    "detetive",
    "piloto",
    "atendente",
    "repórter",
    "cineasta",
    "jurista",
    "psicanalista",
    "fisioterapeuta",
    "terapeuta",
    "massagista",
    "esteticista",
    "especialista",
    "colega",
    "profissional",
    "gênio",
    "sábio",
    "herói",
    "vilão",
    "chef",
    "babá",
    "ama",
]
PT_PROFESSIONS_GENDERED_PAIRS: list[tuple[str, str]] = [
    ("enfermeira", "enfermeiro"),
    ("engenheira", "engenheiro"),
    ("professora", "professor"),
    ("médica", "médico"),
    ("secretária", "secretário"),
    ("cozinheira", "cozinheiro"),
    ("advogada", "advogado"),
    ("empregada", "empregado"),
    ("faxineira", "faxineiro"),
    ("arquiteta", "arquiteto"),
    ("programadora", "programador"),
    ("diretora", "diretor"),
    ("cabeleireira", "cabeleireiro"),
    ("bibliotecária", "bibliotecário"),
    ("deputada", "deputado"),
    ("senadora", "senador"),
    ("vereadora", "vereador"),
    ("ministra", "ministro"),
    ("juíza", "juiz"),
    ("escritora", "escritor"),
    ("cantora", "cantor"),
    ("bailarina", "bailarino"),
    ("costureira", "costureiro"),
    ("filósofa", "filósofo"),
    ("física", "físico"),
    ("matemática", "matemático"),
    ("química", "químico"),
    ("operária", "operário"),
    ("camponesa", "camponês"),
    ("pedreira", "pedreiro"),
    ("mecânica", "mecânico"),
]
PT_STEREOTYPE_NEUTRAL: list[str] = [
    "casa",
    "cozinha",
    "futebol",
    "matemática",
    "beleza",
    "carreira",
    "ciência",
    "família",
    "bebê",
    "criança",
    "amor",
    "força",
    "coragem",
    "inteligência",
    "delicadeza",
    "sensibilidade",
    "violência",
    "poder",
    "guerra",
    "poesia",
    "literatura",
    "moda",
    "filosofia",
    "religião",
    "dinheiro",
    "salário",
    "liderança",
    "cuidado",
    "enfermagem",
    "engenharia",
    "física",
    "computação",
    "programação",
    "maquiagem",
    "negócios",
    "política",
    "música",
    "dança",
    "costura",
    "mecânica",
    "medicina",
    "direito",
    "educação",
    "escola",
    "universidade",
    "laboratório",
    "fábrica",
    "escritório",
    "hospital",
    "igreja",
    "esporte",
    "vôlei",
    "ginástica",
    "boxe",
    "balé",
    "lágrimas",
    "choro",
    "riso",
    "fofoca",
    "piada",
    "bravura",
    "ternura",
    "ambição",
    "paciência",
    "razão",
    "emoção",
    "lógica",
    "intuição",
]
PT_OCCUPATION_LABELS = {  # usado para colorir gráficos
    "estereótipo feminino": [
        "enfermagem",
        "cozinha",
        "costura",
        "maquiagem",
        "beleza",
        "cuidado",
        "balé",
        "dança",
        "moda",
        "babá",
    ],
    "estereótipo masculino": [
        "futebol",
        "engenharia",
        "programação",
        "mecânica",
        "guerra",
        "boxe",
        "negócios",
        "poder",
        "força",
        "liderança",
    ],
}


def load_pt() -> Lexicon:
    return Lexicon(
        lang="pt",
        definitional_pairs=PT_DEFINITIONAL_PAIRS,
        equalize_pairs=PT_EQUALIZE_PAIRS,
        gender_specific=PT_GENDER_SPECIFIC,
        professions=PT_PROFESSIONS_EPICENE,
        stereotype_neutral=PT_STEREOTYPE_NEUTRAL,
        professions_gendered_pairs=PT_PROFESSIONS_GENDERED_PAIRS,
        she_he=("ela", "ele"),
        extra={"labels": PT_OCCUPATION_LABELS},
    )


class _Lazy:
    def __init__(self, loader):
        self._loader, self._v = loader, None

    def __getattr__(self, item):
        if self._v is None:
            self._v = self._loader()
        return getattr(self._v, item)

    def get(self) -> Lexicon:
        if self._v is None:
            self._v = self._loader()
        return self._v


PT = _Lazy(load_pt)
EN = _Lazy(load_en)
