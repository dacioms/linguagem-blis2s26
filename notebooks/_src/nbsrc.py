"""Mini-DSL para escrever notebooks como código Python."""


def md(text: str) -> tuple[str, str]:
    return ("md", text)


def code(text: str) -> tuple[str, str]:
    return ("code", text)


SETUP = code('''
# Configuração comum a todos os notebooks ---------------------------------------------
import sys, warnings, os
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"  # JupyterLab renderiza offline; HTML carrega plotly.js via CDN (notebooks leves)
pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 160)
from replang import PATHS, WordVectors, settings
FAST = settings.fast  # REPLANG_FAST=1 reduz épocas/tamanhos
print("raiz:", ROOT, "| fast:", FAST)
''')

HEADER = """
> **Disciplina:** Representações de Linguagem — *como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto* (BLIS 2S26).
> Este notebook faz parte de uma sequência (`00` → `13`). Código de apoio em `src/replang`; interface interativa em `app/` (`uv run replang app`).
"""


def refs(*keys: str) -> str:
    table = {
        "mikolov13a": "MIKOLOV, T.; CHEN, K.; CORRADO, G.; DEAN, J. **Efficient Estimation of Word Representations in Vector Space.** ICLR Workshop, 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)",
        "mikolov13b": "MIKOLOV, T.; SUTSKEVER, I.; CHEN, K.; CORRADO, G.; DEAN, J. **Distributed Representations of Words and Phrases and their Compositionality.** NIPS, 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)",
        "glove": "PENNINGTON, J.; SOCHER, R.; MANNING, C. **GloVe: Global Vectors for Word Representation.** EMNLP, 2014. [ACL D14-1162](https://aclanthology.org/D14-1162/)",
        "fasttext": "BOJANOWSKI, P.; GRAVE, E.; JOULIN, A.; MIKOLOV, T. **Enriching Word Vectors with Subword Information.** TACL, 2017. [arXiv:1607.04606](https://arxiv.org/abs/1607.04606)",
        "hartmann": "HARTMANN, N. et al. **Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks.** STIL, 2017. [arXiv:1708.06025](https://arxiv.org/abs/1708.06025)",
        "bolukbasi": "BOLUKBASI, T.; CHANG, K.-W.; ZOU, J.; SALIGRAMA, V.; KALAI, A. **Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.** NIPS, 2016. [arXiv:1607.06520](https://arxiv.org/abs/1607.06520)",
        "elmo": "PETERS, M. et al. **Deep contextualized word representations.** NAACL, 2018. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)",
        "doc2vec": "LE, Q.; MIKOLOV, T. **Distributed Representations of Sentences and Documents.** ICML, 2014. [arXiv:1405.4053](https://arxiv.org/abs/1405.4053)",
        "levy": "LEVY, O.; GOLDBERG, Y. **Neural Word Embedding as Implicit Matrix Factorization.** NIPS, 2014; LEVY, GOLDBERG & DAGAN. **Improving Distributional Similarity with Lessons Learned from Word Embeddings.** TACL, 2015.",
        "baroni": "BARONI, M.; DINU, G.; KRUSZEWSKI, G. **Don't count, predict!** ACL, 2014.",
        "faruqui": "FARUQUI, M. et al. **Problems With Evaluation of Word Embeddings Using Word Similarity Tasks.** RepEval, 2016.",
        "harris": "HARRIS, Z. **Distributional Structure.** Word, 1954; FIRTH, J. R. *A synopsis of linguistic theory*, 1957.",
        "rodrigues": "RODRIGUES, J. et al. **LX-DSemVectors: Distributional Semantics Models for Portuguese.** PROPOR, 2016.",
    }
    return "\n".join(f"- {table[k]}" for k in keys)
