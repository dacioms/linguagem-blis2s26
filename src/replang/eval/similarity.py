"""Similaridade de palavras e de sentenças (avaliação intrínseca por julgamentos humanos).

Os conjuntos padrão (WordSim-353, RG-65, SimLex-999; ASSIN para sentenças em PT) exigem download
externo. Para que os notebooks rodem *offline*, incluímos **mini-conjuntos autorais ilustrativos**
(``MINI_WORDSIM_PT``, ``MINI_STS_PT``) — pequenos demais para conclusões científicas, mas
suficientes para demonstrar a métrica (correlação de Spearman/Pearson) e seus problemas
(Faruqui et al., 2016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from replang.embedding import WordVectors

# (palavra1, palavra2, similaridade 0-10) — escala inspirada no WordSim-353
MINI_WORDSIM_PT: list[tuple[str, str, float]] = [
    ("rei", "rainha", 8.5),
    ("rei", "monarca", 9.0),
    ("carro", "automóvel", 9.5),
    ("carro", "estrada", 6.0),
    ("médico", "hospital", 7.0),
    ("médico", "enfermeiro", 7.5),
    ("professor", "escola", 7.0),
    ("gato", "cachorro", 7.0),
    ("gato", "tigre", 6.5),
    ("maçã", "banana", 6.5),
    ("maçã", "computador", 1.0),
    ("café", "xícara", 6.0),
    ("dinheiro", "banco", 7.0),
    ("banco", "praça", 3.5),
    ("amor", "ódio", 5.0),
    ("amor", "carinho", 8.0),
    ("livro", "biblioteca", 7.5),
    ("livro", "papel", 6.0),
    ("sol", "lua", 6.5),
    ("sol", "calor", 7.0),
    ("brasil", "portugal", 6.0),
    ("brasil", "futebol", 5.0),
    ("rio", "mar", 6.5),
    ("rio", "montanha", 4.0),
    ("guerra", "paz", 5.5),
    ("guerra", "soldado", 7.5),
    ("música", "piano", 7.0),
    ("música", "pedra", 1.0),
    ("pão", "padaria", 7.5),
    ("pão", "avião", 0.5),
    ("telefone", "celular", 8.5),
    ("telefone", "cadeira", 1.0),
    ("cidade", "capital", 7.0),
    ("cidade", "aldeia", 6.0),
    ("rápido", "veloz", 9.0),
    ("rápido", "lento", 4.0),
    ("feliz", "alegre", 9.0),
    ("feliz", "triste", 4.0),
    ("grande", "enorme", 8.5),
    ("grande", "pequeno", 4.0),
]

# (sentença1, sentença2, similaridade 1-5) — escala do ASSIN (Hartmann et al., 2017, §4.2)
MINI_STS_PT: list[tuple[str, str, float]] = [
    ("o menino joga futebol no parque", "uma criança brinca com a bola no parque", 4.0),
    ("o menino joga futebol no parque", "o governo anunciou novas medidas econômicas", 1.0),
    (
        "a empresa anunciou lucro recorde no trimestre",
        "a companhia divulgou resultados financeiros positivos",
        4.5,
    ),
    ("a empresa anunciou lucro recorde no trimestre", "o gato dorme no sofá da sala", 1.0),
    (
        "o presidente viajou para a europa ontem",
        "o chefe de estado fez uma viagem ao exterior",
        4.0,
    ),
    ("o presidente viajou para a europa ontem", "a receita do bolo leva três ovos", 1.0),
    ("chove muito na região sul do país", "o tempo está chuvoso no sul", 4.5),
    ("chove muito na região sul do país", "o time venceu a partida por dois a zero", 1.0),
    ("a mulher comprou pão na padaria", "a senhora foi à padaria comprar pão", 5.0),
    ("a mulher comprou pão na padaria", "o homem vendeu o carro usado", 2.0),
    ("o médico atendeu o paciente no hospital", "o doutor examinou o doente na clínica", 4.5),
    ("o médico atendeu o paciente no hospital", "a banda tocou no festival de música", 1.0),
    ("os alunos estudam para a prova de matemática", "os estudantes se preparam para o exame", 4.0),
    ("os alunos estudam para a prova de matemática", "o rio transbordou após a tempestade", 1.0),
    (
        "o filme estreia na próxima semana nos cinemas",
        "a obra chega às salas de cinema em breve",
        4.0,
    ),
    ("o filme estreia na próxima semana nos cinemas", "a inflação subiu em janeiro", 1.0),
    ("o cachorro late para o carteiro", "o cão ladra quando o carteiro chega", 4.5),
    ("o cachorro late para o carteiro", "a professora corrigiu as redações", 1.0),
    ("a ponte foi fechada para reforma", "a estrada está interditada por obras", 3.5),
    ("a ponte foi fechada para reforma", "o cantor lançou um novo álbum", 1.0),
    ("ela comprou um vestido vermelho", "ela comprou uma camisa azul", 3.0),
    ("ele perdeu as chaves de casa", "ele encontrou as chaves do carro", 2.5),
    ("o avião decolou com atraso", "o voo partiu depois do horário previsto", 4.5),
    ("o avião decolou com atraso", "o avião pousou no horário", 2.5),
    ("a cidade ganhou um novo parque", "a prefeitura inaugurou uma praça", 3.5),
    ("a cidade ganhou um novo parque", "o museu recebeu uma exposição", 2.0),
    ("o jantar foi delicioso", "a comida estava ótima", 4.5),
    ("o jantar foi delicioso", "a reunião foi cancelada", 1.0),
    ("a biblioteca abre às oito", "a loja fecha às dezoito", 2.0),
    ("a biblioteca abre às oito", "a biblioteca inicia o atendimento às oito horas", 4.5),
]


def word_similarity_eval(
    wv: WordVectors, dataset: list[tuple[str, str, float]] | None = None
) -> dict:
    """Correlação de Spearman entre cosseno e julgamentos humanos (como Bojanowski et al., Tab. 1)."""
    dataset = dataset or MINI_WORDSIM_PT
    rows = []
    for a, b, s in dataset:
        if a in wv and b in wv:
            rows.append({"w1": a, "w2": b, "humano": s, "cosseno": wv.similarity(a, b)})
    df = pd.DataFrame(rows)
    rho = spearmanr(df.humano, df.cosseno).correlation if len(df) > 2 else np.nan
    return {"spearman": float(rho), "cobertura": len(df) / len(dataset), "tabela": df}
