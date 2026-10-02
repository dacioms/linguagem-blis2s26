from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "13 · Síntese: a linha do tempo, o que o vetor carrega e roteiro da apresentação"

CELLS = [
    md(f"""
# 13 · Síntese: a linha do tempo, o que o vetor carrega e o roteiro de 4 h
{HEADER}

Este notebook fecha a sequência. Ele (1) consolida em uma tabela única os modelos e as métricas vistas, (2) reconta a história em uma linha do tempo, (3) responde às três perguntas do título da apresentação e (4) traz o roteiro minuto a minuto (detalhado em `docs/roteiro_4h.md`).
"""),
    SETUP,
    md("""
## 1. Linha do tempo
"""),
    code('''
timeline = pd.DataFrame([
    (1954, "Harris; Firth (1957)", "hipótese distribucional", "premissa", "01"),
    (1990, "Deerwester et al. (LSA)", "SVD da matriz termo×documento", "contagem", "01"),
    (1996, "Lund & Burgess (HAL)", "matriz palavra×palavra por janela", "contagem", "01"),
    (2003, "Bengio et al. (NNLM)", "modelo neural de linguagem com embeddings", "preditivo", "02"),
    (2013, "Mikolov et al. (2013a)", "CBOW e Skip-gram; analogias", "preditivo", "02"),
    (2013, "Mikolov et al. (2013b)", "NEG, HS, subamostragem, frases, composição aditiva", "preditivo", "03"),
    (2014, "Pennington et al. (GloVe)", "mínimos quadrados ponderados sobre log X_ij", "contagem+predição", "04"),
    (2014, "Le & Mikolov (Doc2Vec)", "vetor por parágrafo", "preditivo", "11"),
    (2014, "Levy & Goldberg", "SGNS = fatoração de PMI deslocado", "teoria", "01/03"),
    (2016, "Bolukbasi et al.", "viés de gênero como direção; hard/soft debias", "análise/ética", "08/09/10"),
    (2017, "Bojanowski et al. (fastText)", "n-gramas de caracteres; OOV", "subpalavra", "05"),
    (2017, "Hartmann et al.", "31 modelos PT; analogias × tarefas", "avaliação", "06/07"),
    (2018, "Peters et al. (ELMo)", "representações contextuais de biLM", "contextual", "12"),
    (2018, "Devlin et al. (BERT)", "Transformers bidirecionais + fine-tuning", "contextual", "—"),
    (2019, "Gonen & Goldberg", "debias esconde, não remove", "análise/ética", "09"),
], columns=["ano", "trabalho", "ideia", "família", "notebook"])
timeline
'''),
    md("""
## 2. Quadro consolidado dos modelos deste projeto

Reunimos, para cada modelo disponível, as métricas calculadas nos notebooks anteriores (recalculadas aqui de forma rápida): analogias PT-BR (cobertura e acurácia), POS tagging, mini-STS, DirectBias de gênero (profissões epicenas + áreas). É a "tabela-resumo" para a apresentação.
"""),
    code('''
import time
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.models.trainers import list_models
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
from replang.eval import pos_tagging_eval, sentence_similarity_eval, MINI_STS_PT
from replang.data.corpora import load_macmorpho
from replang.data.lexicons import PT
from replang.bias import gender_direction, direct_bias
lex = PT.get(); an = load_analogies("pt-br")
train = load_macmorpho(split="train"); test = load_macmorpho(split="test", limit_sentences=300)
models = {"Wikipedia2Vec PT 100d": load_ptwiki()}
for m in list_models():
    if m["algo"] in ("word2vec", "fasttext", "glove-numpy"):
        models[m["name"]] = load_local_model(m["name"])
rows = []
for name, wv in models.items():
    t = time.time()
    r = evaluate_analogies(wv, an, restrict=30000, max_per_category=200).set_index("categoria")
    pos = pos_tagging_eval(wv, train, test, max_train_tokens=20000)["acurácia"]
    sts = sentence_similarity_eval(wv, MINI_STS_PT)["pearson"]
    g = gender_direction(wv, lex.definitional_pairs)
    neutras = [w for w in lex.professions + lex.stereotype_neutral if w in wv]
    rows.append({"modelo": name, "dim": wv.dim, "|V|": len(wv), "cobertura analogias": r.loc["TOTAL", "cobertas"] / r.loc["TOTAL", "n"], "analogias (acurácia)": r.loc["TOTAL", "acurácia"],
                 "POS": pos, "mini-STS ρ": sts, "DirectBias gênero": direct_bias(wv, neutras, g), "s": round(time.time() - t)})
resumo = pd.DataFrame(rows).set_index("modelo")
resumo.style.format({"cobertura analogias": "{:.0%}", "analogias (acurácia)": "{:.1%}", "POS": "{:.3f}", "mini-STS ρ": "{:.2f}", "DirectBias gênero": "{:.3f}"}).background_gradient(cmap="Blues", subset=["analogias (acurácia)", "POS", "mini-STS ρ"]).background_gradient(cmap="Reds", subset=["DirectBias gênero"])
'''),
    md("""
## 3. As três perguntas do título

### Como transformar palavra em vetor?
Contando contextos (01, 04) ou prevendo-os (02, 03, 05) — ambos são formas de aplicar a hipótese distribucional; o resultado é um espaço onde **proximidade = contexto parecido** e **direções = relações** (analogias, gênero, tempo verbal).

### O que esse vetor carrega?
* Semântica lexical e relações (02, 03, 04): *rei–rainha*, *país–capital*.
* Morfologia, se o modelo olha dentro da palavra (05).
* Tópico/estilo, se olha o documento (11).
* Sintaxe e sentido dependentes do contexto, se olha a sentença (12).
* E as **estatísticas sociais do corpus** (08, 10): estereótipos ocupacionais, viés indireto — "o vetor carrega o que o texto carrega".

### O viés que vem junto: dá para tirar?
Em parte (09): Neutralize/Equalize zeram métricas específicas e preservam a utilidade; mas a estrutura residual permanece (Gonen & Goldberg) e a definição do que é "neutro" é uma decisão humana, dependente da língua (10) e da aplicação.
"""),
    md("""
## 4. Roteiro de 4 horas (resumo; detalhe em `docs/roteiro_4h.md`)

| Hora | Bloco | Material | Dinâmica |
|---|---|---|---|
| 0:00 | Abertura: o título como três perguntas | nb 00 | demonstração `homem:rei::mulher:?` e `ele:médico::ela:?` no app (página 4) |
| 0:10 | A. Premissas | nb 01; app p. 1 | construir a matriz de coocorrência ao vivo com o toy corpus; PPMI; SVD |
| 0:35 | B. word2vec | nb 02, 03; app p. 2, 3, 5 | treinar Skip-gram ao vivo; HS × NEG; subamostragem; frases; PCA país→capital |
| 1:20 | C. Variações | nb 04, 05, 11; app p. 6 | GloVe (razões de probabilidade); fastText (OOV ao vivo); Doc2Vec (gêneros de Machado) |
| 1:50 | *intervalo* | | |
| 2:05 | D. Português | nb 06, 07; app p. 7 | corpus e pré-processamento de Hartmann; analogias × POS × STS; discussão de Faruqui |
| 2:45 | E. Viés | nb 08, 09, 10; app p. 8 | direção de gênero (PCA); extremos; analogias geradas; hard/soft debias; gênero gramatical em PT; experimento sintético |
| 3:35 | F. Contexto | nb 12; app p. 9 | *banco* em duas sentenças; camadas × tarefas |
| 3:50 | Síntese e discussão | nb 13 | tabela consolidada; perguntas abertas |

### Perguntas para fechar a discussão
1. Se um embedding "só reflete o corpus", de quem é a responsabilidade pelo viés? (dados, modelo, aplicação)
2. O que uma avaliação de embeddings **para o português** deveria medir que os benchmarks traduzidos não medem?
3. Com modelos contextuais e LLMs, o que muda nessa história — e o que permanece?
"""),
    md(f"""
## Referências completas da disciplina

### Leituras obrigatórias
{refs("mikolov13a", "bolukbasi", "hartmann")}

### Leituras complementares
{refs("mikolov13b", "glove", "fasttext", "elmo", "doc2vec")}

### Citadas ao longo dos notebooks
{refs("harris", "levy", "baroni", "faruqui", "rodrigues")}
- GONEN, H.; GOLDBERG, Y. **Lipstick on a Pig.** NAACL, 2019.
- LING, W. et al. **Two/Too Simple Adaptations of Word2Vec for Syntax Problems** (wang2vec). NAACL, 2015.
- DEVLIN, J. et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** NAACL, 2019.
"""),
]
