from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "04 · GloVe (Pennington, Socher & Manning, 2014): contagem global + predição"

CELLS = [
    md(f"""
# 04 · GloVe: vetores globais a partir de razões de probabilidades
{HEADER}

**Artigo-base (leitura complementar):** Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation* (EMNLP).

## Ideia central
O word2vec percorre o corpus janela a janela e **nunca usa as estatísticas globais** explicitamente; os métodos de contagem (LSA/HAL, notebook 01) usam a matriz global mas ponderam mal (todas as coocorrências pesam igual, e as frequentes dominam). O GloVe propõe treinar diretamente sobre a matriz de coocorrência $X$, com um objetivo de mínimos quadrados **ponderado**:

$$J = \\sum_{{i,j=1}}^{{V}} f(X_{{ij}})\\,\\big(w_i^{{\\top}}\\tilde w_j + b_i + \\tilde b_j - \\log X_{{ij}}\\big)^2,\\qquad f(x) = \\begin{{cases}} (x/x_{{\\max}})^{{\\alpha}} & x < x_{{\\max}} \\\\ 1 & \\text{{caso contrário}}\\end{{cases}}$$

com $x_{{\\max}} = 100$ e $\\alpha = 3/4$. O argumento do artigo: o que carrega significado não é $P(k\\mid i)$, mas a **razão** $P(k\\mid i)/P(k\\mid j)$ — e um modelo log-bilinear $w_i^\\top \\tilde w_k = \\log P(k\\mid i)$ é o jeito mais simples de capturar razões como diferenças de vetores.

Em Hartmann et al. (2017) o GloVe foi o **melhor nas analogias** (sobretudo semânticas) e o **pior nas tarefas extrínsecas** — o paradoxo que discutiremos no notebook 07.
"""),
    SETUP,
    md("""
## 1. A motivação: razões de probabilidade (Tabela 1 do artigo)

O artigo usa *ice*/*steam*: para $k$ = *solid*, $P(k|ice)/P(k|steam) \\gg 1$; para *gas*, $\\ll 1$; para *water* e *fashion*, $\\approx 1$. Repetimos a ideia em português no corpus Machado com os alvos **rei**/**padre** — dois substantivos masculinos de autoridade com contextos distintos.
"""),
    code('''
from replang.data.corpora import load_machado_sentences
from replang.models.cooccurrence import build_vocab, cooccurrence_matrix
from replang.models.glove_np import glove_probability_ratios
sents = load_machado_sentences()
words, index, freq = build_vocab(sents, min_count=5)
X = cooccurrence_matrix(sents, index, window=5, weighting="harmonic")
probes = ["trono", "coroa", "igreja", "missa", "casa", "dia", "mulher", "dinheiro"]
glove_probability_ratios(X, words, "rei", "padre", probes).round(4)
'''),
    md("""
Leia a última coluna: *trono* e *coroa* ≫ 1 (associadas a *rei*), *igreja* e *missa* ≪ 1 (associadas a *padre*), *casa*, *dia*, *dinheiro* ≈ 1 (neutras). A razão **cancela** o que é comum aos dois e realça o que discrimina — é isso que o produto interno $w_i^\\top \\tilde w_k$ deve modelar.
"""),
    md("""
## 2. A função de ponderação $f(x)$

Três exigências do artigo: $f(0)=0$ (pares que nunca coocorrem não entram — a soma só percorre os $X_{ij}>0$, por isso é esparsa); $f$ não decrescente (pares raros pesam menos — são ruidosos); $f$ **saturada** (pares frequentíssimos não dominam — o problema do LSA).
"""),
    code('''
import plotly.express as px
x = np.linspace(0, 150, 300)
df = pd.DataFrame({"x": x, "α=3/4, x_max=100": np.where(x < 100, (x / 100) ** 0.75, 1), "α=1 (linear)": np.where(x < 100, x / 100, 1), "x_max=50": np.where(x < 50, (x / 50) ** 0.75, 1)})
px.line(df.melt(id_vars="x"), x="x", y="value", color="variable", template="plotly_white", title="Função de ponderação f(X_ij) do GloVe").show()
nz = X.data
print(f"pares não nulos: {X.nnz:,} de {len(words)**2:,} possíveis ({X.nnz/len(words)**2:.2%}); mediana de X_ij = {np.median(nz):.2f}; 99º percentil = {np.percentile(nz, 99):.1f}")
'''),
    md("""
## 3. Implementação *from scratch* (numpy + AdaGrad)

`replang.models.glove_np.GloVe` constrói $X$ com ponderação $1/d$ (distância na janela, como no artigo), inicializa $W, \\tilde W, b, \\tilde b$ e minimiza $J$ por SGD com **AdaGrad** (o otimizador do artigo) sobre lotes de pares $(i, j)$ não nulos. No final, o vetor da palavra é $W + \\tilde W$ (§4.2 do artigo: somar os dois reduz ruído).
"""),
    code('''
import inspect
from replang.models.glove_np import GloVe
src = inspect.getsource(GloVe.fit)
print(src[src.index("for epoch"):src.index("rec = {")])
'''),
    code('''
import time
from replang.data.corpora import toy_corpus
toy = toy_corpus() * 20
t = time.time()
g = GloVe(dim=20, window=2, epochs=60, x_max=10, lr=0.05, seed=1).fit(toy)
wv_g = g.to_wordvectors("GloVe toy")
print(f"{time.time()-t:.1f}s | perda final {g.history[-1]['loss']:.4f}")
from replang.viz import training_curve
training_curve(g.history, title="GloVe (toy): J ponderado por época").show()
for w in ["rei", "gato", "paris"]:
    print(f"{w:<6} {[x for x, _ in wv_g.most_similar(w, topn=4)]}")
'''),
    md("""
### 3.1 O mesmo algoritmo em JAX: `jit` + *scatter-add* (CPU ou GPU)

O gargalo da versão numpy é o `np.add.at` (acumulação esparsa, uma *thread*). `replang.models.glove_jax.GloVeJax` compila a época inteira (`jax.jit` + `lax.fori_loop`) e usa `.at[].add`; em CPU é 5–8× mais rápido e, com `uv sync --extra cuda`, roda na GPU sem mudar uma linha. Os resultados são equivalentes (mesma função objetivo, mesmo AdaGrad).
"""),
    code('''
from replang.accel import report
from replang.models.glove_jax import GloVeJax
print(report())
sub = sents[:30000 if FAST else 60000]
t = time.time(); gn = GloVe(dim=100, window=5, epochs=3, min_count=5, x_max=50, batch_size=16384).fit(sub); tn = time.time() - t
t = time.time(); gj = GloVeJax(dim=100, window=5, epochs=3, min_count=5, x_max=50, batch_size=16384).fit(sub); tj = time.time() - t
print(f"3 épocas em {len(sub)} sentenças — numpy: {tn:.1f}s (perda {gn.history[-1]['loss']:.4f}) | JAX: {tj:.1f}s (perda {gj.history[-1]['loss']:.4f}) → {tn/tj:.1f}×")
print("vizinhos de 'amor' — numpy:", [w for w, _ in gn.to_wordvectors().most_similar("amor", topn=5)], "| JAX:", [w for w, _ in gj.to_wordvectors().most_similar("amor", topn=5)])
'''),
    md("""
## 4. GloVe no corpus Machado e comparação com Skip-gram e PPMI-SVD

O modelo `machado_glove100` (numpy, 15 épocas, janela 5) foi treinado por `replang train`. Comparamos com o Skip-gram (gensim) e com o pipeline de contagem do notebook 01 em três frentes: vizinhos, analogias PT-BR (sintáticas) e a **mini-similaridade** de palavras (`MINI_WORDSIM_PT`).
"""),
    code('''
from replang.data.embeddings import load_local_model
from replang.models.cooccurrence import CountModel
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
from replang.eval.similarity import word_similarity_eval
glove = load_local_model("machado_glove100"); sg = load_local_model("machado_sg100")
cm = CountModel(window=5, dim=100, min_count=5, alpha=0.75, weighting="harmonic", max_vocab=25000).fit(sents)
ppmi_svd = cm.to_wordvectors("PPMI-SVD 100d (Machado)")
an = load_analogies("pt-br")
rows = []
for wv in [glove, sg, ppmi_svd]:
    r = evaluate_analogies(wv, an, restrict=None).set_index("categoria")
    ws = word_similarity_eval(wv)
    rows.append({"modelo": wv.name, "analogias sintáticas": r.loc["TOTAL sintática", "acurácia"], "analogias semânticas": r.loc["TOTAL semântica", "acurácia"],
                 "cobertas": int(r.loc["TOTAL", "cobertas"]), "mini-WordSim ρ": ws["spearman"]})
pd.DataFrame(rows).style.format({"analogias sintáticas": "{:.1%}", "analogias semânticas": "{:.1%}", "mini-WordSim ρ": "{:.2f}"})
'''),
    code('''
for w in ["capitu", "amor", "dinheiro", "rua", "ontem"]:
    print(f"{w:<9} GloVe:    {[x for x, _ in glove.most_similar(w, topn=6)]}")
    print(f"{'':<9} SG:       {[x for x, _ in sg.most_similar(w, topn=6)]}")
    print(f"{'':<9} PPMI-SVD: {[x for x, _ in ppmi_svd.most_similar(w, topn=6)]}")
'''),
    md("""
## 5. Contagem × predição: o debate

* Baroni, Dinu & Kruszewski (2014), *Don't count, predict!*: modelos preditivos (word2vec) venceram os de contagem em quase todos os testes.
* Pennington et al. (2014): GloVe, um modelo de contagem bem ponderado, iguala ou supera o word2vec em analogias com menos tempo de treino (Fig. 4 do artigo: GloVe converge em menos iterações/tempo que CBOW/Skip-gram).
* Levy, Goldberg & Dagan (2015): a diferença são **hiperparâmetros** (janela dinâmica, subamostragem, suavização $\\alpha$, deslocamento, vetor de contexto somado, normalização). Transplantados para PPMI-SVD, as diferenças quase somem.
* Hartmann et al. (2017), em PT: GloVe melhor em analogias **semânticas** (48,5 % com 600d), mas pior em POS e similaridade de sentenças.

A lição prática: *escolha pela tarefa e ajuste os hiperparâmetros*, não pela "família" do algoritmo.
"""),
    code('''
# Curva de treino do GloVe 100d (Machado) gravada pelo script de treino
import json
meta = json.loads((PATHS.models / "machado_glove100.json").read_text())
training_curve(meta["history"], title=f"GloVe 100d (Machado): perda por época — {meta['seconds']}s de treino").show()
'''),
    md(f"""
## 6. Síntese e pontos para a apresentação

1. **Razões de probabilidade** como objeto de modelagem: o insight do artigo, independente do algoritmo.
2. O objetivo é um **mínimos quadrados ponderado** sobre $\\log X_{{ij}}$: explícito, esparso, paralelizável; AdaGrad.
3. $W + \\tilde W$: GloVe soma os dois conjuntos de vetores; word2vec descarta $W_{{out}}$.
4. Em PT (Hartmann), GloVe ≠ "melhor" — depende da tarefa.

### Perguntas para discussão
* Por que $\\log X_{{ij}}$ e não $X_{{ij}}$? (lei de Zipf; o log torna a relação aproximadamente linear)
* O que acontece com $f$ se $x_{{\\max}}$ for muito baixo? (todos os pares pesam igual → volta ao LSA)

## Referências
{refs("glove", "baroni", "levy", "hartmann")}
"""),
]
