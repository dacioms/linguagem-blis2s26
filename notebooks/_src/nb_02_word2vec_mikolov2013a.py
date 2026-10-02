from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "02 · word2vec (Mikolov et al., 2013a): CBOW, Skip-gram e a aritmética de vetores"

CELLS = [
    md(f"""
# 02 · word2vec: CBOW, Skip-gram e a aritmética de vetores
{HEADER}

**Artigo-base (leitura obrigatória):** Mikolov, Chen, Corrado & Dean (2013a), *Efficient Estimation of Word Representations in Vector Space*.

## O que o artigo propôs
Até 2013, os melhores vetores vinham de **modelos neurais de linguagem** (Bengio et al., 2003; Collobert & Weston, 2008): redes com camada oculta não linear, caras de treinar (dias para poucas centenas de milhões de palavras).
Mikolov et al. fazem uma pergunta de engenharia: *quanto da qualidade dos vetores sobrevive se tirarmos a camada oculta?* A resposta são duas arquiteturas **log-lineares**:

* **CBOW** (*continuous bag-of-words*): prevê a palavra central a partir da **média** dos vetores do contexto (a ordem é ignorada — daí "bag").
* **Skip-gram**: prevê cada palavra do contexto a partir da palavra central.

Complexidade por exemplo de treino (Tabela do artigo, §3): CBOW $O(N\\cdot D + D\\cdot\\log_2 V)$ e Skip-gram $O(C\\cdot(D + D\\cdot\\log_2 V))$, contra $O(N\\cdot D + N\\cdot D\\cdot H + H\\cdot\\log_2 V)$ do NNLM com camada oculta $H$. O termo $\\log_2 V$ vem do **softmax hierárquico** (notebook 03).

Além da eficiência, o artigo cunhou o teste de **analogias**: `vec("king") − vec("man") + vec("woman") ≈ vec("queen")`, com 19.544 questões em 14 categorias (5 semânticas, 9 sintáticas) — o `questions-words.txt`.
"""),
    SETUP,
    md("""
## 1. Premissas: o modelo log-linear e o softmax

Para a palavra central $w_I$ e uma palavra de contexto $w_O$, o Skip-gram define

$$p(w_O \\mid w_I) = \\frac{\\exp(v'_{w_O}{}^{\\top} v_{w_I})}{\\sum_{w=1}^{V} \\exp(v'_{w}{}^{\\top} v_{w_I})}$$

com **dois** vetores por palavra: $v_w$ ("entrada", matriz $W_{in}$) e $v'_w$ ("saída", $W_{out}$). O objetivo é maximizar a log-verossimilhança média dos contextos numa janela de tamanho $c$:

$$\\frac{1}{T}\\sum_{t=1}^{T}\\sum_{-c\\le j\\le c,\\ j\\ne 0} \\log p(w_{t+j}\\mid w_t).$$

"Log-linear" = o logaritmo da probabilidade é linear nos parâmetros ($v'^{\\top}v$): não há não linearidade entre entrada e saída. É isso que torna o modelo rápido **e** que explica a aritmética de vetores (Mikolov 2013b, §5): somar vetores ≈ multiplicar distribuições de contexto.

O denominador custa $O(V)$ por exemplo — inviável para $V\\sim 10^5$–$10^6$. O artigo usa softmax hierárquico; o 2013b introduz a amostragem negativa (ambos no notebook 03). Aqui usamos amostragem negativa por padrão.
"""),
    md("""
## 2. Implementação *from scratch* (numpy)

A classe `replang.models.word2vec_np.Word2Vec` espelha os parâmetros do `word2vec.c`: `sg` (1 = Skip-gram, 0 = CBOW), `window`, `negative`, `hs`, `sample`, `alpha`.
Vamos abrir o capô: vocabulário, pares (centro, contexto), passo de gradiente.
"""),
    code('''
import inspect
from replang.models.word2vec_np import Word2Vec, Vocab
from replang.data.corpora import toy_corpus
toy = toy_corpus() * 20   # repetimos para ter estatística suficiente
model = Word2Vec(dim=20, window=2, sg=1, negative=5, sample=0, epochs=1, seed=1).build_vocab(toy)
print("|V| =", len(model.vocab), "| 10 mais frequentes:", model.vocab.words[:10])
enc = model._encode_corpus(toy[:1])
pares = list(model._pairs(enc))
print("sentença:", toy[0]); print("pares (centro → contexto), janela efetiva sorteada em [1, window]:")
print([(model.vocab.words[a], model.vocab.words[b]) for a, b in pares][:12])
'''),
    code('''
# O passo de gradiente da amostragem negativa (explicado em detalhe no notebook 03)
print(inspect.getsource(Word2Vec._step_neg))
'''),
    md("""
### 2.1 Treinando Skip-gram e CBOW no toy corpus

Observe a **perda** cair e os vizinhos de `rei` se organizarem. A perda da amostragem negativa é
$-\\log\\sigma(v'_{w_O}{}^{\\top}v_{w_I}) - \\sum_{i=1}^{k}\\log\\sigma(-v'_{w_i}{}^{\\top}v_{w_I})$; com $k=5$ negativos o valor inicial é $\\approx 6\\log 2 \\approx 4{,}16$.
"""),
    code('''
from replang.viz import training_curve, scatter_words
sg = Word2Vec(dim=20, window=2, sg=1, negative=5, sample=0, epochs=25, batch_size=64, seed=1).train(toy)
cb = Word2Vec(dim=20, window=2, sg=0, negative=5, sample=0, epochs=25, batch_size=64, seed=1).train(toy)
hist = pd.concat([pd.DataFrame(sg.history).assign(modelo="Skip-gram"), pd.DataFrame(cb.history).assign(modelo="CBOW")])
import plotly.express as px
px.line(hist, x="epoch", y="loss", color="modelo", markers=True, template="plotly_white", title="Perda média por época (toy corpus)").show()
wv_sg, wv_cb = sg.to_wordvectors("SG toy"), cb.to_wordvectors("CBOW toy")
for w in ["rei", "gato", "paris"]:
    print(f"{w:<6} SG: {[x for x, _ in wv_sg.most_similar(w, topn=4)]}   CBOW: {[x for x, _ in wv_cb.most_similar(w, topn=4)]}")
'''),
    code('''
scatter_words(wv_sg, {"realeza": ["rei", "rainha", "príncipe", "princesa"], "pessoas": ["homem", "mulher", "menino", "menina"],
                      "animais": ["gato", "cachorro"], "frutas": ["maçã", "banana"], "lugares": ["paris", "lisboa", "roma", "madri", "frança", "portugal", "itália", "espanha"]},
              title="Skip-gram (20d) no toy corpus — PCA", arrows=[("rei", "rainha"), ("homem", "mulher"), ("príncipe", "princesa"), ("menino", "menina")])
'''),
    md("""
As setas `rei→rainha`, `homem→mulher`, `príncipe→princesa`, `menino→menina` são aproximadamente **paralelas**: o "deslocamento de gênero" é quase o mesmo vetor. Esta é a origem geométrica tanto das analogias (Mikolov) quanto da **direção de gênero** de Bolukbasi (notebook 08).
"""),
    md("""
## 3. Entrada × saída: por que dois vetores por palavra?

$W_{in}$ representa a palavra **como alvo**; $W_{out}$ a representa **como contexto**. O produto $v'_c{}^\\top v_w$ mede "quão bem $c$ prediz $w$".
Palavras que coocorrem têm $v_w$ parecido com $v'_c$ (relação sintagmática); palavras intercambiáveis têm $v_w$ parecido com $v_{w'}$ (relação paradigmática). O `word2vec.c` descarta $W_{out}$; GloVe soma as duas (notebook 04).
"""),
    code('''
w = "rei"
print("vizinhos em W_in  (paradigmáticos):", [x for x, _ in sg.to_wordvectors(combine="in").most_similar(w, topn=5)])
print("vizinhos em W_out (contextos):      ", [x for x, _ in sg.to_wordvectors(combine="out").most_similar(w, topn=5)])
vin, vout = sg.W_in, sg.W_out
scores = vout @ vin[sg.vocab.index[w]]
p = np.exp(scores - scores.max()); p /= p.sum()
print("p(contexto | rei) via softmax completo:", [(sg.vocab.words[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:6]])
'''),
    md("""
## 4. Escalando: Machado de Assis (2,4 M tokens)

A implementação numpy é didática, não rápida. Para o corpus completo usamos o **gensim** (`replang.models.trainers`), que implementa o mesmo algoritmo em C/Cython. Os modelos foram treinados por `uv run replang train` e ficam em `data/models`:
`machado_sg100` (Skip-gram, NEG-10), `machado_cbow100`, `machado_sg100_hs` (softmax hierárquico), `machado_sg50_nosub` (sem subamostragem) e `machado_ft100` (fastText).
"""),
    code('''
from replang.data.embeddings import load_local_model
from replang.models.trainers import list_models
pd.DataFrame(list_models())[["name", "algo", "dim", "epochs", "vocab", "seconds"]] if list_models() else print("rode: uv run replang train")
'''),
    code('''
sg_m = load_local_model("machado_sg100"); cb_m = load_local_model("machado_cbow100")
for w in ["capitu", "amor", "escravo", "dinheiro", "triste", "rua"]:
    print(f"{w:<9} SG:   {[x for x, _ in sg_m.most_similar(w, topn=6)]}")
    print(f"{'':<9} CBOW: {[x for x, _ in cb_m.most_similar(w, topn=6)]}")
'''),
    md("""
## 5. Analogias: o teste de Mikolov et al.

> *"We find that... vector("King") − vector("Man") + vector("Woman") results in a vector that is closest to vector("Queen")."*

O método **3CosAdd** busca $\\arg\\max_y \\cos(y,\\ b - a + c)$ excluindo $a, b, c$. O conjunto `questions-words.txt` tem 14 categorias; aqui usamos a versão **PT-BR** (Rodrigues et al., 2016) adotada por Hartmann et al. (2017), e a inglesa com o GloVe 50d para comparação.
"""),
    code('''
from replang.data.analogies import load_analogies
pt_an = load_analogies("pt-br"); en_an = load_analogies("en")
print(f"PT-BR: {pt_an.n:,} questões em {len(pt_an.categories)} categorias | EN: {en_an.n:,} questões")
pt_an.summary()
'''),
    code('''
from replang.data.embeddings import load_ptwiki, load_glove_en
pt = load_ptwiki(); en = load_glove_en()
exemplos = [("homem", "rei", "mulher"), ("brasil", "brasília", "frança"), ("bom", "melhor", "ruim"), ("falar", "falando", "comer"), ("rápido", "rapidamente", "feliz"), ("um", "dois", "primeiro")]
for a, b, c in exemplos:
    r = pt.analogy(a, b, c, topn=3) if pt.has(a, b, c) else "OOV"
    print(f"{a} : {b} :: {c} : ?  →  {r}")
'''),
    code('''
from replang.eval.analogies import evaluate_analogies
import time
t = time.time()
res_pt = evaluate_analogies(pt, pt_an, restrict=30000, max_per_category=None if not FAST else 300)
print(f"{time.time()-t:.0f}s")
res_pt.style.format({"acurácia": "{:.1%}"}).background_gradient(subset=["acurácia"], cmap="Blues")
'''),
    code('''
fig = px.bar(res_pt[~res_pt.categoria.str.startswith("TOTAL")], x="categoria", y="acurácia", color="tipo", template="plotly_white",
             title="Wikipedia2Vec PT 100d — acurácia por categoria (3CosAdd, busca nas 30k palavras mais frequentes)")
fig.update_layout(xaxis_tickangle=-35); fig.show()
'''),
    md("""
### 5.1 O que entra na conta: cobertura, restrição de vocabulário e método

* **Cobertura**: questões com qualquer palavra fora do vocabulário são descartadas (como no `compute-accuracy.c` original, que também restringe a busca às 30 mil palavras mais frequentes). Reportar só a acurácia sem a cobertura é enganoso.
* **3CosMul** (Levy & Goldberg, 2014) costuma dar alguns pontos a mais.
* Modelos treinados em **Machado de Assis** cobrem pouco das categorias (capitais do mundo, moedas) — um corpus literário do século XIX não "sabe" que *Bancoque* é capital da Tailândia. Isso ilustra a dependência do corpus.
"""),
    code('''
rows = []
for name, wv in [("Wikipedia2Vec PT 100d", pt), ("Machado SG 100d", sg_m), ("Machado CBOW 100d", cb_m)]:
    for method in ["add", "mul"]:
        r = evaluate_analogies(wv, pt_an, method=method, restrict=30000, max_per_category=300).set_index("categoria")
        rows.append({"modelo": name, "método": method, "sintática": r.loc["TOTAL sintática", "acurácia"], "semântica": r.loc["TOTAL semântica", "acurácia"],
                     "total": r.loc["TOTAL", "acurácia"], "cobertura": r.loc["TOTAL", "cobertas"] / r.loc["TOTAL", "n"]})
pd.DataFrame(rows).style.format({"sintática": "{:.1%}", "semântica": "{:.1%}", "total": "{:.1%}", "cobertura": "{:.1%}"})
'''),
    md("""
## 6. Dimensão e tamanho do corpus (Tabela 2 do artigo)

O artigo mostra que aumentar **só** a dimensão ou **só** o corpus satura; é preciso crescer os dois. Reproduzimos a tendência treinando Skip-gram (gensim) em fatias do corpus Machado com dimensões diferentes e medindo analogias **sintáticas** (as semânticas quase não têm cobertura neste corpus).

> Os números absolutos serão **baixos**: um corpus literário do século XIX (2,4 M tokens) é duas a três ordens de grandeza menor que os dos artigos e o teste é "da Wikipédia" (capitais, nacionalidades). O que interessa é a **tendência**: 4× mais tokens → várias vezes mais acertos; dimensão a mais sem dados a mais não ajuda (Tabela 2 do artigo).
"""),
    code('''
from gensim.models import Word2Vec as GWord2Vec
from replang.data.corpora import load_machado_sentences
sents = load_machado_sentences()
grid = [(0.25, 50), (0.25, 100), (1.0, 50), (1.0, 100)] if not FAST else [(0.25, 50), (1.0, 50)]
rows = []
for frac, dim in grid:
    sub = sents[: int(len(sents) * frac)]
    t = time.time()
    m = GWord2Vec(sub, vector_size=dim, window=5, sg=1, negative=10, min_count=5, epochs=3, sample=1e-4, workers=settings.workers, seed=1)
    wv = WordVectors.from_gensim(m.wv)
    r = evaluate_analogies(wv, pt_an, restrict=None).set_index("categoria")
    rows.append({"fração do corpus": frac, "tokens": sum(map(len, sub)), "dim": dim, "sintática": r.loc["TOTAL sintática", "acurácia"],
                 "cobertas": int(r.loc["TOTAL", "cobertas"]), "segundos": round(time.time() - t)})
pd.DataFrame(rows)
'''),
    md(f"""
## 7. O que o artigo encontrou (resumo dos resultados)

| Modelo (640d, 320 M tokens) | Semântica | Sintática |
|---|---|---|
| RNNLM | 9 % | 36 % |
| NNLM | 23 % | 53 % |
| CBOW | 24 % | 64 % |
| Skip-gram | **55 %** | 59 % |

Com mais dados e dimensões (Skip-gram 1000d, 6 B tokens, DistBelief): 66 % / 65 %. Skip-gram é melhor em semântica; CBOW é mais rápido e competitivo em sintaxe. O treino de um dia em um cluster substituiu semanas do NNLM.

### Pontos para a apresentação
1. A inovação não é "rede neural": é **tirar** a camada oculta e aceitar um modelo mais simples que vê muito mais dados.
2. Dois vetores por palavra; descartar $W_{{out}}$ é uma escolha, não uma necessidade.
3. A analogia por aritmética é uma **propriedade emergente** do objetivo log-linear — e é exatamente o mecanismo que carrega estereótipos (Bolukbasi).
4. Analogias são um *proxy* barato; o notebook 07 (Hartmann) discute seus limites.

### Perguntas para discussão
* Por que o CBOW "dilui" a semântica? (média do contexto apaga a ordem e as palavras raras)
* O que aconteceria com a analogia `homem : rei :: mulher : ?` se *rainha* fosse uma palavra raríssima no corpus?

## Referências
{refs("mikolov13a", "mikolov13b", "levy", "rodrigues", "hartmann")}
"""),
]
