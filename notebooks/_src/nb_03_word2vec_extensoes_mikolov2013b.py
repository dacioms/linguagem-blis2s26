from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "03 · Extensões do Skip-gram (Mikolov et al., 2013b): softmax hierárquico, amostragem negativa, subamostragem, frases e composicionalidade"

CELLS = [
    md(f"""
# 03 · Extensões do Skip-gram: softmax hierárquico, amostragem negativa, subamostragem, frases e composicionalidade
{HEADER}

**Artigo-base (leitura complementar):** Mikolov, Sutskever, Chen, Corrado & Dean (2013b), *Distributed Representations of Words and Phrases and their Compositionality* (NIPS).

O 2013a definiu as arquiteturas; o 2013b resolveu os problemas práticos que fazem o word2vec funcionar em escala:

| Problema | Solução | Seção do artigo |
|---|---|---|
| softmax custa $O(V)$ | **softmax hierárquico** com árvore de Huffman ($O(\\log_2 V)$) | §2.1 |
| ... ou | **amostragem negativa** (NEG): $k$ "ruídos" por exemplo positivo | §2.2 |
| palavras frequentes dominam e informam pouco | **subamostragem** com $P(w_i) = 1 - \\sqrt{{t/f(w_i)}}$ | §2.3 |
| "Boston Globe" não é Boston + Globe | **detecção de frases** por escore de bigramas | §4 |
| o que a soma de vetores significa? | **composicionalidade aditiva** | §5 |

Neste notebook implementamos cada uma em numpy, medimos o efeito no corpus Machado de Assis e reproduzimos as tabelas do artigo em miniatura.
"""),
    SETUP,
    md("""
## 1. O gargalo: softmax completo

Com $V$ palavras, calcular $p(w_O\\mid w_I)$ exige $V$ produtos internos **por par de treino**. Para $V = 10^5$ e bilhões de pares, inviável.
Duas saídas: aproximar a **estrutura** do softmax (árvore) ou trocar o **objetivo** (classificação binária contra ruído).
"""),
    code('''
from replang.data.corpora import load_machado_sentences, toy_corpus
from replang.models.word2vec_np import Word2Vec, HuffmanTree, Vocab
sents = load_machado_sentences()
vocab = Vocab(sents, min_count=5)
V = len(vocab)
print(f"|V| = {V:,} (min_count=5) · tokens = {int(vocab.total):,}")
print(f"produtos internos por par: softmax completo = {V:,} | hierárquico ≈ log2(V) = {np.log2(V):.1f} | NEG-5 = 6")
'''),
    md("""
## 2. Softmax hierárquico com árvore de Huffman (§2.1)

Cada palavra é uma **folha** de uma árvore binária; cada nó interno $n$ tem um vetor $v'_n$. A probabilidade da palavra é o produto, ao longo do caminho da raiz à folha, da probabilidade de virar para o lado certo:

$$p(w\\mid w_I) = \\prod_{j=1}^{L(w)-1} \\sigma\\big([\\![n(w,j+1) = \\mathrm{ch}(n(w,j))]\\!]\\cdot v'_{n(w,j)}{}^{\\top} v_{w_I}\\big)$$

onde $[\\![x]\\!]$ vale $+1$ ou $-1$. Como $\\sigma(x)+\\sigma(-x)=1$, as probabilidades somam 1 automaticamente. A **árvore de Huffman** dá códigos curtos às palavras frequentes → menos nós a atualizar em média.
"""),
    code('''
tree = HuffmanTree(vocab.counts)
print(f"nós internos: {tree.n_inner:,} = |V| − 1 · profundidade máxima: {tree.max_len}")
print(f"comprimento médio do código (ponderado pela frequência): {tree.average_code_length(vocab.counts):.2f} bits  vs  log2|V| = {np.log2(V):.2f}")
amostra = [vocab.words[i] for i in [0, 1, 2, 50, 500, 5000, V - 1]]
pd.DataFrame({"palavra": amostra, "freq": [int(vocab.counts[vocab.index[w]]) for w in amostra],
              "código Huffman": ["".join(map(str, tree.codes[vocab.index[w]])) for w in amostra]})
'''),
    md("""
A implementação vetorizada (`Word2Vec._step_hs`) usa os caminhos preenchidos (`path_arr`, `code_arr`, `mask_arr`) para atualizar só os nós de cada caminho:
"""),
    code('''
import inspect
print(inspect.getsource(Word2Vec._step_hs))
'''),
    md("""
## 3. Amostragem negativa (§2.2)

Em vez de modelar $p(w_O\\mid w_I)$, treinamos um classificador logístico que distingue o par real $(w_I, w_O)$ de $k$ pares com "ruído" $w_i \\sim P_n(w)$:

$$\\log\\sigma(v'_{w_O}{}^{\\top}v_{w_I}) + \\sum_{i=1}^{k}\\mathbb{E}_{w_i\\sim P_n(w)}\\big[\\log\\sigma(-v'_{w_i}{}^{\\top}v_{w_I})\\big] \\qquad (4)$$

Derivada da *Noise Contrastive Estimation* (Gutmann & Hyvärinen, 2012), mas simplificada: não precisa das probabilidades numéricas do ruído. A escolha empírica crucial é $P_n(w) \\propto U(w)^{3/4}$ — achata a distribuição unigrama, dando mais chance a palavras raras como negativos. O artigo recomenda $k = 5$–20 para corpora pequenos e 2–5 para grandes.
"""),
    code('''
import plotly.express as px
f = vocab.counts / vocab.total
for power, label in [(1.0, "U(w)"), (0.75, "U(w)^0.75"), (0.0, "uniforme")]:
    p = vocab.counts ** power; p /= p.sum()
    print(f"{label:<10} P(mais frequente)={p[0]:.3f}  P(100ª)={p[100]:.5f}  P(última)={p[-1]:.2e}  entropia={-(p*np.log2(p)).sum():.1f} bits")
df = pd.DataFrame({"ranking": np.arange(1, 2001), "U(w)": f[:2000], "U(w)^0.75": (vocab.counts**0.75 / (vocab.counts**0.75).sum())[:2000]})
px.line(df.melt(id_vars="ranking"), x="ranking", y="value", color="variable", log_x=True, log_y=True, template="plotly_white",
        title="Distribuição de ruído: unigrama × unigrama^0.75 (Machado)").show()
'''),
    md("""
### 3.1 HS × NEG: tempo e qualidade (Tabela 1 do artigo, em miniatura)

O artigo reporta (300d, 1 B tokens, analogias): NEG-5 59 %, NEG-15 61 %, HS-Huffman 47 %, NCE-5 53 %; com subamostragem $10^{-5}$: 60 %, 61 %, 55 %.
Reproduzimos com o gensim no corpus Machado (modelos já treinados por `replang train`), avaliando em analogias PT-BR **sintáticas** (as únicas com cobertura razoável) e em vizinhos.
"""),
    code('''
from replang.data.embeddings import load_local_model
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
from replang.models.trainers import list_models
an = load_analogies("pt-br")
meta = {m["name"]: m for m in list_models()}
rows = []
for name, label in [("machado_sg100", "Skip-gram NEG-10 (sub 1e-4)"), ("machado_sg100_hs", "Skip-gram HS (sub 1e-4)"), ("machado_sg50_nosub", "Skip-gram NEG-5 50d (sem subamostragem)"), ("machado_cbow100", "CBOW NEG-10 (sub 1e-4)")]:
    wv = load_local_model(name)
    r = evaluate_analogies(wv, an, restrict=None).set_index("categoria")
    rows.append({"modelo": label, "dim": wv.dim, "segundos de treino": meta.get(name, {}).get("seconds"), "sintática": r.loc["TOTAL sintática", "acurácia"],
                 "semântica": r.loc["TOTAL semântica", "acurácia"], "cobertas": int(r.loc["TOTAL", "cobertas"])})
pd.DataFrame(rows).style.format({"sintática": "{:.1%}", "semântica": "{:.1%}"})
'''),
    code('''
sg, hs = load_local_model("machado_sg100"), load_local_model("machado_sg100_hs")
for w in ["capitu", "escravo", "dinheiro", "ontem"]:
    print(f"{w:<8} NEG: {[x for x, _ in sg.most_similar(w, topn=6)]}")
    print(f"{'':<8} HS:  {[x for x, _ in hs.most_similar(w, topn=6)]}")
'''),
    md("""
## 4. Subamostragem de palavras frequentes (§2.3)

Cada ocorrência da palavra $w_i$ é **descartada** com probabilidade

$$P(w_i) = 1 - \\sqrt{\\frac{t}{f(w_i)}}$$

com $t \\approx 10^{-5}$. Palavras com $f(w) < t$ nunca são descartadas; *a*, *de*, *que* (frequência ~5 %) são descartadas em ~98 % das vezes. Efeitos: treino 2–10× mais rápido, janelas efetivas maiores (as palavras funcionais somem e as de conteúdo se aproximam) e vetores melhores para palavras raras.
"""),
    code('''
for t in [1e-3, 1e-4, 1e-5]:
    keep = vocab.keep_prob(t)
    kept_tokens = (keep * vocab.counts).sum() / vocab.total
    print(f"t={t:.0e}: mantém {kept_tokens:.0%} dos tokens | P(manter 'de')={keep[vocab.index['de']]:.3f} | P(manter 'capitu')={keep[vocab.index['capitu']]:.3f}")
top = vocab.words[:15]
pd.DataFrame({"palavra": top, "f(w)": [f"{f[i]:.2%}" for i in range(15)], "P(descartar) t=1e-4": [f"{1-vocab.keep_prob(1e-4)[i]:.0%}" for i in range(15)]}).T
'''),
    code('''
# Efeito na janela efetiva: a mesma sentença antes e depois da subamostragem
m = Word2Vec(sample=1e-4, seed=0).build_vocab(sents)
s = sents[777]
print("original      :", " ".join(s))
ids = m._encode_corpus([s])[0]
print("subamostrada  :", " ".join(m.vocab.words[i] for i in ids))
'''),
    md("""
## 5. Aprendendo frases (§4)

"Nova York", "Dom Casmurro" e "Rio de Janeiro" não são composições das partes. O artigo detecta bigramas com

$$\\mathrm{score}(w_i, w_j) = \\frac{\\mathrm{count}(w_i w_j) - \\delta}{\\mathrm{count}(w_i)\\times \\mathrm{count}(w_j)}$$

(no código original, multiplicado pelo total de tokens), com $\\delta$ descontando bigramas raros; 2–4 passagens com limiar decrescente formam frases mais longas. Bigramas acima do limiar viram um token único (`rio_de_janeiro`) e o Skip-gram é treinado normalmente.
"""),
    code('''
from replang.models.word2vec_np import learn_phrases
import time
t0 = time.time()
sents_ph, scores = learn_phrases(sents, min_count=10, threshold=80, passes=2)
print(f"{len(scores)} frases em {time.time()-t0:.0f}s. Top 25 por escore:")
pd.Series(scores).sort_values(ascending=False).head(25).round(0).to_frame("escore").T
'''),
    code('''
# Treinamos um Skip-gram rápido (gensim) sobre o corpus com frases e consultamos vizinhos de frases
from gensim.models import Word2Vec as GWord2Vec
mph = GWord2Vec(sents_ph if not FAST else sents_ph[:60000], vector_size=100, window=5, sg=1, negative=10, min_count=5, epochs=3, sample=1e-4, workers=settings.workers, seed=1)
wv_ph = WordVectors.from_gensim(mph.wv, "SG + frases")
frases_vocab = [p for p in sorted(scores, key=scores.get, reverse=True) if p in wv_ph and not p[0].isdigit()]
print(f"{len(frases_vocab)} frases no vocabulário do modelo (min_count=5). Vizinhos de algumas:")
for ph in frases_vocab[:4] + [p for p in ["rua_do_ouvidor", "vossa_excelência", "meu_caro"] if p in wv_ph]:
    print(f"{ph:<26} {[x for x, _ in wv_ph.most_similar(ph, topn=6)]}")
print("Nota: 'rio de janeiro' não vira frase porque 'de' é tão frequente que o escore (count(rio de)/count(rio)count(de)) fica baixo — limitação conhecida do word2phrase.")
'''),
    md("""
## 6. Composicionalidade aditiva (§5)

> *"vec("Russia") + vec("river") is close to vec("Volga River")"*

A explicação do artigo: como os vetores são treinados para prever contextos e a saída é log-linear, **somar dois vetores ≈ multiplicar as duas distribuições de contexto** (uma função AND): palavras prováveis nos dois contextos ficam prováveis na soma. Testamos em PT com o Wikipedia2Vec.
"""),
    code('''
from replang.data.embeddings import load_ptwiki
pt = load_ptwiki()
def soma(*ws, topn=5):
    v = sum(pt.unit[pt.index[w]] for w in ws)
    sims = pt.similarities(v)
    for w in ws: sims[pt.index[w]] = -np.inf
    idx = np.argsort(-sims)[:topn]
    return [pt.words[i] for i in idx]
for ws in [("rússia", "rio"), ("alemanha", "capital"), ("brasil", "moeda"), ("frança", "escritor"), ("japão", "comida"), ("vietnã", "capital")]:
    print(f"{' + '.join(ws):<22} → {soma(*ws)}")
'''),
    md("""
## 7. Como o passo de gradiente da NEG se conecta à fatoração de matrizes

Levy & Goldberg (2014) mostraram que o ótimo de (4) satisfaz $v'_c{}^{\\top}v_w = \\mathrm{PMI}(w,c) - \\log k$: a amostragem negativa **fatora implicitamente** a matriz PMI deslocada do notebook 01. O expoente $3/4$ e a subamostragem são o que o word2vec faz "a mais" — e Levy et al. (2015) mostram que, transplantados para o PPMI-SVD, fecham boa parte da diferença.
"""),
    code('''
# Verificação empírica no toy corpus: produto interno W_in·W_out ≈ PMI − log k ?
from replang.models.cooccurrence import cooccurrence_matrix, build_vocab
toy = toy_corpus() * 30
m = Word2Vec(dim=30, window=2, negative=5, sample=0, epochs=60, batch_size=64, seed=3).train(toy)
words, index, freq = build_vocab(toy)
M = cooccurrence_matrix(toy, index, window=2).toarray()
total = M.sum(); pw = M.sum(1) / total; pc = M.sum(0) / total
with np.errstate(divide="ignore"):
    pmi = np.log(M / total) - np.log(pw)[:, None] - np.log(pc)[None, :]
pairs = [(index[a], index[b]) for a in words for b in words if M[index[a], index[b]] > 3]
x = np.array([pmi[i, j] - np.log(5) for i, j in pairs])
y = np.array([m.W_out[m.vocab.index[words[j]]] @ m.W_in[m.vocab.index[words[i]]] for i, j in pairs])
print(f"correlação entre PMI − log k e v'_c·v_w sobre {len(pairs)} pares: {np.corrcoef(x, y)[0, 1]:.2f}")
px.scatter(x=x, y=y, labels={"x": "PMI(w,c) − log k", "y": "v'_c · v_w (SGNS)"}, template="plotly_white", title="Levy & Goldberg (2014): SGNS ≈ fatoração do PMI deslocado").show()
'''),
    md(f"""
## 8. Síntese

* **HS × NEG**: mesma qualidade de ordem de grandeza; NEG é mais simples e melhor para palavras frequentes, HS ajuda com subamostragem (Tabela 3 do artigo).
* **Subamostragem** é o "hiperparâmetro escondido" mais importante: acelera e melhora palavras raras.
* **Frases** e **composição aditiva** mostram que o espaço vetorial tem estrutura além das palavras — mas também que ele é limitado: `vec(não) + vec(bom)` não dá *ruim*.
* O 2013b é também o artigo que apresenta o **PCA de países e capitais** (Fig. 2), reproduzido na página 5 do app e no notebook 02.

### Perguntas para discussão
1. A amostragem negativa transforma um problema multiclasse em binário. O que se perde? (não é mais um modelo de linguagem calibrado)
2. Por que $U(w)^{{3/4}}$ e não $U(w)$? Que relação tem com o PPMI suavizado?
3. Se somar vetores funciona como AND, por que a negação não funciona?

## Referências
{refs("mikolov13b", "mikolov13a", "levy")}
"""),
]
