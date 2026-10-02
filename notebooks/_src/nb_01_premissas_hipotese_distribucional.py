from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "01 · Premissas: de símbolos a vetores (hipótese distribucional, coocorrência, PMI, SVD)"

CELLS = [
    md(f"""
# 01 · Premissas: de símbolos a vetores
{HEADER}

## Por que este notebook vem antes dos artigos?
Todos os artigos da disciplina partem de três ideias que raramente são explicadas neles:

1. **Hipótese distribucional** (Harris, 1954; Firth, 1957): *o significado de uma palavra está nos contextos em que ela ocorre* — "you shall know a word by the company it keeps".
2. **Espaço vetorial**: representar cada palavra como um ponto em $\\mathbb{{R}}^d$ e medir "proximidade de significado" por uma **distância/ângulo** (cosseno).
3. **Compressão**: de uma representação esparsa e enorme (|V| dimensões) para uma densa e pequena (50–1000 dimensões) que **generaliza**.

Vamos percorrer o caminho histórico: *one-hot* → *bag-of-words* → matriz de coocorrência → PMI → SVD (LSA/HAL). No final, veremos por que o word2vec (notebook 02) é "a mesma coisa, aprendida por gradiente" (Levy & Goldberg, 2014).
"""),
    SETUP,
    md("""
## 1. One-hot: a representação sem significado

Com um vocabulário de tamanho $|V|$, a palavra $w_i$ vira o vetor $e_i$ (1 na posição $i$, 0 no resto).
Problemas: (a) dimensão $|V|$ (centenas de milhares); (b) **todos os pares são ortogonais** — `rei` está tão longe de `rainha` quanto de `banana`.
"""),
    code('''
from replang.data.corpora import toy_corpus
from replang.models.cooccurrence import build_vocab, one_hot
toy = toy_corpus()
words, index, freq = build_vocab(toy)
print(f"{len(toy)} sentenças, |V| = {len(words)}")
e_rei, e_rainha, e_banana = (one_hot(words, w) for w in ["rei", "rainha", "banana"])
cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
print("cos(rei, rainha) =", cos(e_rei, e_rainha), "| cos(rei, banana) =", cos(e_rei, e_banana))
'''),
    md("""
## 2. Bag-of-words e TF-IDF: vetores para *documentos*

O primeiro uso de vetores em PLN foi para documentos (recuperação de informação, Salton, anos 1970): um documento é a contagem de suas palavras.
Hartmann (2016) usa exatamente o **cosseno entre TF-IDF** como *baseline* para similaridade de sentenças no ASSIN (notebook 07).
"""),
    code('''
from replang.models.cooccurrence import bag_of_words, tfidf
bow = bag_of_words(toy, words)
tf = tfidf(bow)
docs = [" ".join(s) for s in toy]
sim = tf @ tf.T / (np.linalg.norm(tf, axis=1)[:, None] * np.linalg.norm(tf, axis=1)[None, :] + 1e-9)
i = 0
print("Documento:", docs[i])
for j in np.argsort(-sim[i])[1:4]:
    print(f"  {sim[i, j]:.2f}  {docs[j]}")
'''),
    md("""
## 3. Matriz de coocorrência palavra × contexto

Agora o objeto central: para cada palavra-alvo $w$ contamos quantas vezes cada palavra de contexto $c$ aparece numa **janela** de $\\pm k$ posições.
A linha $M_{w\\cdot}$ é a "assinatura distribucional" de $w$. Duas palavras com linhas parecidas aparecem em contextos parecidos.

A ponderação pela distância (`harmonic` = $1/d$, usada pelo GloVe; `linear`, usada implicitamente pelo word2vec ao sortear a janela efetiva) é um detalhe que importa nos artigos.
"""),
    code('''
from replang.models.cooccurrence import cooccurrence_matrix, CountModel
from replang.viz import cooccurrence_heatmap
M = cooccurrence_matrix(toy, index, window=2, weighting="flat")
sel = ["rei", "rainha", "príncipe", "princesa", "homem", "mulher", "gato", "cachorro", "maçã", "banana", "paris", "lisboa"]
df = pd.DataFrame(M.toarray(), index=words, columns=words).loc[sel, sel]
cooccurrence_heatmap(df, "Coocorrências (janela ±2) — subconjunto do toy corpus")
'''),
    code('''
# As linhas completas (todas as colunas) já aproximam palavras do mesmo "tipo":
Mrow = M.toarray()
def cos_row(a, b):
    return cos(Mrow[index[a]], Mrow[index[b]])
for a, b in [("rei", "rainha"), ("rei", "banana"), ("gato", "cachorro"), ("maçã", "banana"), ("paris", "lisboa"), ("paris", "gato")]:
    print(f"cos({a}, {b}) = {cos_row(a, b):.2f}")
'''),
    md("""
## 4. PMI e PPMI: o que é "coocorrer mais que o acaso"?

Contagens brutas são dominadas por palavras frequentes (*o*, *a*, *e* coocorrem com tudo). A **informação mútua pontual** compara a coocorrência observada com a esperada se $w$ e $c$ fossem independentes:

$$\\mathrm{PMI}(w,c) = \\log \\frac{P(w,c)}{P(w)\\,P(c)}, \\qquad \\mathrm{PPMI}(w,c)=\\max(0,\\ \\mathrm{PMI}(w,c)).$$

Duas variantes aparecem "escondidas" nos artigos:
* **suavização** $P(c)^{\\alpha}$ com $\\alpha=0{,}75$ (Levy, Goldberg & Dagan, 2015) — o mesmo expoente $3/4$ da distribuição de ruído do word2vec (Mikolov 2013b, §2.2);
* **deslocamento** $-\\log k$: Levy & Goldberg (2014) provaram que o Skip-gram com $k$ amostras negativas fatora implicitamente a matriz $\\mathrm{PMI}(w,c) - \\log k$.
"""),
    code('''
from replang.models.cooccurrence import ppmi
P = ppmi(M, alpha=1.0)
dfp = pd.DataFrame(P.toarray(), index=words, columns=words).loc[sel, sel]
cooccurrence_heatmap(dfp, "PPMI (α = 1)")
'''),
    code('''
# Efeito do PPMI sobre palavras funcionais: 'o' e 'a' deixam de dominar as linhas
top_counts = pd.Series(Mrow[index["rei"]], index=words).sort_values(ascending=False).head(6)
top_ppmi = pd.Series(P.toarray()[index["rei"]], index=words).sort_values(ascending=False).head(6)
pd.DataFrame({"contagem": top_counts.round(0), "PPMI": top_ppmi.round(2)})
'''),
    md("""
## 5. SVD: comprimir para generalizar (LSA / HAL)

A matriz PPMI é esparsa e tem |V| colunas. A **decomposição em valores singulares** $M \\approx U_k \\Sigma_k V_k^\\top$ projeta cada palavra em $k$ dimensões densas, mantendo as direções de maior variância. É a *Latent Semantic Analysis* (Deerwester et al., 1990).
Usamos $U_k \\Sigma_k^{1/2}$ como vetores — escolha empiricamente melhor (Levy et al., 2015) do que $U_k\\Sigma_k$.

A classe `CountModel` encadeia contagem → PPMI → SVD e devolve um `WordVectors`, a mesma interface que usaremos para word2vec, GloVe e fastText.
"""),
    code('''
from replang.viz import scatter_words
cm = CountModel(window=2, dim=6, alpha=0.75).fit(toy)
wv_count = cm.to_wordvectors("contagem→PPMI→SVD (toy)")
print("valores singulares:", np.round(cm.sigma_, 2))
for w in ["rei", "gato", "maçã", "paris"]:
    print(f"{w:<8}", [f"{x} ({s:.2f})" for x, s in wv_count.most_similar(w, topn=3)])
scatter_words(wv_count, {"realeza": ["rei", "rainha", "príncipe", "princesa"], "pessoas": ["homem", "mulher", "menino", "menina"],
                         "animais": ["gato", "cachorro"], "frutas": ["maçã", "banana"], "lugares": ["paris", "lisboa", "roma", "madri", "frança", "portugal", "itália", "espanha"]},
              title="Vetores densos (6d) do toy corpus, projetados por PCA")
'''),
    md("""
## 6. O mesmo pipeline num corpus real (Machado de Assis)

Com 2,4 M tokens, a matriz PPMI tem dezenas de milhares de linhas; a SVD truncada esparsa (`scipy.sparse.linalg.svds`) resolve em segundos para $k = 100$.
Compare os vizinhos com os do Wikipedia2Vec (treinado em 200 M+ tokens com Skip-gram).
"""),
    code('''
import time
from replang.data.corpora import load_machado_sentences
sents = load_machado_sentences(40000 if FAST else None)
t = time.time()
cm_m = CountModel(window=4, dim=100, min_count=10, alpha=0.75, weighting="harmonic", max_vocab=15000).fit(sents)
wv_m = cm_m.to_wordvectors("PPMI-SVD 100d (Machado)")
print(f"{wv_m} em {time.time()-t:.0f}s; nnz(PPMI) = {cm_m.ppmi_.nnz:,}")
from replang.data.embeddings import load_ptwiki
pt = load_ptwiki()
for w in ["capitu", "amor", "dinheiro", "rua", "triste"]:
    print(f"{w:<9} PPMI-SVD: {[x for x, _ in wv_m.most_similar(w, topn=5)]}")
    print(f"{'':<9} Wiki2Vec: {[x for x, _ in pt.most_similar(w, topn=5)] if w in pt else '(fora do vocabulário da Wikipédia: personagem de Machado!)'}")
'''),
    md("""
## 7. Cosseno, produto interno e normalização

Com vetores de norma 1, $\\cos(u,v) = u \\cdot v$. Bolukbasi et al. (2016, §3) assumem exatamente isso ("as is common, similarity is measured by inner product; words are normalized to unit length").
A norma de um vetor word2vec cresce com a frequência da palavra — por isso normalizar é importante antes de comparar.
"""),
    code('''
norms = np.linalg.norm(pt.vectors, axis=1)
df = pd.DataFrame({"palavra": pt.words, "posição (freq. decrescente)": np.arange(len(pt)), "norma": norms})
import plotly.express as px
fig = px.scatter(df.iloc[::25], x="posição (freq. decrescente)", y="norma", hover_name="palavra", opacity=0.5,
                 title="Norma do vetor × ranking de frequência (Wikipedia2Vec PT)", log_x=True, template="plotly_white")
fig.show()
u, v = pt["rei"], pt["rainha"]
print("produto interno bruto:", float(u @ v), "| cosseno:", pt.similarity("rei", "rainha"))
'''),
    md(f"""
## 8. Síntese: o que levar para o notebook 02

* **Significado ≈ distribuição de contextos.** Toda representação distribucional, de LSA a BERT, formaliza isso de um jeito.
* **Contagem + reponderação (PPMI) + compressão (SVD)** já produz vizinhos razoáveis. Baroni et al. (2014) mostraram que modelos *preditivos* (word2vec) ganham na maioria dos benchmarks, mas Levy et al. (2015) mostraram que boa parte da diferença são **hiperparâmetros** (janela, suavização, deslocamento, normalização).
* O word2vec aprende, por gradiente estocástico e sem construir a matriz, uma fatoração parecida — com a vantagem de escalar para bilhões de tokens em um dia (Mikolov 2013b, §1).

### Perguntas para discussão
1. Que informação a janela ±2 captura que a janela ±10 não captura, e vice-versa? (sintagmático × paradigmático; tópico × função)
2. O PPMI zera associações *negativas*. O que se perde? (Levy et al. discutem o "shifted PPMI".)
3. Se a hipótese distribucional é a premissa, o que acontece com palavras que **nunca** aparecem no corpus? (gancho para fastText, notebook 05)

## Referências
{refs("harris", "levy", "baroni", "mikolov13a", "glove")}
"""),
]
