from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "11 · De palavras a sentenças e documentos (Le & Mikolov, 2014): Paragraph Vector"

CELLS = [
    md(f"""
# 11 · De palavras a sentenças e documentos: Paragraph Vector / Doc2Vec (Le & Mikolov, 2014)
{HEADER}

**Artigo-base (leitura complementar):** Le & Mikolov (2014), *Distributed Representations of Sentences and Documents* (ICML).

## O problema
Precisamos de vetores para **textos** (sentenças, parágrafos, documentos) — para classificar sentimentos, recuperar documentos, medir similaridade (ASSIN!). As opções anteriores:
* *bag-of-words* / TF-IDF: ignora ordem e semântica (*"bom, não ruim"* ≈ *"ruim, não bom"*);
* **média/soma de vetores de palavras**: usa semântica, ignora ordem; é o baseline de Hartmann (2016) e funciona surpreendentemente bem (notebook 07).

## A proposta
Tratar o **parágrafo como mais uma "palavra"**: um vetor $d$ por documento, treinado junto com os vetores de palavras.
* **PV-DM** (*Distributed Memory*): prevê a próxima palavra a partir da concatenação/média de $d$ **e** dos vetores das palavras do contexto (análogo ao CBOW). $d$ funciona como "memória do tópico".
* **PV-DBOW** (*Distributed Bag of Words*): prevê palavras amostradas do parágrafo só a partir de $d$ (análogo ao Skip-gram, mais barato).
* Na **inferência**, para um documento novo: congelam-se os vetores de palavras e otimiza-se só $d$ por gradiente.

Resultados do artigo: estado da arte no Stanford Sentiment Treebank (erro 12,2 % binário vs. 20,3 % do BoW) e no IMDB (7,42 % vs. 11,1 % do BoW).
"""),
    SETUP,
    md("""
## 1. Documentos: blocos da obra de Machado de Assis

Dividimos os 246 textos em blocos de ~60 sentenças (≈ capítulos), com metadados de **gênero literário** (romance, conto, crônica, poesia, crítica, teatro, tradução, miscelânea). Os modelos `machado_pvdm100` e `machado_pvdbow100` foram treinados por `replang train doc2vec` (gensim).
"""),
    code('''
import json
from gensim.models import Doc2Vec
meta = json.loads((PATHS.models / "machado_docs_meta.json").read_text())
pvdm = Doc2Vec.load(str(PATHS.models / "machado_pvdm100.d2v")); pvdbow = Doc2Vec.load(str(PATHS.models / "machado_pvdbow100.d2v"))
docs_df = pd.DataFrame(meta)
print(f"{len(meta)} blocos · vetores de documento: {pvdm.dv.vectors.shape} · vocabulário: {len(pvdm.wv):,}")
docs_df.genero.value_counts().to_frame("blocos").T
'''),
    md("""
## 2. O que o vetor de um documento captura? Vizinhos entre blocos

Para um bloco de *Dom Casmurro*, quais blocos são mais próximos? Se o vetor captura **tópico/estilo**, esperamos outros blocos do mesmo romance ou de romances parecidos.
"""),
    code('''
def nome(i):
    m = meta[i]; return f"{m['arquivo'].split('/')[-1]} [{m['genero']}] bloco {m['bloco']}"
alvo = next(i for i, m in enumerate(meta) if "marm08" in m["arquivo"] and m["bloco"] == 5)   # Dom Casmurro
print("alvo:", nome(alvo))
for model, label in [(pvdm, "PV-DM"), (pvdbow, "PV-DBOW")]:
    print(f"\\n{label}:")
    for j, s in model.dv.most_similar(alvo, topn=6):
        print(f"   {s:.2f}  {nome(j)}")
'''),
    md("""
## 3. Os vetores de documento separam gêneros literários?

Projetamos todos os blocos em 2D (t-SNE) coloridos pelo gênero. Comparamos PV-DM, PV-DBOW e o baseline **média dos vetores de palavras** (Skip-gram Machado).
"""),
    code('''
from sklearn.manifold import TSNE
from replang.data.embeddings import load_local_model
from replang.data.corpora import iter_machado_raw, _strip_machado_header
from replang.utils.text import iter_sentences
import plotly.express as px
sg = load_local_model("machado_sg100")
# reconstruímos os documentos (mesma segmentação do script de treino) para o baseline de média
docs = []
for fname, text in iter_machado_raw():
    sents = list(iter_sentences(_strip_machado_header(text)))
    for i in range(0, len(sents), 60):
        chunk = [t for s in sents[i:i+60] for t in s]
        if len(chunk) > 100:
            docs.append(chunk)
assert len(docs) == len(meta)
mean_vecs = np.stack([np.mean([sg[w] for w in d if w in sg], axis=0) for d in docs])
reps = {"PV-DM": pvdm.dv.vectors, "PV-DBOW": pvdbow.dv.vectors, "média de word2vec": mean_vecs}
frames = []
for name, X in reps.items():
    Z = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(X)
    frames.append(pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "gênero": docs_df.genero, "obra": docs_df.arquivo.str.split("/").str[-1], "modelo": name}))
fig = px.scatter(pd.concat(frames), x="x", y="y", color="gênero", facet_col="modelo", hover_data=["obra"], template="plotly_white", title="Blocos de Machado de Assis em 2D (t-SNE) — três representações de documento", height=480)
fig.show()
'''),
    code('''
# Quantificando: classificador linear de gênero literário (validação cruzada) sobre cada representação
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
y = docs_df.genero.to_numpy(dtype=str)   # pandas 3: colunas de texto são pyarrow; converter para numpy
rows = []
for name, X in reps.items():
    acc = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5).mean()
    rows.append({"representação": name, "acurácia (gênero literário, CV 5)": acc})
pd.DataFrame(rows).style.format({"acurácia (gênero literário, CV 5)": "{:.1%}"})
'''),
    md("""
## 4. Inferindo o vetor de um texto novo

A inferência otimiza só $d$, com as matrizes de palavras congeladas (`infer_vector`). Testamos com trechos **não vistos**: um parágrafo de romance, uma estrofe e uma frase jornalística moderna.
"""),
    code('''
from replang.utils.text import tokenize
novos = {
    "romance (estilo Machado)": "Capitu olhou para mim com os olhos de ressaca e eu senti que a vida inteira cabia naquele instante da sala",
    "poesia": "ó alma triste e serena que no silêncio da noite chora a estrela que se apaga",
    "jornalismo atual": "o governo anunciou nesta terça-feira um pacote de medidas econômicas para conter a inflação e o desemprego",
}
for label, texto in novos.items():
    v = pvdm.infer_vector(tokenize(texto), epochs=50)
    viz = pvdm.dv.most_similar([v], topn=3)
    print(f"{label:<24} → {[ (nome(j).split(' [')[0] + ' [' + meta[j]['genero'] + ']') for j, _ in viz]}")
'''),
    md("""
## 5. Doc2Vec × média de vetores na similaridade de sentenças (ASSIN-like)

Repetimos o protocolo do notebook 07 (cosseno → regressão → Pearson/MSE) no mini-STS com três representações de sentença. Em sentenças curtas o Doc2Vec costuma **perder** para a média de vetores — a inferência com poucas palavras é ruidosa; a vantagem aparece em documentos longos.
"""),
    code('''
from replang.eval.similarity import MINI_STS_PT
from replang.eval.extrinsic import sentence_vector
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
gold = np.array([s for _, _, s in MINI_STS_PT])
def eval_fn(fn):
    cos = np.array([WordVectors.cosine(fn(a), fn(b)) for a, b, _ in MINI_STS_PT])
    pred = LinearRegression().fit(cos[:, None], gold).predict(cos[:, None])
    return pearsonr(gold, cos)[0], float(np.mean((gold - pred) ** 2))
rows = []
for name, fn in [("média word2vec (SG Machado)", lambda s: sentence_vector(sg, s, mode="mean")), ("PV-DM infer_vector", lambda s: pvdm.infer_vector(tokenize(s), epochs=50)), ("PV-DBOW infer_vector", lambda s: pvdbow.infer_vector(tokenize(s), epochs=50))]:
    r, mse = eval_fn(fn)
    rows.append({"representação": name, "Pearson ρ (cosseno)": r, "MSE": mse})
pd.DataFrame(rows).style.format({"Pearson ρ (cosseno)": "{:.2f}", "MSE": "{:.2f}"})
'''),
    md(f"""
## 6. Síntese

* O Paragraph Vector estende a ideia do word2vec a unidades maiores com uma mudança mínima: **um vetor extra por documento**.
* PV-DM preserva alguma ordem (concatenação de contexto); PV-DBOW é mais simples e, combinado com PV-DM, era a recomendação do artigo.
* Limitações: inferência lenta e ruidosa, não determinística; reproduções posteriores (Lau & Baldwin, 2016) só confirmaram os ganhos com ajustes cuidadosos; em sentenças curtas a média de word2vec é um rival forte.
* É um passo da história que leva a *sentence encoders* (Skip-Thought, InferSent, Sentence-BERT) — e ao ELMo (notebook 12), onde o contexto entra de outra forma.

### Perguntas para discussão
* Por que o vetor do documento "lembra o tópico"? (ele é o único parâmetro compartilhado por todas as janelas do documento)
* O que acontece com o vetor de um documento que mistura dois tópicos?

## Referências
{refs("doc2vec", "mikolov13a", "hartmann")}
- LAU, J. H.; BALDWIN, T. **An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation.** RepL4NLP, 2016.
"""),
]
