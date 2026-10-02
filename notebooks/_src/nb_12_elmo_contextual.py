from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "12 · Representações contextuais (Peters et al., 2018): ELMo em miniatura"

CELLS = [
    md(f"""
# 12 · Representações contextuais: uma palavra, muitos vetores (Peters et al., 2018 — ELMo)
{HEADER}

**Artigo-base (leitura complementar):** Peters, Neumann, Iyyer, Gardner, Clark, Lee & Zettlemoyer (2018), *Deep contextualized word representations* (NAACL).

## O limite de tudo o que vimos até aqui
word2vec, GloVe, fastText e Doc2Vec dão **um vetor por tipo**: *banco* (assento) e *banco* (instituição) compartilham o mesmo ponto; *manga* (fruta/camisa) idem. Faruqui et al. (2016) apontam a polissemia como limite da avaliação intrínseca; Bolukbasi et al. notam que *man* é ambíguo (§5.1).

## A proposta ELMo (*Embeddings from Language Models*)
1. Treinar um **modelo de linguagem bidirecional** (biLM): um LSTM *forward* ($p(t_k\\mid t_1..t_{{k-1}})$) e um *backward* ($p(t_k\\mid t_{{k+1}}..t_N)$), com $L=2$ camadas, entrada por **CNN de caracteres** (sem OOV), parâmetros de entrada e softmax compartilhados entre as direções (§3.1).
2. Para cada token, expor **todas as camadas**: $R_k = \\{{x_k, \\overrightarrow{{h}}_{{k,j}}, \\overleftarrow{{h}}_{{k,j}}\\}}$, $2L+1$ vetores (§3.2).
3. A tarefa aprende uma **mistura**:
$$\\mathrm{{ELMo}}_k^{{task}} = \\gamma^{{task}} \\sum_{{j=0}}^{{L}} s_j^{{task}}\\, h_{{k,j}}^{{LM}} \\qquad (1)$$
com $s$ *softmax-normalizado* e $\\gamma$ um escalar.
4. Concatenar $[x_k; \\mathrm{{ELMo}}_k]$ à entrada (e às vezes à saída) de um modelo supervisionado existente, com o biLM **congelado** (§3.3).

Resultados: estado da arte em 6 tarefas (SQuAD +4,7 F1, SNLI, SRL +3,2, coref +3,2, NER, SST-5 +3,3) com reduções relativas de erro de 6–20 % (Tabela 1). Análise (§5.3): camadas baixas codificam **sintaxe** (POS), camadas altas codificam **sentido** (WSD).

## O que fazemos aqui
Um **ELMo-lite** em JAX (`replang.models.bilm_jax`): biLM de 2 camadas LSTM (128 unidades), *lookup* de palavras (sem CNN de caracteres), treinado por alguns minutos em Machado de Assis (`replang train bilm`). Suficiente para **mostrar** o mecanismo: (a) vetores diferentes por contexto; (b) camadas diferentes servem a tarefas diferentes; (c) a mistura da eq. (1).
"""),
    SETUP,
    code('''
from replang.models.bilm_jax import HAS_JAX, BiLM, BiLMVocab, elmo_mix, contextual_vectors
assert HAS_JAX, "instale o extra: uv sync --extra contextual"
import json
model, vocab = BiLM.load(PATHS.models / "machado_bilm.pkl")
meta = json.loads((PATHS.models / "machado_bilm.json").read_text())
print(f"biLM: {meta['layers']} camadas × {meta['hid']} unidades, embedding {meta['emb']}d, vocabulário {meta['vocab']:,}, {meta['steps']} passos")
from replang.viz import training_curve
training_curve(model.history, x="step", y="ppl", title="Perplexidade de treino do biLM (média das direções)").show()
'''),
    md("""
### Arquitetura em código

O coração do modelo é o `scan` sobre a sequência com uma célula LSTM, uma vez para a frente e outra para trás, por camada; a saída tem forma `(L+1, T, 2·hid)` — a camada 0 é o embedding do token duplicado, como no artigo.
"""),
    code('''
import inspect
from replang.models import bilm_jax
print(inspect.getsource(bilm_jax._lstm_scan))
src = inspect.getsource(BiLM._forward_all); print(src)
'''),
    md("""
## 1. O modelo de linguagem funciona? Perplexidade e previsões

Um biLM minúsculo treinado por minutos não é o `CNN-BIG-LSTM` do artigo (perplexidade 39,7 no 1B Word Benchmark). Mas deve ter aprendido regularidades: artigos antes de substantivos, concordância, colocações de Machado.
"""),
    code('''
import jax, jax.numpy as jnp
from replang.data.corpora import load_machado_sentences
sents = load_machado_sentences()
held = [vocab.encode(s) for s in sents[-300:]]
print(f"perplexidade em 300 sentenças finais (não vistas se o treino amostrou do início): {model.perplexity(held):.1f}  (uniforme seria {len(vocab):,})")
def top_next(prefix, k=6):
    enc = vocab.encode(prefix)[:-1]   # sem </s>
    _, lf, _ = model._fwd_fn(model.params, jnp.asarray(np.array(enc, dtype=np.int32)[None]))
    p = np.asarray(jax.nn.softmax(lf[0, -1]))
    return [(vocab.words[i], round(float(p[i]), 3)) for i in np.argsort(-p)[:k]]
for prefix in [["capitu", "olhou", "para"], ["o", "senhor"], ["não", "é"], ["a", "casa", "de"]]:
    print(" ".join(prefix), "→", top_next(prefix))
'''),
    md("""
## 2. Uma palavra, vários vetores: *banco*, *manga*, *letra*

Construímos sentenças com dois sentidos de cada palavra e comparamos o cosseno entre os vetores contextuais do mesmo token, por camada. Na **camada 0** todos são idênticos (cosseno 1 — é o embedding estático); nas camadas LSTM os sentidos se separam.
"""),
    code('''
from replang.utils import tokenize
exemplos = {
    "banco": ["ele sentou no banco da praça para descansar", "o velho dormia no banco de madeira do jardim", "ela foi ao banco sacar o dinheiro da conta", "o gerente do banco negou o empréstimo ao comerciante"],
    "manga": ["a manga da camisa estava rasgada e suja", "ele dobrou a manga do casaco até o cotovelo", "comi uma manga madura e doce no quintal", "a manga é uma fruta tropical muito saborosa"],
    "letra": ["a letra da música falava de amor e saudade", "ele cantou a letra inteira da canção", "a letra dela era bonita e redonda no papel", "escreveu a carta com letra firme e clara"],
}
def sim_matrix(word, frases, layer):
    V = []
    for f in frases:
        toks = tokenize(f)
        V.append(contextual_vectors(model, vocab, toks, word, layer=layer))
    V = np.stack(V); V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T
for word, frases in exemplos.items():
    print(f"\\n=== {word} ===  (sentenças 1-2: sentido A; 3-4: sentido B)")
    for layer in [0, 1, 2]:
        S = sim_matrix(word, frases, layer)
        same = np.mean([S[0, 1], S[2, 3]]); cross = np.mean([S[0, 2], S[0, 3], S[1, 2], S[1, 3]])
        print(f"  camada {layer}: cos(mesmo sentido) = {same:.2f} | cos(sentidos diferentes) = {cross:.2f} | diferença = {same - cross:+.2f}")
'''),
    code('''
import plotly.express as px
word, frases = "banco", exemplos["banco"]
S = sim_matrix(word, frases, 2)
labels = [f"s{i+1}: " + " ".join(tokenize(f))[:38] for i, f in enumerate(frases)]
px.imshow(S, x=[f"s{i+1}" for i in range(4)], y=labels, zmin=0, zmax=1, color_continuous_scale="Blues", text_auto=".2f", template="plotly_white",
          title="cos entre os vetores contextuais de 'banco' (camada 2)").show()
'''),
    md("""
## 3. Vizinhos contextuais (Tabela 4 do artigo)

O artigo compara os vizinhos de *play* no GloVe (misturam esporte e teatro) com os vizinhos do biLM, que trazem **tokens em contexto** do mesmo sentido. Fazemos o mesmo: para cada ocorrência-alvo, buscamos entre os tokens de um conjunto de sentenças de Machado os mais parecidos.
"""),
    code('''
from replang.data.embeddings import load_local_model
sg = load_local_model("machado_sg100")
print("vizinhos ESTÁTICOS de 'banco' (Skip-gram):", [w for w, _ in sg.most_similar("banco", topn=10)])
pool = [s for s in sents if "banco" in s and len(s) <= 25][:400]
tok_vecs, tok_info = [], []
for s in pool:
    reps = model.representations(vocab.encode(s))
    for i, w in enumerate(s):
        if w == "banco":
            tok_vecs.append(reps[2, i + 1]); tok_info.append(" ".join(s))
T = np.stack(tok_vecs); T /= np.linalg.norm(T, axis=1, keepdims=True)
for f in [frases[0], frases[2]]:
    q = contextual_vectors(model, vocab, tokenize(f), "banco", layer=2); q /= np.linalg.norm(q)
    sims = T @ q
    print(f"\\nalvo: «{f}»")
    for i in np.argsort(-sims)[:5]:
        print(f"   {sims[i]:.2f}  {tok_info[i][:110]}")
'''),
    md("""
## 4. Que camada serve a que tarefa? (§5.3: POS × WSD)

O artigo mede, com classificadores lineares sobre representações **congeladas**: POS tagging é melhor com a **1ª camada** (97,3 vs 96,8), desambiguação de sentido com a **2ª** (69,0 vs 67,4). Reproduzimos o lado da sintaxe com o Mac-Morpho: regressão logística sobre a representação de cada camada (e sobre o Skip-gram estático).
"""),
    code('''
from replang.data.corpora import load_macmorpho
from sklearn.linear_model import LogisticRegression
train = load_macmorpho(split="train", limit_sentences=1200 if FAST else 2500); test = load_macmorpho(split="test", limit_sentences=400)
# representações de todas as camadas, calculadas UMA vez por sentença (sem <s> e </s>)
def all_layers(sent_tags):
    toks = [w.lower() for w, _ in sent_tags]
    return model.representations(vocab.encode(toks))[:, 1:-1]
import time; t = time.time()
reps_train = [all_layers(s) for s in train]; reps_test = [all_layers(s) for s in test]
ytr = [t_ for s in train for _, t_ in s]; yte = np.array([t_ for s in test for _, t_ in s])
print(f"representações de {len(train)+len(test)} sentenças em {time.time()-t:.0f}s")
rows = []
for layer in [0, 1, 2]:
    Xtr = np.vstack([r[layer] for r in reps_train]); Xte = np.vstack([r[layer] for r in reps_test])
    clf = LogisticRegression(max_iter=400, C=1.0).fit(Xtr, ytr)
    rows.append({"representação": f"biLM camada {layer}", "dim": Xtr.shape[1], "acurácia POS (linear, sem janela)": float((clf.predict(Xte) == yte).mean())})
static = lambda s: np.stack([sg.get(w.lower(), np.zeros(sg.dim)) for w, _ in s])
Xtr = np.vstack([static(s) for s in train]); Xte = np.vstack([static(s) for s in test])
clf = LogisticRegression(max_iter=400).fit(Xtr, ytr)
rows.append({"representação": "Skip-gram estático (sem janela)", "dim": sg.dim, "acurácia POS (linear, sem janela)": float((clf.predict(Xte) == yte).mean())})
pos_layers = pd.DataFrame(rows)
pos_layers.style.format({"acurácia POS (linear, sem janela)": "{:.3f}"}).background_gradient(subset=["acurácia POS (linear, sem janela)"], cmap="Greens")
'''),
    md("""
Sem janela de contexto, o vetor estático "não sabe" se *a* é artigo ou preposição; o biLM sabe, porque o vetor do token já viu a sentença inteira. É a diferença entre **representar o tipo** e **representar o token**.
"""),
    md("""
## 5. A mistura ELMo: aprendendo $s^{task}$ (eq. 1, Fig. 2)

Em vez de escolher uma camada, a tarefa aprende pesos. Fazemos uma busca simples sobre $s$ (softmax de 3 parâmetros) para a tarefa de POS e para a tarefa de "separar sentidos" (maximizar a diferença intra/inter-sentido da seção 2). O artigo observa: tarefas sintáticas pesam a camada 1; semânticas, a camada 2 (Fig. 2).
"""),
    code('''
from itertools import product
grid = [np.array(w) for w in product([0.0, 1.0, 2.0], repeat=3)]
def softmax(w): e = np.exp(w - w.max()); return e / e.sum()
# tarefa A: POS (subconjunto menor para rapidez)
tr, te = train[:600], test[:150]
reps_tr, reps_te = reps_train[:600], reps_test[:150]
ytr_s = [t_ for s in tr for _, t_ in s]; yte_s = np.array([t_ for s in te for _, t_ in s])
best_pos = None
for w in grid[::2]:
    s = softmax(w)
    Xtr = np.vstack([np.tensordot(s, r, axes=(0, 0)) for r in reps_tr]); Xte = np.vstack([np.tensordot(s, r, axes=(0, 0)) for r in reps_te])
    acc = float((LogisticRegression(max_iter=300).fit(Xtr, ytr_s).predict(Xte) == yte_s).mean())
    if best_pos is None or acc > best_pos[0]: best_pos = (acc, s)
# tarefa B: separação de sentidos
best_wsd = None
for w in grid:
    s = softmax(w); diffs = []
    for word, frases in exemplos.items():
        V = np.stack([elmo_mix(model.representations(vocab.encode(tokenize(f))), np.log(s + 1e-9))[tokenize(f).index(word) + 1] for f in frases])
        V /= np.linalg.norm(V, axis=1, keepdims=True); S = V @ V.T
        diffs.append(np.mean([S[0, 1], S[2, 3]]) - np.mean([S[0, 2], S[0, 3], S[1, 2], S[1, 3]]))
    score = float(np.mean(diffs))
    if best_wsd is None or score > best_wsd[0]: best_wsd = (score, s)
from replang.viz import layer_weights_figure
print(f"POS: melhor acurácia {best_pos[0]:.3f} com pesos {np.round(best_pos[1], 2)}")
print(f"separação de sentidos: melhor diferença {best_wsd[0]:+.3f} com pesos {np.round(best_wsd[1], 2)}")
layer_weights_figure({"POS tagging": best_pos[1], "separar sentidos": best_wsd[1]}, title="Pesos s^task aprendidos (busca em grade) — cf. Fig. 2 de Peters et al.").show()
'''),
    md("""
## 6. Eficiência amostral (§5.4, Fig. 1 do artigo)

Com ELMo, o modelo SRL com **1 %** dos dados rotulados iguala o baseline com **10 %**. Reproduzimos a forma da curva no POS: acurácia × fração de sentenças rotuladas, com a camada 1 do biLM × o vetor estático.
"""),
    code('''
rows = []
for n in ([100, 300, 1000] if FAST else [100, 300, 1000, 2500]):
    for name, Xall_tr, Xall_te in [("biLM camada 1", [r[1] for r in reps_train], [r[1] for r in reps_test]), ("Skip-gram estático", [static(s) for s in train], [static(s) for s in test])]:
        Xtr = np.vstack(Xall_tr[:n]); ytr_n = [t_ for s in train[:n] for _, t_ in s]
        Xte = np.vstack(Xall_te)
        acc = float((LogisticRegression(max_iter=300).fit(Xtr, ytr_n).predict(Xte) == yte).mean())
        rows.append({"sentenças rotuladas": n, "representação": name, "acurácia": acc})
px.line(pd.DataFrame(rows), x="sentenças rotuladas", y="acurácia", color="representação", markers=True, log_x=True, template="plotly_white", title="Eficiência amostral: POS × tamanho do treino rotulado").show()
'''),
    md(f"""
## 7. Síntese: de word2vec a ELMo (e ao que veio depois)

| | word2vec / GloVe / fastText | ELMo |
|---|---|---|
| unidade representada | **tipo** (uma entrada de tabela) | **token** (função da sentença) |
| polissemia | um vetor para todos os sentidos | um vetor por ocorrência |
| treino | predizer vizinhos numa janela | modelo de linguagem bidirecional profundo |
| uso | tabela de consulta | rede congelada + mistura de camadas por tarefa |
| OOV | fastText: n-gramas | CNN de caracteres |

O passo seguinte (BERT, 2018) troca LSTMs por *Transformers* e a mistura de camadas por *fine-tuning* — mas a ideia-chave é a do ELMo: **o vetor é uma função do contexto**. E o viés? Continua lá, agora dependente do contexto (Zhao et al., 2019; May et al., 2019): a geometria mudou, o problema não desapareceu.

### Pontos para a apresentação
1. Mostre a matriz de cossenos de *banco*: camada 0 (tudo 1,0) × camada 2.
2. Mostre a tabela POS por camada: sem janela, o estático falha onde o contextual acerta.
3. Enfatize a **mistura aprendida** — a tarefa escolhe o que usar.

## Referências
{refs("elmo", "fasttext", "mikolov13a")}
- ZHAO, J. et al. **Gender Bias in Contextualized Word Embeddings.** NAACL, 2019.
"""),
]
