from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "06 · Embeddings em português (Hartmann et al., 2017): corpus, modelos e avaliação intrínseca"

CELLS = [
    md(f"""
# 06 · Embeddings em português: corpus, modelos e avaliação intrínseca (Hartmann et al., 2017)
{HEADER}

**Artigo-base (leitura obrigatória):** Hartmann, Fonseca, Shulby, Treviso, Rodrigues & Aluísio (2017), *Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks* (STIL).

## O que o artigo fez
1. Montou um **corpus de 1,39 bilhão de tokens** em português (17 fontes, PT-BR e PT-EU: LX-Corpus, Wikipédia, GoogleNews, SubIMDB, G1, PLN-Br, literatura de domínio público, ...).
2. Definiu um **pré-processamento** cuidadoso (notebook 00, §2.2).
3. Treinou **31 modelos** com quatro algoritmos — Word2Vec (CBOW/Skip-gram), **Wang2Vec** (Ling et al., 2015: variante sensível à ordem), GloVe e FastText — em 50, 100, 300, 600 e 1000 dimensões; publicou todos em `nilc.icmc.usp.br/embeddings`.
4. Avaliou **intrinsecamente** (analogias LX-4WAnalogies, PT-BR e PT-EU) e **extrinsecamente** (POS tagging no Mac-Morpho; similaridade de sentenças no ASSIN).
5. Concluiu que **os rankings não concordam**: analogias não são um bom critério para escolher embeddings (alinhado a Faruqui et al., 2016).

Este notebook cobre 1–4 (intrínseca); o notebook 07 cobre a avaliação extrínseca e a conclusão 5.
"""),
    SETUP,
    md("""
## 1. O corpus do artigo × o nosso

Não podemos treinar em 1,39 bi de tokens aqui. Trabalhamos com dois substitutos: o **Wikipedia2Vec PT** (pré-treinado na Wikipédia PT, ~200 M tokens, skip-gram 100d) e os modelos treinados **ao vivo** em Machado de Assis (2,4 M tokens). Compare as escalas (Tabela 1 do artigo):
"""),
    code('''
tabela1 = pd.DataFrame([
    ("LX-Corpus (Rodrigues et al., 2016)", 714_286_638, "misto (maioria PT-EU)"), ("Wikipedia", 219_293_003, "enciclopédico"), ("GoogleNews", 160_396_456, "notícias"),
    ("SubIMDB-PT", 129_975_149, "legendas (fala)"), ("G1", 105_341_070, "notícias"), ("PLN-Br", 31_196_395, "notícias"), ("Literatura de domínio público", 23_750_521, "prosa"),
    ("Lacio-web", 8_962_718, "misto"), ("Portuguese e-books", 1_299_008, "prosa"), ("Mundo Estranho", 1_047_108, "revista"), ("CHC", 941_032, "ciência p/ crianças"),
    ("FAPESP", 499_008, "divulgação científica"), ("Textbooks", 96_209, "didático"), ("Folhinha", 73_575, "notícias p/ crianças"), ("NILC subcorpus", 32_868, "didático"),
    ("Para Seu Filho Ler", 21_224, "notícias p/ crianças"), ("SARESP", 13_308, "avaliação escolar")], columns=["fonte", "tokens", "gênero"])
print(f"Total do artigo: {tabela1.tokens.sum():,} tokens (3.827.725 tipos)")
from replang.data.corpora import load_machado_sentences
sents = load_machado_sentences()
print(f"Nosso corpus Machado: {sum(map(len, sents)):,} tokens ({sum(map(len, sents)) / tabela1.tokens.sum():.3%} do corpus do artigo)")
tabela1.assign(**{"% do total": (tabela1.tokens / tabela1.tokens.sum() * 100).round(2)})
'''),
    md("""
### 1.1 Pré-processamento (§2.1) aplicado ao nosso corpus

Já aplicamos no notebook 00: tokens com < 5 ocorrências → `UNKNOWN` (nos modelos, `min_count=5`), numerais → 0, URL/EMAIL, clíticos preservados, sentenças ≥ 5 tokens. O efeito do `min_count` no vocabulário:
"""),
    code('''
from collections import Counter
cnt = Counter(t for s in sents for t in s)
for mc in [1, 2, 5, 10]:
    v = sum(1 for c in cnt.values() if c >= mc)
    cov = sum(c for c in cnt.values() if c >= mc) / sum(cnt.values())
    print(f"min_count={mc:<3} |V| = {v:>7,}  cobertura de tokens = {cov:.1%}")
print("\\nExemplos de tipos com 1 ocorrência (viram UNKNOWN):", [w for w, c in cnt.items() if c == 1][:12])
'''),
    md("""
## 2. Os quatro algoritmos (§3)

| Algoritmo | Família | O que captura | Implementação neste projeto |
|---|---|---|---|
| **GloVe** | contagem (matriz global) | semântica; bom em analogias semânticas | `replang.models.glove_np` (notebook 04) |
| **Word2Vec** CBOW / Skip-gram | preditivo, janela sem ordem | semântica (Skip-gram) / rapidez (CBOW) | `replang.models.word2vec_np` + gensim (02, 03) |
| **Wang2Vec** (Ling et al., 2015) | preditivo **com ordem** (janela estruturada: um conjunto de parâmetros por posição) | sintaxe; melhor em POS tagging e no ASSIN | não implementado (ver nota abaixo) |
| **FastText** | preditivo com **subpalavras** | morfologia; melhor em analogias sintáticas | `replang.models.fasttext_np` + gensim (05) |

> **Wang2Vec**: a *Structured Skip-gram* usa matrizes de saída diferentes para cada posição relativa ($-c..+c$), de modo que "prever a palavra à esquerda" ≠ "prever a palavra à direita". É uma modificação simples do nosso `Word2Vec` numpy (basta indexar `W_out` por posição). Fica como exercício sugerido; o gensim não o inclui.

Abaixo, a Tabela 2 do artigo (melhor dimensão por modelo) para referência.
"""),
    code('''
tabela2 = pd.DataFrame([
    ("FastText CBOW", 300, 52.0, 8.4, 30.1), ("FastText Skip-gram", 300, 58.7, 32.2, 45.4), ("GloVe", 300, 45.8, 45.8, 46.7), ("GloVe", 600, 42.3, 48.5, 45.4),
    ("Wang2Vec CBOW", 300, 49.9, 40.3, 45.1), ("Wang2Vec Skip-gram", 600, 52.9, 35.0, 43.9), ("Word2Vec CBOW", 300, 24.7, 4.6, 23.9), ("Word2Vec Skip-gram", 600, 35.6, 20.0, 33.4)],
    columns=["modelo", "dim", "sintática (%)", "semântica (%)", "total (%)"])
tabela2.style.set_caption("Hartmann et al. (2017), Tabela 2 — analogias PT-BR, melhores configurações").background_gradient(subset=["total (%)"], cmap="Blues")
'''),
    md("""
## 3. Avaliação intrínseca: analogias PT-BR e PT-EU (§4.1)

O conjunto LX-4WAnalogies (Rodrigues et al., 2016) traduz e adapta o `questions-words` de Mikolov: 5 categorias semânticas e 9 sintáticas, em duas variantes (BR: *ônibus*, *trem*; EU: *autocarro*, *comboio*). Avaliamos todos os modelos disponíveis, no estilo da Tabela 2.
"""),
    code('''
from replang.data.analogies import load_analogies
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.eval.analogies import evaluate_analogies, compare_models
from replang.models.trainers import list_models
br, eu = load_analogies("pt-br"), load_analogies("pt-eu")
models = {"Wikipedia2Vec PT 100d": load_ptwiki()}
for m in list_models():
    if m["algo"] in ("word2vec", "fasttext") or m["algo"].startswith("glove"):
        models[f"{m['name']} ({m['algo']})"] = load_local_model(m["name"])
cmp_br = compare_models(models, br, restrict=30000, max_per_category=None if not FAST else 300).assign(variante="PT-BR")
cmp_eu = compare_models(models, eu, restrict=30000, max_per_category=None if not FAST else 300).assign(variante="PT-EU")
cmp = pd.concat([cmp_br, cmp_eu]).sort_values(["variante", "total"], ascending=[True, False]).reset_index(drop=True)
cmp.style.format({"sintática": "{:.1%}", "semântica": "{:.1%}", "total": "{:.1%}", "cobertura": "{:.1%}"}).background_gradient(subset=["total"], cmap="Blues")
'''),
    md("""
Dois efeitos saltam aos olhos:

* **Cobertura**: os modelos Machado cobrem ~20 % das questões (nada de capitais do mundo, moedas, estados americanos); o Wikipedia2Vec cobre muito mais. Comparar acurácias com coberturas diferentes é comparar provas diferentes.
* **PT-BR × PT-EU**: diferenças pequenas (o artigo também achou: Tabela 2 tem colunas quase idênticas), porque poucas questões envolvem palavras que mudam entre variantes.
"""),
    code('''
# Por categoria, para o Wikipedia2Vec e o melhor modelo Machado
best_local = cmp_br[cmp_br.modelo != "Wikipedia2Vec PT 100d"].sort_values("total", ascending=False).iloc[0].modelo
import plotly.express as px
frames = []
for name in ["Wikipedia2Vec PT 100d", best_local]:
    r = evaluate_analogies(models[name], br, restrict=30000, max_per_category=None if not FAST else 300)
    frames.append(r[~r.categoria.str.startswith("TOTAL")].assign(modelo=name))
df = pd.concat(frames)
fig = px.bar(df, x="categoria", y="acurácia", color="modelo", barmode="group", template="plotly_white", title="Acurácia por categoria (PT-BR)", hover_data=["cobertas", "n"])
fig.update_layout(xaxis_tickangle=-35); fig.show()
'''),
    md("""
## 4. O efeito da dimensão (50 → 1000 no artigo)

No artigo, a acurácia das analogias **sobe até 300 dimensões e cai depois** (FastText SG: 36,8 → 58,7 → 45,1 % em 50/300/1000d), enquanto no POS tagging (notebook 07) *quanto maior, melhor*. Com o corpus Machado reproduzimos a parte barata: 50 × 100 × 200 dimensões com Skip-gram (gensim, 3 épocas).
"""),
    code('''
import time
from gensim.models import Word2Vec as GW2V
rows = []
for dim in ([50, 100, 200] if not FAST else [50, 100]):
    t = time.time()
    m = GW2V(sents, vector_size=dim, window=5, sg=1, negative=10, min_count=5, epochs=3, sample=1e-4, workers=settings.workers, seed=1)
    wv = WordVectors.from_gensim(m.wv, f"SG {dim}d")
    r = evaluate_analogies(wv, br, restrict=None).set_index("categoria")
    rows.append({"dim": dim, "sintática": r.loc["TOTAL sintática", "acurácia"], "semântica": r.loc["TOTAL semântica", "acurácia"], "total": r.loc["TOTAL", "acurácia"], "s": round(time.time() - t)})
pd.DataFrame(rows).style.format({"sintática": "{:.1%}", "semântica": "{:.1%}", "total": "{:.1%}"})
'''),
    md("""
## 5. Conector para os modelos do NILC

Para usar os embeddings originais do artigo (quando a rede permitir), baixe um arquivo em http://nilc.icmc.usp.br/embeddings (ex.: `skip_s100.zip`) e aponte o carregador. Os arquivos estão no formato texto do word2vec e podem ser truncados às `top_n` palavras mais frequentes.
"""),
    code('''
from replang.data.embeddings import load_nilc
nilc_path = PATHS.raw / "skip_s100.zip"   # ajuste para o arquivo baixado
if nilc_path.exists():
    nilc = load_nilc(nilc_path, top_n=100_000)
    print(nilc, nilc.most_similar("rei", topn=5))
    models["NILC skip_s100"] = nilc
else:
    print("Arquivo do NILC não encontrado em", nilc_path, "— usando Wikipedia2Vec PT como substituto.")
'''),
    md("""
## 6. Variantes BR × EU no espaço vetorial

Mesmo sem o corpus do artigo, podemos observar como a Wikipédia PT (que mistura variantes) posiciona pares BR/EU: *ônibus–autocarro*, *trem–comboio*, *celular–telemóvel*, *grama–relva*.
"""),
    code('''
pt = models["Wikipedia2Vec PT 100d"]
pares = [("ônibus", "autocarro"), ("trem", "comboio"), ("celular", "telemóvel"), ("geladeira", "frigorífico"), ("suco", "sumo"), ("time", "equipa")]
for a, b in pares:
    if pt.has(a, b):
        print(f"cos({a}, {b}) = {pt.similarity(a, b):.2f} | vizinhos de {a}: {[x for x, _ in pt.most_similar(a, topn=4)]} | de {b}: {[x for x, _ in pt.most_similar(b, topn=4)]}")
'''),
    md(f"""
## 7. Síntese

* O artigo é, acima de tudo, um **recurso**: 31 modelos públicos + script de pré-processamento, com avaliação sistemática em PT.
* Nas analogias: GloVe (semântica) e FastText (sintática) lideram; CBOW do Word2Vec é fraco em semântica (como já notava Mikolov 2013a); **Wang2Vec** é consistentemente bom.
* Dimensão: 300 é o "ponto doce" para analogias; para tarefas, maior é melhor.
* A **cobertura** e o **domínio do corpus** pesam tanto quanto o algoritmo — nossos modelos Machado mostram isso de forma extrema.

### Perguntas para discussão
1. Faz sentido um único corpus misturando PT-BR e PT-EU? (Fonseca & Aluísio 2016 dizem que sim para POS; e para aplicações lexicais?)
2. O LX-4WAnalogies é uma tradução de um teste anglocêntrico (estados americanos, moedas). Que categorias um teste "nativo" de PT teria? (gênero gramatical, conjugações, diminutivos...)

## Referências
{refs("hartmann", "rodrigues", "mikolov13a", "glove", "fasttext", "faruqui")}
"""),
]
