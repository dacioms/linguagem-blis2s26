from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "07 · Avaliação extrínseca (Hartmann et al., 2017): POS tagging, similaridade de sentenças e o problema das analogias"

CELLS = [
    md(f"""
# 07 · Avaliação extrínseca: POS tagging, similaridade de sentenças e por que analogias não bastam
{HEADER}

**Artigo-base:** Hartmann et al. (2017), §4.2 e §6; **crítica de referência:** Faruqui et al. (2016).

## A pergunta
Um embedding "bom em analogias" é bom **em uma tarefa real**? Hartmann et al. responderam **não necessariamente**:

| Tarefa | Melhor modelo no artigo | Posição desse modelo nas analogias |
|---|---|---|
| Analogias (intrínseca) | GloVe 300d (46,7 %) | 1º |
| POS tagging, Mac-Morpho (Tab. 3) | Wang2Vec Skip-gram 1000d (95,94 %) | mediano |
| Similaridade ASSIN PT-BR (Tab. 4) | Wang2Vec Skip-gram 1000d (ρ = 0,60) | mediano |
| Similaridade ASSIN PT-EU (Tab. 4) | Word2Vec CBOW 1000d (ρ = 0,55) | **último** nas semânticas |

GloVe e FastText, os melhores nas analogias, foram os **piores** nas duas tarefas. Neste notebook reproduzimos as duas avaliações extrínsecas com os modelos disponíveis e discutimos a crítica de Faruqui et al.
"""),
    SETUP,
    md("""
## 1. POS tagging no Mac-Morpho (§4.2, Tabela 3)

O artigo usa o *tagger* neural **nlpnet** (Fonseca et al., 2015): janela de palavras → embeddings concatenados → camada oculta → *tags*. Nossa versão simplificada (`replang.eval.extrinsic.pos_tagging_eval`) usa a mesma entrada (janela ±2 de embeddings + traços de capitalização) com um classificador **logístico linear**, para rodar em segundos. O que importa é a comparação **relativa** entre embeddings, com o mesmo classificador e os mesmos dados.

O Mac-Morpho revisado tem ~1,2 M tokens e 26 etiquetas (N, V, ADJ, PREP, ART, ...).
"""),
    code('''
from replang.data.corpora import load_macmorpho
train = load_macmorpho(split="train"); test = load_macmorpho(split="test", limit_sentences=600 if not FAST else 250)
tags = pd.Series([t for s in train for _, t in s]).value_counts()
print(f"treino: {len(train):,} sentenças; teste: {len(test)} sentenças; {len(tags)} etiquetas")
print("exemplo:", train[10][:10])
tags.head(12).to_frame("ocorrências").T
'''),
    code('''
import time
from replang.eval.extrinsic import pos_tagging_eval
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.models.trainers import list_models, load_fasttext_model
models = {"Wikipedia2Vec PT 100d": load_ptwiki()}
for m in list_models():
    if m["algo"] in ("word2vec", "fasttext") or m["algo"].startswith("glove"):
        models[f"{m['name']} ({m['algo']})"] = load_local_model(m["name"])
ftm = load_fasttext_model("machado_ft100")
oov_fn = (lambda w: ftm.wv[w]) if ftm is not None else None
n_train = 25000 if FAST else 60000
from replang.accel import report
from replang.eval.parallel import parallel_map
print("aceleração:", report())
def _eval_pos(item):
    name, wv = item
    t = time.time()
    r = pos_tagging_eval(wv, train, test, max_train_tokens=n_train, oov_fn=(lambda w: ftm.wv[w]) if ("ft100" in name and ftm is not None) else None)
    return {"modelo": name, "dim": wv.dim, "acurácia POS": r["acurácia"], "acurácia nos OOV": r["acurácia_oov"], "taxa OOV (teste)": r["taxa_oov"], "s": round(time.time() - t)}
t0 = time.time()
rows = parallel_map(_eval_pos, list(models.items()))   # um processo por modelo (REPLANG_JOBS); backend do classificador: JAX se instalado
print(f"{len(rows)} avaliações em {time.time()-t0:.0f}s (paralelo)")
pos_df = pd.DataFrame(rows).sort_values("acurácia POS", ascending=False)
pos_df.style.format({"acurácia POS": "{:.3f}", "acurácia nos OOV": "{:.3f}", "taxa OOV (teste)": "{:.1%}"}).background_gradient(subset=["acurácia POS"], cmap="Greens")
'''),
    md("""
Dois pontos para comparar com a Tabela 3 do artigo:

* **Taxa de OOV**: os modelos Machado não conhecem o vocabulário jornalístico de 1994 (*Cr$*, *privatização*); o fastText com `oov_fn` compõe vetores para essas palavras — note a coluna "acurácia nos OOV".
* No artigo, fastText foi fraco em POS; os autores suspeitam da **tokenização de clíticos** (o Mac-Morpho separa *disse-lhe* em *disse* + *lhe*; os embeddings não). Aqui o mesmo descasamento existe.
"""),
    md("""
## 2. Similaridade semântica de sentenças (§4.2, Tabela 4) — estilo ASSIN

O ASSIN (PROPOR 2016) pede uma nota de 1 a 5 para pares de sentenças. O *baseline* de Hartmann (2016): vetor da sentença = **soma dos vetores das palavras**; similaridade = cosseno; regressão linear → nota; métricas **Pearson ρ** e **MSE**. O artigo substitui o embedding e compara (melhor: Wang2Vec SG 1000d, ρ = 0,60 / MSE 0,49 em PT-BR).

O ASSIN exige download externo; incluímos um **mini-conjunto autoral** de 30 pares (`MINI_STS_PT`) só para demonstrar o protocolo. **Não** tire conclusões científicas dele. O conector abaixo lê o ASSIN original se os arquivos XML estiverem em `data/raw/assin/`.
"""),
    code('''
from replang.eval.similarity import MINI_STS_PT
from replang.eval.extrinsic import sentence_similarity_eval, sentence_vector
pd.DataFrame(MINI_STS_PT, columns=["sentença 1", "sentença 2", "nota (1–5)"]).head(8)
'''),
    code('''
import glob, xml.etree.ElementTree as ET
def load_assin(folder=PATHS.raw / "assin"):
    """Lê assin-ptbr-*.xml / assin-ptpt-*.xml (formato: <pair similarity="3.5"><t>..</t><h>..</h></pair>)."""
    pairs = []
    for f in sorted(glob.glob(str(folder / "*.xml"))):
        for p in ET.parse(f).getroot().iter("pair"):
            pairs.append((p.find("t").text, p.find("h").text, float(p.get("similarity"))))
    return pairs
assin = load_assin()
dataset = assin if assin else MINI_STS_PT
print(f"usando {'ASSIN' if assin else 'MINI_STS_PT'}: {len(dataset)} pares")
'''),
    code('''
rows = []
for name, wv in models.items():
    for mode in ["sum", "mean"]:
        r = sentence_similarity_eval(wv, dataset, mode=mode, oov_fn=oov_fn if "ft100" in name else None)
        rows.append({"modelo": name, "agregação": mode, "Pearson ρ": r["pearson"], "MSE": r["mse"], "ρ do cosseno bruto": r["pearson_cosseno"]})
sts_df = pd.DataFrame(rows)
sts_df[sts_df.agregação == "sum"].sort_values("Pearson ρ", ascending=False).style.format({"Pearson ρ": "{:.2f}", "MSE": "{:.2f}", "ρ do cosseno bruto": "{:.2f}"}).background_gradient(subset=["Pearson ρ"], cmap="Greens")
'''),
    code('''
# Baseline TF-IDF (o outro traço de Hartmann 2016) para o mesmo conjunto
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
s1, s2, gold = zip(*dataset)
vec = TfidfVectorizer().fit(list(s1) + list(s2))
A, B = vec.transform(s1), vec.transform(s2)
cos_tfidf = np.asarray(A.multiply(B).sum(1)).ravel() / (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) * np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9)
print(f"TF-IDF: Pearson entre cosseno e nota = {pearsonr(cos_tfidf, gold)[0]:.2f}")
'''),
    md("""
## 3. Os rankings concordam? (o achado central, §6)

Juntamos as três métricas por modelo e calculamos a **correlação de Spearman entre rankings**. No artigo, GloVe é 1º nas analogias e último nas tarefas; aqui, com modelos menores, veremos o nosso próprio quadro.
"""),
    code('''
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
from scipy.stats import spearmanr
an = load_analogies("pt-br")
an_rows = []
for name, wv in models.items():
    r = evaluate_analogies(wv, an, restrict=30000, max_per_category=300).set_index("categoria")
    an_rows.append({"modelo": name, "analogias total": r.loc["TOTAL", "acurácia"], "analogias sintáticas": r.loc["TOTAL sintática", "acurácia"], "analogias semânticas": r.loc["TOTAL semântica", "acurácia"]})
summary = pd.DataFrame(an_rows).merge(pos_df[["modelo", "acurácia POS"]], on="modelo").merge(sts_df[sts_df.agregação == "sum"][["modelo", "Pearson ρ"]], on="modelo").set_index("modelo")
ranks = summary.rank(ascending=False).astype(int)
display(summary.style.format("{:.3f}").background_gradient(cmap="Blues"))
print("Spearman entre rankings:")
for a, b in [("analogias total", "acurácia POS"), ("analogias total", "Pearson ρ"), ("acurácia POS", "Pearson ρ"), ("analogias sintáticas", "acurácia POS"), ("analogias semânticas", "Pearson ρ")]:
    print(f"  {a:<22} × {b:<14} ρ = {spearmanr(summary[a], summary[b]).correlation:+.2f}")
ranks
'''),
    md("""
> **Como ler o nosso resultado.** Entre modelos de **corpora muito diferentes** (Wikipédia × Machado) todos os rankings tendem a concordar: o modelo com mais dados ganha em tudo. A discordância que Hartmann et al. encontraram aparece **entre algoritmos treinados no mesmo corpus** — compare só as linhas `machado_*`: o fastText é o melhor em POS e analogias, mas fica no meio na similaridade de sentenças; o Skip-gram HS é o 2º em similaridade e o 6º em POS. Note também que o **mini-STS** tem 30 pares — a incerteza do ρ é enorme.

## 4. A crítica de Faruqui et al. (2016) — por que a avaliação intrínseca engana

1. **Subjetividade**: julgamentos humanos de similaridade confundem *similaridade* (carro–automóvel) com *relação* (carro–estrada); correlações entre anotadores são baixas.
2. **Sem conjunto de validação**: ajustar hiperparâmetros no próprio teste (overfitting ao benchmark).
3. **Baixa correlação com tarefas**: o que se otimiza (cosseno entre pares) não é o que a tarefa usa (combinações lineares, janelas, classificadores).
4. **Frequência**: desempenho em pares raros × frequentes é muito diferente; benchmarks misturam tudo.
5. **Polissemia**: um vetor por palavra não pode acertar *banco* nos dois sentidos (gancho para o notebook 12).

A recomendação do artigo (e de Hartmann et al.): avaliar **na tarefa-alvo**, com conjuntos de validação próprios.
"""),
    md("""
## 5. Experimento extra: quanto dado de treino a tarefa precisa?

Uma pergunta prática para quem escolhe embeddings: com poucos dados rotulados, a qualidade do embedding importa mais. Variamos o número de tokens rotulados no POS tagging para dois modelos.
"""),
    code('''
import plotly.express as px
sel = ["Wikipedia2Vec PT 100d"] + [k for k in models if "sg100 (" in k][:1]
rows = []
for n in ([3000, 10000, 30000] if FAST else [3000, 10000, 30000, 100000]):
    for name in sel:
        r = pos_tagging_eval(models[name], train, test, max_train_tokens=n)
        rows.append({"tokens rotulados": n, "modelo": name, "acurácia": r["acurácia"]})
px.line(pd.DataFrame(rows), x="tokens rotulados", y="acurácia", color="modelo", markers=True, log_x=True, template="plotly_white",
        title="POS tagging: acurácia × tamanho do treino rotulado").show()
'''),
    md(f"""
## 6. Síntese e pontos para a apresentação

1. **Avaliar na tarefa.** A Tabela 2 (analogias) e as Tabelas 3–4 (tarefas) de Hartmann et al. contam histórias diferentes.
2. **OOV e tokenização** são decisões de engenharia com efeito maior que a escolha do algoritmo.
3. O protocolo "soma de vetores + cosseno + regressão" é um *baseline* forte e barato para similaridade de sentenças — e é o ponto de partida do Doc2Vec (notebook 11) e de modelos de sentença.
4. Para a plateia: *qual é a sua tarefa?* define *qual é o seu embedding*.

### Perguntas para discussão
* O POS tagging é uma tarefa "sintática"; por que o GloVe (semântico) e o fastText (morfológico) ficaram atrás do Wang2Vec?
* Se analogias são um proxy ruim, por que continuam sendo usadas? (custo, interpretabilidade, tradição)

## Referências
{refs("hartmann", "faruqui", "fasttext", "glove")}
"""),
]
