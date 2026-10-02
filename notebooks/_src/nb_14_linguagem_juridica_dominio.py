from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "14 · Contexto jurídico I: a língua do Direito e o deslocamento de domínio (geral × jurídico)"

CELLS = [
    md(f"""
# 14 · Contexto jurídico I: a língua do Direito e o deslocamento de domínio
{HEADER}

**Pergunta deste notebook:** o que acontece com tudo o que vimos (coocorrência, word2vec, fastText, avaliação, analogias) quando o corpus é **texto jurídico brasileiro**? Comparamos sistematicamente um embedding **geral** (Wikipedia2Vec PT), um **literário** (Machado de Assis) e embeddings **jurídicos** treinados aqui em decisões do STF.

Leitura de apoio: [`docs/juridico.md`](../docs/juridico.md) (peculiaridades da língua do Direito, particularidades brasileiras, aplicações e debates).

## Corpora jurídicos públicos usados
| Corpus | Conteúdo | Uso aqui |
|---|---|---|
| **RulingBR** (Feijó & Moreira, 2018) | 10.574 decisões do STF (2011–2018): ementa, relatório, voto, área, classe, relator | treino dos embeddings jurídicos (6 M tokens); classificação de área |
| **LeNER-Br** (Luz de Araujo et al., 2018) | decisões anotadas com PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA | avaliação extrínseca: NER jurídico |
| **JurisBERT STS** (Viegas et al., 2023) | pares de ementas (STJ, TJMS) rotulados como similares ou não | avaliação extrínseca: similaridade de ementas |

Os modelos `legal_sg100`, `legal_cbow100` e `legal_ft100` são treinados por `uv run replang train legal` com **o mesmo pré-processamento e hiperparâmetros** dos modelos Machado — para que a comparação isole o *domínio*.
"""),
    SETUP,
    md("""
## 1. Perfil dos corpora: a língua do Direito em números

Comparamos o corpus jurídico (amostra de 1,5 M tokens, mesma ordem de grandeza do Machado) com o literário: tamanho de sentença, diversidade lexical, numerais, latim, a palavra *art*.
"""),
    code('''
from replang.data.corpora import load_machado_sentences
from replang.data.legal import load_legal_sentences, corpus_profile, LEGAL
mach = load_machado_sentences()
jur = load_legal_sentences(prefer_full=False)   # amostra versionada (1,5 M tokens)
prof = pd.DataFrame([corpus_profile(mach, "Machado de Assis (literário)"), corpus_profile(jur, "STF / RulingBR (jurídico)")]).set_index("corpus").T
prof.style.format("{:,.4f}")
'''),
    code('''
import plotly.express as px
lens = pd.DataFrame({"tokens por sentença": [len(s) for s in mach[:40000]] + [len(s) for s in jur[:40000]], "corpus": ["literário"] * 40000 + ["jurídico"] * 40000})
px.histogram(lens[lens["tokens por sentença"] <= 120], x="tokens por sentença", color="corpus", barmode="overlay", nbins=60, histnorm="probability", template="plotly_white",
             title="Distribuição do tamanho das sentenças (≤ 120 tokens)").show()
'''),
    md("""
Dois traços saltam: as sentenças jurídicas são **mais longas** (cauda pesada acima de 40 tokens — períodos subordinados, enumerações) e os **numerais** são muito mais frequentes (artigos, leis, processos, datas, valores). Isso já afeta decisões de modelagem: janela de contexto, e o que fazer com números.
"""),
    md("""
### 1.1 O pré-processamento de Hartmann et al. aplicado a texto jurídico

A normalização "numerais → 0" (notebook 00) é razoável em notícias; em Direito, **o número é o nome da norma**. Veja o que o pipeline faz com uma citação típica:
"""),
    code('''
from replang.utils.text import normalize_text, tokenize
ex = "Nos termos do art. 5º, inciso LV, da CF/88 e do art. 1.022 do CPC (Lei nº 13.105/2015), conheço do REsp 1.234.567/SP e nego-lhe provimento, com fundamento na Súmula 7/STJ."
print("normalizado:", normalize_text(ex))
print("tokens     :", tokenize(ex))
'''),
    md("""
*Lei 13.105/2015* vira *lei 0.000/0000*; *art. 5º* e *art. 1.022* viram o mesmo token; *§* desaparece (é pontuação); *CF/88* vira *cf/00*. Para **treinar embeddings de palavras** isso é aceitável (números são ruído distribucional); para **extrair entidades** (LeNER-Br: LEGISLACAO, JURISPRUDENCIA) é destrutivo. Uma boa prática de domínio: tokenizar números e siglas como tokens próprios *antes* de qualquer normalização, ou tratá-los como entidades.
"""),
    md("""
## 2. Coocorrência no domínio: a radiografia do vocabulário técnico

A hipótese distribucional (notebook 01) aplicada ao corpus do STF: com que palavras *sentença*, *ação* e *parte* coocorrem? Compare com o que o leitor leigo esperaria.
"""),
    code('''
from replang.models.cooccurrence import CountModel
cm = CountModel(window=4, dim=100, min_count=10, alpha=0.75, weighting="harmonic", max_vocab=20000).fit(jur)
ppmi = cm.ppmi_
for w in ["sentença", "ação", "parte", "título", "pena", "agravo"]:
    i = cm.index[w]
    row = ppmi[i].toarray().ravel()
    top = [cm.words[j] for j in np.argsort(-row)[:10]]
    print(f"{w:<9} PPMI mais alto com: {top}")
'''),
    md("""
## 3. Deslocamento de domínio: vizinhos no geral × no jurídico

Carregamos os embeddings e comparamos os vizinhos das palavras **polissêmicas** do léxico jurídico (`replang.data.legal.LEGAL.polysemous`). Como espaços diferentes não são comparáveis por cosseno, medimos a **sobreposição das vizinhanças** (Jaccard dos 10 vizinhos).
"""),
    code('''
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.data.legal import neighborhood_overlap
pt = load_ptwiki(); pt.name = "Wikipédia"
leg = load_local_model("legal_sg100"); leg.name = "STF"
mac = load_local_model("machado_sg100"); mac.name = "Machado"
print(pt, "|", leg, "|", mac)
for w in ["sentença", "ação", "parte", "título", "pena", "agravo", "competência", "prescrição", "trânsito", "remédio"]:
    print(f"\\n{w}")
    for wv in (pt, leg, mac):
        print(f"   {wv.name:<9} {[x for x, _ in wv.most_similar(w, topn=7)] if w in wv else 'OOV'}")
'''),
    code('''
probe = [w for w in LEGAL.probe_words if w in pt and w in leg]
shift = neighborhood_overlap(pt, leg, probe, k=10)
print(f"sobreposição média (Jaccard@10) entre Wikipédia e STF nas {len(probe)} palavras polissêmicas: {shift['jaccard@k'].mean():.3f}")
shift.head(15).style.format({"jaccard@k": "{:.2f}"})
'''),
    code('''
# Controle: palavras NÃO técnicas (frequentes nos dois corpora) devem ter mais sobreposição
comuns = [w for w in ["casa", "cidade", "tempo", "dia", "homem", "mulher", "trabalho", "família", "governo", "dinheiro", "saúde", "escola", "água", "carro", "livro"] if w in pt and w in leg]
ctrl = neighborhood_overlap(pt, leg, comuns, k=10)
glossa = {p[0]: p for p in LEGAL.polysemous}
print(f"Jaccard@10 médio — polissêmicas jurídicas: {shift['jaccard@k'].mean():.3f} | palavras comuns: {ctrl['jaccard@k'].mean():.3f}")
fig = px.bar(pd.concat([shift.assign(grupo="polissêmicas jurídicas"), ctrl.assign(grupo="comuns")]), x="palavra", y="jaccard@k", color="grupo", template="plotly_white",
             title="Sobreposição das vizinhanças Wikipédia × STF por palavra (menor = maior deslocamento de domínio)")
fig.update_layout(xaxis_tickangle=-40); fig.show()
'''),
    md("""
As palavras com Jaccard ≈ 0 são as que **mudam de sentido** ao entrar no Direito: *sentença* (frase → decisão), *título* (nome → documento de crédito), *pena* (dó → sanção), *trânsito* (tráfego → trânsito em julgado), *remédio* (medicamento → remédio constitucional). O glossário:
"""),
    code('''
pd.DataFrame([{"palavra": w, "sentido geral": g, "sentido jurídico": j} for w, g, j in LEGAL.polysemous if w in shift.palavra.values]).head(20)
'''),
    md("""
## 4. Analogias: as gerais perdem cobertura; as jurídicas aparecem

O LX-4WAnalogies (capitais, moedas, família) é um teste "da Wikipédia". No modelo jurídico a **cobertura** cai e a acurácia fica sem sentido — reforçando a conclusão de Hartmann et al. Em compensação, relações próprias do domínio funcionam: *autor : réu :: apelante : apelado*, *juiz : sentença :: tribunal : acórdão*, *civil : CPC :: penal : CPP*.
"""),
    code('''
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
an = load_analogies("pt-br")
rows = []
for wv in (pt, leg, mac):
    r = evaluate_analogies(wv, an, restrict=30000, max_per_category=300).set_index("categoria")
    rows.append({"modelo": wv.name, "cobertura": r.loc["TOTAL", "cobertas"] / r.loc["TOTAL", "n"], "acurácia (cobertas)": r.loc["TOTAL", "acurácia"]})
pd.DataFrame(rows).style.format({"cobertura": "{:.1%}", "acurácia (cobertas)": "{:.1%}"})
'''),
    code('''
rows = []
for a, b, c, d in LEGAL.analogies:
    for wv in (pt, leg):
        if wv.has(a, b, c):
            top = [x for x, _ in wv.analogy(a, b, c, topn=5)]
            rows.append({"analogia": f"{a} : {b} :: {c} : ?", "esperado": d, "modelo": wv.name, "top-5": ", ".join(top), "acerto@5": d in top})
        else:
            rows.append({"analogia": f"{a} : {b} :: {c} : ?", "esperado": d, "modelo": wv.name, "top-5": "OOV", "acerto@5": False})
df = pd.DataFrame(rows)
print(df.groupby("modelo")["acerto@5"].mean().rename("acerto@5 nas analogias jurídicas"))
df
'''),
    md("""
## 5. Avaliação extrínseca I: NER jurídico (LeNER-Br)

Como no notebook 07 (POS), usamos janela de embeddings + regressão logística para prever as etiquetas BIO do LeNER-Br. O que importa é a **comparação** geral × jurídico × fastText (OOV). Métrica: F1 por token nas entidades (ignorando `O`), além da acurácia.
"""),
    code('''
import time
from replang.data.legal import load_lener
from replang.eval.extrinsic import pos_tagging_eval
from replang.models.trainers import load_fasttext_model
from sklearn.metrics import f1_score
train = load_lener("train"); test = load_lener("test")
tags = pd.Series([t for s in train for _, t in s]).value_counts()
print(f"treino: {len(train)} sentenças / {tags.sum():,} tokens · teste: {len(test)} sentenças"); print(tags.head(8).to_dict())
ftm = load_fasttext_model("legal_ft100"); ft_leg = load_local_model("legal_ft100"); ft_leg.name = "STF fastText"
rows = []
for wv, oov_fn in [(pt, None), (mac, None), (leg, None), (ft_leg, (lambda w: ftm.wv[w]) if ftm is not None else None)]:
    t = time.time()
    r = pos_tagging_eval(wv, train, test, window=2, max_train_tokens=60000 if not FAST else 25000, oov_fn=oov_fn)
    clf = r["clf"]
    from replang.eval.extrinsic import _window_features
    Xte = np.vstack([_window_features(wv, [w for w, _ in s], 2, oov_fn) for s in test]); yte = np.array([t_ for s in test for _, t_ in s])
    pred = clf.predict(Xte)
    ents = [t_ for t_ in sorted(set(yte)) if t_ != "O"]
    rows.append({"embedding": wv.name, "acurácia": r["acurácia"], "F1 entidades (macro, token)": f1_score(yte, pred, labels=ents, average="macro"), "taxa OOV": r["taxa_oov"], "s": round(time.time() - t)})
ner = pd.DataFrame(rows)
ner.style.format({"acurácia": "{:.3f}", "F1 entidades (macro, token)": "{:.3f}", "taxa OOV": "{:.1%}"}).background_gradient(subset=["F1 entidades (macro, token)"], cmap="Greens")
'''),
    md("""
## 6. Avaliação extrínseca II: similaridade de ementas (JurisBERT STS)

O "ASSIN jurídico": pares de ementas do STJ e do TJMS rotulados como **similares** (mesmo grupo temático) ou não. Protocolo: vetor da ementa = soma dos vetores das palavras; cosseno; medimos a capacidade de separar similares de não similares (AUC) e comparamos com TF-IDF.
"""),
    code('''
from replang.data.legal import load_legal_sts
from replang.eval.extrinsic import sentence_vector
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
sts = load_legal_sts()
print(sts.shape, sts.tribunal.value_counts().to_dict(), "similar:", sts.similar.mean())
rows = []
for wv, oov_fn in [(pt, None), (mac, None), (leg, None), (ft_leg, (lambda w: ftm.wv[w]) if ftm is not None else None)]:
    cos = np.array([WordVectors.cosine(sentence_vector(wv, a, mode="mean", oov_fn=oov_fn), sentence_vector(wv, b, mode="mean", oov_fn=oov_fn)) for a, b in zip(sts.ementa1, sts.ementa2)])
    rows.append({"representação": wv.name, "AUC (similar × não similar)": roc_auc_score(sts.similar, cos), "cos médio similares": cos[sts.similar == 1].mean(), "cos médio não similares": cos[sts.similar == 0].mean()})
vec = TfidfVectorizer(min_df=2).fit(list(sts.ementa1) + list(sts.ementa2))
A, B = vec.transform(sts.ementa1), vec.transform(sts.ementa2)
cos_tfidf = np.asarray(A.multiply(B).sum(1)).ravel() / (np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) * np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-9)
rows.append({"representação": "TF-IDF", "AUC (similar × não similar)": roc_auc_score(sts.similar, cos_tfidf), "cos médio similares": cos_tfidf[sts.similar == 1].mean(), "cos médio não similares": cos_tfidf[sts.similar == 0].mean()})
pd.DataFrame(rows).style.format({c: "{:.3f}" for c in rows[0] if c != "representação"}).background_gradient(subset=["AUC (similar × não similar)"], cmap="Greens")
'''),
    md("""
## 7. Avaliação extrínseca III: classificar a área do Direito pela ementa (RulingBR)

A tarefa do projeto Victor (STF) em miniatura: dada a ementa, prever a **área** (penal, administrativo, tributário…). Média de vetores → regressão logística, validação cruzada; comparamos geral × jurídico × TF-IDF.
"""),
    code('''
from replang.data.legal import load_rulingbr_ementas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
em = load_rulingbr_ementas()
top_areas = em.area.value_counts().head(6).index
em = em[em.area.isin(top_areas)].reset_index(drop=True)
y = em.area.to_numpy(dtype=str)
print(f"{len(em)} ementas em {len(top_areas)} áreas:", em.area.value_counts().to_dict())
rows = []
for wv in (pt, mac, leg):
    X = np.stack([sentence_vector(wv, e, mode="mean") for e in em.ementa])
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    rows.append({"representação": wv.name, "acurácia (CV 5)": acc})
Xt = TfidfVectorizer(min_df=3, max_features=30000).fit_transform(em.ementa)
rows.append({"representação": "TF-IDF", "acurácia (CV 5)": cross_val_score(LogisticRegression(max_iter=2000), Xt, y, cv=5).mean()})
pd.DataFrame(rows).style.format({"acurácia (CV 5)": "{:.1%}"}).background_gradient(subset=["acurácia (CV 5)"], cmap="Greens")
'''),
    md("""
## 8. Subpalavras no jurídico: papéis em *-ante/-ado* e termos longos

A morfologia jurídica é regular: *agravante/agravado*, *embargante/embargado*, *impetrante/impetrado*, *apelante/apelado*, *recorrente/recorrido*. O fastText aprende a estrutura e cobre OOV (siglas, compostos).
"""),
    code('''
pares = [("agravante", "agravado"), ("embargante", "embargado"), ("impetrante", "impetrado"), ("apelante", "apelado"), ("recorrente", "recorrido"), ("exequente", "executado"), ("locador", "locatário")]
rows = [{"par": f"{a}–{b}", "STF SG": leg.similarity(a, b) if leg.has(a, b) else np.nan, "STF fastText": float(ftm.wv.similarity(a, b)) if ftm else np.nan, "Wikipédia": pt.similarity(a, b) if pt.has(a, b) else np.nan} for a, b in pares]
display(pd.DataFrame(rows).style.format({c: "{:.2f}" for c in ["STF SG", "STF fastText", "Wikipédia"]}))
for w in ["inconstitucionalidade", "desconsideração", "repercussão", "hipossuficiência", "adpf"]:
    print(f"{w:<22} fastText STF: {[x for x, _ in ftm.wv.most_similar(w, topn=6)]}")
'''),
    md("""
## 9. Frases e composição aditiva no domínio

`word2phrase` sobre o corpus do STF encontra os sintagmas que o jurista usa como unidades: *habeas_corpus*, *repercussão_geral*, *dano_moral*, *devido_processo*. E a soma de vetores: *dano + moral*, *recurso + extraordinário*.
"""),
    code('''
from replang.models.word2vec_np import learn_phrases
ph, scores = learn_phrases(jur, min_count=20, threshold=60, passes=1)
top = sorted(scores.items(), key=lambda x: -x[1])
print([p for p, _ in top[:40]])
def soma(wv, *ws, topn=6):
    v = sum(wv.unit[wv.index[w]] for w in ws); sims = wv.similarities(v)
    for w in ws: sims[wv.index[w]] = -np.inf
    return [wv.words[i] for i in np.argsort(-sims)[:topn]]
for ws in [("dano", "moral"), ("recurso", "extraordinário"), ("prisão", "preventiva"), ("união", "estável")]:
    print(f"{' + '.join(ws):<24} STF: {soma(leg, *ws)}   | Wikipédia: {soma(pt, *ws) if pt.has(*ws) else 'OOV'}")
'''),
    md(f"""
## 10. Síntese: o que o domínio muda

1. **Vocabulário e forma**: sentenças mais longas, numerais e siglas como entidades, latim; o pré-processamento genérico destrói informação jurídica.
2. **Semântica**: palavras polissêmicas deslocam-se por completo (Jaccard ≈ 0); embeddings gerais "não sabem Direito" nem com 200 M tokens de Wikipédia.
3. **Avaliação**: a lição de Hartmann et al. vale em dobro — analogias gerais são irrelevantes; nas tarefas jurídicas (NER, STS de ementas, área) o embedding **de domínio** e o **fastText** (OOV) tendem a vencer, e o TF-IDF é um rival sério porque o vocabulário técnico já é discriminativo.
4. **Aplicações** (ver `docs/juridico.md` §5): pesquisa de jurisprudência, triagem/classificação (Victor), agrupamento de recursos repetitivos (Athos), extração de entidades, sumarização/linguagem simples.

### Perguntas para discussão
* A representação ideal para busca de precedentes é a que maximiza AUC em STS — ou a que o jurista consegue *explicar*?
* Um embedding treinado só em decisões do STF serve para um Juizado Especial? (deslocamento *dentro* do domínio: tribunal, instância, ano)
* O que fazer com os números: tokens, entidades ou ruído?

## Referências
{refs("hartmann", "faruqui", "fasttext", "mikolov13b")}
- FEIJÓ, D.; MOREIRA, V. **RulingBR: A Summarization Dataset for Legal Texts.** PROPOR, 2018.
- LUZ DE ARAUJO, P. H. et al. **LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text.** PROPOR, 2018.
- VIEGAS, C. F. O.; COSTA, B. C.; ISHII, R. P. **JurisBERT: A New Approach that Converts a Classification Corpus into an STS One.** ICCSA, 2023.
- POLO, F. M. et al. **LegalNLP.** ENIAC, 2021.
"""),
]
