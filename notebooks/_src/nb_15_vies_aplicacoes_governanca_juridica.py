from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "15 · Contexto jurídico II: o viés que vem junto no Direito, aplicações e governança no Brasil"

CELLS = [
    md(f"""
# 15 · Contexto jurídico II: o viés que vem junto no Direito, aplicações e governança
{HEADER}

O notebook 14 mostrou **o que o vetor jurídico carrega de conhecimento** (vocabulário técnico, relações entre papéis, similaridade entre ementas). Este mostra **o que mais ele carrega** — padrões sociais do Judiciário que o produziu — e discute o que fazer com isso quando o uso é jurídico: triagem, recomendação de precedentes, avaliação de risco.

Roteiro:
1. direção de gênero no corpus do STF (método de Bolukbasi et al., notebooks 08–10) e o problema do gênero gramatical nos **papéis processuais**;
2. termos jurídicos neutros no eixo de gênero: crimes, família, trabalho — e a comparação com a Wikipédia;
3. viés indireto entre termos jurídicos;
4. um experimento de **amplificação**: um "ranqueador" de ementas que usa embeddings prefere textos com marcas de um gênero?
5. *debias* e o que ele não resolve (Gonen & Goldberg) — e por que, para auditoria, pode ser melhor **não** corrigir;
6. aplicações, regulação brasileira (CNJ 332/2020 e 615/2025, LGPD, PL 2338/2023) e debates para a sala.

> **Aviso metodológico.** O corpus (RulingBR) tem 6 M tokens de decisões do STF 2011–2018; os embeddings são pequenos (100d). Os números abaixo ilustram o *método*; conclusões sobre o Judiciário exigiriam corpora maiores, controle por tribunal/ano e análise qualitativa.
"""),
    SETUP,
    code('''
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.data.lexicons import PT
from replang.data.legal import LEGAL
from replang.bias import gender_direction, pca_pairs, random_pca_baseline, project, extremes, direct_bias, indirect_bias, top_indirect_bias_pairs, generate_analogies, split_analogies_pt, hard_debias, pair_bias
from replang.bias.geometry import profile
from replang.viz import pca_variance_figure, bias_axis_figure
lex = PT.get()
pt = load_ptwiki().normalized("Wikipédia"); leg = load_local_model("legal_sg100").normalized("STF")
pairs_leg = [p for p in lex.definitional_pairs if leg.has(*p)]
print(leg, "| pares definicionais presentes no STF:", pairs_leg)
'''),
    md("""
## 1. A direção de gênero existe no corpus jurídico?

Os pares definicionais (*mulher/homem, ela/ele, mãe/pai…*) aparecem em decisões (direito de família, previdenciário, penal). Repetimos a PCA das diferenças (Fig. 6 de Bolukbasi et al.).
"""),
    code('''
comps, ev, used = pca_pairs(leg, pairs_leg)
g_leg = gender_direction(leg, pairs_leg); g_pt = gender_direction(pt, lex.definitional_pairs)
print(f"STF: variância explicada {np.round(ev[:4], 3)} (aleatório {random_pca_baseline(leg.dim, len(used))[0]:.3f}) · cos(ela,g)={float(leg['ela'] @ g_leg):+.2f} cos(ele,g)={float(leg['ele'] @ g_leg):+.2f}")
pca_variance_figure(ev, random_pca_baseline(leg.dim, len(used)), title="STF (RulingBR): variância explicada pela PCA das diferenças dos pares de gênero")
'''),
    md("""
> **Leitura.** No corpus do STF a 1ª componente explica pouco mais que o piso aleatório: a direção de gênero existe, mas é **fraca e inclinada** — *ele* projeta forte (−0,27) e *ela* fraco (+0,11). Faz sentido: o texto jurídico usa o **masculino genérico** (*o réu, o autor, o servidor*) como forma não marcada, e pronomes pessoais contrastivos são raros em acórdãos. Consequência prática: no domínio, a direção obtida com pares "gerais" é menos confiável; pares de **papéis processuais** (*autora/autor, ré/réu*) são sementes melhores — e é o que o próximo painel explora. Todas as projeções abaixo devem ser lidas **relativamente** (um papel em relação aos outros), não em valor absoluto.
"""),
    md("""
## 2. Papéis processuais: separação gramatical × deslocamento do centro

Como no notebook 10, para cada par de papéis (*juíza/juiz, ré/réu, autora/autor, advogada/advogado…*) medimos a **separação** (marca morfológica: esperada) e o **centro** (para onde o *conceito* pende). Papéis **epicenos** (*recorrente, agravante, contribuinte, vítima, testemunha*) só têm centro — se pendem, é estereótipo ou estatística do corpus (quem é vítima, quem é testemunha…).
"""),
    code('''
rows = []
for f, m in LEGAL.gendered_roles:
    if f == m:
        if f in leg:
            rows.append({"papel": f, "tipo": "epiceno", "proj. fem.": np.nan, "proj. masc.": np.nan, "separação": np.nan, "centro": float(leg[f] @ g_leg)})
    elif leg.has(f, m):
        pf, pm = float(leg[f] @ g_leg), float(leg[m] @ g_leg)
        rows.append({"papel": f"{f}/{m}", "tipo": "par morfológico", "proj. fem.": pf, "proj. masc.": pm, "separação": pf - pm, "centro": (pf + pm) / 2})
roles = pd.DataFrame(rows).sort_values("centro")
roles.style.format({c: "{:+.3f}" for c in ["proj. fem.", "proj. masc.", "separação", "centro"]}, na_rep="—").background_gradient(subset=["centro"], cmap="RdBu_r")
'''),
    md("""
Leia com cuidado (e relativamente — a direção é inclinada para *ele*, então quase todos os centros são negativos): *ré/réu*, *delegada/delegado* e *acusada/acusado* têm separação morfológica clara; *promotora/promotor* e *procuradora/procurador* quase nenhuma (formas raras no corpus). O **centro** reflete a frequência com que cada papel aparece em contextos "femininos" (direito de família, previdenciário, violência doméstica) ou "masculinos" (penal). Isso **não** é um estereótipo do modelo: é a estatística do que chega ao STF — que, por sua vez, reflete a sociedade. O ponto é que um sistema treinado nisso **herda** a estatística como se fosse semântica.
"""),
    md("""
## 3. Termos jurídicos neutros no eixo de gênero: STF × Wikipédia

Projetamos substantivos jurídicos sem gênero semântico (*crime, furto, roubo, tráfico, pensão, guarda, alimentos, salário, assédio, aposentadoria, licitação…*) na direção de gênero de cada embedding. Quais pendem para *ela*, quais para *ele*, e os dois corpora concordam?
"""),
    code('''
neutros = [w for w in LEGAL.neutral_terms if w in leg and w in pt]
p_leg = project(leg, neutros, g_leg); p_pt = project(pt, neutros, g_pt)
from scipy.stats import spearmanr
print(f"{len(neutros)} termos · Spearman entre as projeções STF × Wikipédia: {spearmanr(p_leg, p_pt[p_leg.index]).correlation:.2f}")
print("STF — lado ELA:", list(p_leg.index[-10:][::-1])); print("STF — lado ELE:", list(p_leg.index[:10]))
print("Wikipédia — lado ELA:", list(p_pt.sort_values().index[-10:][::-1])); print("Wikipédia — lado ELE:", list(p_pt.sort_values().index[:10]))
print(f"DirectBias_1: STF = {direct_bias(leg, neutros, g_leg):.3f} | Wikipédia = {direct_bias(pt, neutros, g_pt):.3f}  (piso aleatório em 100d ≈ 0.08)")
'''),
    code('''
from replang.viz import bias_scatter_two_embeddings
bias_scatter_two_embeddings(p_leg, p_pt, "STF (direção de gênero)", "Wikipédia (direção de gênero)")
'''),
    code('''
labels = {"crime e pena": ["crime", "violência", "homicídio", "furto", "roubo", "tráfico", "estupro", "lesão", "ameaça", "prisão", "fiança", "pena", "multa"],
          "família": ["família", "guarda", "pensão", "alimentos", "divórcio", "casamento", "maternidade", "paternidade", "filiação"],
          "trabalho e previdência": ["trabalho", "salário", "assédio", "demissão", "aposentadoria", "benefício", "auxílio"]}
df = profile(leg, neutros, g_leg, labels)
bias_axis_figure(df, title="Termos jurídicos projetados na direção de gênero (STF)", axis_label="← ele          ela →")
'''),
    md("""
## 4. Viés indireto entre termos jurídicos

Quais pares de termos neutros têm similaridade explicada pela direção de gênero (β de Bolukbasi, §5.3)? É o mecanismo pelo qual um sistema de recomendação pode aproximar *pensão* de *vulnerabilidade* "por causa do gênero", mesmo sem nenhuma palavra de gênero na consulta.
"""),
    code('''
top_indirect_bias_pairs(leg, neutros, g_leg, topn=15, min_cos=0.25).style.format({"cos": "{:.2f}", "β": "{:.0%}", "proj_g(w)": "{:+.2f}", "proj_g(v)": "{:+.2f}"})
'''),
    md("""
## 5. As "direções de papel" são consistentes entre si?

A geração automática de analogias (eq. 1 de Bolukbasi) exige uma direção de gênero nítida; com a direção fraca do corpus do STF ela devolve ruído (tente: `generate_analogies(leg, "ela", "ele", ...)`). Fazemos a pergunta de forma mais direta: os deslocamentos *autora→autor*, *ré→réu*, *juíza→juiz*, *advogada→advogado* são **paralelos** (uma única "direção feminino→masculino", como no notebook 02) ou cada papel tem a sua? A matriz de cossenos entre os vetores-diferença responde.
"""),
    code('''
import plotly.express as px
pairs_roles = [(f, m) for f, m in LEGAL.gendered_roles if f != m and leg.has(f, m)]
D = np.stack([leg[f] - leg[m] for f, m in pairs_roles]); D /= np.linalg.norm(D, axis=1, keepdims=True)
S = D @ D.T
names = [f"{f}/{m}" for f, m in pairs_roles]
px.imshow(S, x=names, y=names, zmin=-1, zmax=1, color_continuous_scale="RdBu", text_auto=".2f", template="plotly_white", height=620,
          title="cos entre os vetores (feminino − masculino) dos papéis processuais no STF").show()
off = S[np.triu_indices(len(names), 1)]
print(f"cosseno médio entre direções de papel: {off.mean():.2f} (pares de gênero gerais na Wikipédia costumam ficar acima de 0,5)")
'''),
    md("""
Cossenos altos entre pares = uma direção de gênero compartilhada (como *rei/rainha*, *homem/mulher* na Wikipédia); cossenos baixos = cada papel codifica o gênero de forma própria, misturada ao seu contexto típico (*ré* ↔ violência doméstica, *autora* ↔ previdenciário). No corpus jurídico predomina o segundo caso — mais uma razão para medir viés **por tarefa** (seção 6) em vez de confiar numa única direção global.
"""),
    md("""
## 6. Experimento de amplificação: um ranqueador de ementas "vê" gênero?

Simulamos um sistema simples de **recomendação de precedentes**: dada uma consulta, ranqueia ementas pelo cosseno entre a média de vetores. Comparamos o ranking obtido para consultas gêmeas que diferem apenas pela marca de gênero do papel (*"a autora pleiteia…"* vs *"o autor pleiteia…"*). Se as listas divergirem muito, a marca de gênero está influenciando a recuperação — o cenário de *Mary × John* do artigo, agora em petições.
"""),
    code('''
from replang.data.legal import load_rulingbr_ementas
from replang.eval.extrinsic import sentence_vector
em = load_rulingbr_ementas(3000)
E = np.stack([sentence_vector(leg, e, mode="mean") for e in em.ementa]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
def ranking(consulta, k=10):
    q = sentence_vector(leg, consulta, mode="mean"); q /= np.linalg.norm(q) + 1e-9
    return list(np.argsort(-(E @ q))[:k])
pares_consulta = [
    ("a autora pleiteia pensão por morte do companheiro", "o autor pleiteia pensão por morte da companheira"),
    ("a ré foi condenada por tráfico de drogas e requer habeas corpus", "o réu foi condenado por tráfico de drogas e requer habeas corpus"),
    ("a servidora requer a revisão de sua aposentadoria", "o servidor requer a revisão de sua aposentadoria"),
    ("a trabalhadora alega assédio e pede indenização por dano moral", "o trabalhador alega assédio e pede indenização por dano moral"),
    ("a juíza indeferiu a liminar requerida pela defesa", "o juiz indeferiu a liminar requerida pela defesa"),
]
rows = []
for a, b in pares_consulta:
    ra, rb = ranking(a, 10), ranking(b, 10)
    inter = len(set(ra) & set(rb))
    rows.append({"consulta (feminino)": a, "consulta (masculino)": b, "top-10 em comum": inter, "1º resultado igual?": ra[0] == rb[0],
                 "área do 1º (fem.)": em.area[ra[0]], "área do 1º (masc.)": em.area[rb[0]]})
pd.DataFrame(rows)
'''),
    code('''
# Quanto da diferença entre as consultas gêmeas está na direção de gênero? (projeção da diferença dos vetores de consulta em g)
for a, b in pares_consulta[:3]:
    qa = sentence_vector(leg, a, mode="mean"); qb = sentence_vector(leg, b, mode="mean")
    d = qa - qb
    print(f"{a[:45]:<47} ‖Δ‖={np.linalg.norm(d):.3f}  fração de Δ ao longo de g = {abs(float(d @ g_leg)) / (np.linalg.norm(d) + 1e-9):.0%}")
'''),
    md("""
As consultas gêmeas diferem só pela forma do papel (*autora/autor*), mas o *top-10* muda em 3–5 posições e, em alguns casos, até o **1º resultado** e a **área** do precedente recomendado. Só uma parte dessa diferença (10–35 %) está na direção de gênero *g*; o resto é a vizinhança própria de cada forma (*ré* coocorre com violência doméstica, *réu* com tráfico…) — a estatística do corpus virou critério de relevância. Um *debias* das consultas (projetar fora de *g*) tornaria os rankings idênticos; mas isso também apagaria informação legítima quando o gênero **é** juridicamente relevante (Lei Maria da Penha, licença-maternidade). Não há resposta técnica única: é decisão de projeto, documentada.
"""),
    md("""
## 7. Debias no jurídico: o que resolve e o que não resolve

Aplicamos o hard debias ao modelo do STF (equalizando os pares de papéis) e verificamos: (a) DirectBias nos termos neutros; (b) os rankings gêmeos; (c) o teste de Gonen & Goldberg — um classificador ainda recupera o "gênero" dos termos neutros a partir da vizinhança?
"""),
    code('''
B = g_leg[None, :]
eq_pairs = [p for p in lex.equalize_pairs + [(f, m) for f, m in LEGAL.gendered_roles if f != m] if leg.has(*p)]
leg_deb = hard_debias(leg, B, gender_specific=lex.gender_specific, equalize_pairs=eq_pairs, name="STF hard-debiased")
print(f"DirectBias termos neutros: antes {direct_bias(leg, neutros, g_leg):.3f} → depois {direct_bias(leg_deb, neutros, g_leg):.4f}")
print(f"PairBias: antes {pair_bias(leg, neutros, pairs_leg):.3f} → depois {pair_bias(leg_deb, neutros, pairs_leg):.1e}")
E2 = np.stack([sentence_vector(leg_deb, e, mode="mean") for e in em.ementa]); E2 /= np.linalg.norm(E2, axis=1, keepdims=True) + 1e-9
def ranking2(consulta, k=10):
    q = sentence_vector(leg_deb, consulta, mode="mean"); q /= np.linalg.norm(q) + 1e-9
    return list(np.argsort(-(E2 @ q))[:k])
for a, b in pares_consulta[:3]:
    print(f"depois do debias — top-10 em comum: {len(set(ranking2(a)) & set(ranking2(b)))}/10   ({a[:40]}…)")
'''),
    code('''
# Teste "Lipstick on a Pig": agrupar termos neutros por gênero ANTES do debias e ver se um k-NN sobre o embedding DEPOIS ainda recupera os grupos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
lab = (p_leg > p_leg.median()).astype(int)   # rótulo: metade "feminina" × metade "masculina" segundo o embedding ORIGINAL
X_before = leg.matrix(lab.index); X_after = leg_deb.matrix(lab.index)
for name, X in [("antes", X_before), ("depois (hard debias)", X_after)]:
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, lab.values, cv=5).mean()
    print(f"k-NN recupera o 'lado' de gênero dos termos neutros — {name}: {acc:.0%} (acaso = 50%)")
'''),
    md("""
Dois resultados sóbrios. (i) O hard debias zera o DirectBias e o PairBias, mas os rankings gêmeos melhoram **pouco** (7–8 em 10): as ementas também contêm marcas de gênero e a diferença entre *autora* e *autor* não está só em *g*. Igualar os rankings exigiria neutralizar consultas **e** documentos — apagando informação juridicamente relevante (Lei Maria da Penha, licença-maternidade). (ii) Como em Gonen & Goldberg (2019): a projeção em *g* vai a zero, mas os termos que estavam "do lado feminino" continuam **próximos entre si** (o k-NN ainda acerta acima do acaso) — a estrutura de agrupamento sobrevive. Para um sistema de decisão, o viés foi escondido, não removido; para uma **auditoria**, o embedding original é mais útil que o corrigido.
"""),
    md("""
## 8. Aplicações, governança e debates

### Onde as representações já são usadas no Judiciário brasileiro (exemplos públicos)
| Sistema | Órgão | Tarefa | Representação subjacente |
|---|---|---|---|
| Victor | STF | classificar peças e temas de repercussão geral | texto → vetores → redes neurais |
| Athos | STJ | agrupar recursos por similaridade (demandas repetitivas) | similaridade entre documentos |
| Sinapses | CNJ / TJRO | plataforma para treinar e distribuir modelos | — |
| Elis | TJPE | triagem de execuções fiscais | classificação de petições |
| Radar | TJMG | identificar demandas repetitivas | similaridade |

### O arcabouço normativo (resumo — verificar o estágio atual)
* **CF/88**: fundamentação (art. 93, IX); devido processo, contraditório e ampla defesa (art. 5º, LIV–LV); igualdade.
* **LGPD** (Lei 13.709/2018), art. 20: revisão de decisões automatizadas; transparência sobre critérios.
* **Resolução CNJ 332/2020**: não discriminação, supervisão humana, auditabilidade, publicidade; art. 23 desestimula IA em matéria penal, sobretudo decisões preditivas. **Resolução CNJ 615/2025**: atualização com classificação de risco e regras para IA generativa.
* **PL 2338/2023** (Marco Legal da IA): administração da justiça como alto risco; em tramitação.

### Debates (ver `docs/juridico.md`, §7)
1. Embeddings para **auditoria** (preservar o viés para medi-lo) × embeddings para **decisão** (mitigar e avaliar na tarefa).
2. Uso *instrumental* (busca, extração) × uso *decisório* (previsão de resultado, risco) — onde passa a linha do art. 23?
3. Explicabilidade: o magistrado fundamenta, mas a sugestão ancora. Que transparência mínima um sistema de recomendação de precedentes deve oferecer?
4. Gênero gramatical: preservar (geração de minutas) × neutralizar (ranqueamento). Quem decide, e como se documenta?
5. Responsabilidade: fornecedor, tribunal, magistrado, advogado.

### Síntese
* A geometria do viés de Bolukbasi et al. **transfere** para o domínio jurídico e dá instrumentos de **auditoria** (direção, DirectBias, β, rankings gêmeos).
* No Direito, "viés" mistura três coisas: gênero gramatical (necessário), estatística do que chega aos tribunais (informativa, mas perigosa como critério) e estereótipo (indesejável). As métricas ajudam a separá-las; a decisão sobre o que fazer é normativa.
* O *debias* geométrico é insuficiente para sistemas de decisão; a resposta institucional é governança: documentação, avaliação na tarefa com métricas de equidade, supervisão humana e transparência — exatamente o que a regulação brasileira exige.
"""),
    md(f"""
## Referências
{refs("bolukbasi", "hartmann")}
- GONEN, H.; GOLDBERG, Y. **Lipstick on a Pig.** NAACL, 2019.
- ANGWIN, J. et al. **Machine Bias.** ProPublica, 2016.
- CNJ. **Resolução nº 332/2020**; **Resolução nº 615/2025**. BRASIL. **Lei nº 13.709/2018**; **PL 2338/2023**.
- FEIJÓ, D.; MOREIRA, V. **RulingBR.** PROPOR, 2018.
"""),
]
