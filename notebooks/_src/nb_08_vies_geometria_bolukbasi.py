from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "08 · O viés que vem junto (Bolukbasi et al., 2016): geometria do gênero em embeddings"

CELLS = [
    md(f"""
# 08 · O viés que vem junto: a geometria do gênero em embeddings (Bolukbasi et al., 2016, §1–§5)
{HEADER}

**Artigo-base (leitura obrigatória):** Bolukbasi, Chang, Zou, Saligrama & Kalai (2016), *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* (NIPS).

## A tese
O mesmo mecanismo que resolve `man : king :: woman : queen` responde `man : computer programmer :: woman : homemaker` e `father : doctor :: mother : nurse`. O embedding (w2vNEWS: word2vec, 300d, 3 M palavras de notícias do Google) **codifica estereótipos como geometria** — e, usado em busca, recrutamento ou ranking, pode **amplificá-los** (exemplo do artigo: currículos de *Mary* e *John* ranqueados de forma diferente para "computer science PhD student").

## Contribuições (e onde estão neste projeto)
| Contribuição | Seção do artigo | Função em `replang.bias` |
|---|---|---|
| mostrar que o viés existe e coincide com estereótipos humanos (crowd) | §4 | `project`, `extremes` |
| gerar analogias automaticamente e medir estereótipos | §4, Ap. A, eq. (1) | `generate_analogies` |
| identificar a **direção/subespaço de gênero** por PCA de pares definicionais | §5.1, Fig. 6 | `pca_pairs`, `gender_direction`, `bias_subspace` |
| **DirectBias** | §5.2 | `direct_bias` |
| **viés indireto** β(w, v) | §5.3, Fig. 3 | `indirect_bias` |
| algoritmos de *debias* (hard/soft) | §6 | notebook 09 |
| classificar palavras específicas de gênero (SVM) | §7, Fig. 7 | `gender_classifier` |

Reproduzimos tudo com o **GloVe 6B 50d** (inglês; o w2vNEWS tem 3,6 GB) e, no notebook 10, em português.
"""),
    SETUP,
    md("""
## 1. Preliminares (§3): vetores unitários, cosseno, pares de gênero

O artigo assume $\\|\\vec w\\| = 1$ para todas as palavras, similaridade = produto interno, um conjunto $N$ de palavras **neutras** (sem gênero por definição: *flight attendant*, *shoes*) e um conjunto $P$ de **pares** de gênero (*she–he*, *mother–father*). O embedding original foi filtrado às 50 mil palavras mais frequentes, minúsculas, sem dígitos (26.377 restantes). Fazemos o mesmo.
"""),
    code('''
from replang.data.embeddings import load_glove_en
from replang.data.lexicons import EN
en_raw = load_glove_en()
keep = [w for w in en_raw.words if w.isalpha() and len(w) < 20]
en = en_raw.subset(keep, "GloVe 50d (filtrado)").normalized()
lex = EN.get()
print(en, "| normas =", np.unique(np.round(np.linalg.norm(en.vectors, axis=1), 3)))
print("pares definicionais (Fig. 5):", lex.definitional_pairs)
print(f"{len(lex.gender_specific)} palavras específicas de gênero (Ap. C) · {len(lex.professions)} profissões · {len(lex.equalize_pairs)} pares de equalização")
'''),
    md("""
## 2. Estereótipos ocupacionais (§4, Fig. 1)

Projetamos cada profissão no eixo $\\vec{she} - \\vec{he}$ e listamos os extremos. O artigo pediu a *crowd workers* que avaliassem cada ocupação (0–10 em estereotipicidade) e obteve correlação de Spearman 0,51 com a projeção; a lista inclui, no lado *she*: homemaker, nurse, receptionist, librarian, socialite, hairdresser, nanny, bookkeeper, stylist, housekeeper; no lado *he*: maestro, skipper, protege, philosopher, captain, architect, financier, warrior, broadcaster, magician.
"""),
    code('''
from replang.bias import project, extremes
she_he = en["she"] - en["he"]
profs = [p for p in lex.professions if p in en]
fem, masc = extremes(en, she_he, profs, 12)
print(f"{len(profs)} profissões no vocabulário")
print("extremo SHE:", fem); print("extremo HE: ", masc)
'''),
    code('''
from replang.viz import bias_axis_figure
from replang.bias.geometry import profile
meta = {p[0]: p for p in lex.extra["professions_meta"]}
df = profile(en, profs, she_he)
df["estereótipo (crowd, −1 fem … +1 masc)"] = [meta[w][2] for w in df.palavra]
df["definicional (crowd)"] = [meta[w][1] for w in df.palavra]
sel = pd.concat([df.nsmallest(18, "proj_g"), df.nlargest(18, "proj_g")])
bias_axis_figure(sel, title="Profissões projetadas no eixo she − he (GloVe 50d)", axis_label="← he          she →", color_col=None)
'''),
    code('''
# Correlação com os julgamentos humanos distribuídos com o repositório debiaswe (coluna 'stereotype' de professions.json)
from scipy.stats import spearmanr
neutras = df[df["definicional (crowd)"].abs() < 0.5]   # exclui profissões "definicionalmente" gendered (actress, businesswoman...)
rho = spearmanr(neutras.proj_g, -neutras["estereótipo (crowd, −1 fem … +1 masc)"]).correlation
print(f"Spearman entre projeção no eixo she−he e estereótipo julgado pela crowd: {rho:.2f}  (artigo, w2vNEWS: 0,51)")
import plotly.express as px
px.scatter(neutras, x="proj_g", y="estereótipo (crowd, −1 fem … +1 masc)", hover_name="palavra", template="plotly_white",
           title="Geometria × julgamento humano (profissões neutras por definição)", labels={"proj_g": "projeção em she − he"}).show()
'''),
    md("""
## 3. Comparando embeddings (§4, Fig. 4)

O viés não é um artefato do w2vNEWS: projetando as mesmas ocupações em outro embedding (GloVe web crawl), o artigo obteve ρ = 0,81. Comparamos o GloVe 50d com a versão **100d** do mesmo corpus? Não temos; usamos o eixo *she–he* em dois **subconjuntos de dimensões**... — não: o certo é comparar embeddings diferentes. Usamos o GloVe EN 50d × GloVe EN 50d *treinado em outro corpus*? Também indisponível offline. Mostramos então a robustez interna: eixo *she–he* × eixo *woman–man* × eixo *her–his*.
"""),
    code('''
from replang.viz import bias_scatter_two_embeddings
p1 = project(en, profs, en["she"] - en["he"]); p2 = project(en, profs, en["woman"] - en["man"]); p3 = project(en, profs, en["her"] - en["his"])
print(f"Spearman she−he × woman−man: {spearmanr(p1, p2[p1.index]).correlation:.2f} | she−he × her−his: {spearmanr(p1, p3[p1.index]).correlation:.2f}")
bias_scatter_two_embeddings(p1, p2, "eixo she − he", "eixo woman − man")
'''),
    md("""
## 4. Analogias que exibem estereótipos (§4, Ap. A, eq. 1)

Dado o par-semente $(a, b)$ = (*she*, *he*), o artigo busca pares $(x, y)$ tais que $a : x :: b : y$ é uma boa analogia:

$$S_{(a,b)}(x,y) = \\begin{cases}\\cos(\\vec a - \\vec b,\\ \\vec x - \\vec y) & \\text{se } \\|\\vec x - \\vec y\\| \\le \\delta \\\\ 0 & \\text{caso contrário}\\end{cases}$$

com $\\delta = 1$ (vetores unitários ⇒ ângulo ≤ 60°; $x$ e $y$ precisam ser *semanticamente próximos*). Diferente do "paralelogramo" clássico ($\\min \\|(a-b)-(x-y)\\|$), que gera pares pouco ligados a gênero (Fig. 9). A *crowd* julgou 150 analogias: 29 estereotipadas (*sewing–carpentry*, *nurse–surgeon*, *interior designer–architect*), 72 apropriadas (*queen–king*, *sister–brother*).
"""),
    code('''
from replang.bias import generate_analogies, expand_gender_specific
an_raw = generate_analogies(en, "she", "he", delta=1.0, topn=30, restrict=25000, exclude=lex.all_gender_words())
print("excluindo só a lista-semente (218 palavras):", list(zip(an_raw.x[:14], an_raw.y[:14])))
'''),
    md("""
Quase tudo são **nomes próprios** (*amy–barry*, *hannah–wallace*): o embedding sabe que nomes têm gênero — analogias *apropriadas*, não estereótipos, mas que escondem o resto. O artigo resolve isso no §7 generalizando a lista de 218 palavras a todo o vocabulário com um **SVM linear** (6.449 palavras específicas de gênero no w2vNEWS). Fazemos o mesmo e excluímos o conjunto expandido.
"""),
    code('''
from replang.bias import predicted_names
from replang.data.lexicons import EN_FIRST_NAMES_F, EN_FIRST_NAMES_M
expanded, svm_acc = expand_gender_specific(en, lex.gender_specific, threshold=1.0)
print(f"lista expandida pelo SVM (§7): {len(expanded)} palavras (acurácia balanceada {svm_acc:.1%})")
# Nomes próprios: um segundo classificador (nomes-semente × substantivos comuns) estima os nomes do vocabulário
non_names = profs[:120] + [w for w in en.words[200:1200] if w not in lex.all_gender_words()][:200]
names = predicted_names(en, EN_FIRST_NAMES_F + EN_FIRST_NAMES_M, non_names)
print(f"nomes próprios estimados: {len(names)} (amostra: {sorted(names)[:12]})")
exclude_all = set(expanded) | names
an = generate_analogies(en, "she", "he", delta=1.0, topn=40, restrict=25000, exclude=exclude_all)
an.head(40).T
'''),
    md("""
Leia a lista com o olhar do artigo: há pares **apropriados** (relacionados a gênero por definição, que escaparam dos filtros — alguns nomes, *marries–nobleman*), pares **estereotipados** (*lovely–magnificent*, *beauty–great*, *baby/marriage/kids* de um lado, *league/rule/offense/field* do outro, *pink–red*) e **ruído** (o GloVe 50d é pequeno e ‖x − y‖ ≤ 1 deixa passar pares vagos). O método é indiferente a isso; o julgamento humano é que separa — é o que a Fig. 2 do artigo mostra, e por isso os autores recorreram à *crowd*.
"""),
    md("""
## 5. Identificando o subespaço de gênero (§5.1, Fig. 6)

Um único par (*she–he*) é ruidoso (*man* também é "mankind", verbo *to man* ...). O artigo toma 10 pares, **centra** cada par pela média $\\mu_i$ e faz PCA das diferenças. Se o gênero fosse "ruído", a variância se espalharia (como em pares aleatórios — Fig. 6, direita); em vez disso, **uma componente domina**: a direção $g$.
"""),
    code('''
from replang.bias import pca_pairs, gender_direction, random_pca_baseline
from replang.viz import pca_variance_figure
comps, ev, used = pca_pairs(en, lex.definitional_pairs)
g = gender_direction(en, lex.definitional_pairs)
base = random_pca_baseline(en.dim, n_pairs=len(used), trials=300)
print("pares usados:", used)
print("variância explicada:", np.round(ev, 3), "| aleatório:", np.round(base[:5], 3))
print(f"cos(she, g) = {float(en['she'] @ g):+.2f}  cos(he, g) = {float(en['he'] @ g):+.2f}  cos(nurse, g) = {float(en['nurse'] @ g):+.2f}  cos(engineer, g) = {float(en['engineer'] @ g):+.2f}")
pca_variance_figure(ev, base, title="Fig. 6: variância explicada pela PCA das diferenças centradas (GloVe 50d, 10 pares) × pares aleatórios")
'''),
    md("""
> No GloVe 50d a 1ª componente explica ~48 % e a 2ª ~41 % — menos "limpo" que no w2vNEWS do artigo (onde a 1ª domina com folga). Possível razão: o par *mary–john* carrega também "nome próprio" e o par *gal–guy* é informal (e, em 50 dimensões, há menos espaço para separar os fatores). Experimente remover pares (célula abaixo) — é um bom exercício de sensibilidade.
"""),
    code('''
for drop in [None, ("mary", "john"), ("gal", "guy")]:
    pairs = [p for p in lex.definitional_pairs if p != drop]
    _, ev2, _ = pca_pairs(en, pairs)
    print(f"sem {drop}: {np.round(ev2[:3], 3)}")
'''),
    md("""
Validação dos pares (Fig. 5 do artigo): cada par deve "classificar" corretamente palavras femininas/masculinas sugeridas pela *crowd* (acurácia 75–93 %). Verificamos com as palavras específicas de gênero da lista do Apêndice C, separadas por heurística simples.
"""),
    code('''
fem_words = [w for w in lex.gender_specific if w in en and w in {"she", "her", "woman", "women", "girl", "girls", "mother", "daughter", "sister", "wife", "female", "queen", "actress", "lady", "mom", "aunt", "niece", "bride", "widow", "princess", "nun", "grandmother", "lesbian", "herself", "hers", "mrs", "ms", "mary", "feminine", "maiden", "heroine", "waitress"}]
masc_words = [w for w in lex.gender_specific if w in en and w in {"he", "his", "man", "men", "boy", "boys", "father", "son", "brother", "husband", "male", "king", "actor", "gentleman", "dad", "uncle", "nephew", "groom", "widower", "prince", "monk", "grandfather", "gay", "himself", "mr", "john", "masculine", "hero", "waiter", "lad", "guy"}]
rows = []
for a, b in lex.definitional_pairs:
    d = en[a] - en[b]
    acc = np.mean([float(en[w] @ d) > 0 for w in fem_words] + [float(en[w] @ d) < 0 for w in masc_words])
    rows.append({"par": f"{a}–{b}", "acurácia": acc})
pd.DataFrame(rows).style.format({"acurácia": "{:.0%}"})
'''),
    md("""
## 6. DirectBias (§5.2)

$$\\mathrm{DirectBias}_c = \\frac{1}{|N|}\\sum_{w \\in N} |\\cos(\\vec w, g)|^c$$

$c$ controla o rigor: $c=0$ conta qualquer sobreposição; $c=1$ é a média dos cossenos absolutos. No artigo, para as 327 profissões, $\\mathrm{DirectBias}_1 = 0{,}08$.
"""),
    code('''
from replang.bias import direct_bias
for c in [0.5, 1.0, 2.0]:
    print(f"c={c}: DirectBias = {direct_bias(en, profs, g, c):.4f}")
# comparação: palavras "aleatórias" (as 300 mais frequentes que não são de gênero) e palavras de gênero
freq_words = [w for w in en.words[:500] if w not in lex.all_gender_words()][:300]
print(f"DirectBias_1 profissões = {direct_bias(en, profs, g):.4f} | 300 palavras frequentes = {direct_bias(en, freq_words, g):.4f} | palavras específicas de gênero = {direct_bias(en, [w for w in lex.gender_specific if w in en], g):.4f}")
rng = np.random.default_rng(0)
for d in [50, 300]:
    R = rng.normal(size=(5000, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    print(f"|cos| esperado entre vetores unitários ALEATÓRIOS em {d}d: {np.abs(R[:, 0]).mean():.3f}")
'''),
    md("""
> **Cuidado com a dimensão.** Em 50 dimensões, dois vetores aleatórios têm |cos| ≈ 0,11; em 300, ≈ 0,05. Por isso o nosso DirectBias (0,18) não é comparável ao 0,08 do artigo (300d) em valor absoluto — compare sempre com o **piso aleatório** do mesmo espaço e entre grupos de palavras do mesmo embedding.
"""),
    md("""
## 7. Viés indireto β(w, v) (§5.3, Fig. 3)

Mesmo removendo *she/he* da conversa, *receptionist* fica mais perto de *softball* que de *football*. O artigo decompõe $\\vec w = \\vec w_g + \\vec w_\\perp$ e mede quanto da similaridade $\\vec w\\cdot\\vec v$ vem da componente de gênero:

$$\\beta(w, v) = \\Big(\\vec w\\cdot\\vec v - \\frac{\\vec w_\\perp\\cdot\\vec v_\\perp}{\\|\\vec w_\\perp\\|\\|\\vec v_\\perp\\|}\\Big)\\Big/\\ \\vec w\\cdot\\vec v$$

No artigo: β(softball, receptionist) = 67 %, β(softball, waitress) = 35 %, β(softball, homemaker) = 38 %, β(football, businessman) = 31 %, β(football, maestro) = 42 %; já *pitcher*/*footballer* têm β ≈ 0 (a similaridade é "legítima").
"""),
    code('''
from replang.bias import indirect_bias
axis = en["softball"] - en["football"]
soft, foot = extremes(en, axis, profs, 8)
print("mais perto de softball:", soft); print("mais perto de football:", foot)
rows = [{"w": "softball", "v": v, "cos": en.similarity("softball", v), "β": indirect_bias(en, "softball", v, g)} for v in soft[:6]]
rows += [{"w": "football", "v": v, "cos": en.similarity("football", v), "β": indirect_bias(en, "football", v, g)} for v in foot[:6]]
pd.DataFrame(rows).style.format({"cos": "{:.2f}", "β": "{:.0%}"})
'''),
    md("""
No GloVe 50d o eixo *softball–football* **não** reproduz o exemplo do artigo (os extremos são *welder*, *lifeguard*… e β ≈ 0): o viés indireto é específico de cada embedding. Em vez de escolher o par à mão, deixamos os dados falarem: entre todas as duplas de profissões com similaridade ≥ 0,3, quais têm a **maior fração da similaridade explicada por g**?
"""),
    code('''
from replang.bias import top_indirect_bias_pairs
top_indirect_bias_pairs(en, profs, g, topn=15, min_cos=0.3).style.format({"cos": "{:.2f}", "β": "{:.0%}", "proj_g(w)": "{:+.2f}", "proj_g(v)": "{:+.2f}"})
'''),
    md("""
## 8. Palavras específicas de gênero × neutras (§7, Fig. 7)

Para *debiasar* é preciso saber **o que não neutralizar**: *king*, *mother*, *beard* têm gênero por definição. O artigo parte de 218 palavras (Ap. C) e treina um **SVM linear** para generalizar aos 3 M de palavras (F ≈ 0,63; acurácia balanceada 95 %). A Fig. 7 mostra as palavras em dois eixos: projeção em *she–he* (x) e distância à fronteira do SVM (y).
"""),
    code('''
from replang.bias import gender_classifier
neutral_pool = [w for w in en.words[:8000] if w not in lex.all_gender_words()]
clf, acc = gender_classifier(en, [w for w in lex.gender_specific if w in en], neutral_pool[:3000])
print(f"acurácia balanceada (validação cruzada): {acc:.2%}")
cands = [w for w in en.words[:20000] if w not in lex.all_gender_words()]
scores = clf.decision_function(en.matrix(cands))
pred_specific = [cands[i] for i in np.argsort(-scores)[:30]]
print("palavras fora da lista que o SVM considera específicas de gênero:", pred_specific)
'''),
    code('''
words_plot = [w for w in lex.gender_specific if w in en][:80] + profs[:80] + ["beard", "uterus", "pregnant", "mustache", "bikini", "tuxedo", "dress", "cosmetics"]
words_plot = [w for w in dict.fromkeys(words_plot) if w in en]
X = en.matrix(words_plot)
fig_df = pd.DataFrame({"palavra": words_plot, "proj she−he": X @ ((en["she"] - en["he"]) / np.linalg.norm(en["she"] - en["he"])), "SVM (específica ↑ / neutra ↓)": clf.decision_function(X),
                       "grupo": ["específica (lista)" if w in lex.gender_specific else "profissão/neutra" for w in words_plot]})
px.scatter(fig_df, x="proj she−he", y="SVM (específica ↑ / neutra ↓)", color="grupo", text="palavra", template="plotly_white", title="Fig. 7: eixo de gênero × fronteira específica/neutra", height=650).update_traces(textposition="top center", textfont_size=9).show()
'''),
    md(f"""
## 9. Síntese e pontos para a apresentação

1. O viés **não é um bug do word2vec**: é a hipótese distribucional funcionando sobre um corpus que reflete a sociedade (e o *reporting bias*: *male nurse* é dito, *female nurse* não).
2. Ele é **mensurável** (DirectBias, β) porque é **geométrico** (direção $g$).
3. Está em **todos** os embeddings estáticos testados e **concorda com estereótipos humanos** (ρ = 0,51–0,81).
4. Afeta também pares de palavras neutras (viés indireto) — não basta tirar *he/she*.
5. A geometria que causa o problema é a mesma que permite a solução (notebook 09).

### Perguntas para discussão
* "O embedding só reflete o mundo": é argumento para não corrigir? (§9: *we recommend erring on the side of neutrality*)
* Quais outros subespaços existem? (§9: *minorities − whites* → *parliamentarian, lawyer* × *butler, footballer, crooner*)
* O que muda em línguas com gênero gramatical? (notebook 10)

## Referências
{refs("bolukbasi", "mikolov13a", "glove")}
"""),
]
