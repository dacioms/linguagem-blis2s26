from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "09 · Removendo o viés (Bolukbasi et al., 2016, §6–§8): Neutralize, Equalize, Soft debias e o que se preserva"

CELLS = [
    md(f"""
# 09 · Removendo o viés: Neutralize, Equalize e Soft debias (Bolukbasi et al., 2016, §6–§8)
{HEADER}

## Objetivos do *debias* (§1)
1. **Reduzir o viés**: (a) palavras neutras equidistantes dos pares de gênero (*nurse* igualmente perto de *he* e *she*); (b) reduzir associações de gênero entre palavras neutras (viés indireto).
2. **Manter a utilidade**: (a) preservar associações não ligadas a gênero (moda, futebol); (b) preservar o gênero definicional (*man–father*, *she:grandmother :: he:grandfather*).

## Os algoritmos
* **Passo 1 — Identificar o subespaço de viés** $B$ (notebook 08): $k$ primeiras componentes da PCA das diferenças centradas dos pares definicionais ($k = 1$: a direção $g$).
* **Passo 2a — Hard de-biasing** = *Neutralize* + *Equalize*.
* **Passo 2b — Soft bias correction**: transformação linear $T$ que preserva produtos internos enquanto reduz a projeção das neutras em $B$, com compromisso $\\lambda$.

Implementações: `replang.bias.debias.hard_debias`, `soft_debias`, `pair_bias`.
"""),
    SETUP,
    code('''
from replang.data.embeddings import load_glove_en
from replang.data.lexicons import EN
from replang.bias import gender_direction, direct_bias, indirect_bias, pair_bias, hard_debias, soft_debias, project, extremes
lex = EN.get()
en = load_glove_en().subset([w for w in load_glove_en().words if w.isalpha() and len(w) < 20], "GloVe 50d").normalized()
g = gender_direction(en, lex.definitional_pairs)
B = g[None, :]
profs = [p for p in lex.professions if p in en]
neutral = [w for w in en.words if w not in lex.all_gender_words()]
print(en, "| direção g pronta | profissões:", len(profs))
'''),
    md("""
## 1. Neutralize (§6, passo 2a)

Para cada palavra neutra $w \\in N$: remover a projeção no subespaço e renormalizar,

$$\\vec w := \\frac{\\vec w - \\vec w_B}{\\|\\vec w - \\vec w_B\\|}, \\qquad \\vec w_B = \\sum_{j=1}^{k} (\\vec w\\cdot b_j)\\, b_j .$$

É uma **projeção ortogonal**: tudo o que $w$ "sabia" fora de $B$ é mantido; a coordenada de gênero vira zero. O conjunto $N$ é definido como complemento das palavras específicas de gênero ($N = W \\setminus S$, §7).
"""),
    code('''
def neutralize(v, B):
    v_B = (v @ B.T) @ B
    out = v - v_B
    return out / np.linalg.norm(out)
nurse = en["nurse"]
print(f"cos(nurse, g) antes = {float(nurse @ g):+.3f}  →  depois = {float(neutralize(nurse, B) @ g):+.3f}")
print("vizinhos de nurse antes :", [w for w, _ in en.most_similar("nurse", topn=8)])
'''),
    md("""
## 2. Equalize (§6, passo 2a)

Para cada conjunto de igualdade $E$ (ex.: {grandmother, grandfather}, {guy, gal}): as palavras ficam **simétricas** em torno de um centro $\\nu$ fora de $B$, mantendo norma 1:

$$\\mu := \\frac{1}{|E|}\\sum_{w\\in E}\\vec w,\\qquad \\nu := \\mu - \\mu_B,\\qquad \\vec w := \\nu + \\sqrt{1 - \\|\\nu\\|^2}\;\\frac{\\vec w_B - \\mu_B}{\\|\\vec w_B - \\mu_B\\|}.$$

Fora de $B$, as duas palavras ficam **iguais** ($\\nu$); dentro de $B$, ficam **centradas** (opostas) e reescaladas para que $\\|\\vec w\\| = 1$. Por que centrar? Porque no embedding original *female* tem projeção de gênero maior que *male* (ambas positivas!) — sem centrar, *male* e *female* ficariam idênticas e perderíamos analogias como *father:male :: mother:female*.

**Observação 1 do artigo**: após Neutralize + Equalize, para toda palavra neutra $w$ e todo par $(e_1, e_2)$ de um conjunto de igualdade, $\\vec w\\cdot\\vec e_1 = \\vec w\\cdot\\vec e_2$ e $\\|\\vec w - \\vec e_1\\| = \\|\\vec w - \\vec e_2\\|$; logo $\\mathrm{PairBias} = 0$.
"""),
    code('''
import inspect
from replang.bias import debias
src = inspect.getsource(debias.hard_debias)
print(src[src.index("# Neutralize"):src.index("out.invalidate()")])
'''),
    code('''
EQ = list(dict.fromkeys(lex.equalize_pairs + lex.definitional_pairs))   # conjuntos de igualdade E (inclui os pares definicionais)
hard = hard_debias(en, B, gender_specific=lex.gender_specific, equalize_pairs=EQ, name="GloVe 50d hard-debiased")
# Verificação da Observação 1
checks = []
for w in ["nurse", "engineer", "receptionist", "homemaker", "programmer"]:
    for a, b in [("she", "he"), ("woman", "man"), ("grandmother", "grandfather")]:
        checks.append({"w": w, "par": f"{a}/{b}", "antes |cos(w,a)−cos(w,b)|": abs(en.similarity(w, a) - en.similarity(w, b)), "depois": abs(hard.similarity(w, a) - hard.similarity(w, b))})
pd.DataFrame(checks).style.format({"antes |cos(w,a)−cos(w,b)|": "{:.3f}", "depois": "{:.1e}"})
'''),
    code('''
print(f"DirectBias_1 (profissões): antes {direct_bias(en, profs, g):.4f} → depois {direct_bias(hard, profs, g):.5f}")
print(f"PairBias (200 neutras × pares definicionais): antes {pair_bias(en, neutral[:200], lex.definitional_pairs):.4f} → depois {pair_bias(hard, neutral[:200], lex.definitional_pairs):.1e}")
print(f"gênero definicional preservado: cos(she,g)={float(hard['she'] @ g):+.2f} cos(he,g)={float(hard['he'] @ g):+.2f} | cos(king,queen)={hard.similarity('king','queen'):.2f} (antes {en.similarity('king','queen'):.2f})")
'''),
    md("""
### 2.1 Geometria do Equalize em 2D

Visualizamos *grandmother*/*grandfather* e *gal*/*guy* no plano formado por $g$ e uma direção ortogonal: antes (assimétricos) e depois (simétricos em torno de $\\nu$).
"""),
    code('''
import plotly.graph_objects as go
def plane_coords(wv, words, g):
    X = wv.matrix(words)
    x = X @ g
    resid = X - np.outer(x, g)
    u = resid.mean(0); u -= (u @ g) * g; u /= np.linalg.norm(u)
    return x, resid @ u
fig = go.Figure()
for name, wv, sym in [("antes", en, "circle"), ("depois (hard)", hard, "diamond")]:
    ws = [w for w in ["grandmother", "grandfather", "gal", "guy", "babysitter", "wisdom", "nurse"] if w in wv]
    x, y = plane_coords(wv, ws, g)
    fig.add_scatter(x=x, y=y, mode="markers+text", text=ws, name=name, textposition="top center", marker=dict(size=11, symbol=sym))
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(template="plotly_white", title="Equalize: pares ficam simétricos em g; neutras (babysit, wisdom) vão para g = 0", xaxis_title="componente em g (gênero)", yaxis_title="componente ortogonal (média)")
fig.show()
'''),
    md("""
## 3. Soft bias correction (§6, passo 2b; Apêndice B)

Em vez de projetar, aprende-se uma transformação linear $T \\in \\mathbb{R}^{d\\times d}$:

$$\\min_T\\ \\|(TW)^\\top(TW) - W^\\top W\\|_F^2 + \\lambda\\,\\|(TN)^\\top(TB)\\|_F^2$$

O primeiro termo preserva **todos** os produtos internos (a "utilidade"); o segundo empurra a projeção das neutras $N$ no subespaço $B$ para zero. $\\lambda \\to \\infty$ recupera o Neutralize; $\\lambda = 0{,}2$ no artigo. O artigo resolve como um **programa semidefinido** (com $X = T^\\top T$, e a SVD de $W$ para reduzir a $d\\times d$). Nossa versão didática otimiza $T$ por gradiente — mesma função objetivo, solução aproximada.
"""),
    code('''
import time
rows, hists = [], {}
for lam in [0.05, 0.2, 1.0, 5.0]:
    t = time.time()
    soft, T, hist = soft_debias(en, B, neutral_words=profs, lam=lam, steps=200, lr=0.05, sample=2500, seed=0)
    hists[lam] = hist
    rows.append({"λ": lam, "DirectBias (profissões)": direct_bias(soft, profs, g), "cos(king,queen)": soft.similarity("king", "queen"), "cos(nurse,doctor)": soft.similarity("nurse", "doctor"),
                 "desvio médio dos produtos internos": hist[-1]["preservação"] ** 0.5, "s": round(time.time() - t)})
soft_df = pd.DataFrame(rows)
soft_df.style.format({"DirectBias (profissões)": "{:.4f}", "cos(king,queen)": "{:.3f}", "cos(nurse,doctor)": "{:.3f}", "desvio médio dos produtos internos": "{:.4f}"})
'''),
    code('''
import plotly.express as px
h = pd.concat([pd.DataFrame(v).assign(λ=str(k)) for k, v in hists.items()])
fig = px.line(h, x="step", y="viés", color="λ", log_y=True, template="plotly_white", title="Soft debias: termo de viés ao longo da otimização, para vários λ")
fig.show()
'''),
    md("""
## 4. A utilidade é preservada? (§8, Tabela 1 e Fig. 8)

O artigo mede, antes/depois: RG-65 e WS-353 (similaridade) e MSR-analogy — praticamente inalterados (62,3/54,5/57,0 → 62,4/54,1/57,0 hard). E, com a *crowd*: analogias estereotipadas caem de 19 % para 6 % (hard), com o número de analogias apropriadas mantido (Fig. 8). Reproduzimos com o conjunto `questions-words` (restrito, para ser rápido) e com os vizinhos.
"""),
    code('''
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
qs = load_analogies("en")
rows = []
for name, wv in [("original", en), ("hard-debiased", hard), ("soft λ=0.2", soft_debias(en, B, neutral_words=profs, lam=0.2, steps=200, lr=0.05, sample=2500)[0])]:
    r = evaluate_analogies(wv, qs, restrict=30000, max_per_category=300 if FAST else 600).set_index("categoria")
    rows.append({"embedding": name, "semântica": r.loc["TOTAL semântica", "acurácia"], "sintática": r.loc["TOTAL sintática", "acurácia"], "total": r.loc["TOTAL", "acurácia"], "family": r.loc["family", "acurácia"]})
pd.DataFrame(rows).style.format({"semântica": "{:.1%}", "sintática": "{:.1%}", "total": "{:.1%}", "family": "{:.1%}"})
'''),
    code('''
for w in ["nurse", "programmer", "receptionist", "football", "wisdom"]:
    print(f"{w:<13} antes : {[x for x, _ in en.most_similar(w, topn=7)]}")
    print(f"{'':<13} depois: {[x for x, _ in hard.most_similar(w, topn=7)]}")
'''),
    code('''
# As analogias do artigo: he:doctor :: she:? e she:ovarian_cancer :: he:?
for a, b, c in [("he", "doctor", "she"), ("man", "programmer", "woman"), ("father", "doctor", "mother"), ("she", "nurse", "he")]:
    print(f"{a}:{b} :: {c}:?   antes → {[x for x, _ in en.analogy(a, b, c, topn=4, exclude_words=lex.gender_specific)]}   depois → {[x for x, _ in hard.analogy(a, b, c, topn=4, exclude_words=lex.gender_specific)]}")
print("preservadas (gênero definicional): she:queen :: he:?", [x for x, _ in hard.analogy("she", "queen", "he", topn=3)], "| she:sister :: he:?", [x for x, _ in hard.analogy("she", "sister", "he", topn=3)])
'''),
    md("""
## 5. Viés indireto após o hard debias (§8)

O artigo repete o experimento *softball–football*: *receptionist*, *waitress*, *homemaker* saem do topo; *pitcher* e *footballer* continuam (similaridade legítima).
"""),
    code('''
axis_before = en["softball"] - en["football"]; axis_after = hard["softball"] - hard["football"]
print("softball antes :", extremes(en, axis_before, profs, 6)[0]); print("softball depois:", extremes(hard, axis_after, profs, 6)[0])
print("football antes :", extremes(en, axis_before, profs, 6)[1]); print("football depois:", extremes(hard, axis_after, profs, 6)[1])
'''),
    md("""
## 6. Geração de analogias antes × depois (Fig. 8 do artigo, sem a *crowd*)

Sem avaliadores humanos, contamos quantas das analogias geradas envolvem **profissões** (um proxy grosseiro de "estereótipo ocupacional").
"""),
    code('''
from replang.bias import generate_analogies, expand_gender_specific
expanded, _ = expand_gender_specific(en, lex.gender_specific)   # §7: nomes próprios etc. (ver notebook 08)
prof_set = set(profs)
for name, wv in [("antes", en), ("depois", hard)]:
    an = generate_analogies(wv, "she", "he", delta=1.0, topn=100, restrict=25000, exclude=expanded)
    n_prof = int((an.x.isin(prof_set) | an.y.isin(prof_set)).sum())
    print(f"{name}: {len(an)} analogias geradas; {n_prof} envolvem profissões; exemplos: {list(zip(an.x[:8], an.y[:8]))}")
'''),
    md(f"""
## 7. Limitações e a discussão posterior

* **Gonen & Goldberg (2019), "Lipstick on a Pig"**: o hard debias zera a projeção em $g$, mas as palavras "femininas" continuam **agrupadas entre si** (a estrutura relativa sobrevive); um classificador recupera o gênero das neutras com alta acurácia. O viés foi *escondido*, não removido.
* A escolha de $N$, $S$ e dos pares é **subjetiva** e específica da aplicação (§7: *the choice of words is subjective*); erros em $S$ neutralizam o que não deveriam (*beard*, *uterus*).
* Só gênero binário; só uma direção; só inglês. Outras línguas (gênero gramatical!) e outros vieses (raça, etnia, religião: §9) ficam como trabalho futuro — o notebook 10 explora o português.
* O artigo é claro sobre o objetivo mínimo: *machine learning should not be used to inadvertently amplify these biases*.

### Pontos para a apresentação
1. Neutralize = projeção; Equalize = simetrização; Soft = otimização com compromisso. Três operações lineares simples.
2. A **Observação 1** é verificável em uma linha de código — mostre a tabela da seção 2.
3. Utilidade preservada (Tabela 1) é o argumento de venda; "Lipstick on a Pig" é o contraponto obrigatório.

## Referências
{refs("bolukbasi")}
- GONEN, H.; GOLDBERG, Y. **Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them.** NAACL, 2019.
"""),
]
