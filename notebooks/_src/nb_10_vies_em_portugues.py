from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "10 · Viés de gênero em português: gênero gramatical, profissões epicenas, corpus sintético e debias em PT"

CELLS = [
    md(f"""
# 10 · Viés de gênero em português: gênero gramatical, profissões epicenas e um experimento controlado
{HEADER}

Bolukbasi et al. (2016) terminam com uma pergunta: *"it is also an interesting direction to consider how the approach and findings here would apply to other languages, especially languages with grammatical gender where the definitions of most nouns carry a gender marker"* (§9). O português é exatamente esse caso. Este notebook:

1. aplica o método ao **Wikipedia2Vec PT** e ao **Machado de Assis** (direção de gênero, extremos, DirectBias);
2. discute o problema do **gênero gramatical**: *enfermeira/enfermeiro* carregam gênero na ortografia — isso é viés? (não: é morfologia) — e por isso usa **profissões epicenas** (*dentista, gerente, cientista*) e **áreas** (*enfermagem, engenharia*);
3. separa, nas analogias geradas, pares **morfológicos** de **estereótipos**;
4. faz um **experimento controlado** com corpus sintético, variando o viés do corpus de 50 % a 100 % e medindo o viés geométrico resultante;
5. aplica o hard debias em PT e discute o que ele faz com o gênero gramatical.
"""),
    SETUP,
    code('''
from replang.data.embeddings import load_ptwiki, load_local_model
from replang.data.lexicons import PT
from replang.bias import gender_direction, pca_pairs, random_pca_baseline, project, extremes, direct_bias, indirect_bias, generate_analogies, split_analogies_pt, hard_debias, pair_bias
from replang.bias.geometry import profile
from replang.viz import pca_variance_figure, bias_axis_figure
lex = PT.get()
pt = load_ptwiki().normalized("Wikipedia2Vec PT 100d"); mach = load_local_model("machado_sg100").normalized("Machado SG 100d")
print(pt, "|", mach)
print("pares definicionais PT:", lex.definitional_pairs)
print(f"profissões epicenas: {len(lex.professions)} | áreas/substantivos neutros: {len(lex.stereotype_neutral)} | pares morfológicos de profissão: {len(lex.professions_gendered_pairs)}")
'''),
    md("""
## 1. A direção de gênero em PT (Fig. 6, versão PT)

Em português a PCA das diferenças dos pares é ainda mais "limpa" que em inglês: a 1ª componente explica mais de 60 % da variância no Wikipedia2Vec. Hipótese: o gênero gramatical faz *toda* a concordância (artigos, adjetivos, particípios) coocorrer com o gênero, reforçando a direção.
"""),
    code('''
for wv in [pt, mach]:
    comps, ev, used = pca_pairs(wv, lex.definitional_pairs)
    g = gender_direction(wv, lex.definitional_pairs)
    print(f"{wv.name}: pares usados {len(used)} · variância explicada {np.round(ev[:4], 3)} · aleatório {random_pca_baseline(wv.dim, len(used))[0]:.3f} · cos(ela,g)={float(wv['ela'] @ g):+.2f} cos(ele,g)={float(wv['ele'] @ g):+.2f}")
g_pt = gender_direction(pt, lex.definitional_pairs); g_m = gender_direction(mach, lex.definitional_pairs)
_, ev_pt, used_pt = pca_pairs(pt, lex.definitional_pairs)
pca_variance_figure(ev_pt, random_pca_baseline(pt.dim, len(used_pt)), title="PT (Wikipedia2Vec): variância explicada pela PCA das diferenças dos pares × aleatório")
'''),
    md("""
## 2. O que está nos extremos do eixo *ela–ele*

Projetamos três grupos: (a) **profissões epicenas** (forma única para os dois gêneros), (b) **áreas e substantivos abstratos** (*cozinha, engenharia, coragem, delicadeza*) e (c) **pares morfológicos** (*enfermeira/enfermeiro*). Em (a) e (b) a ortografia não explica a posição; em (c) explica — e esse é o ponto da discussão.
"""),
    code('''
epicenas = [w for w in lex.professions if w in pt]; areas = [w for w in lex.stereotype_neutral if w in pt]
fem, masc = extremes(pt, g_pt, epicenas + areas, 15)
print("extremo ELA:", fem); print("extremo ELE:", masc)
print(f"\\nDirectBias_1 (Wikipedia2Vec PT): epicenas = {direct_bias(pt, epicenas, g_pt):.3f} | áreas = {direct_bias(pt, areas, g_pt):.3f} | 300 palavras frequentes não-gênero = {direct_bias(pt, [w for w in pt.words[:600] if w not in lex.all_gender_words()][:300], g_pt):.3f}")
print(f"DirectBias_1 (Machado SG):        epicenas = {direct_bias(mach, [w for w in epicenas if w in mach], g_m):.3f} | áreas = {direct_bias(mach, [w for w in areas if w in mach], g_m):.3f}")
'''),
    code('''
df = profile(pt, epicenas + areas, g_pt, lex.extra["labels"])
sel = pd.concat([df.nsmallest(20, "proj_g"), df.nlargest(20, "proj_g")])
bias_axis_figure(sel, title="Profissões epicenas e áreas projetadas na direção de gênero (Wikipedia2Vec PT)", axis_label="← ele          ela →")
'''),
    code('''
# Pares morfológicos: os dois membros se separam em g (esperado — gênero gramatical), mas o CENTRO do par também pode estar deslocado (estereótipo)
rows = []
for f, m in lex.professions_gendered_pairs:
    if pt.has(f, m):
        pf, pm = float(pt[f] @ g_pt), float(pt[m] @ g_pt)
        rows.append({"par": f"{f}/{m}", "proj. feminina": pf, "proj. masculina": pm, "separação (gramatical)": pf - pm, "centro do par (estereótipo?)": (pf + pm) / 2})
pairs_df = pd.DataFrame(rows).sort_values("centro do par (estereótipo?)")
pairs_df.style.format({c: "{:+.3f}" for c in pairs_df.columns[1:]}).background_gradient(subset=["centro do par (estereótipo?)"], cmap="RdBu_r")
'''),
    md("""
Leitura: a coluna **separação** é o gênero gramatical (sempre positiva: a forma feminina está do lado *ela*); a coluna **centro** diz se o *conceito* da profissão, independentemente da forma, pende para um lado. *Enfermeira/enfermeiro*, *secretária/secretário*, *cozinheira/cozinheiro* tendem a centro positivo; *engenheira/engenheiro*, *físico/física*, *pedreiro/pedreira* a centro negativo. É uma maneira de **separar morfologia de estereótipo** usando a própria geometria.
"""),
    md("""
## 3. Viés indireto em PT

Análogo a *softball–football*: tomamos **cozinha–futebol** e medimos quanto da similaridade com outras palavras neutras vem da direção de gênero.
"""),
    code('''
axis = pt["cozinha"] - pt["futebol"]
coz, fut = extremes(pt, axis, epicenas + areas, 8)
rows = [{"w": "cozinha", "v": v, "cos": pt.similarity("cozinha", v), "β(w,v)": indirect_bias(pt, "cozinha", v, g_pt)} for v in coz[:6]]
rows += [{"w": "futebol", "v": v, "cos": pt.similarity("futebol", v), "β(w,v)": indirect_bias(pt, "futebol", v, g_pt)} for v in fut[:6]]
pd.DataFrame(rows).style.format({"cos": "{:.2f}", "β(w,v)": "{:+.0%}"})
'''),
    code('''
# Versão orientada a dados: pares de palavras neutras (epicenas + áreas) cuja similaridade mais depende de g
from replang.bias import top_indirect_bias_pairs
top_indirect_bias_pairs(pt, epicenas + areas, g_pt, topn=15, min_cos=0.3).style.format({"cos": "{:.2f}", "β": "{:.0%}", "proj_g(w)": "{:+.2f}", "proj_g(v)": "{:+.2f}"})
'''),
    md("""
## 4. Analogias geradas: separando gramática de estereótipo

Com a semente (*ela*, *ele*) a eq. (1) de Bolukbasi devolve, em PT, sobretudo pares **morfológicos** (*escritora–escritor*) — corretos e esperados. A função `split_analogies_pt` usa uma heurística ortográfica para separá-los dos pares **sem relação morfológica**, que são os candidatos a estereótipo.
"""),
    code('''
an = generate_analogies(pt, "ela", "ele", delta=1.0, topn=120, restrict=30000, exclude=lex.all_gender_words())
gram, stereo = split_analogies_pt(an)
print(f"{len(an)} analogias geradas → {len(gram)} gramaticais, {len(stereo)} sem relação morfológica")
print("gramaticais (amostra):", list(zip(gram.x[:10], gram.y[:10])))
print("candidatos a estereótipo:")
stereo.head(25).T
'''),
    code('''
# Com uma semente "conceitual" em vez de pronominal: mulher–homem, mãe–pai
for a, b in [("mulher", "homem"), ("mãe", "pai")]:
    an2 = generate_analogies(pt, a, b, delta=1.0, topn=60, restrict=30000, exclude=lex.all_gender_words())
    _, st2 = split_analogies_pt(an2)
    print(f"{a}:{b} → candidatos a estereótipo: {list(zip(st2.x[:12], st2.y[:12]))}")
'''),
    md("""
## 5. Experimento controlado: o viés geométrico reflete o viés do corpus?

Bolukbasi et al. argumentam que o viés vem do texto. Testamos **causalmente** com um corpus sintético (`synthetic_gender_corpus`): frases-molde em que profissões "estereotipadas" coocorrem com o gênero esperado com probabilidade `bias` ∈ {0,5; 0,6; …; 1,0} e profissões "neutras" com 50 %. Treinamos um Skip-gram (numpy) para cada valor e medimos a projeção das profissões na direção de gênero. Se a hipótese está certa, o DirectBias cresce com `bias` **e** as neutras ficam em zero.
"""),
    code('''
import time
from replang.data.corpora import synthetic_gender_corpus, SYNTHETIC_PROFESSIONS
from replang.models.word2vec_np import Word2Vec
pairs_syn = [("ela", "ele"), ("mulher", "homem"), ("mãe", "pai"), ("filha", "filho"), ("irmã", "irmão"), ("menina", "menino"), ("rainha", "rei"), ("esposa", "marido")]
rows, profiles = [], {}
for bias in ([0.5, 0.75, 1.0] if FAST else [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    t = time.time()
    corpus = synthetic_gender_corpus(15000, bias=bias, seed=0)
    # sem subamostragem: no corpus sintético as palavras de gênero são frequentíssimas e seriam descartadas (t=1e-3 apaga o sinal!)
    m = Word2Vec(dim=50, window=5, negative=10, epochs=8, sample=0, batch_size=128, seed=0).train(corpus)
    wv = m.to_wordvectors().normalized()
    g = gender_direction(wv, pairs_syn)
    pf = project(wv, SYNTHETIC_PROFESSIONS["estereotipo_feminino"], g); pm = project(wv, SYNTHETIC_PROFESSIONS["estereotipo_masculino"], g); pn = project(wv, SYNTHETIC_PROFESSIONS["neutras"], g)
    profiles[bias] = (pf, pm, pn)
    rows.append({"viés do corpus": bias, "média proj. estereótipo fem.": pf.mean(), "média proj. estereótipo masc.": pm.mean(), "separação fem − masc": pf.mean() - pm.mean(), "média |proj.| neutras": pn.abs().mean(),
                 "DirectBias (estereotipadas)": direct_bias(wv, list(pf.index) + list(pm.index), g), "DirectBias (neutras)": direct_bias(wv, list(pn.index), g), "s": round(time.time() - t)})
syn_df = pd.DataFrame(rows)
syn_df.style.format({c: "{:+.3f}" for c in syn_df.columns[1:-1]})
'''),
    code('''
import plotly.express as px
m = syn_df.melt(id_vars="viés do corpus", value_vars=["separação fem − masc", "DirectBias (estereotipadas)", "DirectBias (neutras)"], var_name="métrica", value_name="valor")
px.line(m, x="viés do corpus", y="valor", color="métrica", markers=True, template="plotly_white", title="Viés geométrico cresce com o viés estatístico do corpus; profissões neutras ficam perto de zero").show()
'''),
    md("""
Com `bias = 0.5` (corpus "justo") a direção de gênero **ainda existe** (os pares definicionais a criam), mas as profissões estereotipadas e as neutras ficam todas perto de zero (ruído). A partir de `0.7` a separação fem − masc dispara e satura perto de `1.0`. A geometria é um **termômetro** das coocorrências: nenhum algoritmo "inventa" o estereótipo, ele o **mede e reproduz**.

Detalhe técnico que vale um comentário em sala: a **subamostragem** (`sample`) precisou ser desligada — no corpus sintético as palavras de gênero são tão frequentes que `t = 10⁻³` descartaria quase todas as suas ocorrências e apagaria o sinal. Hiperparâmetros "inocentes" mudam o que o embedding carrega.
"""),
    md("""
## 6. Hard debias em português

Aplicamos Neutralize + Equalize ao Wikipedia2Vec PT com $N$ = tudo menos as palavras específicas de gênero e $E$ = pares de equalização PT (inclui pares morfológicos de parentesco e títulos). **Decisão de projeto**: os pares morfológicos de **profissão** (*enfermeira/enfermeiro*) entram como conjuntos de igualdade? Se sim, *enfermeira* e *enfermeiro* ficam simétricos e o conceito "enfermagem" é neutralizado; se não, eles são tratados como neutras e **perdem** a marca de gênero — o que destruiria concordância em aplicações gerativas. Mostramos as duas opções.
"""),
    code('''
B = g_pt[None, :]
opt_a = hard_debias(pt, B, gender_specific=lex.gender_specific, equalize_pairs=lex.equalize_pairs + lex.professions_gendered_pairs, name="PT hard (profissões equalizadas)")
spec_b = [w for w in lex.gender_specific if w not in {x for p in lex.professions_gendered_pairs for x in p}]
opt_b = hard_debias(pt, B, gender_specific=spec_b, equalize_pairs=lex.equalize_pairs, name="PT hard (profissões neutralizadas)")
neutras = epicenas + areas
rows = []
for wv in [pt, opt_a, opt_b]:
    rows.append({"embedding": wv.name, "DirectBias epicenas+áreas": direct_bias(wv, neutras, g_pt), "PairBias": pair_bias(wv, neutras, lex.definitional_pairs),
                 "cos(enfermeira,g)": float(wv["enfermeira"] @ g_pt), "cos(enfermeiro,g)": float(wv["enfermeiro"] @ g_pt), "cos(enfermeira,enfermeiro)": wv.similarity("enfermeira", "enfermeiro"), "cos(rei,rainha)": wv.similarity("rei", "rainha")})
pd.DataFrame(rows).style.format({c: "{:.3f}" for c in rows[0] if c != "embedding"})
'''),
    code('''
for a, b, c in [("ele", "médico", "ela"), ("homem", "programador", "mulher"), ("ela", "enfermeira", "ele"), ("ele", "engenheiro", "ela")]:
    print(f"{a}:{b} :: {c}:?")
    for wv in [pt, opt_a]:
        print(f"   {wv.name:<38} {[x for x, _ in wv.analogy(a, b, c, topn=5, exclude_words=lex.gender_specific)]}")
print("\\nvizinhos de 'enfermagem' antes :", [x for x, _ in pt.most_similar("enfermagem", topn=8)])
print("vizinhos de 'enfermagem' depois:", [x for x, _ in opt_a.most_similar("enfermagem", topn=8)])
'''),
    md("""
## 7. E no corpus literário? (Machado de Assis)

Um corpus do século XIX tem uma estrutura de gênero muito marcada (papéis sociais, vocabulário de salão). O eixo *ela–ele* do modelo Machado mostra isso de forma crua — e lembra que o "viés" depende do corpus **e da época**.
"""),
    code('''
cands = [w for w in mach.words[:6000] if w not in lex.all_gender_words() and w.isalpha() and len(w) > 3]
fem_m, masc_m = extremes(mach, g_m, cands, 25)
print("lado ELA (Machado):", fem_m); print("lado ELE (Machado):", masc_m)
'''),
    md(f"""
## 8. Síntese

1. O método de Bolukbasi **transfere** para o português: a direção existe (até mais nítida) e os estereótipos ocupacionais aparecem nas profissões epicenas e nas áreas.
2. **Gênero gramatical ≠ viés**: é preciso separar *separação dentro do par* (morfologia) de *deslocamento do centro do par* (estereótipo), e filtrar analogias morfológicas.
3. O experimento sintético confirma a relação causal corpus → geometria, com um "falso positivo" a evitar: a direção existe mesmo sem viés nas profissões.
4. O *debias* em PT exige decidir **o que preservar**: concordância gramatical é informação, não preconceito.

### Perguntas para discussão
* Neutralizar *enfermeira* destrói a concordância (*a enfermeira competente*). Como um modelo gerativo em PT deveria lidar com isso? (gancho: modelos contextuais e *debias* em nível de tarefa)
* Que outras direções valeria medir em PT? (regional, racial, de classe — Bolukbasi §9; e com que pares definicionais?)

## Referências
{refs("bolukbasi", "hartmann")}
- GONEN, H.; GOLDBERG, Y. **Lipstick on a Pig.** NAACL, 2019.
"""),
]
