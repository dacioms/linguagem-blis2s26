from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "05 · fastText (Bojanowski et al., 2017): a palavra como saco de n-gramas de caracteres"

CELLS = [
    md(f"""
# 05 · fastText: subpalavras, morfologia e palavras fora do vocabulário
{HEADER}

**Artigo-base (leitura complementar):** Bojanowski, Grave, Joulin & Mikolov (2017), *Enriching Word Vectors with Subword Information* (TACL).

## O problema
O word2vec dá um vetor **a cada forma**: *casa*, *casas*, *casinha*, *casarão* são tratadas como símbolos sem relação. Em línguas morfologicamente ricas (turco, finlandês, tcheco, russo — e o português, com ~50 formas por verbo) muitas formas são raras ou inéditas no corpus; palavras fora do vocabulário (**OOV**) simplesmente não têm vetor.

## A proposta
Representar cada palavra como o conjunto $\\mathcal{{G}}_w$ dos seus **n-gramas de caracteres** ($3 \\le n \\le 6$), delimitados por `<` e `>`, mais a palavra inteira. A pontuação do Skip-gram com amostragem negativa vira

$$s(w, c) = \\sum_{{g \\in \\mathcal{{G}}_w}} \\mathbf{{z}}_g^{{\\top}} \\mathbf{{v}}_c$$

Os n-gramas são mapeados em $K = 2\\cdot10^6$ *buckets* por *hashing* (FNV-1a) para limitar a memória. Consequências: parâmetros compartilhados entre formas aparentadas; vetores para OOV (soma dos n-gramas); treino só ~1,5× mais lento.
"""),
    SETUP,
    md("""
## 1. N-gramas de caracteres e *hashing*

Exemplo do artigo: `where` com $n=3$ → `<wh, whe, her, ere, re>` + `<where>`. Note que `her` (dentro de *where*) é diferente de `<her>` (a palavra *her*) graças aos delimitadores.
"""),
    code('''
from replang.models.fasttext_np import char_ngrams, fnv1a
for w in ["where", "onde", "casinha", "incrivelmente"]:
    grams = char_ngrams(w, 3, 6)
    print(f"{w:<14} {len(grams):>2} n-gramas: {grams[:8]}{' …' if len(grams) > 8 else ''}")
print("\\nhash FNV-1a (mod 2·10⁶):", {g: fnv1a(g) % 2_000_000 for g in char_ngrams("casa", 3, 4)})
'''),
    md("""
## 2. Implementação *from scratch*

`replang.models.fasttext_np.FastText` herda do Skip-gram numpy e troca o vetor de entrada pela **média** dos vetores dos n-gramas (+ palavra). O gradiente de cada par é distribuído por todos os n-gramas da palavra central — é assim que *casinha* "empresta" o que aprendeu a *casa*.
"""),
    code('''
import inspect, time
from replang.models.fasttext_np import FastText
src = inspect.getsource(FastText.train)
print(src[src.index("centers = np.array"):src.index("losses.append")])
'''),
    code('''
from replang.data.corpora import load_machado_sentences
sents = load_machado_sentences()
sub = sents[:12000 if FAST else 25000]
t = time.time()
ft = FastText(dim=50, window=3, epochs=2, negative=5, min_count=3, bucket=100_000, batch_size=256, seed=1).train(sub)
wv_ft = ft.to_wordvectors("fastText numpy (40k sentenças)")
print(f"{time.time()-t:.0f}s | |V| = {len(wv_ft):,} | perda final {ft.history[-1]['loss']:.3f}")
for w in ["amor", "triste", "casa", "capitu"]:
    print(f"{w:<8} {[x for x, _ in wv_ft.most_similar(w, topn=6)] if w in wv_ft else '(fora do vocabulário do subconjunto — mas o fastText ainda dá um vetor; veja abaixo)'}")
'''),
    md("""
### 2.1 Vetores para palavras **fora do vocabulário**
"""),
    code('''
for oov in ["tristíssimo", "capituzinha", "amorzão", "desamoroso", "xyzzyq"]:
    v = ft.word_vector(oov)
    sims = wv_ft.similarities(v)
    print(f"{oov:<14} OOV → vizinhos: {[wv_ft.words[i] for i in np.argsort(-sims)[:6]]}")
'''),
    md("""
### 2.2 Quais n-gramas importam? (§6.2 e Tabela 6 do artigo)

O artigo remove um n-grama de cada vez e mede o quanto a representação muda. Os n-gramas mais importantes tendem a ser **morfemas**: para *Autofahrer*, `Auto` e `fahrer`; para *kindness*, `ness>`.
"""),
    code('''
from replang.viz import ngram_bar
for w in ["tristeza", "casinha", "amorosamente"]:
    imp = ft.ngram_importance(w)
    print(f"{w:<13} n-gramas mais importantes: {[g for g, _ in imp[:6]]}")
ngram_bar(ft.ngram_importance("amorosamente")[:15], title="'amorosamente': queda do cosseno ao remover cada n-grama (maior = mais importante)").show()
'''),
    md("""
## 3. Modelo completo (gensim) no corpus Machado: fastText × Skip-gram

`machado_ft100` (gensim, Skip-gram, 100d, n ∈ [3,6]) e `machado_sg100` foram treinados com os mesmos hiperparâmetros. Comparamos:
1. vizinhos de palavras raras e OOV;
2. pares morfológicos (similaridade esperada alta);
3. analogias **sintáticas** × **semânticas** (Tabela 2 do artigo: subpalavras ajudam na sintaxe e podem atrapalhar na semântica).
"""),
    code('''
from replang.data.embeddings import load_local_model
from replang.models.trainers import load_fasttext_model
sg = load_local_model("machado_sg100")
ftm = load_fasttext_model("machado_ft100")
wv_ftm = load_local_model("machado_ft100")
raras = ["capituzinha", "escravocrata", "tristíssimo", "desconfiadíssimo", "aborrecimento", "cavalheirismo"]
for w in raras:
    nn_ft = [x for x, _ in ftm.wv.most_similar(w, topn=5)]
    nn_sg = [x for x, _ in sg.most_similar(w, topn=5)] if w in sg else "OOV (sem vetor)"
    print(f"{w:<18} fastText: {nn_ft}\\n{'':<18} Skip-gram: {nn_sg}")
'''),
    code('''
pares = [("menino", "menininho"), ("casa", "casinha"), ("amor", "amoroso"), ("escrever", "escrevia"), ("feliz", "felicidade"), ("triste", "tristeza"), ("rei", "rainha"), ("gato", "gatinho")]
rows = [{"par": f"{a} – {b}", "fastText": float(ftm.wv.similarity(a, b)), "Skip-gram": sg.similarity(a, b) if sg.has(a, b) else np.nan} for a, b in pares]
pd.DataFrame(rows).style.format({"fastText": "{:.2f}", "Skip-gram": "{:.2f}"})
'''),
    code('''
from replang.data.analogies import load_analogies
from replang.eval.analogies import evaluate_analogies
an = load_analogies("pt-br")
rows = []
for wv in [sg, wv_ftm]:
    r = evaluate_analogies(wv, an, restrict=None).set_index("categoria")
    rows.append({"modelo": wv.name, "sintática": r.loc["TOTAL sintática", "acurácia"], "semântica": r.loc["TOTAL semântica", "acurácia"], "cobertas": int(r.loc["TOTAL", "cobertas"])})
pd.DataFrame(rows).style.format({"sintática": "{:.1%}", "semântica": "{:.1%}"})
'''),
    md("""
## 4. Efeito do tamanho do corpus (§5.4 e Fig. 1 do artigo)

O artigo mostra que o fastText **satura cedo**: com 5 % da Wikipédia alemã supera o CBOW com 100 %. Reproduzimos a tendência treinando os dois modelos (gensim, 50d, 3 épocas) em 5 %, 20 % e 100 % do corpus Machado e medindo as analogias **sintáticas/morfológicas** do LX-4WAnalogies (plural, particípio, passado, plural de verbos) — a competência que os n-gramas capturam.
"""),
    code('''
from gensim.models import Word2Vec as GW2V, FastText as GFT
cats = ["gram5-present-participle", "gram7-past-tense", "gram8-plural", "gram9-plural-verbs"]
fracs = [0.05, 0.2, 1.0] if not FAST else [0.05, 0.2]
rows = []
for frac in fracs:
    sub = sents[: int(len(sents) * frac)]
    for name, cls, kw in [("Skip-gram", GW2V, {}), ("fastText", GFT, {"min_n": 3, "max_n": 6, "bucket": 100_000})]:
        t = time.time()
        m = cls(sub, vector_size=50, window=5, sg=1, negative=5, min_count=5, epochs=3, sample=1e-4, workers=settings.workers, seed=1, **kw)
        wv = WordVectors.from_gensim(m.wv, name)
        r = evaluate_analogies(wv, an, restrict=None).set_index("categoria")
        cob = int(r.loc[cats, "cobertas"].sum()); ac = int(r.loc[cats, "acertos"].sum())
        rows.append({"fração": frac, "tokens": sum(map(len, sub)), "modelo": name, "|V|": len(wv), "questões cobertas": cob, "acurácia sintática": ac / max(cob, 1), "s": round(time.time() - t)})
df = pd.DataFrame(rows)
df.style.format({"acurácia sintática": "{:.1%}"})
'''),
    code('''
import plotly.express as px
px.line(df, x="tokens", y="acurácia sintática", color="modelo", markers=True, log_x=True, template="plotly_white",
        title="Analogias morfológicas × tamanho do corpus: fastText aprende com muito menos dados (cf. Fig. 1 de Bojanowski et al.)").show()
'''),
    md("""
## 5. O que o artigo encontrou

* **Similaridade de palavras** (Tabela 1): `sisg` vence `sg`/`cbow` em 8 de 9 línguas; ganho maior em alemão, árabe, russo; empate em inglês (WS353), onde as palavras são comuns.
* **Analogias** (Tabela 2): grande ganho nas **sintáticas** (tcheco 52,8 → 77,8; alemão 44,5 → 56,4); nas semânticas, igual ou pior.
* **Tamanho dos n-gramas** (Tabela 4): 3–6 é razoável; 2-gramas não ajudam (um caractere + um delimitador).
* **Modelagem de linguagem** (Tabela 5): inicializar um LSTM com `sisg` reduz a perplexidade (−8 % tcheco, −13 % russo).
* **Hartmann et al. (2017)**: em PT, fastText foi o melhor nas analogias sintáticas (58,7 % com Skip-gram 300d) e surpreendentemente fraco em POS tagging (a tokenização dos clíticos é apontada como causa).

### Pontos para a apresentação
1. A hipótese distribucional continua valendo — mas agora **compartilhada entre formas**: a distribuição de `<cas` informa *casa*, *casas*, *casinha*.
2. OOV deixa de ser um buraco: vetores para erros de digitação, neologismos e nomes raros.
3. O custo: n-gramas espúrios (`ame` em *amesma*?) e perda em analogias semânticas.
4. É a ponte para o **ELMo** (notebook 12), que usa uma CNN de caracteres como camada de entrada.

### Perguntas para discussão
* Por que o *hashing* funciona apesar das colisões? (2 M buckets × n-gramas raros; colisões são ruído distribuído)
* O que o fastText faz com *manga* (fruta) e *manga* (camisa)? (nada — ainda é um vetor por forma; gancho para ELMo)

## Referências
{refs("fasttext", "mikolov13b", "hartmann", "elmo")}
"""),
]
