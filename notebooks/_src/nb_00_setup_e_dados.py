from nbsrc import HEADER, SETUP, code, md, refs

TITLE = "00 · Preparação do ambiente, dados públicos e mapa da apresentação"

CELLS = [
    md(f"""
# 00 · Preparação: ambiente, dados públicos e mapa da apresentação
{HEADER}

## Objetivos deste notebook
1. Verificar o ambiente (`uv`, pacote `replang`, JAX opcional).
2. Baixar/preparar os **dados públicos** usados em toda a sequência e entender de onde vêm.
3. Apresentar o **mapa da apresentação** (4h) e como os notebooks se encadeiam com os artigos.

## Mapa da apresentação (4 h)

| Bloco | Tempo | Pergunta-guia | Notebooks | Artigos |
|---|---|---|---|---|
| A. Premissas | 0:00–0:35 | Por que vetores? O que é "significado distribucional"? | 01 | Harris (1954); LSA/HAL; Levy & Goldberg (2014) |
| B. word2vec | 0:35–1:20 | Como um modelo log-linear aprende `rei − homem + mulher ≈ rainha`? | 02, 03 | Mikolov et al. (2013a, 2013b) |
| C. Variações | 1:20–1:50 | Contagem *vs.* predição (GloVe); morfologia (fastText); frases/documentos (Doc2Vec) | 04, 05, 11 | Pennington (2014); Bojanowski (2017); Le & Mikolov (2014) |
| *intervalo* | 1:50–2:05 | | | |
| D. Português | 2:05–2:45 | O que muda em PT? Como avaliar? Analogias servem? | 06, 07 | Hartmann et al. (2017) |
| E. Viés | 2:45–3:35 | O que mais o vetor carrega? Dá para tirar? | 08, 09, 10 | Bolukbasi et al. (2016) |
| F. Contexto e síntese | 3:35–4:00 | E se a palavra tiver um vetor por contexto? | 12, 13 | Peters et al. (2018) |

A interface Streamlit (`uv run replang app`) tem uma página por bloco para exploração ao vivo.
"""),
    SETUP,
    md("""
## 1. Ambiente

O projeto usa [`uv`](https://docs.astral.sh/uv/) para gerir o ambiente (`pyproject.toml` + `uv.lock`). Dependências principais:
`numpy`, `scipy`, `scikit-learn`, `gensim` (modelos "de produção"), `plotly` (figuras), `streamlit` (interface), `jax` (extra `contextual`, para o biLM).
"""),
    code('''
import importlib
for pkg in ["numpy", "scipy", "sklearn", "gensim", "plotly", "streamlit", "jax"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:<10} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{pkg:<10} ausente ({type(e).__name__}) — opcional" if pkg == "jax" else f"{pkg:<10} ERRO: {e}")
import replang
print("replang", replang.__version__, "| dados em", PATHS.data)
'''),
    md("""
## 2. Fontes de dados públicas

Todos os dados são públicos e têm licença permissiva ou são de domínio público. O módulo `replang.config.SOURCES` documenta cada um.

| Chave | Uso na apresentação | Artigo relacionado |
|---|---|---|
| `machado` | corpus PT-BR (2,4 M tokens) para treinar word2vec/GloVe/fastText/Doc2Vec/biLM **ao vivo** | todos |
| `mac_morpho` | POS tagging (avaliação extrínseca) | Hartmann et al. (2017) |
| `ptwiki2vec_100d` | embeddings PT pré-treinados na Wikipédia (Wikipedia2Vec, skip-gram 100d) | Hartmann; Bolukbasi (em PT) |
| `glove_en_50d` | embeddings EN para reproduzir Bolukbasi et al. | Bolukbasi et al. (2016) |
| `questions_words`, `lx_analogies_*` | analogias EN e PT-BR/PT-EU | Mikolov (2013a); Rodrigues (2016); Hartmann (2017) |
| `debiaswe` | pares definicionais, pares de equalização, profissões, palavras específicas de gênero | Bolukbasi et al. (2016) |
| `nilc` (conector opcional) | os 31 modelos de Hartmann et al. — exige download manual | Hartmann et al. (2017) |

> Os embeddings do NILC (`nilc.icmc.usp.br/embeddings`) não são acessíveis de todas as redes; por isso o padrão é o Wikipedia2Vec PT. O notebook 06 mostra como trocar.
"""),
    code('''
from replang.config import SOURCES
pd.DataFrame([{"chave": k, "descrição": v["desc"], "licença": v["license"]} for k, v in SOURCES.items()])
'''),
    md("""
### 2.1 Download e pré-processamento

`replang download` baixa os arquivos brutos para `data/raw`; `replang prepare` gera as versões processadas
(`data/processed`) e os caches truncados de embeddings (`data/samples`, versionados). Aqui chamamos as funções diretamente;
se os artefatos já existem, nada é baixado.
"""),
    code('''
from replang.data.corpora import prepare_machado, prepare_macmorpho, load_machado_sentences, load_macmorpho
from replang.data.embeddings import load_ptwiki, load_glove_en

p_mach = prepare_machado(); p_mm = prepare_macmorpho()
sents = load_machado_sentences()
print(f"Machado de Assis: {len(sents):,} sentenças · {sum(map(len, sents)):,} tokens · exemplo: {' '.join(sents[1234][:14])} …")
mm = load_macmorpho()
print(f"Mac-Morpho: {len(mm):,} sentenças anotadas · exemplo: {mm[3][:8]}")
pt = load_ptwiki(); en = load_glove_en()
print(pt, "|", en)
'''),
    md("""
### 2.2 Pré-processamento (seguindo Hartmann et al., 2017, §2.1)

As decisões de normalização afetam o vocabulário e, portanto, os vetores. Reproduzimos as do artigo:
minúsculas; numerais → `0`; URLs → `URL`; e-mails → `EMAIL`; clíticos hifenizados preservados (*machucou-se*);
sentenças com < 5 tokens descartadas; tipos raros (< 5 ocorrências) tratados como `UNKNOWN` (nos modelos, via `min_count`).
"""),
    code('''
from replang.utils.text import normalize_text, tokenize, iter_sentences
exemplo = "Em 1899, Machado publicou Dom Casmurro. Veja em http://machado.mec.gov.br ou escreva para bentinho@capitu.br! Ele machucou-se."
print(normalize_text(exemplo))
print(tokenize(exemplo))
print(list(iter_sentences(exemplo, min_tokens=5)))
'''),
    md("""
## 3. Primeira olhada: o que um embedding "sabe"

Antes de qualquer teoria, uma amostra do que vem pela frente. O método `most_similar` calcula o **cosseno** entre o vetor da
palavra e todos os outros; `analogy` resolve `a : b :: c : ?` por aritmética vetorial (Mikolov et al., 2013a).
"""),
    code('''
for w in ["rei", "computador", "feliz", "capitu"]:
    print(f"{w:<12}", [x for x, _ in pt.most_similar(w, topn=6)] if w in pt else "fora do vocabulário")
print("homem : rei :: mulher : ?", pt.analogy("homem", "rei", "mulher", topn=3))
print("brasil : brasília :: frança : ?", pt.analogy("brasil", "brasília", "frança", topn=3))
print("man : king :: woman : ?", en.analogy("man", "king", "woman", topn=3))
'''),
    md("""
E uma amostra do **viés** que também vem junto (tema do bloco E):
"""),
    code('''
from replang.data import EN, PT
print("EN  he : doctor :: she : ?", en.analogy("he", "doctor", "she", topn=3, exclude_words=EN.get().gender_specific))
print("PT  ele : médico :: ela : ?", pt.analogy("ele", "médico", "ela", topn=3, exclude_words=PT.get().gender_specific))
print("PT  homem : programador :: mulher : ?", pt.analogy("homem", "programador", "mulher", topn=3, exclude_words=PT.get().gender_specific))
'''),
    md(f"""
## 4. Como os notebooks se organizam

Cada notebook segue o mesmo padrão: **premissas → mecanismo (com código *from scratch*) → experimento → o que o artigo encontrou → discussão**.
As implementações didáticas estão em `replang.models` (numpy); os modelos "de produção" (gensim) são treinados por `uv run replang train` e carregados de `data/models`.

## Referências
{refs("mikolov13a", "mikolov13b", "glove", "fasttext", "hartmann", "bolukbasi", "elmo", "doc2vec")}
"""),
]
