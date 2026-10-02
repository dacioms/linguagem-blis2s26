# Representações de Linguagem — BLIS 2S26

> **Como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto.**
> Material completo de uma apresentação acadêmica de 4 h: estrutura didática, código *from scratch* e de produção, dados públicos, interface interativa e 14 notebooks executados — tudo gerido com [`uv`](https://docs.astral.sh/uv/).

## Leituras

| | Artigo | Notebooks |
|---|---|---|
| **Obrigatória** | Mikolov et al. (2013a) — *Efficient Estimation of Word Representations in Vector Space* | 02 |
| **Obrigatória** | Bolukbasi et al. (2016) — *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* | 08, 09, 10 |
| **Obrigatória** | Hartmann et al. (2017) — *Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks* | 06, 07 |
| Complementar | Mikolov et al. (2013b) — *Distributed Representations of Words and Phrases and their Compositionality* | 03 |
| Complementar | Pennington, Socher & Manning (2014) — *GloVe* | 04 |
| Complementar | Bojanowski et al. (2017) — *Enriching Word Vectors with Subword Information* | 05 |
| Complementar | Le & Mikolov (2014) — *Distributed Representations of Sentences and Documents* | 11 |
| Complementar | Peters et al. (2018) — *Deep contextualized word representations* (ELMo) | 12 |

Resumos detalhados de cada artigo: [`docs/artigos/`](docs/artigos/README.md). **Contexto jurídico** (aplicações, peculiaridades, Brasil, debates): [`docs/juridico.md`](docs/juridico.md). Estrutura didática completa: [`docs/00_estrutura_didatica.md`](docs/00_estrutura_didatica.md). Roteiro minuto a minuto: [`docs/roteiro_4h.md`](docs/roteiro_4h.md). Glossário: [`docs/glossario.md`](docs/glossario.md). Arquitetura: [`docs/arquitetura.md`](docs/arquitetura.md).

## Início rápido

```bash
# 1. ambiente (Python ≥ 3.11; instale o uv: https://docs.astral.sh/uv/getting-started/installation/)
uv sync --extra dev --extra contextual      # 'contextual' = JAX para o biLM (ELMo-lite)

# 2. dados públicos (os artefatos pequenos já vêm no repositório: data/samples e data/processed)
uv run replang download && uv run replang prepare

# 3. modelos locais sobre o corpus Machado de Assis (≈ 15 min em 4 CPUs; --fast ≈ 6 min)
uv run replang train           # + `uv run replang train legal` para os modelos jurídicos (≈ 4 min)

# 4. interface interativa
uv run replang app                            # http://localhost:8501

# 5. notebooks (JupyterLab) — já estão executados; para regenerar as saídas:
uv run jupyter lab notebooks/
uv run replang notebooks run                  # ≈ 1 h em 4 CPUs (REPLANG_FAST=1 para a versão rápida)

# 6. testes e lint
uv run pytest -q && uv run ruff check src tests scripts app
```

Ou, com `make`: `make setup data train app` / `make test`.

## Os notebooks (ordem da apresentação)

| # | Notebook | Bloco | Conteúdo |
|---|---|---|---|
| 00 | `00_setup_e_dados` | — | ambiente, fontes públicas, pré-processamento (Hartmann §2.1), mapa das 4 h |
| 01 | `01_premissas_hipotese_distribucional` | A | one-hot, BoW/TF-IDF, coocorrência, PMI/PPMI, SVD (LSA/HAL), cosseno |
| 02 | `02_word2vec_mikolov2013a` | B | CBOW e Skip-gram *from scratch*, W_in/W_out, analogias, dimensão × corpus |
| 03 | `03_word2vec_extensoes_mikolov2013b` | B | softmax hierárquico (Huffman), amostragem negativa, subamostragem, frases, composição aditiva, SGNS = PMI |
| 04 | `04_glove` | C | razões de probabilidade, objetivo ponderado, AdaGrad, contagem × predição |
| 05 | `05_fasttext_subpalavras` | C | n-gramas, hashing FNV, OOV, importância de n-gramas, efeito do tamanho do corpus |
| 06 | `06_embeddings_portugues_hartmann` | D | corpus de 1,39 bi tokens, 4 algoritmos (incl. Wang2Vec), analogias PT-BR/PT-EU, conector NILC |
| 07 | `07_avaliacao_extrinseca` | D | POS tagging (Mac-Morpho), similaridade de sentenças (ASSIN-like), concordância de rankings, Faruqui 2016 |
| 08 | `08_vies_geometria_bolukbasi` | E | eixo she−he, geração de analogias (eq. 1), PCA dos pares (Fig. 6), DirectBias, viés indireto β, SVM (§7) |
| 09 | `09_debiasing_hard_soft` | E | Neutralize, Equalize (Observação 1), soft debias (λ), utilidade preservada, "Lipstick on a Pig" |
| 10 | `10_vies_em_portugues` | E | gênero gramatical × estereótipo, profissões epicenas, experimento controlado com corpus sintético, debias em PT |
| 11 | `11_doc2vec_sentencas_documentos` | C | PV-DM/PV-DBOW em Machado, gêneros literários, inferência, comparação com média de vetores |
| 12 | `12_elmo_contextual` | F | biLM em JAX, polissemia (*banco*, *manga*), camadas × tarefas, mistura ELMo, eficiência amostral |
| 13 | `13_sintese_e_roteiro` | — | linha do tempo, quadro consolidado, as três perguntas, roteiro |
| 14 | `14_linguagem_juridica_dominio` | ⚖️ | a língua do Direito; deslocamento de domínio geral × jurídico (STF); NER (LeNER-Br), similaridade de ementas (JurisBERT), área (RulingBR) |
| 15 | `15_vies_aplicacoes_governanca_juridica` | ⚖️ | viés de gênero no corpus do STF, papéis processuais, ranqueador com consultas gêmeas, debias e limites, aplicações e governança (CNJ, LGPD) |

Os notebooks são gerados a partir de fontes Python em `notebooks/_src/` (`uv run replang notebooks build`), o que mantém o texto revisável e os *diffs* legíveis.

## A interface (`app/`)

Onze páginas Streamlit espelhando os blocos (a 10ª compara embeddings gerais × jurídicos): coocorrência → PPMI → SVD interativo; treinar word2vec ao vivo (numpy); vizinhos e similaridade entre modelos; analogias (3CosAdd/3CosMul + benchmark); projeções 2D; subpalavras e OOV (fastText); avaliação intrínseca × extrínseca; viés de gênero e *debias* (direção, extremos, DirectBias, β, analogias geradas, hard/soft antes × depois); representações contextuais (ELMo-lite).

## Dados públicos

| Fonte | Uso | Licença |
|---|---|---|
| Obra completa de Machado de Assis (NLTK `machado`) | corpus PT-BR para treinar todos os modelos ao vivo | domínio público |
| Mac-Morpho (NILC) | POS tagging (avaliação extrínseca) | CC BY 4.0 |
| Wikipedia2Vec PT 100d (2018) | embeddings PT pré-treinados (50k palavras versionadas) | Apache 2.0 / CC BY-SA |
| GloVe 6B 50d (gensim-data) | reprodução de Bolukbasi et al. em inglês | PDDL |
| `questions-words.txt`; LX-4WAnalogies (BR/EU) | analogias | Apache 2.0; NLX |
| `tolga-b/debiaswe` | pares definicionais, profissões, palavras específicas de gênero | MIT |
| NILC embeddings (conector opcional) | os 31 modelos de Hartmann et al. | uso acadêmico |
| RulingBR (STF, 2011–2018) | corpus jurídico: embeddings de domínio, classificação de área | dados públicos do STF; dataset acadêmico |
| LeNER-Br | NER jurídico (avaliação extrínseca) | acadêmico |
| JurisBERT STS (STJ/TJMS) | similaridade de ementas | dados públicos dos tribunais; dataset acadêmico |

## O pacote `replang`

`src/replang` implementa, de forma legível e testada, tudo o que os artigos descrevem: `models/` (contagem→PPMI→SVD; Skip-gram/CBOW com NEG e softmax hierárquico; GloVe; fastText; biLM em JAX; treinadores gensim), `eval/` (analogias, similaridade, POS, STS), `bias/` (direção de gênero, DirectBias, β, geração de analogias, hard/soft debias, filtro morfológico PT), `viz/` (figuras Plotly), `data/` (corpora, léxicos PT/EN, analogias). Veja [`docs/arquitetura.md`](docs/arquitetura.md).

## Licença

Código sob MIT (ver `LICENSE`). Dados conforme as licenças de cada fonte (tabela acima). Textos didáticos: CC BY 4.0.
