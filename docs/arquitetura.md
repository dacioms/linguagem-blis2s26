# Arquitetura do repositório

```
linguagem-blis2s26/
├── pyproject.toml / uv.lock      gestão de pacotes com uv (extras: dev, contextual=JAX)
├── Makefile                      atalhos (setup, data, train, notebooks, app, test, lint)
├── configs/training.yaml         hiperparâmetros de referência (espelham os artigos)
├── src/replang/                  pacote Python
│   ├── config.py                 caminhos, fontes públicas (SOURCES), flags (REPLANG_FAST, REPLANG_TOP_N)
│   ├── embedding.py              WordVectors: cosseno, vizinhos, 3CosAdd/3CosMul, I/O (npz, word2vec txt, gensim)
│   ├── utils/   text.py          normalização/tokenização PT (Hartmann §2.1);  download.py  cache de downloads
│   ├── data/    corpora.py       Machado de Assis, Mac-Morpho, corpus sintético de gênero, toy corpus
│   │            embeddings.py    Wikipedia2Vec PT, GloVe EN (truncados), conector NILC, modelos locais
│   │            lexicons.py      pares definicionais/equalização, profissões, palavras de gênero (EN: debiaswe; PT: autoral)
│   │            analogies.py     questions-words (EN), LX-4WAnalogies (PT-BR/EU)
│   │            legal.py         corpora jurídicos (RulingBR, LeNER-Br, JurisBERT STS), léxico jurídico, perfil de corpus, deslocamento de domínio
│   ├── models/  cooccurrence.py  contagem → PPMI → SVD; one-hot; BoW/TF-IDF
│   │            word2vec_np.py   Skip-gram/CBOW, NEG/HS (Huffman), subamostragem, frases — numpy
│   │            glove_np.py      GloVe com AdaGrad — numpy
│   │            fasttext_np.py   n-gramas, FNV-1a, Skip-gram com subpalavras, OOV, importância de n-gramas
│   │            trainers.py      gensim (Word2Vec, FastText, Doc2Vec) com cache em data/models
│   │            bilm_jax.py      biLM de 2 camadas (ELMo-lite) em JAX; mistura de camadas
│   ├── eval/    analogies.py     acurácia por categoria;  similarity.py  Spearman + mini-conjuntos PT
│   │            extrinsic.py     POS tagging (janela + regressão logística), similaridade de sentenças (Pearson/MSE)
│   ├── bias/    geometry.py      PCA dos pares, direção g, DirectBias, β indireto, geração de analogias, SVM, filtro morfológico PT
│   │            debias.py        hard (Neutralize+Equalize), soft (gradiente), PairBias
│   ├── viz/plots.py              figuras Plotly reutilizadas por notebooks e app
│   └── cli.py                    `replang info|download|prepare|train|notebooks|app|test`
├── scripts/  train_models.py     treina todos os modelos locais;  build_notebooks.py  gera/executa .ipynb de notebooks/_src
├── notebooks/_src/nb_*.py        fonte dos notebooks (md/code) — editar aqui e rodar `replang notebooks build`
├── notebooks/*.ipynb             notebooks gerados e executados (saídas incluídas)
├── app/streamlit_app.py + views/ interface multipágina (11 páginas; a 10ª é o contexto jurídico)
├── tests/                        pytest (27 testes, sem rede)
├── data/samples/                 artefatos pequenos versionados (embeddings truncados, léxicos, analogias)
├── data/processed/               corpora normalizados (versionados: Machado, Mac-Morpho)
├── data/raw/, data/models/       downloads e modelos treinados (ignorados pelo git)
└── docs/                         estrutura didática, roteiro 4h, glossário, resumos dos artigos, juridico.md (eixo jurídico)
```

## Fluxo de dados

1. `replang download` → `data/raw` (GitHub raw, releases do gensim-data, S3 do Wikipedia2Vec).
2. `replang prepare` → `data/processed` (corpora) e `data/samples` (embeddings truncados a 50k palavras, float16).
3. `replang train` → `data/models` (gensim, numpy, JAX) com JSON de metadados.
4. `replang notebooks both` → executa `notebooks/*.ipynb` (≈ 1 h em 4 CPUs; `REPLANG_FAST=1` para versão rápida).
5. `replang app` → Streamlit lê `data/samples` e `data/models`.

## Decisões de projeto

* **Implementações didáticas em numpy** (legíveis, vetorizadas por mini-lote) + **gensim** para os modelos "de produção". As duas expõem a mesma interface `WordVectors`.
* **Dados públicos com fallback offline**: embeddings truncados versionados (≈ 14 MB) e corpora processados garantem que os notebooks rodem sem rede; conectores para NILC e ASSIN quando disponíveis.
* **JAX opcional** para o biLM (PyTorch CPU não estava acessível no ambiente de construção; JAX via PyPI é leve).
* **Notebooks gerados de fontes Python**: diffs legíveis, texto revisável, execução reprodutível via nbclient.
* **Figuras Plotly** com renderer `plotly_mimetype+notebook_connected`: JupyterLab renderiza offline; HTML carrega plotly.js via CDN, mantendo os `.ipynb` leves.
