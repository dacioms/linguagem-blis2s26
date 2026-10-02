# notebooks/

Os `.ipynb` desta pasta são **gerados** a partir das fontes Python em `_src/` (uma lista de células `md`/`code` por notebook) e **executados** com `nbclient`, de modo que as saídas (tabelas, figuras Plotly) estão gravadas.

```bash
uv run replang notebooks build            # regenera os .ipynb (sem saídas) a partir de _src/
uv run replang notebooks run              # executa todos (≈ 30 min em 4 CPUs; --jobs N em máquinas com 8+ núcleos; REPLANG_FAST=1 reduz)
uv run replang notebooks both --only 08   # regenera e executa um só
uv run jupyter lab .                      # abrir
```

Pré-requisitos: `uv sync --extra dev --extra contextual`, dados em `data/samples` (já versionados) e modelos em `data/models` (`uv run replang train` e `uv run replang train legal`). Sem o extra `contextual` (JAX), o notebook 12 não roda.

| # | Tema | Tempo aprox. de execução |
|---|---|---|
| 00 | setup, dados, mapa | 10 s |
| 01 | premissas: coocorrência, PPMI, SVD | 30 s |
| 02 | word2vec (Mikolov 2013a) | 1 min |
| 03 | extensões (Mikolov 2013b) | 40 s |
| 04 | GloVe | 1,5 min |
| 05 | fastText | 3 min |
| 06 | embeddings PT (Hartmann) | 2 min |
| 07 | avaliação extrínseca | 1 min (JAX) / 7 min (sklearn) |
| 08 | viés: geometria (Bolukbasi) | 1–5 min |
| 09 | debias | 3–8 min |
| 10 | viés em PT + experimento sintético | 2,5 min |
| 11 | Doc2Vec | 6 min |
| 12 | ELMo-lite (JAX) | 5 min |
| 13 | síntese | 0,5 min |
| 14 | contexto jurídico I: língua do Direito, deslocamento de domínio, NER/STS/área | 2 min |
| 15 | contexto jurídico II: viés, ranqueador gêmeo, debias, governança | 3 min |

As figuras usam o renderer `plotly_mimetype+notebook_connected`: o JupyterLab renderiza offline; a visualização HTML carrega `plotly.js` de um CDN.
