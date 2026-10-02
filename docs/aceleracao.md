# Aceleração: paralelismo e GPU

Resposta curta: **sim, há ganho**, mas ele vem de lugares diferentes do que se costuma supor.

| Onde o tempo ia | O que foi feito | Ganho medido (CPU, 4 núcleos) | Com GPU (RTX 4060 / 5050) |
|---|---|---|---|
| GloVe numpy (`np.add.at`, 1 *thread*) | `GloVeJax`: época compilada (`jit` + `fori_loop`) com *scatter-add* | **5–8×** (32,5 s → 6,1 s por 3 épocas em 60k sentenças) | lotes de 2^14–2^16 pares inteiros na placa; esperado > 20× |
| Skip-gram/fastText numpy *from scratch* | `SkipGramJax` (NEG, subpalavras opcionais), lotes de 512 (CPU) / 4096 (GPU) com recorte do gradiente agregado | **1,7×** (30,9 s → 18,5 s; 8 épocas, corpus sintético) | o passo é dominado por *gather/scatter* de linhas: grande ganho em GPU |
| Sondagens lineares (POS, NER, probes do ELMo) com `LogisticRegression` (lbfgs) | `LinearClassifier(backend="jax")`: softmax + Adam em mini-lotes, compilado | **6,7–9×** (24–28 s → 3,6 s; 60k tokens × 503 traços × 26 classes); acurácia 0,905 → 0,898 | frações de segundo |
| biLM (ELMo-lite) em JAX | já em JAX; *bucketing* de comprimento evita recompilações | 5× (recompilação por sentença eliminada) | treino de 2500 passos cai de ~6 min para < 1 min |
| Avaliar 7 modelos em sequência (nb 07, 13, 14) | `parallel_map` (joblib/loky) | **não ajuda em 4 núcleos** (0,5×: cada processo já usa 4 *threads* e paga a serialização); padrão `REPLANG_JOBS = cpus // 4` | — |
| Executar 16 notebooks em sequência | `replang notebooks run --jobs N` | em 4 núcleos, `--jobs 2` ≈ 1,3×; em 8–16 núcleos, `--jobs 3–4` ≈ 2–3× | — |
| gensim (word2vec, fastText, Doc2Vec) | já multi-*thread* (`REPLANG_WORKERS`) | — | gensim não usa GPU |

Números completos reproduzíveis com `uv run python scripts/benchmark_accel.py` (tabela impressa em Markdown). Os valores de GPU não foram medidos aqui (o ambiente de construção não tem GPU) e são estimativas a partir do perfil dos kernels; execute o *benchmark* na sua máquina.

## Como ligar

```bash
# CPU (padrão): o backend JAX é usado automaticamente quando instalado
uv sync --extra dev --extra contextual
uv run replang info            # mostra "aceleração: backend=jax · dispositivo=CPU (...)"

# GPU NVIDIA — Linux ou WSL2 (JAX não tem build nativo para Windows com CUDA)
uv sync --extra dev --extra cuda
uv run replang info            # deve mostrar dispositivo=GPU (NVIDIA GeForce RTX 4060 ...)
```

Variáveis de ambiente: `REPLANG_BACKEND=numpy|jax|auto`, `REPLANG_DEVICE=cpu|gpu|auto`, `REPLANG_JOBS=N`, `REPLANG_WORKERS=N` (gensim).

### RTX 4060 (Ada Lovelace, sm_89)
Funciona com as *wheels* `jax[cuda12]` atuais; *driver* NVIDIA ≥ 525 (CUDA 12). 8 GB de VRAM são mais do que suficientes para todos os modelos deste projeto (os maiores têm ~ 50k × 100 floats).

### RTX 5050 (Blackwell, sm_120)
Exige CUDA **≥ 12.8** e *driver* recente (série 570+); versões antigas do `jaxlib` não incluem kernels para sm_120 e caem em CPU silenciosamente — confira com `uv run replang info` ou `python -c "import jax; print(jax.devices())"`. Se aparecer só `CpuDevice`, atualize `jax`/`jaxlib` (`uv lock --upgrade-package jax`) e o *driver*.

### Windows
Use WSL2 (Ubuntu) com o *driver* NVIDIA do Windows; dentro do WSL, `uv sync --extra cuda`. Alternativa sem GPU: o backend JAX em CPU já traz os ganhos da primeira coluna.

## O que **não** muda com GPU
* O gensim (modelos "de produção" treinados em Machado e no STF) é C/Cython multi-*thread*; para 6 M tokens leva ~1–2 min por modelo em 4 núcleos — a GPU não o acelera.
* A leitura dos corpora e a extração de janelas de traços para POS/NER são Python puro; o ganho vem de paralelismo de processos (`REPLANG_JOBS`) em máquinas com muitos núcleos.
* As figuras Plotly e o Streamlit não usam GPU.

## Onde cada backend é usado

| Componente | numpy | JAX |
|---|---|---|
| `replang.models.glove_np.GloVe` (didático) | ✔ | — |
| `replang.models.glove_jax.GloVeJax` (treino em `replang train`, nb 04) | — | ✔ |
| `replang.models.word2vec_np.Word2Vec`, `fasttext_np.FastText` (didáticos, mostram o gradiente) | ✔ | — |
| `replang.models.word2vec_jax.SkipGramJax` (nb 10, experimentos repetidos) | — | ✔ |
| `replang.eval.linear.LinearClassifier` (POS, NER, sondagens; `backend="auto"`) | sklearn | ✔ (padrão quando instalado) |
| `replang.models.bilm_jax.BiLM` (ELMo-lite) | — | ✔ |
| `replang.eval.parallel.parallel_map` (vários modelos) | processos | processos |

Os módulos numpy foram mantidos **intencionalmente**: são os que os notebooks abrem para mostrar o passo de gradiente linha a linha. As versões JAX têm a mesma API e os mesmos resultados (verificado nos testes `tests/test_accel.py`).
