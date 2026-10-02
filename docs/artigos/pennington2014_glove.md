# Pennington, Socher & Manning (2014) — *GloVe: Global Vectors for Word Representation*

> **Nota:** resumo a partir de conhecimento prévio (sem acesso ao PDF nesta sessão); confira os números no original antes de citá-los em sala.

## Ficha

- **Título:** GloVe: Global Vectors for Word Representation
- **Autores:** Jeffrey Pennington, Richard Socher, Christopher D. Manning (Stanford)
- **Ano:** 2014
- **Venue:** EMNLP 2014, pp. 1532–1543
- **Link:** https://aclanthology.org/D14-1162/ — projeto: https://nlp.stanford.edu/projects/glove/
- **Classificação na disciplina:** leitura **complementar** (eixo "modelos de contagem")

## Em uma frase

GloVe aprende vetores de palavras por regressão de mínimos quadrados ponderada sobre o **logaritmo das coocorrências globais** do corpus, unindo a eficiência estatística dos métodos de fatoração de matriz à estrutura linear (analogias) dos métodos preditivos como o word2vec.

## Problema e motivação

Em 2014 havia duas famílias de modelos de semântica distribucional. (1) **Fatoração de matriz global** — LSA (Deerwester 1990), HAL (Lund & Burgess 1996), e variantes com PMI e PPMI — que usa toda a estatística de coocorrência do corpus, mas produz espaços cujas dimensões principais são dominadas por palavras muito frequentes e que vão mal em analogias. (2) **Janelas locais com predição** — Skip-gram e CBOW (Mikolov 2013a/b) — que aprendem boa estrutura linear, mas varrem o corpus janela a janela, sem explorar a redundância das contagens repetidas, e cuja origem da propriedade de analogia era mal compreendida.

A pergunta dos autores é: que propriedade das estatísticas de coocorrência faz surgir a linearidade das analogias, e como construir um modelo que a capture diretamente? A resposta é que o significado está nas **razões** de probabilidades de coocorrência, não nas probabilidades brutas.

## Premissas necessárias para entender

- **Hipótese distribucional** e matriz de coocorrência palavra–contexto.
- **Probabilidade condicional** $P_{ij} = P(j\mid i)$ e razões de probabilidade.
- **Logaritmo** e por que $\log$ transforma produtos (razões) em somas (diferenças de vetores).
- **Mínimos quadrados ponderados** e função de perda.
- **Homomorfismo** entre grupos $(\mathbb R,+)$ e $(\mathbb R_{>0},\times)$: a função exponencial.
- **AdaGrad** (otimizador com taxa adaptativa).
- **Vetores de palavra e de contexto** ($w$ e $\tilde w$) e sua simetria.
- Noções de **PMI** e **LSA** para situar o modelo entre os de contagem.

## Método / modelo em detalhe

### Intuição: razões de coocorrência

Seja $X$ a matriz de coocorrência, $X_{ij}$ o número de vezes que a palavra $j$ aparece no contexto de $i$, $X_i = \sum_k X_{ik}$ e $P_{ij} = X_{ij}/X_i$. O exemplo do artigo compara $i = \text{ice}$ e $j = \text{steam}$ com palavras-sonda $k$: para $k = \text{solid}$, a razão $P_{ik}/P_{jk}$ é grande (≈ 8,9); para $k = \text{gas}$, é pequena (≈ 0,085); para $k = \text{water}$ (relacionada a ambas) e $k = \text{fashion}$ (a nenhuma), fica perto de 1. As **razões** discriminam melhor que as probabilidades isoladas, que são dominadas por palavras genéricas.

### Derivação do modelo

Procura-se uma função $F$ dos vetores de palavra $w_i, w_j$ e do vetor de contexto $\tilde w_k$ que capture a razão:

$$F(w_i, w_j, \tilde w_k) = \frac{P_{ik}}{P_{jk}}$$

Como se quer que a informação de $P_{ik}/P_{jk}$ seja codificada em **diferenças** de vetores (espaços vetoriais são lineares), toma-se $F(w_i - w_j, \tilde w_k)$; para transformar vetores em escalares de modo simples, usa-se o produto interno:

$$F\big((w_i - w_j)^\top\tilde w_k\big) = \frac{P_{ik}}{P_{jk}}$$

Exigindo que $F$ seja um homomorfismo entre a soma (nos argumentos) e a razão (nos valores), obtém-se $F = \exp$, de onde

$$w_i^\top\tilde w_k = \log P_{ik} = \log X_{ik} - \log X_i$$

O termo $\log X_i$ não depende de $k$, quebrando a simetria entre palavra e contexto; absorve-se em um viés $b_i$ e adiciona-se $\tilde b_k$ para restaurar a simetria:

$$w_i^\top\tilde w_k + b_i + \tilde b_k = \log X_{ik}$$

### Função de perda ponderada

A equação acima é mal definida quando $X_{ik} = 0$ e daria peso igual a coocorrências raras (ruidosas) e frequentes. Introduz-se uma ponderação $f(X_{ij})$:

$$J = \sum_{i,j=1}^{V} f(X_{ij})\left(w_i^\top\tilde w_j + b_i + \tilde b_j - \log X_{ij}\right)^2$$

com $V$ o tamanho do vocabulário e

$$f(x) = \begin{cases}(x/x_{\max})^{\alpha} & \text{se } x < x_{\max}\\ 1 & \text{caso contrário}\end{cases}$$

Propriedades desejadas: $f(0) = 0$ (zeros não contribuem; a soma percorre só entradas não nulas), $f$ não decrescente (raras pesam menos) e saturação (frequentíssimas não dominam). Os autores usam $x_{\max} = 100$ e $\alpha = 3/4$ — o mesmo expoente do negative sampling de Mikolov 2013b.

### Relação com o Skip-gram

Os autores mostram que o objetivo do Skip-gram, reescrito agrupando coocorrências iguais, equivale a uma entropia cruzada ponderada por $X_i$ sobre as mesmas estatísticas globais; trocar a entropia cruzada (que tem caudas pesadas e exige normalização custosa) por mínimos quadrados sobre logaritmos, com ponderação $f$, leva ao GloVe. O modelo é portanto um "meio-termo": usa contagens globais como LSA, mas com um objetivo que produz estrutura linear como word2vec.

### Complexidade e treinamento

O custo depende do número de entradas não nulas de $X$; assumindo uma lei de potência para as coocorrências, os autores estimam $|X| = O(|C|^{0{,}8})$, com $|C|$ o tamanho do corpus — melhor que o $O(|C|)$ de métodos por janela. Treinamento por AdaGrad, amostrando entradas não nulas, com taxa inicial 0,05; 50 iterações para dimensões < 300 e 100 para as demais. Janela de contexto de 10 à esquerda e 10 à direita, com peso decrescente $1/d$ pela distância $d$. Como $W$ e $\tilde W$ são equivalentes, o vetor final usado é $W + \tilde W$, o que dá um pequeno ganho (efeito de *ensemble*).

## Experimentos e resultados principais

Corpora: Wikipédia 2010 (1 bi de tokens), Wikipédia 2014 (1,6 bi), Gigaword 5 (4,3 bi), Gigaword 5 + Wikipédia 2014 (6 bi) e Common Crawl (42 bi); vocabulário das 400 mil palavras mais frequentes. Números aproximados, de memória:

**Analogias (Mikolov 2013a), acurácia %:**

| Modelo | Dim | Tokens | Semântica | Sintática | Total |
|---|---|---|---|---|---|
| CBOW | 300 | 1,6 bi | ~16 | ~53 | ~36 |
| Skip-gram | 300 | 1 bi | ~61 | ~61 | ~61 |
| GloVe | 300 | 1,6 bi | ~80 | ~62 | ~70 |
| GloVe | 300 | 6 bi | ~77 | ~67 | ~72 |
| GloVe | 300 | 42 bi | ~82 | ~69 | ~75 |

GloVe supera word2vec e os métodos de fatoração (SVD, SVD-S, SVD-L) com os mesmos dados, e mais dados ajudam (42 bi → 75%).

**Similaridade de palavras** (WordSim-353, MC, RG, SCWS, RW): correlação de Spearman superior a SVD-L, CBOW e Skip-gram na maior parte dos conjuntos; o ganho em RW é menor.

**NER (CoNLL-2003):** usando os vetores como features em um CRF, GloVe fica ligeiramente acima de CBOW/Skip-gram/HPCA nos conjuntos de teste (F1 em torno de 88–89 no CoNLL, aproximadamente).

**Ablações:** o desempenho cresce com a dimensão até ~300 e satura; janelas simétricas maiores ajudam a semântica, enquanto janelas assimétricas (só à esquerda) favorecem a sintaxe; sintaxe satura cedo com o tamanho do corpus, semântica continua a crescer. Em tempo de treino igual, GloVe supera o word2vec (comparação criticada posteriormente por favorecer o GloVe na escolha de épocas vs. número de negativos).

## Análises e discussões do artigo

- A principal contribuição conceitual é a explicação de **por que** analogias funcionam: razões de coocorrência são lineares no log, e o modelo as codifica diretamente.
- A ponderação $f$ é apresentada como essencial: sem ela, palavras raras ruidosas e palavras frequentíssimas distorcem o ajuste.
- A combinação $W + \tilde W$ e a simetria dos vieses são escolhas de design justificadas empiricamente.
- Os autores reconhecem que o modelo ignora a ordem das palavras dentro da janela (como CBOW) e que a comparação de tempo de treino com word2vec depende de muitos hiperparâmetros.
- O modelo é apresentado como global (usa todo o corpus de uma vez) e paralelizável, com complexidade sublinear no tamanho do corpus.

## Críticas e limitações (visão contemporânea)

- Levy, Goldberg & Dagan (2015), "Improving Distributional Similarity with Lessons Learned from Word Embeddings": com hiperparâmetros equalizados (janela, subamostragem, suavização de contexto, $W+\tilde W$), SGNS e PPMI-SVD empatam ou superam GloVe; muito da vantagem reportada vem de configuração, não do modelo.
- A derivação via homomorfismo é elegante mas heurística; outras funções $F$ também satisfariam as restrições.
- A matriz de coocorrência de um corpus de 42 bi de tokens consome muita memória; na prática, GloVe e word2vec têm custos comparáveis.
- Herda os vieses do corpus tanto quanto o word2vec — Bolukbasi et al. (2016) mostram correlação de 0,81 entre os vieses ocupacionais de GloVe e w2vNEWS.
- Vetor estático por palavra: sem polissemia, sem morfologia; em português, Hartmann et al. (2017) encontram GloVe ótimo em analogias e **pior** em POS tagging e similaridade de sentenças.
- Ainda assim, os vetores GloVe pré-treinados (6B, 42B, 840B, Twitter) foram a inicialização padrão de modelos de PLN até BERT, e o modelo permanece a referência da família "contagem".

## Conexões com os outros artigos da disciplina

- É o representante da família **contagem** na linha "contagem → predição", e explicitamente dialoga com **Mikolov 2013a/b**, reinterpretando o Skip-gram como entropia cruzada sobre coocorrências e emprestando o expoente 3/4.
- **Hartmann 2017** inclui GloVe como o melhor em analogias (46,7% total PT-BR) e o pior nas tarefas — o caso mais claro da desconexão intrínseco/extrínseco.
- **Bolukbasi 2016** usa GloVe treinado em web crawl para mostrar que o viés de gênero não é artefato do word2vec.
- **fastText** (Bojanowski 2017) e **ELMo** (Peters 2018) adicionam o que GloVe não tem: morfologia e contexto; ELMo usa GloVe como entrada em vários baselines.
- A ideia de "fatorar uma matriz de estatísticas" liga GloVe ao resultado de Levy & Goldberg sobre SGNS ≈ PMI deslocado.

## Pontos para a apresentação

- Começar pelo exemplo *ice/steam* com *solid, gas, water, fashion*: razões revelam significado, probabilidades não.
- Mostrar o caminho $F(\text{razão}) \to \exp \to \log X_{ik} = w_i^\top\tilde w_k + b_i + \tilde b_k$ no quadro.
- Explicar a função $f$ e seus três requisitos; destacar $\alpha = 3/4$ e $x_{\max} = 100$.
- Posicionar GloVe como "LSA que aprendeu com o word2vec" e "word2vec que usa contagens globais".
- Resultados: ~75% em analogias com 42 bi de tokens; mais dados ajudam semântica.
- Trazer a crítica de Levy et al. (2015): hiperparâmetros importam mais que o modelo.
- Ligar à Tabela 2 de Hartmann: melhor em analogias, pior nas tarefas.
- Mencionar que GloVe tem o mesmo viés que word2vec (Figura 4 de Bolukbasi).

**Perguntas para discussão**
1. Se SGNS e GloVe fatoram estatísticas parecidas, a dicotomia "contagem vs. predição" ainda faz sentido?
2. A ponderação $f$ silencia coocorrências raras: que tipo de palavras (e de falantes) são sub-representadas por isso?
3. O que a "explicação" das analogias via razões de coocorrência de fato explica — e o que deixa de fora (por exemplo, por que falha em muitas analogias)?

## Glossário mínimo

- **Matriz de coocorrência $X$:** contagens de pares palavra–contexto no corpus inteiro.
- **$P_{ij}$:** probabilidade de $j$ ocorrer no contexto de $i$.
- **Vetor de palavra $w$ / de contexto $\tilde w$:** os dois conjuntos de parâmetros, somados ao final.
- **Vieses $b_i, \tilde b_j$:** escalares que absorvem $\log X_i$ e restauram a simetria.
- **$f(X_{ij})$:** função de ponderação com corte $x_{\max}$ e expoente $\alpha$.
- **LSA / HAL / PPMI-SVD:** métodos de contagem anteriores.
- **AdaGrad:** otimizador com taxa de aprendizado adaptativa por parâmetro.
- **Lei de potência:** suposição sobre a distribuição das coocorrências usada para estimar a complexidade.
- **Analogia linear:** propriedade de que $w_b - w_a \approx w_d - w_c$ para relações análogas.
