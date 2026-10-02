# Le & Mikolov (2014) — *Distributed Representations of Sentences and Documents* (Paragraph Vector / Doc2Vec)

> **Nota:** resumo a partir de conhecimento prévio (sem acesso ao PDF nesta sessão); confira os números no original antes de citá-los em sala.

## Ficha

- **Título:** Distributed Representations of Sentences and Documents
- **Autores:** Quoc V. Le, Tomas Mikolov (Google)
- **Ano:** 2014
- **Venue:** ICML 2014 (PMLR vol. 32); arXiv 1405.4053
- **Link:** https://arxiv.org/abs/1405.4053
- **Classificação na disciplina:** leitura **complementar** (eixo "do vetor de palavra ao vetor de texto")

## Em uma frase

O **Paragraph Vector** estende o word2vec para textos de tamanho arbitrário: cada sentença, parágrafo ou documento recebe um vetor próprio, treinado para ajudar a prever as palavras do texto (PV-DM) ou prever palavras amostradas dele (PV-DBOW), produzindo representações densas que superam *bag-of-words* em sentimento e recuperação de informação.

## Problema e motivação

A representação dominante de textos em 2014 era o **saco de palavras** (BoW) ou de n-gramas, eventualmente com pesos TF-IDF. Ela tem dois problemas: perde a **ordem** das palavras (*"bom, não ruim"* e *"ruim, não bom"* viram o mesmo vetor) e ignora a **semântica** (*powerful*, *strong* e *Paris* ficam igualmente distantes). Alternativas que usavam vetores de palavras — média ponderada dos vetores, ou composição por árvores sintáticas (redes recursivas de Socher) — exigiam *parsing* ou descartavam a ordem, e eram difíceis de aplicar a documentos longos.

A proposta é um método **não supervisionado** que aprende um vetor por "parágrafo" (qualquer trecho, de frase a documento) com o mesmo mecanismo de predição de contexto do word2vec, sem necessidade de análise sintática nem rótulos, e que pode ser usado como feature em classificadores convencionais.

## Premissas necessárias para entender

- **CBOW e Skip-gram** (Mikolov 2013a): prever uma palavra a partir do contexto por um classificador log-linear.
- **Softmax hierárquico** e **negative sampling** (Mikolov 2013b) para tornar a predição viável.
- **Matriz de projeção (lookup):** vetores armazenados como linhas de uma matriz, indexados por id.
- **Descida de gradiente estocástica** e *backpropagation* através da tabela de vetores.
- **Bag-of-words / TF-IDF** e por que ignoram ordem e semântica.
- Tarefas de **análise de sentimento** (Stanford Sentiment Treebank, IMDB) e métricas de erro.
- Noção de **inferência** de um vetor novo em tempo de teste (otimização com parâmetros congelados).

## Método / modelo em detalhe

### Ponto de partida: word2vec como predição

No modelo de palavras, dada a sequência $w_1,\dots,w_T$, maximiza-se

$$\frac{1}{T}\sum_{t=k}^{T-k}\log p(w_t\mid w_{t-k},\dots,w_{t+k})$$

com $p$ dado por um softmax (hierárquico, na prática) sobre $y = b + Uh(w_{t-k},\dots,w_{t+k};W)$, em que $W$ é a matriz de vetores de palavras, $h$ concatena ou media os vetores de contexto, e $U, b$ são os parâmetros do softmax.

### PV-DM (Distributed Memory)

Cada parágrafo recebe um id e um vetor em uma matriz $D$, e cada palavra um vetor em $W$. Para prever a palavra $w_t$ em uma janela deslizante dentro do parágrafo, o vetor do parágrafo é **concatenado** (ou mediado) com os vetores das palavras de contexto:

$$h = h\big(d,\ w_{t-k},\dots,w_{t+k};\ D, W\big)$$

e $p(w_t\mid \cdot) = \mathrm{softmax}(b + Uh)$. O vetor $d$ é o mesmo para todas as janelas do mesmo parágrafo (daí "memória distribuída": ele guarda "o que falta" no contexto, o tópico do texto), enquanto $W$ é compartilhado entre todos os parágrafos. Treinam-se $D$, $W$, $U$ e $b$ por SGD. Com concatenação, a ordem dentro da janela é preservada, como em um modelo de linguagem com janela.

### PV-DBOW (Distributed Bag of Words)

Variante análoga ao Skip-gram: ignora-se o contexto e usa-se **apenas** o vetor do parágrafo para prever palavras sorteadas de uma janela aleatória do texto. Exige guardar menos parâmetros (não precisa de $W$) e é mais rápido; é tipicamente mais fraco sozinho, mas os autores recomendam **concatenar PV-DM e PV-DBOW** como representação final, pois a combinação é mais consistente.

### Inferência para textos novos

Em tempo de teste, o parágrafo novo não tem vetor em $D$. Adiciona-se uma coluna nova a $D$ e faz-se descida de gradiente **só nela**, mantendo $W$, $U$ e $b$ congelados, até convergir. O vetor resultante é então usado como feature em um classificador (regressão logística, SVM ou rede neural). A inferência é, portanto, uma pequena otimização por documento, não um simples *lookup*.

### Hiperparâmetros usados

Nos experimentos de sentimento, os autores usaram vetores de 400 dimensões para parágrafos e palavras, janela de 8 palavras no Stanford Sentiment Treebank (aproximadamente) e janela de 10 no IMDB, com softmax hierárquico, combinando PV-DM e PV-DBOW por concatenação (800 dimensões no total). A janela e a dimensão foram escolhidas por validação.

## Experimentos e resultados principais

Números aproximados, de memória; verificar no original.

### Stanford Sentiment Treebank (SST)

Frases de críticas de filmes, com rótulos em 5 classes (muito negativo a muito positivo) e em 2 classes (positivo/negativo), usando um classificador logístico sobre o vetor do parágrafo:

| Modelo | Erro 5 classes (%) | Erro 2 classes (%) |
|---|---|---|
| Naive Bayes / SVM com bigramas | ~58–59 | ~16–18 |
| Média de vetores de palavras | ~67 | ~20 |
| Rede recursiva (RNN, Socher) | ~56,8 | ~17,6 |
| Matrix-Vector RNN | ~55,6 | ~17,1 |
| Recursive Neural Tensor Network (RNTN) | ~54,3 | ~14,6 |
| **Paragraph Vector** | **~51,3** | **~12,2** |

O Paragraph Vector supera o RNTN, que exige árvores de *parsing* e supervisão em cada nó, sem usar estrutura sintática.

### IMDB (documentos longos)

100 mil críticas de filmes (25 mil treino, 25 mil teste, 50 mil sem rótulo, usadas para treinar os vetores). Os parágrafos têm várias sentenças, onde métodos recursivos não se aplicam diretamente:

| Modelo | Erro (%) |
|---|---|
| BoW / bigramas + classificador | ~11–12 |
| LDA, LSA e variantes | ~10–17 |
| NBSVM com bigramas (Wang & Manning 2012) | ~8,8 |
| **Paragraph Vector** | **~7,4** |

Redução relativa de erro de cerca de 15% sobre o melhor método anterior.

### Recuperação de informação

Tarefa construída pelos autores: dados três parágrafos (dois resultados do mesmo query de busca e um de outro), identificar qual não pertence; mede-se a distância entre vetores. O Paragraph Vector reduziu o erro para aproximadamente 3,8%, contra cerca de 5,6% do melhor BoW/bigrama e acima de 10% para a média de vetores de palavras.

### Análises adicionais

PV-DM é consistentemente melhor que PV-DBOW, mas a combinação dos dois é a mais robusta; a janela de contexto afeta o desempenho em sentimento, com faixa ótima entre 5 e 12 palavras; o custo de inferência em teste é paralelizável e proporcional ao tamanho do documento.

## Análises e discussões do artigo

- O vetor de parágrafo é interpretado como uma **memória** do que o contexto local não diz — o tópico ou a "essência" do texto.
- Como o treinamento é não supervisionado, o método funciona com poucos dados rotulados e pode aproveitar grandes volumes de texto sem rótulo (os 50 mil documentos do IMDB).
- A abordagem é agnóstica ao tamanho do texto: frases, parágrafos e documentos tratados do mesmo modo.
- Os autores admitem que a inferência em teste exige otimização (mais lenta que um *lookup*) e que a escolha da janela e da combinação PV-DM/PV-DBOW é empírica.
- Trabalhos futuros apontados: aplicar os vetores a parsing, tradução e outras tarefas; estudar composição além de parágrafos.

## Críticas e limitações (visão contemporânea)

- **Reprodutibilidade:** os resultados do IMDB foram difíceis de replicar; Mesnil et al. (2014) e a implementação do `gensim` obtiveram números piores, e o próprio Mikolov sugeriu em listas de discussão que o desempenho depende de detalhes (ordem de treinamento, taxa de aprendizado). Lau & Baldwin (2016) fizeram uma avaliação empírica cuidadosa e mostraram que PV-DBOW com vetores de palavras pré-treinados é a configuração mais confiável — invertendo a recomendação original.
- O método é **transdutivo**: vetores de novos documentos exigem otimização e são sensíveis à semente; documentos semelhantes podem receber vetores diferentes a cada inferência.
- O modelo continua, no fundo, um saco de janelas: a ordem só é capturada localmente (na concatenação do PV-DM); negação e estrutura de longo alcance não são modeladas.
- Foi superado por codificadores de sentença supervisionados ou contrastivos (InferSent, Universal Sentence Encoder, Sentence-BERT) e por modelos contextuais; mas permanece um baseline leve e sem supervisão útil para agrupar documentos.
- Os vieses dos vetores de palavras (Bolukbasi 2016) se propagam para os vetores de documentos e, por consequência, para classificadores de sentimento, triagem de currículos etc.

## Conexões com os outros artigos da disciplina

- Extensão direta de **Mikolov 2013a/b**: PV-DM é CBOW com um token extra "id do parágrafo"; PV-DBOW é Skip-gram com o parágrafo como palavra central; usa softmax hierárquico e os mesmos truques de treinamento.
- Fornece o elo entre vetor de **palavra** e vetor de **texto**, pergunta que a apresentação precisa responder ("o que o vetor carrega?" no nível do documento).
- Contrasta com a abordagem simples de **Hartmann 2017** para similaridade de sentenças (soma de vetores de palavras): Doc2Vec é a alternativa "aprendida" — e a média de vetores aparece como baseline fraco no artigo.
- **GloVe** e **fastText** também podem alimentar o PV-DBOW com vetores pré-treinados (Lau & Baldwin 2016).
- **ELMo** (Peters 2018) representa o passo seguinte: em vez de um vetor fixo por documento, um vetor por token dependente do contexto, consumido diretamente por modelos de tarefa.

## Pontos para a apresentação

- Motivar com o problema do BoW: "bom, não ruim" vs. "ruim, não bom".
- Mostrar PV-DM como CBOW com um "token extra" que é o id do parágrafo; PV-DBOW como Skip-gram do parágrafo.
- Enfatizar a inferência em teste: o vetor do documento novo é *otimizado*, não procurado.
- Resultados-chave: supera RNTN no SST e NBSVM no IMDB (erro ~7,4%).
- Alertar sobre reprodutibilidade (Lau & Baldwin 2016) — bom exemplo de ciência aberta e seus limites.
- Ligar à prática: `gensim.models.Doc2Vec` para agrupar textos em português com os embeddings NILC como inicialização.
- Perguntar o que o vetor de um documento "carrega" além do tópico — e que vieses herda.

**Perguntas para discussão**
1. Por que a média dos vetores das palavras vai tão mal em sentimento, enquanto o Paragraph Vector vai bem, se ambos ignoram grande parte da ordem?
2. A inferência por otimização torna o vetor de um mesmo documento não determinístico; isso é um problema para aplicações como busca ou auditoria?
3. Como comparar Doc2Vec com a média de vetores (Hartmann 2017) na tarefa ASSIN em português?

## Glossário mínimo

- **Paragraph Vector (PV):** vetor denso aprendido para um trecho de texto de tamanho arbitrário.
- **PV-DM:** Distributed Memory; o vetor do parágrafo é concatenado ao contexto para prever a palavra.
- **PV-DBOW:** Distributed Bag of Words; o vetor do parágrafo prevê palavras sorteadas do texto.
- **Matriz $D$ / $W$:** tabelas de vetores de parágrafos e de palavras.
- **Inferência:** otimização do vetor de um documento novo com os demais parâmetros fixos.
- **Bag-of-words (BoW):** contagem de palavras sem ordem; **TF-IDF** pondera por raridade.
- **SST:** Stanford Sentiment Treebank, frases com rótulos de sentimento em 5 e 2 classes.
- **IMDB:** corpus de 100 mil críticas de filmes para classificação binária.
- **RNTN:** Recursive Neural Tensor Network (Socher et al. 2013), composição por árvore sintática.
- **NBSVM:** SVM com features de Naive Bayes (Wang & Manning 2012), baseline forte de sentimento.
