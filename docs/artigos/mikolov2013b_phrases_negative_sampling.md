# Mikolov et al. (2013b) — *Distributed Representations of Words and Phrases and their Compositionality*

## Ficha

- **Título:** Distributed Representations of Words and Phrases and their Compositionality
- **Autores:** Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean (Google)
- **Ano:** 2013
- **Venue:** NIPS 2013; arXiv 1310.4546 (16 out. 2013)
- **Link:** https://arxiv.org/abs/1310.4546
- **Classificação na disciplina:** leitura **complementar** (eixo "word2vec")

## Em uma frase

O artigo torna o Skip-gram prático e melhor em três frentes — *negative sampling* como alternativa simples ao softmax hierárquico, subamostragem de palavras frequentes e detecção de frases por pontuação de bigramas — e mostra que os vetores resultantes têm estrutura linear que permite tanto analogias quanto composição aditiva de significados.

## Problema e motivação

Mikolov et al. (2013a) haviam introduzido o Skip-gram, treinável em mais de 100 bilhões de palavras por dia em uma única máquina, porque dispensa multiplicações de matrizes densas. Mas restavam gargalos: o softmax completo sobre um vocabulário $W$ de $10^5$–$10^7$ palavras é inviável; o softmax hierárquico de Morin & Bengio é uma aproximação com custo $\log_2 W$, mas sua qualidade depende da árvore; palavras muito frequentes (*the*, *in*) dominam o treinamento sem trazer informação; e vetores de palavras isoladas não representam expressões idiomáticas (*Boston Globe* não é Boston + Globe; *Air Canada* não é ar + Canadá). O artigo propõe extensões simples para cada um desses problemas, publica os resultados em um conjunto de analogias com frases e descreve a propriedade de composicionalidade aditiva.

## Premissas necessárias para entender

- **Hipótese distribucional:** palavras com contextos semelhantes têm significados semelhantes.
- **Modelo de linguagem neural** e a função **softmax** $p(y) = \exp(z_y)/\sum_j \exp(z_j)$.
- **Vetores de entrada e saída** ($v_w$ e $v'_w$): cada palavra tem duas representações no Skip-gram.
- **Função logística** $\sigma(x) = 1/(1+e^{-x})$ e regressão logística binária.
- **Noise Contrastive Estimation (NCE):** treinar um modelo para distinguir dados de ruído.
- **Árvore de Huffman:** árvore binária que dá códigos curtos a símbolos frequentes.
- **Distribuição unigrama** $U(w)$: frequência relativa de cada palavra no corpus.
- **Analogias vetoriais** e vizinho mais próximo por cosseno (Mikolov 2013a).

## Método / modelo em detalhe

### Objetivo do Skip-gram (eq. 1)

Dada a sequência $w_1,\dots,w_T$, maximizar a log-verossimilhança média dos contextos:

$$\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\le j\le c,\ j\ne 0}\log p(w_{t+j}\mid w_t)$$

onde $c$ é o tamanho da janela (maior $c$ = mais exemplos e mais acurácia, ao custo de tempo).

### Softmax completo (eq. 2)

$$p(w_O\mid w_I) = \frac{\exp\left(v'^{\top}_{w_O} v_{w_I}\right)}{\sum_{w=1}^{W}\exp\left(v'^{\top}_{w} v_{w_I}\right)}$$

$v_w$ é o vetor de entrada, $v'_w$ o de saída, $W$ o tamanho do vocabulário. O gradiente custa $O(W)$ por exemplo — impraticável.

### Softmax hierárquico (eq. 3)

Representa a saída como uma árvore binária com as $W$ palavras nas folhas. Seja $n(w,j)$ o $j$-ésimo nó do caminho da raiz até $w$, $L(w)$ o comprimento do caminho, $\mathrm{ch}(n)$ um filho fixo de $n$ e $[\![x]\!]$ igual a 1 se $x$ é verdadeiro e $-1$ caso contrário:

$$p(w\mid w_I) = \prod_{j=1}^{L(w)-1}\sigma\!\left([\![\,n(w,j+1) = \mathrm{ch}(n(w,j))\,]\!]\cdot v'^{\top}_{n(w,j)} v_{w_I}\right)$$

Cada passo é uma decisão binária "vai para a esquerda ou direita"; o custo cai para $L(w_O) \approx \log_2 W$. Há um vetor $v'_n$ por nó interno em vez de um por palavra. Os autores usam uma **árvore de Huffman**, que dá caminhos curtos às palavras frequentes e acelera o treinamento.

### Negative sampling (eq. 4)

Simplificação do NCE: como só interessa a qualidade dos vetores (não a verossimilhança exata), substitui-se cada termo $\log p(w_O\mid w_I)$ por

$$\log\sigma\!\left(v'^{\top}_{w_O} v_{w_I}\right) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}\left[\log\sigma\!\left(-v'^{\top}_{w_i} v_{w_I}\right)\right]$$

A tarefa vira distinguir, por regressão logística, o contexto verdadeiro $w_O$ de $k$ "negativos" sorteados da distribuição de ruído $P_n(w)$. Valores de $k$ entre 5 e 20 servem para corpora pequenos; 2 a 5 bastam para grandes. Diferença para o NCE: NEG usa só amostras, sem as probabilidades numéricas do ruído. Empiricamente a melhor $P_n$ é a **unigrama elevada a 3/4**:

$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

que superou tanto a unigrama quanto a uniforme em todas as tarefas; o expoente achata a distribuição, dando mais chance a palavras raras como negativos.

### Subamostragem de palavras frequentes (eq. 5)

Cada ocorrência de $w_i$ é **descartada** com probabilidade

$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

em que $f(w_i)$ é a frequência da palavra e $t$ um limiar (tipicamente $10^{-5}$). Palavras com $f > t$ são agressivamente reduzidas, mas a ordem das frequências é preservada. A fórmula é heurística, mas acelera o treino (2x a 10x) e melhora os vetores de palavras raras — pois *France–Paris* informa mais que *France–the*.

### Aprendizado de frases (eq. 6)

Bigramas que ocorrem juntos com frequência, mas raramente em outros contextos, viram tokens únicos:

$$\mathrm{score}(w_i,w_j) = \frac{\mathrm{count}(w_i w_j) - \delta}{\mathrm{count}(w_i)\times\mathrm{count}(w_j)}$$

$\delta$ é um desconto que evita frases feitas de palavras muito raras. Bigramas acima de um limiar são unidos; rodam-se 2 a 4 passadas com limiar decrescente para formar frases de mais de duas palavras (*New York Times*, *Toronto Maple Leafs*), enquanto *this is* permanece separado.

### Composicionalidade aditiva

Os autores observam que somas simples são significativas: $\mathrm{vec}(\text{Russia}) + \mathrm{vec}(\text{river}) \approx \mathrm{vec}(\text{Volga River})$; $\mathrm{vec}(\text{Germany}) + \mathrm{vec}(\text{capital}) \approx \mathrm{vec}(\text{Berlin})$. A explicação: os vetores estão em relação linear com a entrada do softmax e representam, de forma logarítmica, a distribuição de contextos; a soma de dois vetores corresponde ao **produto** das duas distribuições de contexto — uma operação "E" — que favorece contextos prováveis para ambas as palavras.

## Experimentos e resultados principais

Dados: corpus interno de notícias do Google com 1 bilhão de palavras; vocabulário de 692 mil (mínimo 5 ocorrências); vetores de 300 dimensões; janela 5. Tarefa: analogias de Mikolov 2013a (*Germany:Berlin :: France:?*).

**Tabela 1 — acurácia (%) em analogias de palavras:**

| Método | Tempo (min) | Sintática | Semântica | Total |
|---|---|---|---|---|
| NEG-5 | 38 | 63 | 54 | 59 |
| NEG-15 | 97 | 63 | 58 | **61** |
| HS-Huffman | 41 | 53 | 40 | 47 |
| NCE-5 | 38 | 60 | 45 | 53 |
| *com subamostragem $10^{-5}$* | | | | |
| NEG-5 | 14 | 61 | 58 | 60 |
| NEG-15 | 36 | 61 | 61 | **61** |
| HS-Huffman | 21 | 52 | 59 | 55 |

Negative sampling supera o softmax hierárquico e o NCE; a subamostragem reduz o tempo em ~2,7x e melhora a parte semântica.

**Tabela 3 — analogias com frases (3 218 exemplos):** sem subamostragem NEG-5 24%, NEG-15 27%, HS-Huffman 19%; com subamostragem $10^{-5}$: 27%, 42% e **47%** (HS vira o melhor). Com 33 bilhões de palavras, 1 000 dimensões, HS e janela = sentença inteira: **72%**; reduzindo para 6 bilhões, 66% — a quantidade de dados é crucial.

**Tabelas 4–6 (qualitativas):** vizinhos de frases raras (*Vasco de Gama → Lingsugur / Italian explorer*), somas (*Czech + currency → koruna*; *French + actress → Juliette Binoche*) e comparação com Collobert & Weston, Turian e Mnih: o Skip-gram treinado em 30 bilhões de palavras em um dia dá vizinhos muito melhores para palavras raras (*ninjutsu → ninja, martial arts*).

## Análises e discussões do artigo

- A escolha de algoritmo e hiperparâmetros é **específica da tarefa**: os fatores mais importantes são arquitetura, dimensão, taxa de subamostragem e janela.
- HS com subamostragem foi o melhor para frases, mostrando que a subamostragem pode melhorar não só velocidade mas acurácia.
- A linearidade dos vetores é atribuída ao modelo log-linear, mas RNNs não lineares também mostram essa propriedade com mais dados.
- O trabalho se posiciona como **complementar** aos métodos recursivos de Socher para frases: tokens de frase + adição vetorial dão uma forma barata de representar trechos maiores.
- Limitações reconhecidas: a fórmula de subamostragem é heurística; a detecção de frases é "simples e orientada por dados", sem comparação com técnicas da literatura; resultados de modelagem de linguagem com $U^{3/4}$ não são reportados.

## Críticas e limitações (visão contemporânea)

- Levy & Goldberg (2014) mostraram que o Skip-gram com negative sampling fatora implicitamente uma matriz **PMI deslocada** ($\mathrm{PMI}(w,c) - \log k$), aproximando o modelo "preditivo" dos de contagem; Levy, Goldberg & Dagan (2015) mostraram que boa parte da vantagem vem de hiperparâmetros (subamostragem, $k$, $\alpha=3/4$) transferíveis a métodos de contagem.
- Um vetor por palavra ignora polissemia; a composição aditiva funciona para alguns casos nomeados, mas não é uma teoria de composicionalidade (negação, ordem e escopo não são capturados).
- O conjunto de analogias com frases é dominado por entidades nomeadas (times, companhias aéreas, executivos) — avalia conhecimento enciclopédico do corpus, não linguagem.
- Vieses dos dados (Bolukbasi 2016) se propagam e são amplificados pela mesma linearidade celebrada aqui.
- Mesmo assim, as três técnicas (NEG, subamostragem, frases) se tornaram padrão e estão presentes no `gensim` e no `fastText` até hoje.

## Conexões com os outros artigos da disciplina

- É a continuação direta de **Mikolov 2013a**: mesmo Skip-gram, agora com treinamento eficiente e melhor qualidade; o benchmark de analogias é o mesmo.
- **GloVe** (Pennington 2014) se apresenta como alternativa de contagem ao Skip-gram e usa exatamente o expoente 3/4 em sua função de ponderação — uma coincidência que não é casual.
- **fastText** (Bojanowski 2017) parte do Skip-gram com negative sampling descrito aqui (sua Seção 3.1 reproduz a eq. 4) e troca $v_w$ pela soma de n-gramas.
- **Hartmann 2017** treina Word2Vec e Wang2Vec com essas técnicas em português.
- **Bolukbasi 2016** usa o embedding do Google News treinado com estas técnicas (w2vNEWS) como objeto de estudo.
- **Doc2Vec** (Le & Mikolov 2014) estende a mesma ideia de predição de contexto para parágrafos.

## Pontos para a apresentação

- Explicar o problema do softmax com números: $W = 10^6$ exige um milhão de produtos escalares por palavra de treino.
- Mostrar negative sampling como "um classificador binário em vez de um de um milhão de classes".
- Comentar o expoente 3/4: o que acontece com a frequência de *the* vs. *koruna* quando elevadas a 0,75.
- Subamostragem: fórmula simples que dá 2–10x de velocidade e melhora palavras raras.
- Detecção de frases: *New York Times* como um token — ligação com NER e entidades.
- Demonstração ao vivo: $\mathrm{vec}(\text{Rússia}) + \mathrm{vec}(\text{rio})$ nos embeddings NILC.
- Resultado central: Tabela 1 (NEG-15 com subamostragem: 61%).
- Encaixar a crítica de Levy & Goldberg: predição ≈ fatoração de PMI.

**Perguntas para discussão**
1. Por que o produto de distribuições de contexto (soma de vetores) funciona como um "E" lógico e onde isso falha?
2. Negative sampling abandona a interpretação probabilística do modelo; isso importa se só queremos vetores?
3. Que vieses a detecção de frases pode introduzir (por exemplo, nomes próprios vs. expressões comuns)?

## Glossário mínimo

- **Skip-gram:** modelo que prevê palavras de contexto a partir da palavra central.
- **Softmax hierárquico (HS):** aproximação do softmax com árvore binária, custo $\log_2 W$.
- **Árvore de Huffman:** árvore que dá códigos curtos a palavras frequentes.
- **NCE:** estimação contrastiva de ruído; distingue dados de amostras de ruído.
- **Negative sampling (NEG):** versão simplificada do NCE com $k$ negativos por positivo.
- **$U(w)^{3/4}$:** distribuição unigrama suavizada usada para sortear negativos.
- **Subamostragem:** descarte probabilístico de palavras frequentes com limiar $t$.
- **Frase (phrase):** sequência de palavras tratada como token único após pontuação de bigramas.
- **Composicionalidade aditiva:** soma de vetores aproxima o vetor de um conceito combinado.
- **Vetores de entrada/saída ($v$, $v'$):** as duas matrizes de parâmetros do Skip-gram.
