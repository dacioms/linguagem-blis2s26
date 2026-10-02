# Peters et al. (2018) — *Deep contextualized word representations* (ELMo)

## Ficha

- **Título:** Deep contextualized word representations
- **Autores:** Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner (Allen Institute for AI); Christopher Clark, Kenton Lee, Luke Zettlemoyer (University of Washington)
- **Ano:** 2018
- **Venue:** NAACL-HLT 2018 (Best Paper); arXiv 1802.05365 (v2 em 22 mar. 2018)
- **Link:** https://arxiv.org/abs/1802.05365 — http://allennlp.org/elmo
- **Classificação na disciplina:** leitura **complementar** (eixo "representações contextuais")

## Em uma frase

ELMo substitui o vetor fixo por palavra por uma **função de toda a sentença**: uma combinação linear, aprendida por tarefa, dos estados internos de um modelo de linguagem bidirecional profundo pré-treinado, o que melhora o estado da arte em seis tarefas e revela que camadas baixas codificam sintaxe e camadas altas, semântica.

## Problema e motivação

Embeddings pré-treinados (word2vec, GloVe) eram componentes padrão de quase todos os sistemas neurais de PLN em 2017, mas atribuem **um único vetor independente de contexto** a cada palavra. Uma boa representação deveria modelar (1) características complexas do uso (sintaxe e semântica) e (2) como esse uso varia entre contextos — a **polissemia**. *Play* em "spectacular play on the grounder" (jogada) e "Broadway play" (peça) recebe o mesmo vetor do GloVe.

Tentativas anteriores: enriquecer com subpalavras (fastText), aprender um vetor por sentido (Neelakantan et al. 2014), ou representações contextuais como context2vec, CoVe (codificador de tradução automática, limitado pela escassez de corpora paralelos) e TagLM (Peters et al. 2017, que usava só a camada superior de um LM). ELMo generaliza esses esforços: usa dados monolíngues abundantes (~30 milhões de sentenças), um biLM profundo, e — a inovação central — **expõe todas as camadas** ao modelo da tarefa, deixando-o escolher a mistura de sinais.

## Premissas necessárias para entender

- **Modelo de linguagem (LM):** $p(t_1,\dots,t_N) = \prod_k p(t_k\mid t_1,\dots,t_{k-1})$; perplexidade como métrica.
- **LSTM** e **biLSTM** (redes recorrentes com memória, nas duas direções); conexões residuais.
- **CNN sobre caracteres** e **highway layers** para formar a representação de um token a partir de seus caracteres (ligação com fastText).
- **Softmax** sobre vocabulário e **softmax-normalização** de pesos escalares.
- **Aprendizado semissupervisionado / transferência:** pré-treinar sem rótulos, depois usar em tarefas supervisionadas com pesos congelados.
- **Regularização** ($\ell_2$, dropout) e **layer normalization**.
- Noções das seis tarefas: SQuAD (QA extrativa), SNLI (inferência textual), SRL (papéis semânticos), correferência, NER e SST-5 (sentimento em 5 classes); métricas $F_1$ e acurácia.
- **WSD** (desambiguação de sentidos) e **POS tagging** como sondas intrínsecas.

## Método / modelo em detalhe

### Modelo de linguagem bidirecional (Seção 3.1)

Para $N$ tokens $(t_1,\dots,t_N)$, o LM direto computa

$$p(t_1,\dots,t_N) = \prod_{k=1}^{N}p(t_k\mid t_1,\dots,t_{k-1})$$

Cada token recebe uma representação independente de contexto $\mathbf x_k^{LM}$ (via CNN de caracteres), passada por $L$ camadas de LSTM; a camada $j$ produz o estado $\overrightarrow{\mathbf h}_{k,j}^{LM}$, e o topo $\overrightarrow{\mathbf h}_{k,L}^{LM}$ alimenta um softmax que prevê $t_{k+1}$. O LM reverso faz o mesmo da direita para a esquerda:

$$p(t_1,\dots,t_N) = \prod_{k=1}^{N}p(t_k\mid t_{k+1},\dots,t_N)$$

produzindo $\overleftarrow{\mathbf h}_{k,j}^{LM}$. O **biLM** maximiza a soma das log-verossimilhanças das duas direções:

$$\sum_{k=1}^{N}\Big(\log p(t_k\mid t_1,\dots,t_{k-1};\Theta_x,\overrightarrow\Theta_{LSTM},\Theta_s) + \log p(t_k\mid t_{k+1},\dots,t_N;\Theta_x,\overleftarrow\Theta_{LSTM},\Theta_s)\Big)$$

com os parâmetros da representação de tokens $\Theta_x$ e do softmax $\Theta_s$ **compartilhados** entre direções, e LSTMs separadas. As duas direções não se veem (não é um transformer com atenção bidirecional).

### ELMo (Seção 3.2, equação 1)

Para cada token $t_k$, um biLM de $L$ camadas gera $2L+1$ vetores:

$$R_k = \{\mathbf x_k^{LM},\ \overrightarrow{\mathbf h}_{k,j}^{LM},\ \overleftarrow{\mathbf h}_{k,j}^{LM}\mid j=1,\dots,L\} = \{\mathbf h_{k,j}^{LM}\mid j = 0,\dots,L\}$$

onde $\mathbf h_{k,0}^{LM}$ é a camada de token e $\mathbf h_{k,j}^{LM} = [\overrightarrow{\mathbf h}_{k,j}^{LM};\overleftarrow{\mathbf h}_{k,j}^{LM}]$ a concatenação das duas direções. ELMo colapsa $R_k$ em um vetor por tarefa:

$$\mathbf{ELMo}_k^{task} = E(R_k;\Theta^{task}) = \gamma^{task}\sum_{j=0}^{L}s_j^{task}\,\mathbf h_{k,j}^{LM} \qquad (1)$$

- $s^{task} = \mathrm{softmax}(\cdot)$: pesos escalares normalizados, um por camada, **aprendidos pela tarefa**;
- $\gamma^{task}$: escalar que ajusta a escala do vetor inteiro, "de importância prática para a otimização" (sem ele, o caso só-última-camada falhou em SNLI e nem treinou em SRL);
- opcionalmente aplica-se layer normalization a cada camada antes da soma, pois as ativações têm distribuições distintas.

O caso particular $s_L = 1$ recupera TagLM e CoVe (só a camada do topo).

### Como incluir ELMo em uma tarefa (Seção 3.3)

1. Rodar o biLM pré-treinado (pesos **congelados**) e gravar todas as camadas para cada token.
2. No modelo supervisionado, que normalmente forma $\mathbf x_k$ (embedding + caracteres) e depois $\mathbf h_k$ (biRNN/CNN), concatenar $[\mathbf x_k;\mathbf{ELMo}_k^{task}]$ na entrada.
3. Para algumas tarefas (SNLI, SQuAD), também concatenar na saída: $[\mathbf h_k;\mathbf{ELMo}_k^{task}]$, com um segundo conjunto de pesos.
4. Regularizar: dropout sobre ELMo e, em alguns casos, penalidade $\lambda\|\mathbf w\|_2^2$ nos pesos de camada, que empurra $s^{task}$ para a média das camadas.

### Arquitetura pré-treinada (Seção 3.4)

Baseada em Józefowicz et al. (2016) com treino conjunto das direções e conexão residual entre camadas. Dimensões reduzidas à metade do CNN-BIG-LSTM: $L = 2$ camadas biLSTM com 4 096 unidades e projeções de 512; entrada por **2 048 filtros convolucionais** de n-gramas de caracteres, duas highway layers e projeção para 512. Treinado por 10 épocas no 1B Word Benchmark: perplexidade média (direto/reverso) de **39,7** contra 30,0 do CNN-BIG-LSTM direto. Em geral, afina-se o biLM (*fine-tuning* de uma época, sem rótulos) nos dados da tarefa: a perplexidade cai muito (SNLI: 72,1 → 16,8) e o ganho na tarefa é dependente do domínio.

## Experimentos e resultados principais

### Tabela 1 — seis tarefas (teste)

| Tarefa | SOTA anterior | Baseline próprio | ELMo + baseline | Ganho abs. / rel. |
|---|---|---|---|---|
| SQuAD ($F_1$) | 84,4 | 81,1 | **85,8** | +4,7 / 24,9% |
| SNLI (acur.) | 88,6 | 88,0 | **88,7 ± 0,17** | +0,7 / 5,8% |
| SRL ($F_1$) | 81,7 | 81,4 | **84,6** | +3,2 / 17,2% |
| Coref ($F_1$ médio) | 67,2 | 67,2 | **70,4** | +3,2 / 9,8% |
| NER ($F_1$) | 91,93 ± 0,19 | 90,15 | **92,22 ± 0,10** | +2,06 / 21% |
| SST-5 (acur.) | 53,7 | 51,4 | **54,7 ± 0,5** | +3,3 / 6,8% |

Em toda tarefa, simplesmente adicionar ELMo estabelece novo estado da arte, com reduções relativas de erro de 6 a 25%. Em SQuAD, o ganho (+4,7) é bem maior que o de CoVe (+1,8); um ensemble de 11 modelos atinge 87,4. Em SST-5, trocar CoVe por ELMo no BCN dá +1,0.

### Tabela 2 — todas as camadas vs. só a última (dev)

| Tarefa | Baseline | Só última | Todas, $\lambda=1$ | Todas, $\lambda=0{,}001$ |
|---|---|---|---|---|
| SQuAD | 80,8 | 84,7 | 85,0 | **85,2** |
| SNLI | 88,1 | 89,1 | 89,3 | **89,5** |
| SRL | 81,6 | 84,1 | 84,6 | **84,8** |

Usar todas as camadas supera só a última; deixar os pesos variarem livremente ($\lambda$ pequeno) é melhor que forçá-los à média ($\lambda=1$), exceto em NER (pouco dado), insensível a $\lambda$.

### Tabela 3 — onde incluir ELMo (dev)

SQuAD: entrada 85,1; entrada+saída **85,6**; saída 84,8. SNLI: 88,9 / **89,5** / 88,7. SRL: **84,7** / 84,3 / 80,9. Tarefas com atenção após o biRNN (SNLI, SQuAD) lucram com ELMo também na saída; SRL (e correferência) preferem só a entrada.

### Tabelas 5 e 6 — o que cada camada captura

| Sonda | 1ª camada biLM | 2ª camada biLM | CoVe 1ª / 2ª | Referência |
|---|---|---|---|---|
| WSD ($F_1$, vizinho mais próximo) | 67,4 | **69,0** | 59,4 / 64,7 | 70,1 (Iacobacci 2016), 65,9 (1º sentido WordNet) |
| POS (acur., classificador linear) | **97,3** | 96,8 | 93,3 / 92,8 | 97,8 (Ling 2015) |

A **segunda camada** é melhor para desambiguar sentidos (semântica); a **primeira** é melhor para classes gramaticais (sintaxe). Em ambas as sondas, o biLM supera o CoVe, o que explica a vantagem nas tarefas finais. A Figura 2 mostra os pesos $s^{task}$ aprendidos: na entrada, as tarefas favorecem a primeira camada (fortemente em correferência e SQuAD); na saída, os pesos são mais equilibrados.

### Eficiência amostral (Seção 5.4, Figura 1)

Com ELMo, o modelo de SRL supera o máximo do baseline (que levou 486 épocas) já na época 10 — 98% menos atualizações. Variando o tamanho do conjunto de treino de 0,1% a 100%: os ganhos são maiores com pouco dado; em SRL, ELMo com **1%** dos dados iguala o baseline com **10%**.

## Análises e discussões do artigo

- A tese validada por ablação: representações **profundas** (mistura de camadas) superam a camada do topo, seja de um biLM ou de um codificador de tradução.
- Diferentes camadas codificam diferentes tipos de informação, coerente com Belinkov et al. (2017) e Søgaard & Goldberg (2016) em outras redes profundas; daí a utilidade de expor todas.
- O biLM fornece representações mais transferíveis que o CoVe para WSD e POS.
- ELMo age como um tipo de semissupervisão: o modelo da tarefa seleciona os sinais úteis.
- Limitações admitidas: o parâmetro $\gamma$ é um truque de otimização necessário; a melhor configuração (onde incluir, $\lambda$, layer norm, fine-tuning do biLM) varia por tarefa e foi escolhida empiricamente; o biLM é menor que o melhor LM disponível, por custo computacional.

## Críticas e limitações (visão contemporânea)

- O biLM é apenas "superficialmente bidirecional": cada direção vê só um lado; BERT (Devlin et al. 2018) substituiu LSTMs por transformers com atenção bidirecional e *masked LM*, superando ELMo em todas as tarefas em poucos meses, e o paradigma mudou de "features congeladas" para **fine-tuning** do modelo inteiro.
- Custo computacional alto para a época (LSTMs de 4 096 unidades; rodar o biLM em cada sentença).
- A mistura **escalar** por camada é rígida; a análise de "sintaxe vs. semântica" é sugestiva, mas sondas lineares são métricas limitadas (Hewitt & Liang 2019).
- Como os vetores dependem do contexto, as métricas de viés de Bolukbasi não se aplicam diretamente; trabalhos posteriores (May et al. 2019; Zhao et al. 2019) mostraram que ELMo também codifica viés de gênero, propagado para tarefas como correferência.
- Treinado só em inglês; versões para outras línguas (incluindo português, via AllenNLP e trabalhos da comunidade) vieram depois.

## Conexões com os outros artigos da disciplina

- Fecha a linha "contagem → predição → subpalavra → **contexto**": o vetor deixa de ser uma entrada de tabela (word2vec, GloVe) ou uma soma de n-gramas (fastText) e vira a saída de uma rede sobre a sentença.
- Herda do **fastText** a ideia de construir tokens a partir de caracteres (CNN de caracteres resolve OOV).
- Usa **GloVe** como embedding de entrada nos baselines de SQuAD, SRL e SNLI — ELMo é *adicionado*, não substitui, mostrando complementaridade.
- A Tabela 4 (*play*) é a resposta direta à limitação de polissemia que **Mikolov 2013a/b** e **Hartmann 2017** assumem sem resolver.
- Responde à crítica metodológica de Hartmann/Faruqui: avaliação extrínseca em seis tarefas, com sondas intrínsecas apenas como análise.
- Para **Bolukbasi 2016**, muda o objeto do debiasing: não há mais um vetor por palavra para neutralizar.

## Pontos para a apresentação

- Começar com *play* na Tabela 4: GloVe mistura *played, game, players*; o biLM separa "jogada" de "peça de teatro".
- Explicar o biLM como dois modelos de linguagem (esquerda→direita e direita→esquerda) com embedding de caracteres compartilhado.
- Desenhar a equação (1) como "média ponderada das camadas, com pesos que cada tarefa aprende, vezes um escalar $\gamma$".
- Receita de uso em 3 passos: congelar o biLM, concatenar na entrada (e às vezes na saída), treinar a tarefa.
- Tabela 1: seis tarefas, seis novos estados da arte, 6–25% menos erro.
- Tabela 5/6: camada 1 = sintaxe (POS 97,3), camada 2 = semântica (WSD 69,0).
- Eficiência amostral: 1% dos dados com ELMo ≈ 10% sem.
- Fechar com o contexto histórico: 8 meses depois, BERT.

**Perguntas para discussão**
1. Por que expor *todas* as camadas é melhor do que usar a camada final, se a final "já viu" as anteriores?
2. Como medir viés de gênero em uma representação que muda a cada sentença?
3. O que se perde ao abandonar o vetor estático (interpretabilidade, aritmética de analogias, custo)?

## Glossário mínimo

- **ELMo:** Embeddings from Language Models; representação contextual por token.
- **biLM:** modelo de linguagem bidirecional (LM direto + LM reverso treinados juntos).
- **$\mathbf h_{k,j}^{LM}$:** estado da camada $j$ para o token $k$ (concatenação das duas direções).
- **$s^{task}$, $\gamma^{task}$:** pesos de camada (softmax) e escalar global aprendidos por tarefa.
- **CNN de caracteres + highway:** rede que forma $\mathbf x_k$ a partir dos caracteres do token.
- **CoVe:** vetores contextuais extraídos de um codificador de tradução automática (McCann et al. 2017).
- **TagLM:** uso da última camada de um LM como feature para etiquetagem (Peters et al. 2017).
- **Perplexidade:** exponencial da entropia cruzada do LM.
- **WSD:** desambiguação de sentido de palavra.
- **SRL:** rotulação de papéis semânticos ("quem fez o quê a quem").
- **Eficiência amostral:** quanto dado rotulado é necessário para atingir dado desempenho.
