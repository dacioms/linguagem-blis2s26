# Mikolov et al. (2013a) — *Efficient Estimation of Word Representations in Vector Space* (word2vec)

> **Nota:** resumo a partir de conhecimento prévio (sem acesso ao PDF nesta sessão); confira os números no original antes de citá-los em sala.

## Ficha

- **Título:** Efficient Estimation of Word Representations in Vector Space
- **Autores:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (Google)
- **Ano:** 2013
- **Venue:** ICLR 2013 (Workshop track); arXiv 1301.3781
- **Link:** https://arxiv.org/abs/1301.3781 — código original: https://code.google.com/archive/p/word2vec/
- **Classificação na disciplina:** leitura **obrigatória** (eixo "word2vec")

## Em uma frase

O artigo propõe duas arquiteturas log-lineares simples — **CBOW** e **Skip-gram** — que, ao eliminar a camada oculta não linear dos modelos de linguagem neurais, aprendem vetores de palavras de alta qualidade a partir de bilhões de palavras em horas, e introduz o teste de analogias ("rei − homem + mulher ≈ rainha") que se tornou o padrão de avaliação.

## Problema e motivação

Até 2013, a maioria dos sistemas de PLN tratava palavras como símbolos atômicos (índices em um vocabulário, *one-hot*), sem noção de similaridade. Representações distribuídas aprendidas por redes neurais existiam desde Bengio et al. (2003) (NNLM) e tinham sido usadas com sucesso (Collobert & Weston 2008; Turian et al. 2010; Mikolov et al. 2010 com RNNLM), mas o custo de treino limitava-as a corpora de centenas de milhões de palavras e dimensões de 50–100. Com dados realmente grandes, modelos simples (n-gramas) continuavam vencendo por mera escala.

A pergunta do artigo é: como aprender vetores de boa qualidade a partir de **bilhões** de palavras e vocabulários de **milhões** de tipos? A hipótese é que o gargalo é a camada oculta não linear do NNLM; ao removê-la, perde-se expressividade, mas ganha-se a capacidade de treinar em ordens de grandeza mais dados — e isso compensa. Um segundo objetivo é mostrar que os vetores capturam **regularidades linguísticas** além de similaridade simples: há múltiplos "graus de similaridade" (*big–bigger* como *small–smaller*) que podem ser expressos por operações algébricas.

## Premissas necessárias para entender

- **Hipótese distribucional** e representações *one-hot* vs. distribuídas (densas, de baixa dimensão).
- **Modelo de linguagem neural (NNLM, Bengio 2003):** camadas de entrada (projeção), oculta e softmax de saída.
- **Softmax** e **softmax hierárquico** com árvore de Huffman (reduz o custo de $V$ para $\log_2 V$).
- **Complexidade computacional** expressa como número de parâmetros acessados por exemplo.
- **Descida de gradiente estocástica** e *backpropagation*; treinamento distribuído (DistBelief).
- **Similaridade por cosseno** e busca de vizinho mais próximo.
- **LSA/LDA** como alternativas de contagem anteriores.

## Método / modelo em detalhe

### Medida de complexidade

Os autores comparam arquiteturas pela complexidade de treinamento

$$O = E\times T\times Q$$

em que $E$ é o número de épocas (tipicamente 3–50), $T$ o número de palavras do corpus (até $10^9$) e $Q$ o custo por exemplo, específico de cada modelo.

**NNLM:** com $N$ palavras anteriores codificadas em $D$ dimensões, camada oculta $H$ e vocabulário $V$:

$$Q = N\times D + N\times D\times H + H\times V$$

O termo $H\times V$ é dominado pelo softmax (resolvido por softmax hierárquico, $H\times\log_2 V$), restando $N\times D\times H$ como gargalo.

**RNNLM:** $Q = H\times H + H\times V$; a recorrência $H\times H$ é o gargalo.

### CBOW (Continuous Bag-of-Words)

Semelhante ao NNLM sem camada oculta: os vetores das $N$ palavras de contexto (por exemplo, 4 à esquerda e 4 à direita) são **somados/mediados** na camada de projeção — a ordem não importa, daí "saco de palavras" — e usados para prever a palavra central por um classificador log-linear:

$$Q = N\times D + D\times\log_2 V$$

### Skip-gram

Inverte a direção: a palavra central prevê cada palavra do contexto dentro de uma janela de raio $C$. Como palavras distantes são menos relacionadas, sorteia-se para cada exemplo um raio $R\in\{1,\dots,C\}$ e preveem-se $2R$ palavras, dando menos peso às distantes:

$$Q = C\times(D + D\times\log_2 V)$$

Em ambos, a saída usa softmax hierárquico com árvore de Huffman. Cada palavra tem um vetor de entrada (a matriz de projeção) e um vetor de saída; o resultado final é a matriz de entrada.

### Aritmética de analogias

Para responder $a : b :: c : ?$ computa-se $\mathbf y = \mathbf x_b - \mathbf x_a + \mathbf x_c$ e busca-se a palavra cujo vetor tem o maior cosseno com $\mathbf y$ (excluindo as três palavras da pergunta). O exemplo célebre: $\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman}) \approx \mathrm{vec}(\text{queen})$.

### Conjunto de teste

O **Semantic-Syntactic Word Relationship test set** contém 8 869 analogias semânticas (5 tipos: capital comum, todas as capitais, moeda, cidade–estado, família) e 10 675 sintáticas (9 tipos: adjetivo–advérbio, oposto, comparativo, superlativo, particípio presente, nacionalidade, passado, plural de substantivo, plural de verbo), geradas combinando pares de palavras. Só respostas exatas contam; sinônimos são erros.

## Experimentos e resultados principais

Corpus de treino: notícias do Google com cerca de 6 bilhões de tokens; vocabulário restrito ao 1 milhão de palavras mais frequentes. Os números abaixo são aproximados (de memória).

**Efeito de dimensão e dados (CBOW, subconjunto de 30 mil palavras):** aumentar só a dimensão ou só o corpus satura; é preciso crescer os dois juntos. Dobrar o corpus custa aproximadamente o mesmo que dobrar a dimensão.

**Comparação de arquiteturas (mesmos dados, 640 dimensões, aproximadamente):**

| Modelo | Semântica (%) | Sintática (%) |
|---|---|---|
| RNNLM | ~9 | ~36 |
| NNLM | ~23 | ~53 |
| CBOW | ~24 | ~64 |
| Skip-gram | ~55 | ~59 |

CBOW é bom em sintaxe e fraco em semântica; Skip-gram é muito melhor em semântica — um padrão que Hartmann et al. (2017) reencontram em português.

**Modelos públicos anteriores:** vetores de Collobert & Weston, Turian, Mnih e Huang (50–100 dimensões, centenas de milhões de palavras) ficam abaixo de 20% no total; Skip-gram com 300 dimensões e 783 milhões de palavras chega a cerca de 50% em semântica e 56% em sintaxe (total ≈ 53%), e CBOW nos mesmos dados a ~16% / ~53%.

**Escala com DistBelief:** treinando em paralelo (dezenas a centenas de réplicas), Skip-gram de 1 000 dimensões em ~6 bilhões de palavras atinge cerca de 66% de acurácia total; o NNLM equivalente seria muito mais caro. O artigo também reporta resultado competitivo na tarefa de "escolha de sentença" da Microsoft Research (MSR Sentence Completion), com Skip-gram combinado a RNNLM.

**Exemplos qualitativos:** *France – Paris + Italy ≈ Rome*; *Einstein – scientist + Messi ≈ midfielder*; *copper – Cu + zinc ≈ Zn*; *Microsoft – Windows + Google ≈ Android*.

## Análises e discussões do artigo

- A simplicidade do modelo compensa a perda de não linearidade porque permite usar mais dados e mais dimensões; os vetores resultantes superam os de redes mais complexas.
- Semântica e sintaxe são capturadas em graus diferentes por CBOW e Skip-gram; as janelas e a ponderação por distância importam.
- Os autores reconhecem que a acurácia em analogias (≈60%) está longe de perfeita e que o teste é exigente (só respostas exatas).
- O artigo discute que vetores podem ser úteis em tradução automática, sistemas de perguntas e respostas e extensão de bases de conhecimento.
- Limitações admitidas: o softmax hierárquico ainda é um custo; o modelo ignora a ordem das palavras (CBOW) e não trata frases/expressões — temas do artigo seguinte (Mikolov 2013b).

## Críticas e limitações (visão contemporânea)

- A versão ICLR é um relatório técnico curto; muitos detalhes (hiperparâmetros, protocolo exato) só ficaram claros com o código e com o artigo de NIPS.
- O teste de analogias é enviesado para entidades nomeadas e flexões regulares; Linzen (2016) e outros mostram que a exclusão das palavras da pergunta e a estrutura de vizinhança inflam os resultados; Faruqui et al. (2016) e Hartmann et al. (2017) mostram fraca correlação com tarefas finais.
- Um vetor por tipo: nenhum tratamento de polissemia nem de morfologia (resolvidos, em parte, por fastText e ELMo).
- Levy & Goldberg (2014) demonstram que o Skip-gram é uma fatoração implícita de PMI, o que relativiza a dicotomia "contagem vs. predição".
- Os vetores herdam e amplificam vieses do corpus (Bolukbasi 2016; Caliskan 2017).
- Apesar disso, o artigo é um dos mais citados da história do PLN e definiu a *interface* (vetor denso por palavra, analogias por aritmética) sobre a qual toda a disciplina se organiza.

## Conexões com os outros artigos da disciplina

- É o ponto de partida da linha: substitui métodos de **contagem** (LSA) por **predição** e cria o benchmark de analogias usado por Mikolov 2013b, GloVe, fastText e Hartmann.
- **Mikolov 2013b** completa o modelo com negative sampling, subamostragem e frases.
- **GloVe** (Pennington 2014) propõe-se como alternativa que combina contagem global com a propriedade de analogias linear.
- **fastText** (Bojanowski 2017) é "Skip-gram + n-gramas de caracteres".
- **Doc2Vec** (Le & Mikolov 2014) estende CBOW/Skip-gram para parágrafos.
- **Hartmann 2017** treina CBOW e Skip-gram em português e reproduz o padrão CBOW fraco em semântica.
- **Bolukbasi 2016** mostra o lado sombrio da aritmética de analogias; **ELMo** (Peters 2018) abandona o vetor estático.

## Pontos para a apresentação

- Contextualizar: em 2013, *one-hot* e n-gramas dominavam; redes neurais eram lentas demais para a escala do Google.
- Desenhar CBOW e Skip-gram lado a lado como "contexto → palavra" e "palavra → contexto", sem camada oculta.
- Explicar a complexidade $O = E\times T\times Q$ e por que remover $N\times D\times H$ muda tudo.
- Demonstrar *rei − homem + mulher* nos embeddings NILC e discutir quando falha.
- Mostrar que CBOW e Skip-gram capturam coisas diferentes (semântica vs. sintaxe).
- Enfatizar a lição "dados + dimensão juntos": escala como fator de qualidade.
- Antecipar as críticas: analogias como métrica, viés, polissemia.
- Conectar com a Tabela 2 de Hartmann (CBOW < 10% em semântica).

**Perguntas para discussão**
1. Por que a remoção da camada oculta não destrói a capacidade de capturar relações semânticas?
2. O teste de analogias mede "entendimento" ou regularidades estatísticas do corpus? Qual a diferença prática?
3. Que relações da língua portuguesa (gênero, concordância, clíticos) o teste de analogias original não cobre?

## Glossário mínimo

- **Representação distribuída:** vetor denso em que o significado está espalhado pelas dimensões.
- **CBOW:** Continuous Bag-of-Words; prevê a palavra central a partir da média do contexto.
- **Skip-gram:** prevê o contexto a partir da palavra central.
- **Camada de projeção:** tabela de vetores (matriz $V\times D$) que substitui a entrada *one-hot*.
- **Softmax hierárquico:** saída organizada em árvore binária (Huffman), custo $\log_2 V$.
- **NNLM / RNNLM:** modelos de linguagem neurais feed-forward / recorrentes.
- **DistBelief:** infraestrutura de treino distribuído do Google.
- **Analogia vetorial:** $\mathbf x_b - \mathbf x_a + \mathbf x_c$ seguido de vizinho mais próximo por cosseno.
- **Semantic-Syntactic Word Relationship test set:** 19 544 analogias em 14 categorias.
- **Janela de contexto $C$:** número máximo de palavras à esquerda e à direita consideradas.
