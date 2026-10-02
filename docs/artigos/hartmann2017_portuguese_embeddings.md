# Hartmann et al. (2017) — *Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks*

## Ficha

- **Título:** Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks
- **Autores:** Nathan S. Hartmann, Erick Fonseca, Christopher D. Shulby, Marcos V. Treviso, Jéssica S. Rodrigues, Sandra M. Aluísio (ICMC-USP e UFSCar)
- **Ano:** 2017
- **Venue:** STIL 2017 (Symposium in Information and Human Language Technology); arXiv 1708.06025 (20 ago. 2017)
- **Link:** https://arxiv.org/abs/1708.06025 — modelos em http://nilc.icmc.usp.br/embeddings
- **Classificação na disciplina:** leitura **obrigatória** (eixo "avaliação em português")

## Em uma frase

Os autores treinam 31 modelos de word embeddings (GloVe, Word2Vec, Wang2Vec e FastText; 50 a 1 000 dimensões) em um corpus de 1,39 bilhão de tokens em português brasileiro e europeu, avaliam-nos em analogias e em duas tarefas reais (POS tagging e similaridade de sentenças) e concluem que o desempenho em analogias **não prediz** o desempenho nas tarefas.

## Problema e motivação

Embeddings já eram padrão em PLN em 2017, mas quase todos os modelos públicos e benchmarks eram para inglês. Para o português havia poucos recursos: Rodrigues et al. (2016) treinaram Skip-gram no LX-Corpus e traduziram o conjunto de analogias de Mikolov (LX-4WAnalogies, com versões PT-BR e PT-EU), obtendo 52,8% de acurácia; Sousa (2016) comparou CBOW, Skip-gram e GloVe em uma amostra de Wikipédia (melhor: CBOW 300d, 20,4% geral); Fonseca et al. (2015) e Fonseca & Aluísio (2016) avaliaram embeddings em POS tagging e mostraram que corpora maiores e misturar variantes ajuda.

Faltava uma comparação sistemática, com os mesmos dados, entre as duas famílias de métodos — **contagem** (LSA, HAL, GloVe) e **predição** (Word2Vec e derivados; divisão de Baroni et al. 2014) — e, sobretudo, um teste da crítica de Faruqui et al. (2016): analogias e similaridade intrínseca seriam avaliações inadequadas, devendo-se preferir tarefas finais. As contribuições declaradas são (i) disponibilizar 31 modelos e o script de pré-processamento e (ii) mostrar a falta de correlação entre avaliação intrínseca e extrínseca.

## Premissas necessárias para entender

- **Hipótese distribucional** e a diferença entre modelos de contagem (matriz de coocorrência fatorada) e preditivos (rede rasa que prevê contexto).
- **Word2Vec CBOW e Skip-gram** (Mikolov 2013a): prever a palavra central a partir do contexto ou o contexto a partir da palavra central.
- **GloVe** (Pennington 2014): regressão sobre o logaritmo das coocorrências.
- **Analogias vetoriais:** resolver $a : b :: c : ?$ por $\vec b - \vec a + \vec c$ e vizinho mais próximo por cosseno.
- **POS tagging** (etiquetagem morfossintática) e o corpus **Mac-Morpho**; o tagger neural **nlpnet**.
- **Similaridade semântica de sentenças** (ASSIN/PROPOR 2016): nota de 1 a 5 por par de sentenças; métricas correlação de Pearson $\rho$ e erro quadrático médio (MSE).
- **Avaliação intrínseca vs. extrínseca:** medir propriedades do espaço vetorial vs. medir a utilidade em uma tarefa final.

## Método / modelo em detalhe

### Corpus (Tabela 1)

Os autores compilaram 17 fontes, totalizando **1 395 926 282 tokens** e **3 827 725 tipos**, misturando português brasileiro e europeu (seguindo a evidência de que corpus maior é melhor, mesmo misto):

| Corpus | Tokens | Tipos | Gênero |
|---|---|---|---|
| LX-Corpus (Rodrigues et al. 2016) | 714 286 638 | 2 605 393 | Misto (PT-EU) |
| Wikipédia (dump 20/10/16) | 219 293 003 | 1 758 191 | Enciclopédico |
| GoogleNews | 160 396 456 | 664 320 | Informativo |
| SubIMDB-PT | 129 975 149 | 500 302 | Língua falada (legendas) |
| G1 (2014–2015) | 105 341 070 | 392 635 | Informativo |
| PLN-Br | 31 196 395 | 259 762 | Informativo |
| Obras literárias de domínio público | 23 750 521 | 381 697 | Prosa |
| Lacio-web | 8 962 718 | 196 077 | Misto |
| E-books em português | 1 299 008 | 66 706 | Prosa |
| Mundo Estranho | 1 047 108 | 55 000 | Informativo |
| CHC | 941 032 | 36 522 | Informativo |
| FAPESP | 499 008 | 31 746 | Divulgação científica |
| Livros didáticos | 96 209 | 11 597 | Didático |
| Folhinha | 73 575 | 9 207 | Informativo (infantil) |
| Subcorpus NILC | 32 868 | 4 064 | Informativo |
| Para Seu Filho Ler | 21 224 | 3 942 | Informativo |
| SARESP | 13 308 | 3 293 | Didático |
| **Total** | **1 395 926 282** | **3 827 725** | |

### Pré-processamento

Tokenização e normalização para **reduzir o vocabulário**: tipos com menos de 5 ocorrências substituídos por `UNKNOWN`; numerais normalizados para zeros; URLs → `URL`; e-mails → `EMAIL`. Tokenização por espaços e pontuação, com atenção à hifenização: pronomes clíticos como *machucou-se* são mantidos intactos. Como o LX-Corpus usava outra tokenização e é subconjunto do deles, ele foi retokenizado, sua seção de Wikipédia removida e apenas sentenças com 5 ou mais tokens foram mantidas (de 1,72 bi para 714 mi de tokens).

### Os quatro algoritmos

1. **GloVe** (Pennington et al. 2014): constrói matriz de coocorrência $M$ em que $M_{ij}$ reflete a probabilidade de $j$ ocorrer perto de $i$, e ajusta vetores para obedecer $w_i\cdot w_j + b_i + b_j = \log(M_{ij})$, com $b_i, b_j$ vieses escalares. Conhecido por capturar bem semântica.
2. **Word2Vec** (Mikolov et al. 2013): **CBOW** prevê a palavra omitida a partir do contexto; **Skip-gram** prevê os vizinhos a partir da palavra. Modelo log-linear com uma única matriz de pesos além dos embeddings.
3. **Wang2Vec** (Ling et al. 2015): modificação do Word2Vec sensível à **ordem das palavras**. *Continuous Window*: concatena os embeddings de contexto na ordem em que ocorrem. *Structured Skip-gram*: usa um conjunto de parâmetros distinto para prever cada posição relativa ao alvo. Espera-se melhor captura de sintaxe.
4. **FastText** (Bojanowski et al. 2016; Joulin et al. 2016): associa embeddings a **n-gramas de caracteres** e representa a palavra como soma deles, capturando morfologia.

Dimensões treinadas: **50, 100, 300, 600 e 1 000**. Com 4 algoritmos (Word2Vec, Wang2Vec e FastText em CBOW e Skip-gram, mais GloVe) chega-se a 7 configurações; Wang2Vec CBOW foi treinado só até 300 dimensões, dando os 31 modelos.

### Avaliação intrínseca

Analogias sintáticas e semânticas do LX-4WAnalogies (Rodrigues et al. 2016), nas versões PT-BR e PT-EU. O benchmark contém cinco tipos semânticos (capitais comuns, todas as capitais, moedas, cidades–estados, família) e nove sintáticos (adjetivo–advérbio, antônimos, comparativos, superlativos, particípios, nacionalidades, passado, plural de substantivos, plural de verbos).

### Avaliação extrínseca

- **POS tagging:** tagger **nlpnet** sobre o Mac-Morpho revisado, com configuração fixa de Fonseca et al. (2015): 20 épocas, 100 neurônios ocultos, taxa de aprendizado inicial 0,01, atributos de capitalização, sufixo e prefixo. Sem busca de hiperparâmetros: a ideia é comparar só os embeddings.
- **Similaridade de sentenças (ASSIN):** tarefa compartilhada do PROPOR 2016, com conjuntos PT-BR e PT-EU. O baseline de Hartmann (2016), vencedor da tarefa, usa regressão linear com duas features (cosseno TF-IDF e cosseno da soma dos embeddings das palavras). Aqui só a feature de embeddings é usada, para isolar o efeito do modelo. O baseline original (Word2Vec Skip-gram 600d, Wikipédia + G1 + PLN-Br) obteve $\rho = 0{,}58$ / MSE 0,50 (PT-BR) e $\rho = 0{,}55$ / MSE 0,83 (PT-EU).

## Experimentos e resultados principais

### Tabela 2 — analogias (acurácia %, melhores configurações por modelo)

| Modelo | Dim | PT-BR sint. | PT-BR sem. | PT-BR total | PT-EU sint. | PT-EU sem. | PT-EU total |
|---|---|---|---|---|---|---|---|
| FastText Skip-gram | 300 | **58,7** | 32,2 | 45,4 | **58,5** | 31,1 | 44,8 |
| FastText CBOW | 300 | 52,0 | 8,4 | 30,1 | 52,0 | 9,1 | 30,5 |
| GloVe | 300 | 45,8 | 45,8 | **46,7** | 45,9 | 42,3 | **46,2** |
| GloVe | 600 | 42,3 | **48,5** | 45,4 | 42,3 | **43,8** | 43,1 |
| Wang2Vec Skip-gram | 300 | 53,3 | 33,9 | 42,8 | 53,4 | 32,3 | 43,6 |
| Wang2Vec CBOW | 300 | 49,9 | 40,3 | 45,1 | 50,0 | 36,9 | 43,5 |
| Word2Vec Skip-gram | 600 | 35,6 | 20,0 | 33,4 | 35,3 | 17,6 | 33,5 |
| Word2Vec CBOW | 600 | 25,8 | 5,2 | 23,1 | 25,4 | 5,1 | 22,9 |

Leitura: **GloVe** é o melhor na média e em semântica; **FastText Skip-gram** domina as analogias sintáticas (morfologia) seguido de Wang2Vec; todos os CBOW, exceto Wang2Vec CBOW, vão muito mal em semântica (replicando Mikolov 2013a). Dimensões maiores que 300–600 não ajudam nas analogias; Word2Vec puro fica bem abaixo dos demais.

### Tabela 3 — POS tagging (acurácia %)

| Modelo | 50 | 100 | 300 | 600 | 1000 |
|---|---|---|---|---|---|
| FastText CBOW | 91,18 | 92,57 | 93,86 | 93,86 | 94,27 |
| FastText Skip-gram | 93,15 | 93,78 | 94,82 | 95,25 | 95,49 |
| GloVe | 93,13 | 93,73 | 94,76 | 95,23 | 95,57 |
| Wang2Vec CBOW | 95,33 | 95,59 | 95,83 | — | — |
| Wang2Vec Skip-gram | 95,07 | 95,70 | 95,89 | 95,88 | **95,94** |
| Word2Vec CBOW | 95,00 | 95,27 | 95,58 | 95,65 | 95,62 |
| Word2Vec Skip-gram | 94,79 | 95,18 | 95,66 | 95,82 | 95,81 |

Quanto maior a dimensão, melhor (exceção: Word2Vec 1 000 pouco pior que 600). **Wang2Vec** é o melhor (seu Skip-gram de 300d supera o Word2Vec de 1 000d); **GloVe e FastText são os piores**, surpreendentemente, dado que FastText "deveria" ajudar em morfologia. Os autores notam que as acurácias ficam abaixo de Fonseca et al. (2015), provavelmente porque os clíticos não foram separados dos verbos, gerando muitas palavras fora do vocabulário.

### Tabela 4 — similaridade de sentenças ASSIN (ρ de Pearson / MSE)

| Modelo | Dim | PT-BR ρ | PT-BR MSE | PT-EU ρ | PT-EU MSE |
|---|---|---|---|---|---|
| Wang2Vec Skip-gram | 1000 | **0,60** | **0,49** | 0,54 | 0,85 |
| Wang2Vec Skip-gram | 600 | 0,59 | 0,49 | 0,54 | **0,83** |
| Word2Vec CBOW | 600 | 0,57 | 0,51 | **0,55** | 0,86 |
| Word2Vec CBOW | 1000 | 0,58 | 0,50 | **0,55** | 0,87 |
| FastText Skip-gram | 300 | 0,55 | 0,53 | 0,40 | 1,02 |
| GloVe | 1000 | 0,51 | 0,56 | 0,46 | 0,94 |
| Baseline Hartmann (2016) | 600 | 0,58 | 0,50 | 0,55 | 0,83 |

Os modelos melhores em analogias semânticas (GloVe) **não** são os melhores na tarefa semântica. O melhor para PT-BR é Wang2Vec Skip-gram 1 000d (também o melhor em POS); para PT-EU é Word2Vec CBOW — justamente a família pior nas analogias. Nem FastText nem GloVe superam o baseline.

## Análises e discussões do artigo

- A conclusão central: resultados intrínsecos e extrínsecos **não se alinham**. GloVe vence nas analogias e perde (com FastText) nas duas tarefas; CBOW perde nas analogias e vence em ASSIN PT-EU. Isso corrobora Faruqui et al. (2016).
- **Wang2Vec** se mostra robusto em todas as avaliações; a hipótese é que a ordem das palavras ajuda tanto em sintaxe quanto em semântica (a posição de uma negação muda o sentido; um saco de palavras a embaralha).
- O resultado fraco do FastText em POS é descrito como surpreendente; os autores não o explicam, mas a nota sobre clíticos sugere um problema de tokenização que afeta todos os modelos.
- Limitações admitidas: hiperparâmetros das tarefas não foram otimizados (apenas uma configuração comparativa); vocabulário com muitos OOV por causa dos clíticos; trabalho futuro inclui outras tokenizações, lematização de verbos para reduzir vocabulário, e mais tarefas de avaliação.
- A escolha das tarefas foi deliberada: POS é morfossintática (como as analogias "sintáticas", que são na verdade morfológicas) e ASSIN é semântica, para testar a correspondência esperada — que não se confirmou.

## Críticas e limitações (visão contemporânea)

- Apenas **duas** tarefas extrínsecas, uma delas com um tagger neural pequeno e configuração fixa; com um modelo mais forte os embeddings podem fazer menos diferença. A falta de correlação pode ser parcialmente ruído de medição (sem múltiplas sementes nem intervalos de confiança).
- A métrica de similaridade de sentenças (cosseno da soma de vetores) é muito simplificada; conclusões sobre "semântica" dependem dessa escolha.
- Hiperparâmetros dos **embeddings** (janela, número de negativos, épocas, subamostragem) não são reportados em detalhe; Levy, Goldberg & Dagan (2015) mostram que eles importam mais que o algoritmo.
- A avaliação por analogias tem seus próprios problemas (Linzen 2016; Schluter 2018): o protocolo que exclui as três palavras-pergunta infla resultados.
- Não há análise de **viés** nos embeddings; dado o corpus (notícias, legendas, literatura), os estereótipos de Bolukbasi et al. provavelmente estão presentes — e com gênero gramatical.
- Mesmo assim, os **embeddings NILC** se tornaram o recurso padrão para português por anos e o artigo é uma referência de boas práticas: dados e scripts abertos, múltiplas dimensões, avaliação em tarefas.

## Conexões com os outros artigos da disciplina

- Implementa, lado a lado, o modelo de **contagem** (GloVe, Pennington 2014) e os **preditivos** (Word2Vec, Mikolov 2013a/b; Wang2Vec), mais o **subpalavra** (FastText, Bojanowski 2017), tornando-se um laboratório comparativo em português da linha histórica da disciplina.
- Confirma em português o achado de Mikolov 2013a de que CBOW é fraco em analogias semânticas, e o de Bojanowski de que n-gramas ajudam em analogias **sintáticas** (58,7%) — mas acrescenta que isso não se traduz em POS tagging.
- Serve de ponte para **Bolukbasi 2016**: os mesmos modelos podem ser auditados por viés de gênero com o eixo *ela–ele*.
- Antecipa **ELMo** (Peters 2018): se um vetor estático por palavra não prediz bem tarefas, representações contextuais avaliadas diretamente em tarefas são o passo seguinte.
- Reforça a lição metodológica de Faruqui (2016) e Nayak et al. (2016): avaliar embeddings com tarefas práticas.

## Pontos para a apresentação

- Apresentar a Tabela 1 como "o que é um corpus de 1,4 bilhão de tokens": mistura de notícias, legendas, Wikipédia e literatura, PT-BR + PT-EU.
- Destacar as decisões de pré-processamento (clíticos, `UNKNOWN`, numerais) e seu efeito colateral em POS tagging.
- Mostrar o gráfico mental da Tabela 2: GloVe ganha semântica, FastText ganha sintaxe, CBOW puro perde tudo.
- Virar a página: Tabela 3 e 4 invertem o ranking — GloVe e FastText vão para o fim.
- Mensagem central: "analogia não é proxy de utilidade"; citar Faruqui 2016.
- Wang2Vec como o "vencedor silencioso": ordem das palavras importa.
- Lembrar que os modelos NILC são públicos e podem ser carregados no notebook da apresentação.
- Propor o exercício: projetar profissões no eixo *ela–ele* nos embeddings NILC.

**Perguntas para discussão**
1. Se analogias não medem utilidade, o que exatamente elas medem? Vale a pena mantê-las como diagnóstico?
2. Por que a ordem das palavras (Wang2Vec) ajudaria em uma tarefa de similaridade que usa a soma dos vetores?
3. Como a morfologia do português (flexão de gênero e número, clíticos) muda a comparação FastText vs. Word2Vec em relação ao inglês?

## Glossário mínimo

- **Avaliação intrínseca:** testes sobre o próprio espaço vetorial (analogias, similaridade de palavras).
- **Avaliação extrínseca:** desempenho em uma tarefa final (POS, similaridade de sentenças).
- **LX-4WAnalogies:** benchmark de analogias de Mikolov traduzido e adaptado ao português (PT-BR e PT-EU) por Rodrigues et al. (2016).
- **Mac-Morpho:** corpus brasileiro anotado com classes gramaticais, versão revisada.
- **nlpnet:** tagger neural para português (Fonseca et al.).
- **ASSIN:** Avaliação de Similaridade Semântica e Inferência Textual, workshop do PROPOR 2016.
- **Wang2Vec:** Word2Vec sensível à ordem (Continuous Window e Structured Skip-gram, Ling et al. 2015).
- **Clítico:** pronome átono ligado ao verbo por hífen (*machucou-se*).
- **$\rho$ de Pearson / MSE:** correlação entre notas previstas e humanas / erro quadrático médio.
- **OOV (out-of-vocabulary):** palavra ausente no vocabulário do embedding.
