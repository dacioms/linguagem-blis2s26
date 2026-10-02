# Bolukbasi et al. (2016) — *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*

## Ficha

- **Título:** Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
- **Autores:** Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, Adam Kalai (Boston University e Microsoft Research New England)
- **Ano:** 2016
- **Venue:** NIPS 2016 (Advances in Neural Information Processing Systems 29); versão arXiv 1607.06520 (21 jul. 2016)
- **Link:** https://arxiv.org/abs/1607.06520
- **Classificação na disciplina:** leitura **obrigatória** (eixo "o viés que vem junto")

## Em uma frase

Os autores mostram que embeddings word2vec treinados em notícias do Google codificam estereótipos de gênero em uma direção geométrica identificável, definem métricas de viés direto e indireto, e propõem algoritmos (*hard* e *soft debiasing*) que removem essas associações das palavras neutras em gênero sem degradar a utilidade do embedding.

## Problema e motivação

Até 2016 havia centenas de trabalhos aplicando word embeddings (busca web, triagem de currículos, ranqueamento de documentos, análise de sentimento), mas nenhum havia apontado que esses vetores são **flagrantemente sexistas**. O embedding estudado é o word2vec público treinado em notícias do Google (3 milhões de palavras e termos, 300 dimensões; os autores o chamam **w2vNEWS**). A mesma aritmética vetorial que resolve $\vec{man} - \vec{woman} \approx \vec{king} - \vec{queen}$ também produz $\vec{man} - \vec{woman} \approx \vec{computer\ programmer} - \vec{homemaker}$ e "pai está para médico como mãe está para enfermeira".

O risco central é a **amplificação**: um sistema de busca que usa embeddings para melhorar relevância tende a ranquear a página de "John" acima da página idêntica de "Mary" para a consulta *cmu computer science phd student*, porque *computer programmer* está mais perto de nomes masculinos. Os autores observam ainda que contagens de primeira ordem são enganosas (*male nurse* é mais frequente que *female nurse*), mas embeddings, por serem métodos de segunda ordem, capturam o *reporting bias* implícito, funcionando como uma espécie de Teste de Associação Implícita (IAT) do corpus.

A distinção fundamental do artigo é entre palavras **específicas de gênero** por definição (*brother*, *sister*, *businesswoman*) e palavras **neutras em gênero** (*nurse*, *flight attendant*, *shoes*). O objetivo não é apagar o gênero do embedding, mas remover associações de gênero apenas das palavras neutras, preservando analogias definicionais como *she:queen :: he:king*.

## Premissas necessárias para entender

- **Hipótese distribucional** e aritmética de analogias em embeddings (Mikolov 2013a/b): relações aparecem como diferenças de vetores.
- **Produto interno e cosseno:** com vetores normalizados ($\|\vec w\| = 1$), similaridade é $\vec w_1 \cdot \vec w_2 = \cos(\vec w_1, \vec w_2)$.
- **Projeção sobre um subespaço:** para um subespaço $B$ com base ortonormal $\{b_1,\dots,b_k\}$, $v_B = \sum_j (v\cdot b_j)\,b_j$; o resíduo $v - v_B$ é ortogonal a $B$.
- **PCA / SVD:** decomposição que encontra as direções de maior variância de um conjunto de vetores.
- **SVM linear e validação cruzada estratificada:** usados para generalizar a lista de palavras específicas de gênero.
- **Programação semidefinida (SDP):** o *soft debiasing* é formulado como um problema convexo em uma matriz $X = T^\top T \succeq 0$.
- **Crowdsourcing (Amazon Mechanical Turk):** rótulos humanos de "estereótipo" vs. "analogia apropriada" por 10 trabalhadores dos EUA.

## Método / modelo em detalhe

### Preliminares e vocabulário

Embedding: vetor unitário $\vec w \in \mathbb{R}^d$ para cada palavra $w \in W$. $N \subset W$ é o conjunto de palavras neutras; $P \subset W\times W$ é um conjunto de pares F-M (ex.: *she-he*, *mother-father*). Do w2vNEWS, tomaram as 50 000 palavras mais frequentes, filtraram minúsculas com menos de 20 caracteres e ficaram com **26 377 palavras**.

### Geração de analogias (métrica $S_{(a,b)}$)

Dado um par-semente $(a,b)$ (ex.: *she, he*), os autores pontuam todos os pares $(x,y)$ por

$$S_{(a,b)}(x,y) = \begin{cases}\cos(\vec a - \vec b,\ \vec x - \vec y) & \text{se } \|\vec x - \vec y\| \le \delta\\ 0 & \text{caso contrário}\end{cases}$$

onde $\delta$ é um limiar de similaridade semântica. A intuição: a diferença $\vec x - \vec y$ deve ser quase paralela à direção-semente, mas $x$ e $y$ não podem estar longe demais (senão a analogia perde coerência). Usam $\delta = 1$, que, para vetores unitários, equivale a um ângulo $\le \pi/3$. Para evitar redundância, não retornam várias analogias com a mesma palavra $x$. Das 150 analogias *she-he* geradas, **72 foram julgadas apropriadas** e **29 estereotipadas** por ao menos 5 de 10 avaliadores (ex.: *sewing-carpentry*, *nurse-surgeon*, *housewife-shopkeeper*).

O Apêndice A explica por que não usar o "paralelogramo" $\min \|(\vec a - \vec b) - (\vec x - \vec y)\|$ nem 3CosMul: aqueles métodos tendem a retornar $y = x$ ou pares sem relação com gênero.

### Passo 1: direção de gênero via PCA dos pares

Pares individuais são ruidosos (*man* também é interjeição e verbo). Então tomam 10 pares de gênero $\{(x_i,y_i)\}_{i=1}^{10}$ (*she-he, her-his, woman-man, Mary-John, herself-himself, daughter-son, mother-father, gal-guy, girl-boy, female-male*), que classificam palavras femininas/masculinas sugeridas pela multidão com 75–93% de acurácia (Figura 5). Em geral, para conjuntos definidores $D_1,\dots,D_n$:

$$\mu_i := \sum_{w\in D_i} \vec w/|D_i|, \qquad \mathbf{C} := \sum_{i=1}^{n}\sum_{w\in D_i} (\vec w - \mu_i)^\top(\vec w - \mu_i)/|D_i|$$

e o **subespaço de viés** $B$ são as primeiras $k$ linhas de $\mathrm{SVD}(\mathbf C)$. Com $k=1$ obtém-se a **direção de gênero** $g$. A Figura 6 mostra que o primeiro componente explica muito mais variância que os demais (algo em torno de 60%), enquanto 10 vetores aleatórios de 300 dimensões dariam uma queda gradual — evidência de que há uma direção real, não ruído.

### Viés direto

Dado o conjunto de palavras neutras $N$ e a direção $g$:

$$\mathrm{DirectBias}_c = \frac{1}{|N|}\sum_{w\in N} |\cos(\vec w, g)|^c$$

onde $c$ controla a "severidade": $c = 0$ conta qualquer sobreposição com $g$ como viés total; $c = 1$ mede gradualmente. Com $N$ = 327 ocupações no w2vNEWS, $\mathrm{DirectBias}_1 = 0{,}08$, isto é, ocupações têm componente substancial ao longo de $g$. A projeção das ocupações em $\vec{she} - \vec{he}$ correlaciona com o julgamento humano de estereotipicalidade ($\rho$ de Spearman = 0,51) e é consistente entre w2vNEWS e GloVe treinado em web crawl ($\rho = 0{,}81$, Figura 4).

### Viés indireto

Mesmo removendo todas as palavras de gênero, *receptionist* continuaria mais perto de *softball* que de *football*. Decompondo $w = w_g + w_\perp$, com $w_g = (w\cdot g)g$, a contribuição do gênero para a similaridade entre $w$ e $v$ é

$$\beta(w,v) = \left(w\cdot v - \frac{w_\perp\cdot v_\perp}{\|w_\perp\|_2\|v_\perp\|_2}\right)\Big/\ w\cdot v$$

ou seja, a fração do produto interno original que desaparece quando projetamos fora o subespaço de gênero e renormalizamos. Propriedades: $\beta(w,w)=0$; se $w_g = 0 = v_g$ então $\beta = 0$; se $w_\perp = 0 = v_\perp$ então $\beta = 1$. No eixo *softball–football*, os $\beta$ de *receptionist*, *waitress* e *homemaker* com *softball* são 67%, 35% e 38%; os de *businessman* e *maestro* com *football* são 31% e 42% (Figura 3).

### Passo 2a: Hard debiasing (Neutralize + Equalize)

**Neutralize.** Para cada $w \in N$, reembeda-se

$$\vec w := (\vec w - \vec w_B)/\|\vec w - \vec w_B\|$$

removendo a componente no subespaço $B$ e renormalizando.

**Equalize.** Para cada conjunto de igualdade $E \in \mathcal E$ (ex.: $\{grandmother, grandfather\}$, $\{guy, gal\}$):

$$\mu := \sum_{w\in E} w/|E|, \qquad \nu := \mu - \mu_B,$$

$$\text{para cada } w\in E:\quad \vec w := \nu + \sqrt{1 - \|\nu\|^2}\ \frac{\vec w_B - \mu_B}{\|\vec w_B - \mu_B\|}.$$

Interpretação: fora de $B$ todas as palavras do conjunto recebem o mesmo vetor $\nu$ (a média); dentro de $B$ elas são centradas (média zero) e reescaladas pelo fator $\sqrt{1 - \|\nu\|^2}$ para que o vetor resultante tenha norma 1. A centralização importa: no embedding original tanto *male* quanto *female* têm projeção positiva (feminina) em $g$; sem centrar, as duas ficariam idênticas e se perderia *father:male :: mother:female*. A **Observação 1** garante que, após os passos 1 e 2a, toda palavra neutra $w$ é equidistante de qualquer par $e_1,e_2$ de um conjunto de igualdade ($\vec w\cdot\vec e_1 = \vec w\cdot\vec e_2$ e $\|\vec w-\vec e_1\| = \|\vec w-\vec e_2\|$), e portanto o viés de pares é zero.

### Passo 2b: Soft debiasing

Em vez de zerar, aprende-se uma transformação linear $T\in\mathbb R^{d\times d}$ que preserva produtos internos e minimiza a projeção das palavras neutras no subespaço $B$:

$$\min_T\ \|(TW)^\top(TW) - W^\top W\|_F^2 + \lambda\,\|(TN)^\top(TB)\|_F^2$$

onde $W$ é a matriz de todos os vetores, $N$ a matriz das palavras neutras e $\lambda$ equilibra os dois termos; usam $\lambda = 0{,}2$ (para $\lambda$ grande recupera-se o *hard debiasing*). Com $X = T^\top T$ o problema vira um SDP; como $W$ tem $300\times 400\,000$ entradas, aplicam SVD $W = U\Sigma V^\top$ e reduzem o problema a uma matriz $300\times300$ (Apêndice B). O embedding final é normalizado.

### Seção 7: identificando palavras específicas de gênero com SVM

Como o conjunto específico de gênero $S$ é muito menor, define-se $N = W\setminus S$. A partir de definições do WordNet, selecionam manualmente uma base $S_0$ de **218 palavras** nas 26 377. Para estender aos 3 milhões de termos do w2vNEWS, treinam uma **SVM linear** ($C = 1{,}0$) sobre os vetores e obtêm $S = S_0 \cup S_1$ com **6 449 palavras**. Com validação cruzada estratificada de 10 folds: $F$-score $0{,}627 \pm 0{,}102$ (as classes são desbalanceadas); acurácia balanceada $95{,}12\% \pm 1{,}46\%$ (regularização escolhida por CV aninhada). A Figura 7 mostra as palavras projetadas no eixo $\vec{she}-\vec{he}$ (horizontal) e na distância à fronteira da SVM (vertical): as neutras ficam acima da linha e seriam colapsadas ao eixo vertical pelo *hard debiasing*.

## Experimentos e resultados principais

**Preservação de utilidade (Tabela 1, vocabulário filtrado):**

| Embedding | RG | WS | Analogias MSR |
|---|---|---|---|
| Original | 62,3 | 54,5 | 57,0 |
| Hard-debiased | 62,4 | 54,1 | 57,0 |
| Soft-debiased | 62,4 | 54,2 | 56,8 |

No w2vNEWS completo (Tabela 3, Apêndice H): 76,1/70,0/71,2 antes e 76,5/69,7/71,2 após *hard debiasing*.

**Viés direto (Figura 8):** entre as 150 melhores analogias *she-he*, **19%** foram julgadas estereotipadas no embedding original contra **6%** após *hard debiasing*; o número de analogias apropriadas se mantém. Exemplo: *he:doctor :: she:X* passa de *nurse* para *physician*, enquanto *she:ovarian cancer :: he:prostate cancer* é preservada. O *soft debiasing* foi menos eficaz.

**Viés indireto:** após o debiasing, as palavras extremas no eixo *softball–football* passam a ser *pitcher, infielder, major leaguer* (relevantes) em vez de *receptionist, waitress, homemaker*; *pitcher* e *footballer* permanecem porque sua associação é funcional, não de gênero.

## Análises e discussões do artigo

- Uma única direção captura boa parte do gênero, mas o viés real é combinação de gênero e outros fatores (*mathematician*–*geometry* têm viés masculino, mas sua similaridade é legítima); a métrica $\beta$ mede só a parcela atribuível a $g$.
- Os autores admitem que a escolha de palavras neutras é **subjetiva** (*beard*, *estrogen*, *rabbi* são casos ambíguos; *nursing* como profissão vs. amamentar) e deve ser customizada por aplicação.
- Não há *ground truth* para viés indireto; a avaliação é qualitativa.
- O *hard debiasing* apaga distinções úteis (ex.: *to grandfather a regulation*); o *soft* é a alternativa quando se quer preservar mais a geometria.
- Reconhecem a objeção "o embedding só reflete a sociedade", mas argumentam que ao menos sistemas computacionais não devem **amplificar** o viés.
- O mesmo w2vNEWS exibe estereótipos raciais (eixo *minorities–whites*: *parliamentarian, lawyer* vs. *butler, crooner*); gênero gramatical em outras línguas é apontado como trabalho futuro.

## Críticas e limitações (visão contemporânea)

- **Gonen & Goldberg (2019), "Lipstick on a Pig":** mostram que o viés permanece recuperável após o *hard debiasing* — palavras estereotipadas continuam agrupadas entre si (um classificador recupera o gênero original com alta acurácia). Projetar fora uma direção esconde a métrica, não a informação.
- O **gênero binário** é assumido (direção F–M); identidades não binárias e intersecções com raça/classe ficam fora do modelo.
- A lista de pares definidores e as 218 palavras base são específicas do **inglês**; em português, com gênero gramatical em quase todos os substantivos e adjetivos, a separação "definicional vs. neutro" é muito mais difícil (ex.: *enfermeira/enfermeiro*).
- A avaliação por crowdworkers dos EUA codifica estereótipos de uma cultura específica.
- Para modelos contextuais (ELMo, BERT) a noção de "um vetor por palavra" não existe; métodos posteriores (SEAT, CEAT, INLP, contrafactual data augmentation) adaptam a ideia.
- O trabalho inspirou uma vasta literatura de medição (WEAT de Caliskan et al. 2017) e mitigação; sua contribuição mais duradoura é o **vocabulário conceitual** (direção de gênero, viés direto/indireto) e a demonstração clara do problema.

## Conexões com os outros artigos da disciplina

- Depende diretamente da **aritmética de analogias** de Mikolov 2013a/b e do embedding word2vec do Google News; o viés em **GloVe** (Pennington 2014) é mostrado como igualmente presente (Figura 4), evidenciando que o problema vem dos dados, não do algoritmo.
- O **fastText** (Bojanowski 2017) herdaria o mesmo viés, com o agravante de que n-gramas de caracteres carregam marcas morfológicas de gênero (*-eira*, *-eiro*).
- Para **Hartmann 2017**, a pergunta natural é: os embeddings NILC em português carregam os mesmos estereótipos ocupacionais? É o experimento que a apresentação pode propor.
- **ELMo** (Peters 2018) muda o objeto: o viés passa a depender do contexto e exige novas métricas.
- Metodologicamente, ecoa a crítica de Faruqui (2016) citada por Hartmann: analogias são um instrumento tanto de avaliação quanto de **auditoria**.

## Pontos para a apresentação

- Abrir com as duas analogias lado a lado (*king–queen* vs. *programmer–homemaker*): a mesma geometria que impressiona é a que ofende.
- Mostrar a Figura 1 (ocupações extremas *she/he*) e perguntar quais o público "adivinharia".
- Explicar o Passo 1 como "PCA das diferenças de pares": por que 10 pares e por que o primeiro autovalor domina.
- Diferenciar com exemplos **viés direto** (*nurse* perto de *she*) e **indireto** (*receptionist* perto de *softball*).
- Demonstrar Neutralize + Equalize em 2D no quadro (vetor, projeção, renormalização, centralização de *male/female*).
- Resultado-chave: 19% → 6% de analogias estereotipadas, sem perda em RG/WS/analogias.
- Fechar com a crítica de Gonen & Goldberg: "batom no porco" — o viés foi escondido, não removido.
- Ponte para o português: gênero gramatical torna o conjunto "neutro" problemático.

**Perguntas para discussão**
1. Em português, *enfermeira* é específica de gênero por definição ou é um estereótipo codificado na morfologia? Como adaptar $S_0$?
2. Debiasing é responsabilidade de quem treina o embedding, de quem o usa, ou de quem produz o corpus?
3. Se a direção $g$ pode ser removida, ela também pode ser *usada* para medir viés em novos corpora — isso é um instrumento científico legítimo?

## Glossário mínimo

- **w2vNEWS:** embedding word2vec de 300 dimensões treinado em notícias do Google (3 M de termos).
- **Palavra específica de gênero:** associada a um gênero por definição (*brother*, *queen*).
- **Palavra neutra em gênero:** não associada por definição (*nurse*, *softball*).
- **Direção de gênero $g$:** primeiro componente principal das diferenças dos pares F–M.
- **Subespaço de viés $B$:** generalização de $g$ para $k$ direções.
- **Viés direto:** projeção média de palavras neutras em $g$.
- **Viés indireto $\beta(w,v)$:** fração da similaridade entre duas palavras neutras explicada por $g$.
- **Neutralize:** remover a componente em $B$ de palavras neutras.
- **Equalize:** tornar palavras de um conjunto de igualdade equidistantes de todas as neutras.
- **Soft debiasing:** transformação linear $T$ que negocia preservação de produtos internos vs. remoção de viés ($\lambda$).
- **Conjunto de igualdade $E$:** pares/conjuntos como $\{grandmother, grandfather\}$ a serem igualados fora de $B$.
- **Reporting bias:** vieses do que é dito ou omitido no texto (Gordon & Van Durme).
