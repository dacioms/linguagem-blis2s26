# Bojanowski et al. (2017) — *Enriching Word Vectors with Subword Information* (fastText)

## Ficha

- **Título:** Enriching Word Vectors with Subword Information
- **Autores:** Piotr Bojanowski*, Edouard Grave*, Armand Joulin, Tomas Mikolov (Facebook AI Research; * contribuição igual)
- **Ano:** 2017 (arXiv v1 em jul. 2016; v2 em jun. 2017)
- **Venue:** Transactions of the ACL (TACL), vol. 5, pp. 135–146
- **Link:** https://arxiv.org/abs/1607.04606 — código: https://github.com/facebookresearch/fastText
- **Classificação na disciplina:** leitura **complementar** (eixo "subpalavra")

## Em uma frase

O artigo estende o Skip-gram com negative sampling representando cada palavra como a **soma dos vetores de seus n-gramas de caracteres**, o que compartilha parâmetros entre palavras morfologicamente relacionadas, melhora a qualidade em línguas de morfologia rica e permite calcular vetores para palavras nunca vistas no treino.

## Problema e motivação

Word2vec e GloVe atribuem um vetor distinto e independente a cada palavra do vocabulário, ignorando sua **estrutura interna**. Isso é uma limitação séria em línguas morfologicamente ricas: verbos em francês ou espanhol têm mais de quarenta formas flexionadas; o finlandês tem quinze casos nominais; o alemão forma compostos (*Tischtennis* = *Tisch* + *Tennis*). Muitas dessas formas são raras ou ausentes no corpus, logo recebem vetores ruins ou nenhum vetor. Como a formação de palavras segue regras, informação no nível de caracteres pode ser usada para melhorar os vetores.

Trabalhos anteriores exigiam análise morfológica (segmentadores, listas de morfemas, dados anotados) ou arquiteturas pesadas (RNNs sobre caracteres). A proposta aqui é deliberadamente simples: sem pré-processamento ou supervisão, só n-gramas de caracteres, mantendo o Skip-gram rápido o bastante para corpora grandes. A ideia remonta a Schütze (1993), que aprendeu representações de 4-gramas por SVD.

## Premissas necessárias para entender

- **Skip-gram com negative sampling** (Mikolov 2013b): função de pontuação $s(w,c) = u_w^\top v_c$ e perda logística com negativos.
- **Hipótese distribucional** (Harris 1954): aprender a *prever bem* as palavras do contexto.
- **Perda logística** $\ell(x) = \log(1 + e^{-x})$ e descida de gradiente estocástica assíncrona (Hogwild).
- **N-gramas de caracteres** e marcadores de fronteira de palavra.
- **Função hash** (FNV-1a) e o compromisso entre memória e colisões.
- **Correlação de Spearman** para similaridade de palavras e acurácia em analogias.
- **Perplexidade** em modelagem de linguagem (quanto menor, melhor).
- **Palavras fora do vocabulário (OOV)** e por que modelos por palavra não podem representá-las.

## Método / modelo em detalhe

### Modelo geral (Seção 3.1)

Dado um vocabulário de tamanho $W$ e um corpus $w_1,\dots,w_T$, o Skip-gram maximiza

$$\sum_{t=1}^{T}\sum_{c\in\mathcal C_t}\log p(w_c\mid w_t)$$

onde $\mathcal C_t$ é o conjunto de índices do contexto de $w_t$. O softmax $p(w_c\mid w_t) = e^{s(w_t,w_c)}/\sum_{j=1}^{W}e^{s(w_t,j)}$ é inadequado porque implicaria prever uma única palavra de contexto; o problema é reformulado como **classificações binárias independentes**: para cada posição de contexto $c$, os contextos verdadeiros são positivos e $\mathcal N_{t,c}$ é um conjunto de negativos sorteados. Com $\ell(x) = \log(1+e^{-x})$, a perda é

$$\sum_{t=1}^{T}\left[\sum_{c\in\mathcal C_t}\ell\big(s(w_t,w_c)\big) + \sum_{n\in\mathcal N_{t,c}}\ell\big(-s(w_t,n)\big)\right]$$

No Skip-gram original, cada palavra tem dois vetores $u_w, v_w\in\mathbb R^d$ (entrada e saída) e $s(w_t,w_c) = u_{w_t}^\top v_{w_c}$.

### Modelo de subpalavras (Seção 3.2)

Cada palavra $w$ é representada como um **saco de n-gramas de caracteres**. Adicionam-se os símbolos especiais `<` e `>` no início e fim da palavra, para distinguir prefixos e sufixos de sequências internas, e a própria palavra inteira também entra como um "n-grama" especial. Para *where* e $n = 3$:

`<wh`, `whe`, `her`, `ere`, `re>` e a sequência especial `<where>`.

Note que `<her>` (a palavra *her*) é diferente do trigrama `her` dentro de *where*. Na prática, extraem-se todos os n-gramas com $3 \le n \le 6$. Dado um dicionário de n-gramas de tamanho $G$ e $\mathcal G_w\subset\{1,\dots,G\}$ o conjunto de n-gramas de $w$, associa-se um vetor $\mathbf z_g$ a cada n-grama e a função de pontuação vira

$$s(w,c) = \sum_{g\in\mathcal G_w}\mathbf z_g^\top \mathbf v_c$$

ou seja, o vetor de entrada da palavra é a **soma** dos vetores de seus n-gramas (o vetor de contexto $\mathbf v_c$ continua por palavra). Isso compartilha representações entre palavras e permite aprender vetores confiáveis para palavras raras.

**Hashing.** Para limitar memória, os n-gramas são mapeados a inteiros em $\{1,\dots,K\}$ pela função hash **FNV-1a**, com $K = 2\cdot 10^6$. Uma palavra é então representada por seu índice no dicionário mais o conjunto de n-gramas *hasheados*.

### Otimização e implementação (Seções 4.2–4.3)

SGD com decaimento linear do passo: no instante $t$, passo $\gamma_0(1 - t/(TP))$, com $P$ passadas. Paralelização por Hogwild (threads compartilham parâmetros sem travas). Hiperparâmetros: dimensão 300; 5 negativos por positivo, sorteados proporcionalmente à raiz quadrada da frequência unigrama; janela $c$ sorteada uniformemente entre 1 e 5; subamostragem com limiar $10^{-4}$; mínimo de 5 ocorrências; $\gamma_0 = 0{,}025$ para o Skip-gram base e $0{,}05$ para o modelo proposto e o CBOW. Custo: o modelo é ~1,5x mais lento que o Skip-gram (105 mil vs. 145 mil palavras/segundo/thread). Implementado em C++.

Dados: Wikipédia em nove línguas (árabe, tcheco, alemão, inglês, espanhol, francês, italiano, romeno, russo), normalizadas com o script de Matt Mahoney, 5 passadas.

### Representação de palavras OOV

Para uma palavra não vista, basta somar os vetores de seus n-gramas. Nos experimentos, **sisg-** denota o modelo com vetor nulo para OOV e **sisg** (Subword Information Skip-Gram) o modelo que constrói o vetor a partir dos n-gramas.

## Experimentos e resultados principais

### Tabela 1 — similaridade de palavras (Spearman × 100)

| Língua | Dataset | sg | cbow | sisg- | sisg |
|---|---|---|---|---|---|
| Ar | WS353 | 51 | 52 | 54 | **55** |
| De | Gur350 | 61 | 62 | 64 | **70** |
| De | Gur65 | 78 | 78 | **81** | **81** |
| De | ZG222 | 35 | 38 | 41 | **44** |
| En | RW | 43 | 43 | 46 | **47** |
| En | WS353 | 72 | **73** | 71 | 71 |
| Es | WS353 | 57 | 58 | 58 | **59** |
| Fr | RG65 | 70 | 69 | **75** | **75** |
| Ro | WS353 | 48 | 52 | 51 | **54** |
| Ru | HJ | 59 | 60 | 60 | **66** |

O modelo vence em todos os conjuntos, exceto o WS353 inglês (palavras comuns, que já têm bons vetores); o ganho é maior em árabe, alemão e russo (declinações, compostos) e no inglês de palavras raras (RW). Calcular vetores OOV (sisg) nunca é pior que usar vetor nulo (sisg-).

### Tabela 2 — analogias (acurácia %)

| Língua | Tipo | sg | cbow | sisg |
|---|---|---|---|---|
| Cs | Semântica | 25,7 | 27,6 | 27,5 |
| Cs | Sintática | 52,8 | 55,0 | **77,8** |
| De | Semântica | 66,5 | 66,8 | 62,3 |
| De | Sintática | 44,5 | 45,0 | **56,4** |
| En | Semântica | 78,5 | 78,2 | 77,8 |
| En | Sintática | 70,1 | 69,9 | **74,9** |
| It | Semântica | 52,3 | 54,7 | 52,3 |
| It | Sintática | 51,5 | 51,8 | **62,7** |

Morfologia melhora muito as analogias **sintáticas** (tcheco: +25 pontos); não ajuda nas semânticas e chega a prejudicar em alemão e italiano — efeito ligado ao comprimento dos n-gramas.

### Tabela 3 — comparação com métodos morfológicos

Treinando nos mesmos dados dos concorrentes: sisg supera Soricut & Och (2015) (De Gur350: 73 vs. 64; ZG222: 43 vs. 22; En RW: 48 vs. 42; Es: 54 vs. 47; Fr: 69 vs. 67) e Botha & Blunsom (2014) em todos os conjuntos (ex.: En WS353 54 vs. 39). O ganho em alemão vem de modelar compostos, que a análise por prefixos/sufixos não captura.

### Efeito do tamanho dos dados (Figura 1)

Treinando com 1, 2, 5, 10, 20 e 50% da Wikipédia: sisg é melhor em todos os tamanhos e satura rápido; com **5%** dos dados em alemão (Gur350) obtém 66, acima do CBOW com 100% (62); com **1%** dos dados em inglês (RW) obtém 45, acima do CBOW completo (43). Implicação prática: dá para treinar bons vetores em corpora pequenos de domínio específico.

### Efeito do tamanho dos n-gramas (Tabela 4)

Variando $n\in\{i,\dots,j\}$ em inglês e alemão: a escolha 3–6 é razoável; n-gramas longos ($n \le 5$ ou $6$) são importantes (compostos alemães); 2-gramas não informam porque, com os marcadores de fronteira, um 2-grama de sufixo tem só um caractere "real". Para analogias semânticas, n-gramas maiores ajudam.

### Tabela 5 — modelagem de linguagem (perplexidade de teste)

| | Cs | De | Es | Fr | Ru |
|---|---|---|---|---|---|
| CLBL (Botha & Blunsom) | 465 | 296 | 200 | 225 | 304 |
| CANLM (Kim et al.) | 371 | 239 | 165 | 184 | 261 |
| LSTM (sem pré-treino) | 366 | 222 | 157 | 173 | 262 |
| sg | 339 | 216 | 150 | 162 | 237 |
| sisg | **312** | **206** | **145** | **159** | **206** |

Inicializar a tabela de embeddings de um LSTM de 650 unidades com vetores sisg reduz a perplexidade em 8% (tcheco) e 13% (russo) sobre o Skip-gram; o ganho é menor em espanhol (3%) e francês (2%).

## Análises e discussões do artigo

- **Vizinhos mais próximos (Tabela 7):** para palavras raras ou técnicas, sisg dá vizinhos melhores (*tech-rich → tech-dominated, tech-heavy*; *english-born → british-born, polish-born*; *micromanaging → micromanage*), enquanto o Skip-gram devolve ruído (*.ixic*, *defang*).
- **N-gramas mais importantes (Tabela 6):** removendo um n-grama por vez e medindo a mudança do vetor, os mais importantes coincidem com **morfemas**: *Autofahrer → Fahr, Fahrer, Auto*; *kindness → ness>, kind*; *finirais → ais>, finis* (flexão verbal francesa). O método recupera morfologia sem nunca ter sido informado dela.
- **OOV (Figura 2):** mapas de calor da similaridade entre n-gramas mostram que *chip* casa com *micro* e *circuit* em *microcircuit*; *rarity* casa com *scarce* em *scarceness* e o sufixo *-ity* com *-ness*; *young* casa com *adolesc-* em *preadolescent*.
- Limitações admitidas: a faixa 3–6 é arbitrária e não foi validada por língua ou tarefa (falta de dados de teste); n-gramas podem degradar analogias semânticas; o modelo é 1,5x mais lento.

## Críticas e limitações (visão contemporânea)

- O **saco de n-gramas** ignora a ordem interna e produz vetores "borrados" para palavras com morfologia irregular ou para homógrafos; n-gramas superficiais podem fazer *casa* e *casaco* parecerem relacionados.
- O hashing com $K = 2\cdot10^6$ introduz colisões silenciosas.
- Os n-gramas carregam marcas de **gênero gramatical** (*-eira*, *-eiro*, *-a*, *-o*): em português, o modelo pode reforçar associações de gênero por via morfológica, o que complica o debiasing de Bolukbasi (a direção de gênero vira também uma direção de sufixos).
- Hartmann et al. (2017) mostram que, em português, o ganho em analogias sintáticas **não** se traduz em POS tagging, onde FastText foi dos piores.
- A ideia de subpalavra foi absorvida pelos modelos contextuais por meio de tokenização BPE/WordPiece (BERT) e CNNs de caracteres (ELMo), que resolvem OOV de forma diferente; o fastText permanece útil como baseline rápido e para línguas de poucos recursos.

## Conexões com os outros artigos da disciplina

- Parte explicitamente do **Skip-gram com negative sampling** de Mikolov 2013b (mesma equação, mesmo $U^{3/4}$-like, mesma subamostragem) e muda só a parametrização de $s(w,c)$.
- Contrasta com **GloVe** (contagem por palavra inteira): a Tabela 2 mostra que a morfologia ajuda sintaxe, mas o GloVe de Hartmann continua melhor em semântica.
- É um dos quatro algoritmos de **Hartmann 2017**, que confirma em português a força em analogias sintáticas (58,7%) e revela a fraqueza em tarefas finais.
- A representação de OOV por n-gramas é a ponte conceitual para a **CNN de caracteres do ELMo** (Peters 2018), que também representa qualquer token a partir dos caracteres.
- Para **Bolukbasi 2016**, levanta a questão de como a morfologia de gênero entra no subespaço de viés.

## Pontos para a apresentação

- Explicar o problema com um verbo português: *falar* tem dezenas de formas; o word2vec aprende cada uma do zero.
- Mostrar a decomposição de `<where>` em trigramas e o papel de `<` e `>`.
- Fórmula central: vetor da palavra = soma dos vetores dos n-gramas; o contexto continua por palavra.
- Resultado-chave: Tabela 2 — sintaxe sobe muito, semântica não.
- Figura 1: 5% dos dados com sisg supera 100% com CBOW — útil para domínios pequenos.
- Tabela 6: o modelo "descobre" morfemas (*Auto + fahrer*; *kind + ness*).
- Demonstração: pedir o vetor de uma palavra inventada (*desconfigurabilíssimo*) ao fastText.
- Gancho para Hartmann: no português, o ganho ficou nas analogias.

**Perguntas para discussão**
1. Em que casos somar n-gramas prejudica (homógrafos, prefixos enganosos, nomes próprios)?
2. Como a morfologia de gênero do português interage com o viés de gênero do embedding?
3. Dado que BPE/WordPiece resolvem OOV em modelos contextuais, o que ainda justifica o fastText hoje?

## Glossário mínimo

- **N-grama de caracteres:** subsequência de $n$ caracteres de uma palavra, com marcadores `<` e `>`.
- **sisg / sisg-:** Subword Information Skip-Gram, com e sem construção de vetores para OOV.
- **sg / cbow:** baselines word2vec (Skip-gram e CBOW).
- **FNV-1a:** função hash usada para mapear n-gramas a $K = 2\cdot10^6$ posições.
- **Hogwild:** SGD paralelo assíncrono sem travas.
- **OOV:** palavra fora do vocabulário de treino.
- **RW (Rare Words):** conjunto inglês de similaridade com palavras raras (Luong et al. 2013).
- **Gur350 / ZG222 / HJ:** conjuntos de similaridade em alemão e russo.
- **Perplexidade:** medida de qualidade de modelo de linguagem; menor é melhor.
- **Morfema:** menor unidade de significado (*kind* + *ness*).
