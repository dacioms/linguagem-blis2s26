# Roteiro da apresentação (4 h)

**Título:** Representações de Linguagem — como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto.
**Formato:** exposição + demonstrações ao vivo (notebooks e app Streamlit) + discussão. Preparar: `uv sync --extra dev --extra contextual && make data && make train` (≈ 20 min) e abrir `uv run replang app` numa aba e o JupyterLab noutra.

Convenções: **[NB]** notebook; **[APP]** página do app; **[D]** pergunta de discussão; **[Q]** pergunta rápida para a plateia.

---

## 0:00–0:10 · Abertura: o título como três perguntas

- Mostrar `homem : rei :: mulher : ?` → *rainha* e, em seguida, `ele : médico :: ela : ?` **[APP 4]** (marcar "excluir palavras de gênero"). A mesma operação, dois resultados: um encantador, outro desconfortável.
- As três perguntas e os seis blocos (tabela do **[NB 00]**). Regras do jogo: tudo que será mostrado roda em CPU, com dados públicos.

## 0:10–0:35 · Bloco A — Premissas: de símbolos a vetores

**[NB 01] [APP 1]**
1. One-hot: cos(rei, rainha) = cos(rei, banana) = 0. **[Q]** O que um computador "sabe" sobre *rei*? Nada.
2. Hipótese distribucional (Harris 1954). Construir a matriz de coocorrência do *toy corpus* ao vivo **[APP 1]**: mudar a janela de 1 para 3 e ver as linhas mudarem.
3. PPMI: por que *o* e *a* dominam e como corrigir. Mostrar a linha de *rei* antes/depois.
4. SVD: 57 dimensões → 6; o scatter de PCA com realeza/animais/frutas/lugares.
5. Mesmo pipeline em Machado de Assis: vizinhos de *amor*, *dinheiro*, *rua*.
6. Fechar com Levy & Goldberg: o word2vec (próximo bloco) fatora implicitamente essa matriz.
- **[D]** O que a janela ±2 captura que ±10 não captura?

## 0:35–1:20 · Bloco B — word2vec (Mikolov 2013a, 2013b)

**[NB 02] [APP 2, 3, 5]**
1. (0:35) O contexto de 2013: NNLM caros. A pergunta de engenharia: tirar a camada oculta. CBOW × Skip-gram (figura do artigo); tabela de complexidade.
2. (0:45) Abrir o capô: pares (centro, contexto) e o passo de gradiente da amostragem negativa **[NB 02 §2]**. Treinar Skip-gram ao vivo no toy corpus **[APP 2]** com 15 épocas; mostrar a perda caindo e as setas paralelas *rei→rainha*, *homem→mulher*.
3. (0:55) Dois vetores por palavra (W_in / W_out). Analogias: 3CosAdd; benchmark PT-BR por categoria **[NB 02 §5]**; cobertura.
4. (1:00) **[NB 03]** O gargalo do softmax. Softmax hierárquico (árvore de Huffman com códigos reais de *a*, *que*, *saffron*) × amostragem negativa (U^0,75). Tabela HS × NEG no corpus Machado.
5. (1:08) Subamostragem: a sentença antes/depois (*é melhor ter trezentos contos…* → *é trezentos contos trinta sempre cifra*). Frases: `rua_do_ouvidor`, `vossa_excelência`. Composição aditiva: *rússia + rio → volga*.
6. (1:15) PCA país→capital **[APP 5]** (Fig. 2 de Mikolov 2013b). O gráfico de Levy & Goldberg (correlação 0,9 entre PMI−log k e v′·v).
- **[D]** Se somar vetores é um AND, por que a negação não funciona?

## 1:20–1:50 · Bloco C — Variações: GloVe, fastText, Doc2Vec

**[NB 04, 05, 11] [APP 6]**
1. (1:20) GloVe: a tabela de razões de probabilidade (*rei/padre*: trono, coroa × igreja, missa). A função f e o objetivo. Curva de treino do GloVe 100d em Machado. Contagem × predição: Baroni 2014 vs Levy 2015.
2. (1:32) fastText: n-gramas de `<onde>` e o hash FNV. OOV ao vivo **[APP 6]**: digitar *capituzinha*, *tristíssimo*, um erro de digitação. Tabela de pares morfológicos (fastText 0,9 × Skip-gram OOV). Analogias sintáticas: 43 % × 3 % no mesmo corpus.
3. (1:42) Doc2Vec em 8 minutos: PV-DM/PV-DBOW; t-SNE dos blocos de Machado por gênero literário; inferir um parágrafo novo.
- **[Q]** Qual dos três você usaria para um corretor ortográfico? Para um classificador de documentos jurídicos?

## 1:50–2:05 · Intervalo

## 2:05–2:45 · Bloco D — Português: Hartmann et al. (2017)

**[NB 06, 07] [APP 7]**
1. (2:05) O corpus de 1,39 bi tokens (Tabela 1) × os nossos 2,4 M. Pré-processamento: clíticos, numerais, UNKNOWN; efeito do min_count.
2. (2:12) Os quatro algoritmos; Wang2Vec explicado em um slide (janela estruturada). Tabela 2 do artigo.
3. (2:18) Analogias PT-BR × PT-EU por categoria com os nossos modelos; cobertura como protagonista; dimensão 50/100/200.
4. (2:28) **[NB 07]** POS tagging no Mac-Morpho: tabela de acurácia por embedding; OOV e o fastText. Similaridade de sentenças estilo ASSIN; baseline TF-IDF; conector do ASSIN.
5. (2:38) O ranking concorda? Spearman entre métricas. Faruqui et al.: cinco problemas da avaliação intrínseca.
- **[D]** O que um teste de analogias "nativo" do português mediria?

## 2:45–3:35 · Bloco E — O viés que vem junto: Bolukbasi et al. (2016)

**[NB 08, 09, 10] [APP 8]**
1. (2:45) A tese e o exemplo da busca (Mary × John). *Reporting bias*. Específicas × neutras.
2. (2:52) **[NB 08]** Profissões no eixo she−he (Fig. 1); correlação com a crowd. Robustez entre eixos (Fig. 4).
3. (3:00) Geração de analogias (eq. 1, δ = 1): ler a lista ao vivo e separar "apropriada" de "estereótipo" com a plateia **[APP 8, aba 4]**.
4. (3:06) PCA dos pares (Fig. 6): uma direção domina; sensibilidade à remoção de pares. DirectBias (0,08 no artigo). Viés indireto β: *softball–receptionist*.
5. (3:14) **[NB 09]** Neutralize (projeção) e Equalize (simetrização) com a figura 2D de *grandmother/grandfather*; Observação 1 verificada numericamente (PairBias = 0). Soft debias e o papel de λ. Utilidade preservada (Tabela 1); *he:doctor::she:?* antes/depois.
6. (3:24) **[NB 10]** Português: direção ainda mais nítida; profissões epicenas e áreas nos extremos (*beleza, babá, cozinha* × *sábio, capitão, gênio*); gênero gramatical ≠ viés (separação intra-par × centro do par); filtro morfológico nas analogias; **experimento sintético** (viés do corpus 0,5 → 1,0 ⇒ DirectBias cresce); debias em PT e a concordância.
7. (3:32) "Lipstick on a pig" (Gonen & Goldberg): o que o debias não faz.
- **[D]** "O embedding só reflete o mundo" — argumento para não corrigir? De quem é a responsabilidade?

## 3:35–3:50 · Bloco F — Contexto: ELMo (Peters et al., 2018)

**[NB 12] [APP 9]**
1. (3:35) O limite: *banco* tem um vetor. biLM, 2L+1 representações, eq. (1).
2. (3:40) Demonstração: matriz de cossenos de *banco* por camada (camada 0 = 1,0 em tudo; camada 2 separa sentidos) **[APP 9]**; vizinhos contextuais.
3. (3:45) Camadas × tarefas: POS por camada (linear, sem janela) × estático; eficiência amostral. Um slide sobre BERT e o que permanece (viés em modelos contextuais).

## 3:50–4:00 · Síntese e discussão

**[NB 13]**
- Linha do tempo 1954–2019; tabela consolidada dos modelos (analogias, POS, STS, DirectBias).
- As três respostas às três perguntas do título.
- Perguntas finais (ver NB 13 §4). Encerrar com o repositório e como reproduzir tudo em casa (`make setup data train run-notebooks`).

---

## Plano B (se algo falhar ao vivo)

| Falha | Alternativa |
|---|---|
| App não sobe | todos os notebooks têm as saídas gravadas; abrir o `.ipynb` |
| Modelos locais ausentes | `REPLANG_FAST=1 uv run replang train` leva ~6 min; ou usar só o Wikipedia2Vec PT (páginas 3–5, 8 funcionam) |
| JAX ausente | pular a demo ao vivo do ELMo e mostrar as saídas do NB 12 |
| Sem rede | tudo roda offline a partir de `data/samples` + `data/processed` |

## Checklist de preparação

- [ ] `uv sync --extra dev --extra contextual`
- [ ] `uv run replang download && uv run replang prepare` (ou confirmar `data/samples/*.npz`)
- [ ] `uv run replang train` (≈ 15 min; gera `data/models`)
- [ ] `uv run pytest` verde
- [ ] `uv run replang notebooks run` (opcional: regenerar saídas; ≈ 30 min)
- [ ] `uv run replang app` e abrir as 10 páginas uma vez (cache aquecido)

---

## Extensão: contexto jurídico (duas formas de uso)

**Forma 1 — "janelas jurídicas" dentro dos 4 h (≈ 3 min por bloco, sem alterar o tempo total):**

| Bloco | Janela (3 min) | Material |
|---|---|---|
| A | a linha de *sentença* na matriz de coocorrência do STF: *prolatada, reforma, anulação* — não *frase* | nb 14 §2 |
| B | analogia *autor : réu :: apelante : apelado* no modelo do STF × Wikipédia | app p. 10 |
| C | fastText e os papéis *-ante/-ado*; *habeas_corpus* como frase | nb 14 §8–9 |
| D | a normalização "numerais → 0" apaga *Lei 13.105/2015*; NER e STS jurídicos como avaliação na tarefa | nb 14 §1.1, §5–6 |
| E | papéis processuais no eixo de gênero (separação × centro) e o ranqueador com consultas gêmeas | nb 15 §2, §6 |
| F | polissemia técnica (*sentença*) como caso ideal de representação contextual; alucinação de precedentes | `docs/juridico.md` §3-F |

**Forma 2 — módulo autônomo de 50 min (para turma de Direito/Tecnologia):**

| Tempo | Conteúdo | Material |
|---|---|---|
| 0:00 | Por que o Direito é especial; a língua do Direito; particularidades brasileiras | `docs/juridico.md` §1–2 |
| 0:10 | Deslocamento de domínio ao vivo: *sentença, título, pena, trânsito, remédio* | app p. 10; nb 14 §3 |
| 0:20 | Avaliar na tarefa: NER (LeNER-Br), similaridade de ementas (JurisBERT), área (RulingBR) | nb 14 §5–7 |
| 0:30 | O viés que vem junto no Direito: papéis, termos neutros, ranqueador gêmeo, debias e seus limites | nb 15 §2–7 |
| 0:42 | Governança: CNJ 332/2020 e 615/2025, LGPD, PL 2338; Victor, Athos, Sinapses | nb 15 §8 |
| 0:46 | Debate: auditoria × decisão; IA em matéria penal; explicabilidade; responsabilidade | `docs/juridico.md` §7 |

Preparação adicional: `uv run replang prepare` (baixa RulingBR, LeNER-Br e JurisBERT) e `uv run replang train legal` (≈ 4 min).
