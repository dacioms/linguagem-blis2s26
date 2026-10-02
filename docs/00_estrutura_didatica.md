# Estrutura didática — Representações de Linguagem

> **Como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto.**
> Organização do conteúdo dos artigos da disciplina em uma sequência pedagógica, das premissas basais aos detalhes de cada trabalho. Cada nível indica os notebooks (`notebooks/NN_*.ipynb`), a página da interface (`app/`) e o módulo de código (`src/replang/`) que o materializam.

## Visão geral: três perguntas, seis blocos

| Pergunta do título | Blocos | Artigos |
|---|---|---|
| Como transformar palavra em vetor? | A. Premissas · B. word2vec · C. Variações | Harris/LSA; Mikolov 2013a, 2013b; GloVe; fastText; Doc2Vec |
| O que esse vetor carrega? | D. Português e avaliação · (C) · F. Contexto | Hartmann 2017; Faruqui 2016; ELMo |
| O viés que vem junto | E. Viés | Bolukbasi 2016; Gonen & Goldberg 2019 |

---

## Nível 0 — Premissas basais (o que o público precisa saber antes de qualquer artigo)

**Notebook 01 · App p. 1 · `replang.models.cooccurrence`, `replang.embedding`**

0.1. **Símbolos não têm significado interno.** Representação *one-hot*: |V| dimensões, todos os pares ortogonais. Problema: *rei* está tão longe de *rainha* quanto de *banana*.

0.2. **Hipótese distribucional** (Harris 1954; Firth 1957). O significado de uma palavra é caracterizado pelos contextos em que ela ocorre. Toda representação distribucional (de LSA a BERT) operacionaliza esta hipótese.

0.3. **Espaço vetorial e similaridade.** Palavras como pontos de ℝ^d; proximidade medida pelo **cosseno** (= produto interno com vetores unitários). Normalização: a norma de um vetor word2vec cresce com a frequência; comparar exige normalizar (Bolukbasi §3 assume ‖w‖ = 1).

0.4. **Matriz de coocorrência** palavra × contexto (janela de ±k). Ponderação por distância (1/d no GloVe; janela dinâmica no word2vec). Relações **sintagmáticas** (coocorrência) × **paradigmáticas** (substituibilidade).

0.5. **Reponderação: PMI/PPMI.** Coocorrer *mais que o acaso*: PMI = log P(w,c)/(P(w)P(c)). Palavras funcionais deixam de dominar. Suavização P(c)^0,75 e deslocamento −log k antecipam ideias do word2vec (Levy et al. 2015).

0.6. **Compressão: SVD truncada** (LSA, Deerwester 1990; HAL, Lund & Burgess 1996). De esparso e enorme para denso e pequeno; generalização.

0.7. **Vetores de documento**: *bag-of-words*, TF-IDF; cosseno entre documentos — o baseline de similaridade de sentenças usado por Hartmann (2016).

0.8. **Ferramental**: softmax, regressão logística, gradiente estocástico, PCA — revisados no ponto em que cada artigo os usa.

---

## Nível 1 — word2vec: o modelo log-linear (Mikolov et al., 2013a) — leitura obrigatória

**Notebook 02 · App p. 2–5 · `replang.models.word2vec_np`, `replang.eval.analogies`**

1.1. Contexto: NNLM (Bengio 2003) e Collobert & Weston — bons vetores, treino caro (camada oculta não linear).
1.2. A pergunta de engenharia: quanto sobrevive sem a camada oculta? Dois modelos **log-lineares**: **CBOW** (contexto → palavra; média, sem ordem) e **Skip-gram** (palavra → cada contexto).
1.3. Complexidade por exemplo: CBOW O(N·D + D·log₂V), Skip-gram O(C·(D + D·log₂V)) vs NNLM O(N·D + N·D·H + H·log₂V). O termo log₂V vem do softmax hierárquico.
1.4. Dois vetores por palavra (W_in, W_out); o word2vec descarta W_out — uma escolha.
1.5. **Analogias por aritmética**: vec(king) − vec(man) + vec(woman) ≈ vec(queen). Teste `questions-words`: 19.544 questões, 14 categorias (5 semânticas, 9 sintáticas). Métodos 3CosAdd / 3CosMul (Levy & Goldberg 2014). Cobertura e restrição de vocabulário.
1.6. Resultados: Skip-gram 55 % semântica / 59 % sintática (640d); CBOW mais rápido; dimensão e corpus precisam crescer juntos (Tabela 2).
1.7. Propriedade emergente: direções = relações. É o mecanismo que depois carrega o viés (Nível 5).

---

## Nível 2 — Fazer o word2vec funcionar em escala (Mikolov et al., 2013b) — complementar

**Notebook 03 · App p. 2 · `replang.models.word2vec_np` (HS, NEG, subamostragem, `learn_phrases`)**

2.1. O gargalo do softmax: O(V) por par.
2.2. **Softmax hierárquico** com árvore de Huffman: produto de sigmoides ao longo do caminho; códigos curtos para palavras frequentes; custo ~log₂V.
2.3. **Amostragem negativa** (eq. 4): classificação binária real × ruído; k = 5–20 (pequeno) ou 2–5 (grande); distribuição U(w)^{3/4}. Relação com NCE.
2.4. **Subamostragem** (eq. 5): P(descartar w) = 1 − √(t/f(w)); t ≈ 10⁻⁵; acelera e melhora palavras raras.
2.5. **Frases** (eq. 6): score(a,b) = (count(ab) − δ)/(count(a)·count(b)); passagens com limiar decrescente.
2.6. **Composicionalidade aditiva** (§5): soma de vetores ≈ AND de distribuições de contexto (vec(Russia)+vec(river) ≈ Volga River). Limites (negação).
2.7. Resultados (Tabela 1): NEG-15 61 %, HS 47 %, NCE 53 %; com subamostragem NEG ≈ HS ≈ 60 %.
2.8. Teoria: SGNS fatora implicitamente PMI − log k (Levy & Goldberg 2014) — fecha o ciclo com o Nível 0.

---

## Nível 3 — Variações: contagem global, subpalavras, documentos

### 3A. GloVe (Pennington, Socher & Manning, 2014) — complementar
**Notebook 04 · `replang.models.glove_np`**
- Razões de probabilidade P(k|i)/P(k|j) como objeto de modelagem (Tabela 1: ice/steam).
- Objetivo J = Σ f(X_ij)(w_i·w̃_j + b_i + b̃_j − log X_ij)²; f saturada com x_max = 100, α = 3/4; AdaGrad; W + W̃.
- Debate contagem × predição (Baroni 2014; Levy 2015).
- Em PT (Hartmann): melhor em analogias semânticas, pior nas tarefas.

### 3B. fastText (Bojanowski et al., 2017) — complementar
**Notebook 05 · App p. 6 · `replang.models.fasttext_np`**
- Palavra = bag de n-gramas de caracteres (3–6) com `<` `>`; s(w,c) = Σ_g z_g·v_c; hashing FNV-1a, K = 2·10⁶.
- Parâmetros compartilhados entre formas; vetores OOV; +1,5× tempo.
- Resultados: similaridade (Tabela 1, 8/9 línguas), analogias sintáticas (Tabela 2), tamanho de n-gramas (Tabela 4), dados (Fig. 1: satura cedo), LM (Tabela 5).
- Análise qualitativa: n-gramas importantes ≈ morfemas (Tabela 6).

### 3C. Doc2Vec / Paragraph Vector (Le & Mikolov, 2014) — complementar
**Notebook 11 · `replang.models.trainers.train_doc2vec`**
- Vetor por documento treinado junto com os de palavras: PV-DM (memória) e PV-DBOW; inferência por otimização.
- Resultados em SST e IMDB; críticas (Lau & Baldwin 2016); comparação com média de vetores.

---

## Nível 4 — O que o vetor carrega em português e como avaliar (Hartmann et al., 2017) — leitura obrigatória

**Notebooks 06, 07 · App p. 3, 4, 7 · `replang.data.corpora`, `replang.eval.*`**

4.1. Corpus: 1,39 bi tokens, 17 fontes, PT-BR + PT-EU (Tabela 1). Pré-processamento (§2.1): minúsculas, numerais → 0, URL/EMAIL, clíticos, sentenças ≥ 5 tokens, min_count 5.
4.2. Quatro algoritmos: GloVe, Word2Vec (CBOW/SG), **Wang2Vec** (ordem: janela estruturada), FastText; dimensões 50–1000; 31 modelos públicos (NILC).
4.3. **Avaliação intrínseca**: LX-4WAnalogies (Rodrigues 2016), BR e EU. Tabela 2: GloVe melhor (46,7 %); FastText SG melhor em sintáticas (58,7 %); CBOW fraco em semânticas; pico em 300d.
4.4. **Avaliação extrínsica**: POS tagging (Mac-Morpho, nlpnet; Tabela 3: Wang2Vec SG 1000d, 95,94 %; GloVe e FastText piores; maior dimensão melhor) e similaridade de sentenças (ASSIN; soma de vetores + cosseno + regressão; Tabela 4: Wang2Vec SG 1000d ρ 0,60; CBOW 1000d em PT-EU).
4.5. **Conclusão**: rankings intrínseco e extrínseco não concordam → analogias não são proxy (Faruqui 2016: subjetividade, overfitting, baixa correlação, frequência, polissemia).
4.6. Decisões de engenharia (tokenização de clíticos, OOV) pesam tanto quanto o algoritmo.

---

## Nível 5 — O viés que vem junto (Bolukbasi et al., 2016) — leitura obrigatória

**Notebooks 08, 09, 10 · App p. 8 · `replang.bias.*`**

5.1. **Tese**: o mesmo mecanismo das analogias responde *man:computer programmer :: woman:homemaker*; risco de **amplificação** (exemplo da busca por currículos). *Reporting bias*: contagens de 1ª ordem enganam, embeddings (2ª ordem) capturam o implícito.
5.2. Preliminares (§3): vetores unitários; w2vNEWS filtrado a 26.377 palavras; palavras **específicas de gênero** × **neutras**; pares F-M; experimentos com crowd (MTurk, 10 trabalhadores).
5.3. **Estereótipos ocupacionais** (§4, Fig. 1): projeção no eixo she−he; ρ = 0,51 com julgamentos humanos; consistente entre embeddings (Fig. 4, ρ = 0,81 com GloVe).
5.4. **Geração de analogias** (eq. 1): S_(a,b)(x,y) = cos(a−b, x−y) se ‖x−y‖ ≤ δ (δ = 1 ⇒ ângulo ≤ 60°); contraste com o paralelogramo (Ap. A, Fig. 9); crowd: 29 estereotipadas / 72 apropriadas em 150.
5.5. **Viés indireto** (§4/§5.3): softball−football; receptionist, homemaker; β(w,v) = (w·v − w⊥·v⊥/‖w⊥‖‖v⊥‖)/(w·v); β(softball, receptionist) = 67 %.
5.6. **Subespaço de gênero** (§5.1, Fig. 5–6): 10 pares; diferenças centradas; PCA; 1ª componente domina vs aleatório; validação dos pares como classificadores (75–93 %).
5.7. **DirectBias_c** (§5.2) = média de |cos(w,g)|^c; 327 ocupações: 0,08.
5.8. **Algoritmos** (§6): passo 1 (subespaço B via SVD de C = Σ_i Σ_{w∈D_i}(w−μ_i)ᵀ(w−μ_i)/|D_i|); passo 2a **Neutralize** (w := (w−w_B)/‖w−w_B‖) + **Equalize** (μ, ν = μ−μ_B, w := ν + √(1−‖ν‖²)(w_B−μ_B)/‖w_B−μ_B‖; por que centrar); Observação 1 (PairBias = 0); passo 2b **Soft** (min ‖(TW)ᵀ(TW) − WᵀW‖² + λ‖(TN)ᵀ(TB)‖², λ = 0,2, SDP via SVD de W, Ap. B).
5.9. **Palavras específicas de gênero** (§7): 218 sementes (Ap. C, WordNet) → SVM linear → 6.449 (F = 0,63; acurácia balanceada 95 %); Fig. 7.
5.10. **Resultados** (§8): utilidade preservada (Tabela 1: RG 62,3 → 62,4; WS 54,5 → 54,1; analogias 57,0 → 57,0); analogias estereotipadas 19 % → 6 % (hard), apropriadas mantidas (Fig. 8); viés indireto reduzido (softball → pitcher, infielder).
5.11. **Discussão** (§9): refletir ≠ amplificar; "errar para o lado da neutralidade"; outros vieses (minorities−whites); **línguas com gênero gramatical** → Nível 5-PT.
5.12. **Em português** (notebook 10): direção mais nítida (gênero gramatical reforça); profissões **epicenas** e áreas como N; pares morfológicos (separação intra-par = gramática; centro do par = estereótipo); filtro morfológico nas analogias geradas; experimento controlado com corpus sintético (viés geométrico ∝ viés do corpus); decisões de debias em PT (preservar concordância).
5.13. **Crítica posterior**: Gonen & Goldberg (2019) — a estrutura relativa sobrevive; o viés é escondido, não removido.

---

## Nível 6 — Representações contextuais (Peters et al., 2018, ELMo) — complementar

**Notebook 12 · App p. 9 · `replang.models.bilm_jax`**

6.1. Limite de tudo o que veio antes: um vetor por tipo (polissemia: *banco*, *manga*).
6.2. **biLM**: LM forward e backward com LSTMs de 2 camadas, CNN de caracteres na entrada, parâmetros de entrada e softmax compartilhados (§3.1); perplexidade 39,7 no 1B Word Benchmark.
6.3. **ELMo** (eq. 1): ELMo_k = γ Σ_j s_j h_{k,j}, com s softmax-normalizado por tarefa; 2L+1 representações por token (§3.2); inclusão na entrada (e saída) de modelos supervisionados com biLM congelado (§3.3); dropout e regularização λ‖w‖².
6.4. Resultados (Tabela 1): 6 tarefas, +6–20 % de redução relativa de erro (SQuAD 81,1 → 85,8; SNLI 88,0 → 88,7; SRL 81,4 → 84,6; coref 67,2 → 70,4; NER 90,15 → 92,22; SST-5 51,4 → 54,7).
6.5. Análise (§5): todas as camadas > última camada (Tabela 2); onde incluir (Tabela 3); camadas baixas = sintaxe (POS 97,3 na 1ª), altas = sentido (WSD 69,0 na 2ª) (Tabelas 5–6); eficiência amostral (Fig. 1); pesos aprendidos (Fig. 2).
6.6. Depois do ELMo: BERT; o viés persiste em modelos contextuais (Zhao 2019).

---

## Nível 7 — Síntese

**Notebook 13 · `docs/roteiro_4h.md`**

- Linha do tempo 1954 → 2019; quadro consolidado dos modelos; respostas às três perguntas do título; perguntas abertas para a plateia.
