# Resumos dos artigos — Representações de Linguagem

Resumos didáticos, em português do Brasil, dos nove artigos que sustentam a apresentação **"Representações de Linguagem: como transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto"**. Cada arquivo segue a mesma estrutura (ficha, em uma frase, problema, premissas, método com equações, resultados, discussão, críticas, conexões, pontos para a apresentação e glossário), para facilitar a leitura cruzada.

Os resumos marcados com (*) foram escritos a partir de conhecimento prévio, sem acesso ao PDF: confira os números no original antes de citá-los.

## Índice

| Arquivo | Artigo | Classificação | Uma linha |
|---|---|---|---|
| [mikolov2013a_word2vec.md](mikolov2013a_word2vec.md) (*) | Mikolov, Chen, Corrado & Dean (2013a), *Efficient Estimation of Word Representations in Vector Space* | **Obrigatória** | CBOW e Skip-gram: redes log-lineares sem camada oculta que aprendem vetores em bilhões de palavras e criam o teste de analogias. |
| [mikolov2013b_phrases_negative_sampling.md](mikolov2013b_phrases_negative_sampling.md) | Mikolov, Sutskever, Chen, Corrado & Dean (2013b), *Distributed Representations of Words and Phrases and their Compositionality* | Complementar | Negative sampling, subamostragem de frequentes e detecção de frases tornam o Skip-gram rápido e melhor; soma de vetores compõe significados. |
| [pennington2014_glove.md](pennington2014_glove.md) (*) | Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation* | Complementar | Regressão ponderada sobre o log das coocorrências globais; explica as analogias por razões de probabilidades. |
| [le2014_doc2vec.md](le2014_doc2vec.md) (*) | Le & Mikolov (2014), *Distributed Representations of Sentences and Documents* | Complementar | Paragraph Vector (PV-DM / PV-DBOW): um vetor por sentença ou documento treinado como extensão do word2vec. |
| [bolukbasi2016_debiasing.md](bolukbasi2016_debiasing.md) | Bolukbasi, Chang, Zou, Saligrama & Kalai (2016), *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings* | **Obrigatória** | Direção de gênero por PCA, métricas de viés direto e indireto, e algoritmos Neutralize/Equalize e soft debiasing. |
| [bojanowski2017_fasttext.md](bojanowski2017_fasttext.md) | Bojanowski, Grave, Joulin & Mikolov (2017), *Enriching Word Vectors with Subword Information* | Complementar | fastText: palavra = soma de n-gramas de caracteres; morfologia, palavras raras e OOV. |
| [hartmann2017_portuguese_embeddings.md](hartmann2017_portuguese_embeddings.md) | Hartmann, Fonseca, Shulby, Treviso, Rodrigues & Aluísio (2017), *Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks* | **Obrigatória** | 31 modelos (GloVe, Word2Vec, Wang2Vec, FastText) em 1,39 bi de tokens em português; analogias não predizem desempenho em tarefas. |
| [peters2018_elmo.md](peters2018_elmo.md) | Peters, Neumann, Iyyer, Gardner, Clark, Lee & Zettlemoyer (2018), *Deep contextualized word representations* | Complementar | ELMo: combinação por tarefa das camadas de um biLM; camadas baixas = sintaxe, altas = semântica; seis novos estados da arte. |
| README.md | este índice | — | Lista dos resumos e tabela comparativa. |

## Ordem de leitura sugerida (linha histórica da apresentação)

1. **Contagem → predição:** Mikolov 2013a → Mikolov 2013b → Pennington 2014 (GloVe)
2. **Além da palavra isolada:** Le & Mikolov 2014 (documentos) → Bojanowski 2017 (subpalavras)
3. **Avaliação em português:** Hartmann 2017
4. **O viés que vem junto:** Bolukbasi 2016
5. **Contexto:** Peters 2018 (ELMo)

## Tabela comparativa

| Artigo | Ano | Ideia central | Tipo de modelo | Contribuição para a apresentação |
|---|---|---|---|---|
| Mikolov et al. (2013a) — word2vec | 2013 | Remover a camada oculta do NNLM para treinar vetores em bilhões de palavras; analogias por aritmética vetorial | Preditivo (CBOW / Skip-gram) | Define "palavra → vetor" e o teste de analogias; base de tudo o que vem depois |
| Mikolov et al. (2013b) — frases e negative sampling | 2013 | Negative sampling com $U(w)^{3/4}$, subamostragem $1-\sqrt{t/f}$, frases por pontuação de bigramas, composição aditiva | Preditivo (Skip-gram) | Explica como o word2vec é treinado na prática e por que somas de vetores "funcionam" |
| Pennington et al. (2014) — GloVe | 2014 | Mínimos quadrados ponderados sobre $\log X_{ij}$; razões de coocorrência explicam analogias | Contagem (fatoração global) | Representa a família de contagem e mostra a convergência com os preditivos |
| Le & Mikolov (2014) — Doc2Vec | 2014 | Vetor por parágrafo treinado junto com os vetores de palavras (PV-DM, PV-DBOW); inferência por otimização | Preditivo (extensão do word2vec a textos) | Responde "o que o vetor carrega" no nível do documento; baseline para similaridade de sentenças |
| Bolukbasi et al. (2016) — debiasing | 2016 | Direção de gênero por PCA de pares; DirectBias, $\beta(w,v)$; Neutralize + Equalize; soft debiasing | Análise/pós-processamento de embeddings preditivos (w2vNEWS) e de contagem (GloVe) | Núcleo do eixo "o viés que vem junto": medir e mitigar estereótipos; crítica posterior (Gonen & Goldberg) |
| Bojanowski et al. (2017) — fastText | 2017 | Palavra = soma de vetores de n-gramas de caracteres (3–6, com `<` `>`, hashing FNV) | Subpalavra (Skip-gram com n-gramas) | Morfologia e OOV, cruciais para o português; ponte para a CNN de caracteres do ELMo |
| Hartmann et al. (2017) — embeddings NILC | 2017 | 31 modelos em 1,39 bi de tokens PT-BR + PT-EU; analogias vs. POS tagging e ASSIN | Comparativo: contagem (GloVe), preditivo (Word2Vec, Wang2Vec) e subpalavra (FastText) | Dados e modelos em português para a demonstração; lição metodológica "analogia não é proxy de utilidade" |
| Peters et al. (2018) — ELMo | 2018 | Vetor por token como combinação linear, aprendida por tarefa, das camadas de um biLM pré-treinado | Contextual (biLSTM sobre caracteres) | Fecha a linha histórica: polissemia resolvida pelo contexto; sintaxe vs. semântica por camada; antessala do BERT |
