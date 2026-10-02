# Glossário

| Termo | Definição curta | Onde aparece |
|---|---|---|
| **3CosAdd / 3CosMul** | Métodos para resolver analogias a:b::c:?: argmax cos(y, b−a+c) / argmax cos(y,b)cos(y,c)/(cos(y,a)+ε) | nb 02; `WordVectors.analogy` |
| **Amostragem negativa (NEG)** | Objetivo que substitui o softmax por classificação binária real × k ruídos (Mikolov 2013b, eq. 4) | nb 03 |
| **Analogia** | Relação a:b::c:d capturada por diferença de vetores | nb 02, 08 |
| **biLM** | Modelo de linguagem bidirecional (forward + backward) usado pelo ELMo | nb 12 |
| **CBOW** | Continuous bag-of-words: prevê a palavra central pela média do contexto | nb 02 |
| **Cobertura** | Fração das questões de analogia cujas quatro palavras estão no vocabulário | nb 02, 06 |
| **Coocorrência** | Contagem de pares (palavra, contexto) numa janela | nb 01 |
| **Cosseno** | Similaridade u·v/(‖u‖‖v‖); com vetores unitários, = produto interno | nb 01 |
| **Direção de gênero (g)** | 1ª componente principal das diferenças centradas de pares definicionais (Bolukbasi §5.1) | nb 08, 10 |
| **DirectBias_c** | Média de \|cos(w,g)\|^c sobre palavras neutras (Bolukbasi §5.2) | nb 08 |
| **Doc2Vec / Paragraph Vector** | Vetor por documento treinado junto com os de palavras (PV-DM, PV-DBOW) | nb 11 |
| **ELMo** | Embeddings from Language Models: mistura por tarefa das camadas de um biLM (Peters 2018, eq. 1) | nb 12 |
| **Equalize** | Passo do hard debias que torna conjuntos de palavras simétricos em torno de ν fora de B | nb 09 |
| **Epiceno** | Substantivo com forma única para os dois gêneros (dentista, gerente) | nb 10 |
| **fastText** | Skip-gram com n-gramas de caracteres (Bojanowski 2017) | nb 05 |
| **GloVe** | Mínimos quadrados ponderados sobre log das coocorrências globais (Pennington 2014) | nb 04 |
| **Hard debias** | Neutralize + Equalize (Bolukbasi §6, passo 2a) | nb 09 |
| **Hipótese distribucional** | O significado de uma palavra está nos contextos em que ela ocorre (Harris 1954) | nb 01 |
| **Huffman (árvore)** | Árvore binária com códigos curtos para itens frequentes; usada no softmax hierárquico | nb 03 |
| **Intrínseca / extrínseca (avaliação)** | Medir o embedding em si (analogias, similaridade) / numa tarefa (POS, STS) | nb 06, 07 |
| **Janela (window)** | Número de palavras de contexto de cada lado | nb 01–03 |
| **LSA / HAL** | Modelos de contagem com SVD (termo×documento / palavra×palavra) | nb 01 |
| **Mac-Morpho** | Corpus PT-BR anotado com classes gramaticais (POS) | nb 07 |
| **min_count** | Frequência mínima para uma palavra entrar no vocabulário (UNKNOWN abaixo) | nb 00, 06 |
| **Neutralize** | Remover a projeção no subespaço de viés e renormalizar | nb 09 |
| **OOV** | Out-of-vocabulary: palavra sem vetor; o fastText compõe um a partir de n-gramas | nb 05, 07 |
| **PairBias** | Média de \|cos(w,a) − cos(w,b)\| sobre neutras e pares; zero após hard debias | nb 09 |
| **PMI / PPMI** | Informação mútua pontual (positiva): log P(w,c)/(P(w)P(c)) | nb 01 |
| **Polissemia** | Uma forma, vários sentidos (banco, manga); limite dos embeddings estáticos | nb 12 |
| **POS tagging** | Atribuir classe gramatical a cada token | nb 07 |
| **Skip-gram** | Prevê as palavras do contexto a partir da central | nb 02 |
| **Soft debias** | Transformação linear T que preserva produtos internos e reduz projeção das neutras em B (λ) | nb 09 |
| **Softmax hierárquico (HS)** | Aproximação do softmax por caminho numa árvore binária (~log₂V) | nb 03 |
| **Subamostragem** | Descartar ocorrências de palavras frequentes com P = 1 − √(t/f) | nb 03 |
| **SVD** | Decomposição em valores singulares; base da LSA e da compressão de PPMI | nb 01 |
| **TF-IDF** | Ponderação termo-frequência × inverso da frequência em documentos | nb 01, 07 |
| **Viés indireto β(w,v)** | Fração da similaridade entre duas neutras explicada pela componente de gênero (Bolukbasi §5.3) | nb 08 |
| **Wang2Vec** | word2vec sensível à ordem (janela estruturada; Ling 2015); melhor em POS/ASSIN em Hartmann | nb 06 |
| **Word2Vec** | Família CBOW/Skip-gram de Mikolov (2013a) | nb 02 |
| **Acórdão / ementa / relatório / voto** | Partes de uma decisão colegiada; a ementa é o resumo oficial | nb 14 |
| **Deslocamento de domínio** | Mudança da vizinhança (sentido) de uma palavra entre corpora; medido por Jaccard das vizinhanças | nb 14, app p. 10 |
| **LeNER-Br / RulingBR / JurisBERT STS / UlyssesNER-Br** | Corpora jurídicos brasileiros públicos (NER em decisões; decisões do STF; pares de ementas; projetos de lei) | nb 14, 15 |
| **Papéis processuais** | Autor/ré, apelante/apelado, agravante/agravado…; pares morfológicos com marca de gênero | nb 14, 15 |
| **Precedente / demandas repetitivas** | Decisões anteriores com força persuasiva ou vinculante; agrupamento por similaridade (Athos, Radar) | nb 14, 15 |
| **Resolução CNJ 332/2020 (e 615/2025)** | Normas do CNJ sobre ética, transparência e governança da IA no Judiciário | nb 15, docs/juridico.md |
