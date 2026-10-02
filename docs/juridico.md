# Representações de linguagem no contexto jurídico

> Complemento à estrutura didática: o que muda quando as ideias dos artigos (contagem, word2vec, GloVe, fastText, Doc2Vec, avaliação em português, viés, contexto) são aplicadas ao **texto jurídico**, em geral e no **Brasil**. Materializado nos notebooks `14_linguagem_juridica_dominio` e `15_vies_aplicacoes_governanca_juridica`, na página *10 · Contexto jurídico* do app e nos corpora de `replang.data.legal`.

## 1. Por que o Direito é um caso especial

1. **O texto é o objeto, não só o meio.** Lei, contrato, petição, sentença e acórdão são atos que *produzem efeitos* pelo que dizem. Uma representação que confunde *prescrição* (do Código Civil) com *prescrição* (médica) não erra "um pouco": erra o instituto.
2. **O custo do erro é assimétrico e recai sobre pessoas.** Sistemas de triagem, recomendação de precedentes ou avaliação de risco afetam liberdade, patrimônio e família. É o cenário de **amplificação de viés** descrito por Bolukbasi et al. (busca que ranqueia *John* acima de *Mary*) com consequências jurídicas diretas.
3. **Há dever de fundamentação e de contraditório.** No Brasil, a Constituição exige decisões fundamentadas (art. 93, IX) e garante devido processo, contraditório e ampla defesa (art. 5º, LIV e LV). Um vetor de 300 números não é uma fundamentação; a *explicabilidade* deixa de ser desejável e passa a ser requisito.
4. **O corpus é institucional e histórico.** Decisões refletem a jurisprudência *de uma época*; embeddings treinados nelas carregam entendimentos superados (súmulas canceladas, leis revogadas) e os padrões sociais do Judiciário que as produziu.
5. **Há regulação específica.** Resoluções do CNJ sobre IA no Judiciário, a LGPD (revisão de decisões automatizadas, art. 20) e o Marco Legal da IA em tramitação tratam a "administração da justiça" como uso de alto risco.

## 2. A língua do Direito: peculiaridades

### 2.1 Gerais (comuns a várias tradições)

| Traço | Exemplo | Efeito sobre as representações |
|---|---|---|
| **Polissemia técnica** | *sentença, ação, parte, título, pena, agravo, embargos, competência, prescrição, mérito, tutela, foro, vara, juízo, conhecer, provimento* | um vetor por palavra (word2vec/GloVe/fastText) mistura o sentido comum e o técnico; modelos contextuais (ELMo/BERT) separam-nos; embeddings **treinados no domínio** deslocam o vetor para o sentido técnico (nb 14) |
| **Registro formal, latinismos, arcaísmos** | *data venia, in dubio pro reo, erga omnes, ex officio, periculum in mora; "colenda Turma", "ora recorrente", "eis que"* | tokens raros ou inexistentes em corpora gerais → OOV; fastText compõe vetores por n-gramas; frases (word2phrase) capturam *habeas_corpus*, *agravo_de_instrumento* |
| **Sentenças longas, subordinação e nominalização** | períodos de 60–100 tokens; "a inexistência de comprovação da ocorrência de prejuízo" | janelas pequenas perdem dependências; a média de vetores dilui; o biLM de 2 camadas tem contexto curto |
| **Intertextualidade densa** | citações de artigos, incisos, súmulas, precedentes (*art. 5º, LV, CF; Súmula 7/STJ; RE 655.265*) | referências numéricas são **entidades** (LeNER-Br: LEGISLACAO, JURISPRUDENCIA); a normalização "numerais → 0" de Hartmann et al. apaga *Lei 8.078/90* → *lei 0.000/00* |
| **Estrutura de documento** | ementa (resumo em caixa alta, estilo telegráfico), relatório, voto, dispositivo | ementas e votos têm distribuições lexicais diferentes; o mesmo corpus contém "dois gêneros textuais" |
| **Negação e modalidade** | *não conhecer do recurso; nega-se provimento; é de se deferir* | bag-of-words não distingue "provido" de "não provido" — o problema da negação (Mikolov 2013b, composição aditiva) é central no resultado do julgamento |
| **Enunciados performativos** | "Julgo procedente"; "Defiro a liminar" | o significado está no ato, não só na distribuição |

### 2.2 Particularidades brasileiras

* **Tradição de *civil law* com precedentes vinculantes.** Desde a EC 45/2004 (súmula vinculante, repercussão geral) e o CPC/2015 (arts. 926–928: precedentes, IRDR, recursos repetitivos), a *similaridade entre casos* virou operação jurídica cotidiana: é exatamente o que embeddings de documento e STS medem (nb 14, JurisBERT). O STF e o STJ construíram sistemas (Victor, Athos) para agrupar recursos por tema.
* **Gênero gramatical nos papéis processuais.** *autora/autor, ré/réu, juíza/juiz, desembargadora/desembargador, ministra/ministro, advogada/advogado, promotora/promotor*. A questão do §9 de Bolukbasi et al. ("línguas com gênero gramatical") aparece com força: a marca de gênero é informação jurídica necessária (concordância, identificação da parte) e, ao mesmo tempo, veículo de estereótipo (nb 15).
* **Numeração e citação padronizadas.** Leis (*Lei nº 13.709/2018*), processos (CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO), artigos, parágrafos (§), incisos (romanos), alíneas. O tokenizador e a normalização decidem se isso vira ruído ou entidade.
* **Abreviaturas institucionais.** *STF, STJ, TST, TSE, TJ-SP, TRF-1, MPF, DPU, OAB, REsp, RE, AgR, HC, MS, ADI, ADPF, DJe*. Corpora gerais raramente as contêm com o sentido jurídico.
* **Variação PT-BR × PT-PT** e, dentro do Brasil, variação regional e entre tribunais (TJ estaduais, Justiça Federal, Trabalho, Eleitoral, Militar). Hartmann et al. mostraram que misturar variantes não prejudica POS; no jurídico, a terminologia processual portuguesa (*despacho saneador*, *contestação*) difere o bastante para justificar corpora nacionais.
* **Linguagem simples.** O Pacto Nacional do Judiciário pela Linguagem Simples (CNJ, 2023) e iniciativas de tribunais impulsionam sumarização e reescrita — tarefas que dependem de representações de sentenças/documentos (Doc2Vec → modelos neurais de sumarização; o RulingBR nasceu como dataset de sumarização de ementas).
* **Dados abertos, mas heterogêneos.** Decisões são públicas (com exceções de sigilo), o que permitiu corpora como RulingBR (STF), LeNER-Br, UlyssesNER-Br (Câmara), Victor (STF), ACORDÃOS-TCU, Kelsen e os datasets do JurisBERT; mas formatos, OCR e metadados variam muito entre tribunais.

## 3. Bloco a bloco: o que muda no contexto jurídico

### Bloco A — Premissas (coocorrência, PPMI, SVD, TF-IDF)
* **Aplicação direta:** a pesquisa de jurisprudência dos tribunais é, ainda hoje, majoritariamente *bag-of-words* (operadores booleanos, TF-IDF/BM25). É a primeira coisa que um jurista usa — e a mais interpretável.
* **Peculiaridade:** a hipótese distribucional funciona *dentro* do domínio: *sentença* coocorre com *prolatada, reforma, anulação*, não com *frase*. A matriz de coocorrência do corpus do STF é a radiografia do vocabulário técnico (nb 14, §2).
* **Debate:** quanto da "semântica jurídica" é distribucional? Conceitos definidos por lei (tipos penais, prazos) têm significado *estipulado*, não estatístico.

### Bloco B — word2vec (Mikolov 2013a, 2013b)
* **Aplicação:** expansão de consulta (sinônimos técnicos: *demissão ↔ dispensa ↔ rescisão*), classificação de petições, agrupamento de temas.
* **Peculiaridade:** analogias jurídicas existem e são informativas (*autor : réu :: apelante : apelado*; *juiz : sentença :: tribunal : acórdão*) — nb 14, §4. A detecção de frases é especialmente útil (*habeas_corpus, repercussão_geral, dano_moral*). A composição aditiva captura *dano + moral*, mas **não** a negação (*provido* vs *não provido*), que decide o resultado.
* **Brasil:** a biblioteca LegalNLP (Polo et al., 2021) distribui word2vec/fastText/Doc2Vec treinados em textos jurídicos brasileiros; o notebook 14 treina versões menores no RulingBR para comparação com o Wikipedia2Vec.

### Bloco C — GloVe, fastText, Doc2Vec
* **fastText:** o português jurídico é morfologicamente produtivo (*inconstitucionalidade, desconsideração, agravante/agravado, embargante/embargado, impetrante/impetrado*): os n-gramas capturam os pares de papéis (-ante/-ado) e dão vetores a termos raros. OOV é a regra em peças novas (nomes, números).
* **Doc2Vec / vetores de documento:** similaridade entre ementas é a base de recomendação de precedentes e da detecção de **demandas repetitivas**. O JurisBERT (Viegas et al., 2023) converteu um corpus de classificação em STS; o notebook 14 avalia média de vetores gerais × jurídicos × TF-IDF nesses pares.
* **GloVe:** as razões de probabilidade (*rei/padre* no nb 04) viram *penal/tributário* — contextos quase disjuntos, úteis para classificação de área.

### Bloco D — Avaliação (Hartmann et al., 2017; Faruqui et al., 2016)
* **A lição central vale dobrado:** analogias e similaridade de palavras *gerais* não medem utilidade jurídica. A avaliação tem de ser **na tarefa**: NER jurídico (LeNER-Br), STS de ementas (JurisBERT), classificação de área/classe (RulingBR/Victor), recuperação de precedentes.
* **Deslocamento de domínio** (nb 14, §3 e §5): um embedding geral (Wikipédia) e um jurídico (STF) discordam sobretudo nas palavras polissêmicas; a utilidade de cada um depende da tarefa e da **cobertura** do vocabulário técnico.
* **Pré-processamento:** reavaliar as decisões de Hartmann et al. para o domínio: manter numerais de leis/artigos como tokens (ou como entidades), preservar "§", "art.", siglas em maiúsculas, não separar clíticos de forma diferente da base anotada.

### Bloco E — Viés (Bolukbasi et al., 2016)
* **O que o vetor jurídico carrega:** (i) o gênero gramatical dos papéis; (ii) estereótipos sobre profissões e papéis (*juíza* vs *juiz*; *vítima*; *ré*); (iii) correlações socioeconômicas e raciais presentes nos autos — descrições de acusados, bairros, ocupações — que a geometria pode transformar em "direção de periculosidade". O artigo mostra o método para gênero; o §9 aponta raça/etnia como trabalho futuro.
* **Risco concreto:** modelos de avaliação de risco (COMPAS, nos EUA; análise da ProPublica, 2016) e sistemas de triagem reproduzem desigualdades históricas. No Brasil, a Resolução CNJ 332/2020 desestimula modelos de IA em matéria penal, sobretudo decisões preditivas (art. 23), e exige transparência, auditabilidade e supervisão humana; foi atualizada em 2025 (Resolução CNJ 615), com abordagem baseada em risco.
* **Debias no jurídico:** neutralizar *juíza/juiz* destrói concordância; equalizar pares preserva a morfologia e remove o estereótipo do *conceito*; mas "Lipstick on a Pig" (Gonen & Goldberg, 2019) mostra que a estrutura residual permanece — para uso em decisões, **documentar e auditar** importa mais do que "corrigir" (nb 15).

### Bloco F — Contexto (ELMo) e depois
* **Polissemia técnica é o caso de uso ideal de representações contextuais**: *sentença* no mesmo acórdão pode ser "decisão de 1º grau" e "frase do depoimento". Modelos jurídicos em português: BERTikal (LegalNLP), JurisBERT, Legal-BERT (inglês), além do BERTimbau geral.
* **LLMs e alucinação de precedentes:** modelos gerativos inventam ementas e números de processos plausíveis (caso *Mata v. Avianca*, EUA, 2023; episódios semelhantes noticiados no Brasil com sanções a advogados). A resposta técnica é **recuperação** sobre bases oficiais (RAG) — isto é, voltar aos vetores de documento e à similaridade do Bloco C, agora com encoders contextuais.

## 4. Comparação domínio geral × jurídico: o que esperar (e o que o notebook 14 mostra)

| Pergunta | Expectativa | Onde ver |
|---|---|---|
| O vocabulário é diferente? | menor diversidade lexical por token (TTR) com cauda longa de termos técnicos; sentenças mais longas; muito mais numerais e latim | nb 14 §1 |
| Os vizinhos mudam? | para palavras polissêmicas, sim — Jaccard entre vizinhanças próximo de zero (*sentença, ação, parte, título*) | nb 14 §3, app p. 10 |
| Analogias gerais funcionam? | a cobertura do LX-4WAnalogies cai (capitais, moedas são raras em acórdãos); analogias *jurídicas* funcionam melhor no modelo jurídico | nb 14 §4 |
| Qual embedding é melhor em NER jurídico? | o de domínio tende a vencer o geral, e fastText ajuda com OOV (siglas, nomes) | nb 14 §5 |
| E em similaridade de ementas? | o de domínio vence o geral; TF-IDF é um baseline forte por causa do vocabulário técnico compartilhado | nb 14 §6 |
| O viés de gênero é diferente? | a direção existe; papéis jurídicos têm marca morfológica forte; termos de família/trabalho/crime distribuem-se de forma estereotipada | nb 15 |

## 5. Aplicações e a representação adequada

| Aplicação | Representação | Risco a vigiar |
|---|---|---|
| Pesquisa de jurisprudência e expansão de consulta | TF-IDF/BM25 + vizinhos de embeddings de domínio | sinônimos espúrios; polissemia |
| Classificação de peças / área / classe (ex.: Victor no STF) | média de vetores de domínio → classificador; depois BERT jurídico | mudança de distribuição entre tribunais e anos |
| Agrupamento de demandas repetitivas; recomendação de precedentes (ex.: Athos no STJ) | vetores de documento (Doc2Vec, encoders de sentença) + STS | "similar" ≠ "mesmo fundamento"; precedentes superados |
| Extração de entidades (partes, leis, datas, órgãos) | embeddings + classificador sequencial (LeNER-Br, UlyssesNER-Br) | OOV, OCR, variação de citação |
| Sumarização / linguagem simples (ementas) | representações de documento → modelos seq2seq | perda de precisão técnica |
| Análise de contratos e *due diligence* | similaridade de cláusulas; NER de valores/prazos | cláusulas atípicas |
| Previsão de resultado / tempo de tramitação | qualquer vetor → regressão/classificação | **viés histórico**, uso em matéria penal, dever de fundamentação |

## 6. Governança e regulação no Brasil (para o debate)

* **Constituição**: fundamentação das decisões (art. 93, IX); devido processo legal, contraditório e ampla defesa (art. 5º, LIV–LV); igualdade (art. 5º, *caput*).
* **LGPD (Lei 13.709/2018)**: direito à revisão de decisões tomadas unicamente com base em tratamento automatizado (art. 20) e dever de informação sobre critérios.
* **Resolução CNJ 332/2020** (ética, transparência e governança na IA do Judiciário): não discriminação, supervisão humana, auditabilidade, publicidade dos modelos (plataforma Sinapses), restrições em matéria penal; **Resolução CNJ 615/2025** atualiza o regime com classificação de risco e regras para uso de IA generativa.
* **Marco Legal da IA (PL 2338/2023)**: sistemas de "administração da justiça" tratados como alto risco; em tramitação à data de redação — verificar o estágio atual.
* **Iniciativas dos tribunais** (exemplos públicos): Victor (STF, classificação de recursos e temas de repercussão geral), Athos (STJ, agrupamento por similaridade), Sinapses (CNJ/TJRO, plataforma de modelos), Elis (TJPE, triagem de execuções fiscais), Radar (TJMG, demandas repetitivas); no controle externo, Alice/Sofia/Monica (TCU, licitações).

## 7. Debates propostos (com posições para a sala)

1. **"O embedding só espelha a jurisprudência."** *A favor de corrigir*: a amplificação em triagem/ranking é documentada; o Judiciário tem dever de igualdade. *Contra*: corrigir vetores sem corrigir a prática esconde o problema (Gonen & Goldberg) e pode distorcer o retrato necessário para auditoria. *Síntese*: separar os usos — embeddings para **auditoria** devem preservar o viés; embeddings para **decisão/ranking** devem ser avaliados por métricas de equidade na tarefa.
2. **IA preditiva em matéria penal.** O art. 23 da Res. 332/2020 desestimula; COMPAS mostra o porquê. Há espaço para uso *instrumental* (busca, extração) sem uso *decisório*?
3. **Explicabilidade × desempenho.** Vetores e redes profundas são opacos; TF-IDF e regras são legíveis. O dever de fundamentação admite "o modelo sugeriu"? Quem fundamenta é o magistrado — mas a sugestão ancora a decisão.
4. **Gênero gramatical: preservar ou neutralizar?** Para gerar texto (minutas), preservar; para ranquear pessoas/casos, medir o deslocamento do *centro do par* (nb 10, nb 15) e auditar.
5. **Linguagem simples × precisão técnica.** Representações aprendidas em "juridiquês" favorecem o registro formal; simplificar muda a distribuição e pode degradar modelos treinados no registro antigo.
6. **De quem é a responsabilidade** quando um sistema de recomendação de precedentes induz erro: do fornecedor, do tribunal, do magistrado, do advogado que não conferiu? (Res. CNJ 332: supervisão humana e responsabilidade do usuário.)

## 8. Referências (além das da disciplina)

- FEIJÓ, D.; MOREIRA, V. **RulingBR: A Summarization Dataset for Legal Texts.** PROPOR, 2018.
- LUZ DE ARAUJO, P. H. et al. **LeNER-Br: a Dataset for Named Entity Recognition in Brazilian Legal Text.** PROPOR, 2018.
- ALBUQUERQUE, H. O. et al. **UlyssesNER-Br: A Corpus of Brazilian Legislative Documents for NER.** PROPOR, 2022.
- VIEGAS, C. F. O.; COSTA, B. C.; ISHII, R. P. **JurisBERT: A New Approach that Converts a Classification Corpus into an STS One.** ICCSA, 2023.
- POLO, F. M. et al. **LegalNLP — Natural Language Processing Methods for the Brazilian Legal Language.** ENIAC, 2021.
- CHALKIDIS, I. et al. **LEGAL-BERT: The Muppets straight out of Law School.** Findings of EMNLP, 2020.
- ANGWIN, J. et al. **Machine Bias.** ProPublica, 2016 (COMPAS).
- GONEN, H.; GOLDBERG, Y. **Lipstick on a Pig.** NAACL, 2019.
- CNJ. **Resolução nº 332/2020**; **Resolução nº 615/2025**. BRASIL. **Lei nº 13.709/2018 (LGPD)**; **PL 2338/2023**.
- SILVA, N. C. et al. **Document type classification for Brazil's supreme court using a convolutional neural network** (projeto Victor). ICoFCS, 2018.
