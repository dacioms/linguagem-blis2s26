# data/samples — artefatos pequenos versionados

| Arquivo | Origem | Licença |
|---|---|---|
| `ptwiki100d_top50k.npz` | Wikipedia2Vec PT 2018-04-20, 100d, 50k palavras mais frequentes (sem entidades) | Apache 2.0 / CC BY-SA |
| `glove50d_top50k.npz` | GloVe 6B 50d (gensim-data), 50k palavras mais frequentes | PDDL |
| `analogies/questions-words.txt` | Mikolov et al. (2013a), repositório `tmikolov/word2vec` | Apache 2.0 |
| `analogies/LX-4WAnalogies*.txt` | Rodrigues et al. (2016), `nlx-group/lx-dsemvectors` | NLX |
| `lexicons/debiaswe_*.json` | Bolukbasi et al. (2016), `tolga-b/debiaswe` | MIT |

Os corpora grandes (Machado, Mac-Morpho) ficam em `data/raw` (ignorado) e são baixados por
`uv run replang download`; as versões processadas vão para `data/processed`.
