"""replang — Representações de Linguagem.

Pacote didático que acompanha a apresentação *"Representações de Linguagem: como
transformar palavra em vetor, o que esse vetor carrega e o viés que vem junto"*
(BLIS 2S26).

Módulos principais
------------------
- :mod:`replang.embedding`  – contêiner leve ``WordVectors`` (vizinhos, analogias, I/O).
- :mod:`replang.data`       – corpora públicos, léxicos de gênero/profissões, conjuntos de analogias.
- :mod:`replang.models`     – implementações *from scratch* (co-ocorrência/PPMI/SVD, Skip-gram/CBOW,
  softmax hierárquico, amostragem negativa, GloVe, fastText) e invólucros do gensim.
- :mod:`replang.eval`       – avaliação intrínseca (analogias, similaridade) e extrínseca (POS, STS).
- :mod:`replang.bias`       – geometria do viés de gênero (Bolukbasi et al.) e algoritmos de *debias*.
- :mod:`replang.viz`        – figuras Plotly/Matplotlib reutilizadas por notebooks e pela interface.
"""

from replang.config import PATHS, settings
from replang.embedding import WordVectors

__all__ = ["PATHS", "WordVectors", "settings"]
__version__ = "0.1.0"
