from replang.utils.text import build_vocab, iter_sentences, normalize_text, strip_accents, tokenize


def test_normalize_numbers_urls_emails():
    s = normalize_text("Veja http://x.com/a e mail joao@x.com em 2024!")
    assert "URL" in s.upper() and "EMAIL" in s.upper() and "0000" in s


def test_tokenize_keeps_clitics_and_drops_punct():
    toks = tokenize("Ele machucou-se, dir-se-ia; não é?")
    assert "machucou-se" in toks and "dir-se-ia" in toks and "," not in toks
    assert tokenize("A B", keep_punct=True) == ["a", "b"]


def test_iter_sentences_min_tokens():
    text = "Uma frase curta. Esta aqui tem muitas palavras para passar no filtro."
    sents = list(iter_sentences(text, min_tokens=5))
    assert len(sents) == 1 and sents[0][0] == "esta"


def test_build_vocab_order_and_unk():
    words, counts = build_vocab([["a", "b", "a"], ["a", "c"]], min_count=1)
    assert words[0] == "UNKNOWN" and words[1] == "a" and counts["a"] == 3


def test_strip_accents():
    assert strip_accents("coração") == "coracao"
