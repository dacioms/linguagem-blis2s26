import gzip

from replang.data.legal import LEGAL, _parse_conll, corpus_profile, normalize_area


def test_lexicon_consistency():
    assert len(LEGAL.polysemous) > 25 and all(len(p) == 3 for p in LEGAL.polysemous)
    assert ("juíza", "juiz") in LEGAL.gendered_roles
    assert all(len(a) == 4 for a in LEGAL.analogies)
    assert set(LEGAL.probe_words) <= {p[0] for p in LEGAL.polysemous}


def test_normalize_area():
    assert normalize_area("Penal") == "direito penal"
    assert normalize_area("direito civil") == "direito civil"
    assert normalize_area(None) is None


def test_parse_conll():
    lines = [
        "EMENTA O",
        ": O",
        "MINISTÉRIO B-ORGANIZACAO",
        "PÚBLICO I-ORGANIZACAO",
        "",
        "outra O",
        "",
    ]
    sents = _parse_conll(lines)
    assert len(sents) == 2 and sents[0][2] == ("MINISTÉRIO", "B-ORGANIZACAO")


def test_corpus_profile_keys():
    prof = corpus_profile(
        [["art", "0", "da", "lei"], ["habeas", "corpus", "é", "remédio", "constitucional"]], "x"
    )
    assert prof["tokens"] == 9 and prof["'art'"] > 0 and prof["latim (tokens)"] > 0


def test_lener_loader_offline(tmp_path, monkeypatch):
    from replang.data import legal

    p = tmp_path / "lener.conll.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write("# split train\nA\tO\nB\tB-PESSOA\n\n# split test\nC\tO\n\n")
    monkeypatch.setattr(legal, "LENER_PROCESSED", p)
    assert legal.load_lener("train") == [[("A", "O"), ("B", "B-PESSOA")]]
    assert legal.load_lener("test") == [[("C", "O")]]
    assert len(legal.load_lener("all")) == 2
