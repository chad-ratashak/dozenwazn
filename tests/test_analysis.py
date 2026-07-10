"""Tests for src/analysis.py — consensus, parsers, loaders, agreement."""

import csv
import json

from src.analysis import (
    consensus,
    load_gold,
    load_model_votes,
    parse_bool,
    run_agreement,
)


# ---------------------------------------------------------------------------
# parse_bool
# ---------------------------------------------------------------------------

def test_parse_bool_true_variants():
    for v in ("True", "true", "TRUE", "1", "yes", "YES", " true "):
        assert parse_bool(v) is True, v


def test_parse_bool_false_variants():
    for v in ("False", "false", "FALSE", "0", "no", "No", " false "):
        assert parse_bool(v) is False, v


def test_parse_bool_none_and_junk():
    for v in (None, "", "   ", "maybe", "n/a"):
        assert parse_bool(v) is None, repr(v)


# ---------------------------------------------------------------------------
# consensus precedence
# ---------------------------------------------------------------------------

def test_gold_beats_majority():
    # Three models say False, but gold says real -> gold wins.
    votes = {"a": False, "b": False, "c": False}
    c = consensus(votes, gold_real=True)
    assert c["status"] == "real"
    assert c["provenance"] == "gold"
    assert c["n_votes"] == 3 and c["n_yes"] == 0

    # And the reverse: everyone says True, gold says unattested.
    c = consensus({"a": True, "b": True, "c": True}, gold_real=False)
    assert c["status"] == "not_real"
    assert c["provenance"] == "gold"


def test_majority_with_quorum_two():
    c = consensus({"a": True, "b": True}, gold_real=None)
    assert c["status"] == "real"
    assert c["provenance"] == "llm-majority"
    assert c["n_votes"] == 2 and c["n_yes"] == 2

    c = consensus({"a": False, "b": False, "c": True}, gold_real=None)
    assert c["status"] == "not_real"
    assert c["provenance"] == "llm-majority"


def test_single_vote_is_provisional():
    c = consensus({"a": True}, gold_real=None)
    assert c["status"] == "real"
    assert c["provenance"] == "llm-single"

    c = consensus({"a": False}, gold_real=None)
    assert c["status"] == "not_real"
    assert c["provenance"] == "llm-single"


def test_tie_and_no_votes_are_unverified():
    c = consensus({"a": True, "b": False}, gold_real=None)
    assert c["status"] == "unverified"
    assert c["provenance"] is None

    c = consensus({}, gold_real=None)
    assert c["status"] == "unverified"
    assert c["provenance"] is None
    assert c["n_votes"] == 0 and c["n_yes"] == 0


def test_none_votes_do_not_count_toward_quorum():
    # One real vote + one blank: still a single vote.
    c = consensus({"a": True, "b": None}, gold_real=None)
    assert c["provenance"] == "llm-single"
    assert c["n_votes"] == 1


# ---------------------------------------------------------------------------
# Synthetic-data loaders + one agreement run
# ---------------------------------------------------------------------------

MODEL_HEADER = ["ROOT", "ROOT_TYPE", "ARITY", "PATTERN", "WEAK_CLASS",
                "PLAUSIBILITY", "VERB", "EXISTS?", "DIALECT_ONLY?",
                "LIST_OF_DIALECTS", "DIALECT_MEANINGS", "MSA?", "MSA_MEANINGS"]


def _write_model_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MODEL_HEADER)
        writer.writerows(rows)


def _make_dataset(tmp_path):
    """Two populated models, one empty model, gold, tokenizer analysis."""
    _write_model_csv(tmp_path / "haiku.csv", [
        ["كتب", "XYZ", "3", "faʿala", "sound", "1.0", "كَتَبَ",
         "True", "False", "Egyptian|Levantine", "to write", "true", "to write"],
        ["كتب", "XYZ", "3", "faʿula", "sound", "0.8", "كَتُبَ",
         "false", "", "", "", "no", ""],
        ["دحرج", "XYZW", "4", "faʿlala", "sound", "0.9", "دَحْرَجَ",
         "1", "0", "", "", "yes", "to roll"],
        # Blank EXISTS? -> not a vote.
        ["ءبت", "XYZ", "3", "faʿila", "sound", "1.0", "أَبِتَ",
         "", "", "", "", "", ""],
    ])
    _write_model_csv(tmp_path / "sonnet.csv", [
        ["كتب", "XYZ", "3", "faʿala", "sound", "1.0", "كَتَبَ",
         "yes", "", "", "", "", ""],
        ["كتب", "XYZ", "3", "faʿula", "sound", "0.8", "كَتُبَ",
         "True", "", "", "", "", ""],  # disagrees with haiku
        ["دحرج", "XYZW", "4", "faʿlala", "sound", "0.9", "دَحْرَجَ",
         "no", "", "", "", "", ""],   # disagrees with haiku
    ])
    # Header-only file: zero filled rows -> must be skipped entirely.
    _write_model_csv(tmp_path / "gemma.csv", [])

    with open(tmp_path / "gold.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ROOT", "ROOT_TYPE", "ARITY", "PATTERN", "VERB",
                         "ARRAMOOZ", "WIKTIONARY", "CORPUS_FREQ_BARE",
                         "GOLD", "GOLD_GLOSS"])
        writer.writerow(["كتب", "XYZ", "3", "faʿala", "كَتَبَ",
                         "exact", "exact", "9000", "attested_dictionary",
                         "to write"])
        writer.writerow(["كتب", "XYZ", "3", "faʿula", "كَتُبَ",
                         "", "", "0", "unattested", ""])
        writer.writerow(["دحرج", "XYZW", "4", "faʿlala", "دَحْرَجَ",
                         "", "", "12", "attested_corpus_only", ""])

    with open(tmp_path / "tokenizer_analysis.jsonl", "w", encoding="utf-8") as f:
        for root, pattern, single in (("كتب", "faʿala", True),
                                      ("كتب", "faʿula", False),
                                      ("دحرج", "faʿlala", True)):
            f.write(json.dumps({
                "root": root, "root_type": "XYZ", "arity": 3,
                "pattern": pattern, "verb": "x", "verb_bare": "x",
                "weak_class": "sound", "plausibility": 1.0,
                "tokens_bare": {t: 1 if single else 3
                                for t in ("cl100k_base", "o200k_base",
                                          "arabert", "camelbert", "jais",
                                          "qwen2.5")},
                "tokens_diacriticized": {t: 4 for t in
                                         ("cl100k_base", "o200k_base",
                                          "arabert", "camelbert", "jais",
                                          "qwen2.5")},
                "single_token_bare": single,
                "single_token_diacriticized": False,
            }, ensure_ascii=False) + "\n")
    return tmp_path


def test_load_model_votes(tmp_path):
    _make_dataset(tmp_path)
    votes = load_model_votes(tmp_path)

    # gemma has zero filled rows and the other five files don't exist.
    assert set(votes) == {"haiku", "sonnet"}

    h = votes["haiku"]
    # Blank EXISTS? row is not a vote.
    assert ("ءبت", "faʿila") not in h
    assert len(h) == 3

    v = h[("كتب", "faʿala")]
    assert v["exists"] is True
    assert v["dialect_only"] is False
    assert v["msa"] is True
    assert v["dialects"] == ["Egyptian", "Levantine"]  # split on |
    assert v["msa_meanings"] == "to write"

    # Numeric booleans: "1"/"0".
    v = h[("دحرج", "faʿlala")]
    assert v["exists"] is True and v["dialect_only"] is False

    # lowercase false.
    assert h[("كتب", "faʿula")]["exists"] is False


def test_load_gold(tmp_path):
    _make_dataset(tmp_path)
    gold = load_gold(tmp_path)
    assert gold is not None

    g = gold[("كتب", "faʿala")]
    assert g["real"] is True
    assert g["gold"] == "attested_dictionary"
    assert g["corpus_freq"] == 9000
    assert g["gloss"] == "to write"
    assert g["arramooz"] == "exact"

    assert gold[("كتب", "faʿula")]["real"] is False
    # attested_corpus_only -> unverified with corpus flag.
    g = gold[("دحرج", "faʿlala")]
    assert g["real"] is None
    assert g["corpus_flag"] is True
    assert g["corpus_freq"] == 12


def test_load_gold_missing_returns_none(tmp_path):
    assert load_gold(tmp_path) is None


def test_agreement_end_to_end(tmp_path):
    _make_dataset(tmp_path)
    lines = []
    summary = run_agreement(tmp_path, echo=lines.append)

    assert summary["models"] == ["haiku", "sonnet"]

    cell = summary["pairwise"]["haiku|sonnet"]
    assert cell["n"] == 3
    # Agree on 1 of 3 shared keys; kappa refused below n=30.
    assert cell["raw_agreement"].startswith("1/3")
    assert cell["kappa"] is None
    assert cell["note"] == "n-too-small"

    # Marginals go through fmt_rate (k/n = ...% with a CI).
    assert summary["marginals"]["haiku"]["rate"].startswith("2/3")
    assert "95% CI" in summary["marginals"]["haiku"]["rate"]

    # Fleiss on the complete-case subset (both answered all 3 keys).
    assert summary["fleiss"]["n_items"] == 3
    assert isinstance(summary["fleiss"]["kappa"], float)

    # Disagreements: the two mixed keys, joined with single_token_bare.
    dis_path = tmp_path / "analysis" / "disagreements.jsonl"
    rows = {(r["root"], r["pattern"]): r
            for r in map(json.loads, dis_path.read_text().splitlines())}
    assert set(rows) == {("كتب", "faʿula"), ("دحرج", "faʿlala")}
    assert rows[("كتب", "faʿula")]["gold"] == "unattested"
    assert rows[("كتب", "faʿula")]["single_token_bare"] is False
    assert rows[("دحرج", "faʿlala")]["single_token_bare"] is True
    assert rows[("دحرج", "faʿlala")]["votes"] == {"haiku": True, "sonnet": False}

    # agreement.json artifact exists and round-trips.
    saved = json.loads((tmp_path / "analysis" / "agreement.json").read_text())
    assert saved["n_disagreements"] == 2
