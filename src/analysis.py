"""Cross-model agreement, consensus, coverage, and neologism analysis.

Loader contract (display.py will import these later — keep names exact):

    load_model_votes(data_dir) -> dict[model, dict[(root, pattern), vote]]
    load_gold(data_dir)        -> dict[(root, pattern), dict] | None
    consensus(votes, gold_real) -> {status, provenance, n_votes, n_yes}

Join key is always (ROOT, PATTERN) — never VERB (surfaces collide across
roots). Every rate printed here goes through src.stats.fmt_rate.
"""

import csv
import json
from pathlib import Path

from src.stats import cohen_kappa, fleiss_kappa, fmt_rate

MODEL_NAMES = ("haiku", "sonnet", "opus", "gpt", "gemini", "gemma", "llama", "jais")
TOKENIZER_NAMES = ("cl100k_base", "o200k_base", "arabert", "camelbert", "jais", "qwen2.5")

GOLD_REAL_LABELS = frozenset({"attested_dictionary", "root_attested_form1"})
GOLD_NOT_REAL_LABELS = frozenset({"unattested"})
GOLD_CORPUS_ONLY = "attested_corpus_only"

# Minimum answered-pair intersection before a Cohen kappa is reportable.
MIN_KAPPA_N = 30

_TRUE = frozenset({"true", "1", "yes"})
_FALSE = frozenset({"false", "0", "no"})


# ---------------------------------------------------------------------------
# Parsing / loaders
# ---------------------------------------------------------------------------

def parse_bool(value):
    """Case-insensitive boolean parser: True/False/true/1/0/yes/no.

    Blank, None, or unrecognized values parse to None (no vote).
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def load_model_votes(data_dir):
    """Scan the model CSVs -> {model: {(root, pattern): vote_dict}}.

    Files with zero filled EXISTS? rows (or missing files) are skipped.
    Vote dict: {exists, msa, dialect_only, dialects, msa_meanings,
    dialect_meanings}. Rows are streamed via csv.DictReader.
    """
    data_dir = Path(data_dir)
    out = {}
    for model in MODEL_NAMES:
        path = data_dir / f"{model}.csv"
        if not path.exists():
            continue
        votes = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = (row.get("EXISTS?") or "").strip()
                if not raw:
                    continue
                key = ((row.get("ROOT") or "").strip(),
                       (row.get("PATTERN") or "").strip())
                dialects = [d.strip()
                            for d in (row.get("LIST_OF_DIALECTS") or "").split("|")
                            if d.strip()]
                votes[key] = {
                    "exists": parse_bool(raw),
                    "msa": parse_bool(row.get("MSA?")),
                    "dialect_only": parse_bool(row.get("DIALECT_ONLY?")),
                    "dialects": dialects,
                    "msa_meanings": (row.get("MSA_MEANINGS") or "").strip(),
                    "dialect_meanings": (row.get("DIALECT_MEANINGS") or "").strip(),
                }
        if votes:
            out[model] = votes
    return out


def gold_is_real(label):
    """Map a GOLD label to True (real) / False (not real) / None (unverified).

    attested_dictionary and root_attested_form1 are gold-real; unattested is
    gold-not-real; attested_corpus_only is unverified-with-corpus-flag.
    """
    if label in GOLD_REAL_LABELS:
        return True
    if label in GOLD_NOT_REAL_LABELS:
        return False
    return None


def load_gold(data_dir):
    """Parse data/gold.csv -> {(root, pattern): entry} or None if missing.

    Entry: {gold, real, corpus_flag, arramooz, wiktionary, corpus_freq, gloss}.
    """
    path = Path(data_dir) / "gold.csv"
    if not path.exists():
        return None
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("GOLD") or "").strip()
            freq_raw = (row.get("CORPUS_FREQ_BARE") or "").strip()
            key = ((row.get("ROOT") or "").strip(),
                   (row.get("PATTERN") or "").strip())
            gold[key] = {
                "gold": label,
                "real": gold_is_real(label),
                "corpus_flag": label == GOLD_CORPUS_ONLY,
                "arramooz": (row.get("ARRAMOOZ") or "").strip(),
                "wiktionary": (row.get("WIKTIONARY") or "").strip(),
                "corpus_freq": int(freq_raw) if freq_raw else 0,
                "gloss": (row.get("GOLD_GLOSS") or "").strip(),
            }
    return gold


def consensus(votes, gold_real=None):
    """Combine per-model exists votes with the gold label.

    votes: {model: bool | None}. Gold overrides everything; otherwise a
    strict majority of non-None votes with quorum >= 2 decides; a single
    vote counts provisionally (provenance llm-single); ties and no votes
    are unverified.
    """
    answered = [v for v in votes.values() if v is not None]
    n_votes = len(answered)
    n_yes = sum(1 for v in answered if v)
    result = {"status": "unverified", "provenance": None,
              "n_votes": n_votes, "n_yes": n_yes}
    if gold_real is True:
        result.update(status="real", provenance="gold")
    elif gold_real is False:
        result.update(status="not_real", provenance="gold")
    elif n_votes >= 2:
        if 2 * n_yes > n_votes:
            result.update(status="real", provenance="llm-majority")
        elif 2 * (n_votes - n_yes) > n_votes:
            result.update(status="not_real", provenance="llm-majority")
        # exact tie -> unverified, provenance None
    elif n_votes == 1:
        result.update(status="real" if n_yes else "not_real",
                      provenance="llm-single")
    return result


def iter_tokenizer_records(data_dir):
    """Stream data/tokenizer_analysis.jsonl line by line."""
    path = Path(data_dir) / "tokenizer_analysis.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _analysis_dir(data_dir):
    out = Path(data_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _exists_votes(votes_by_model, key):
    """{model: exists-bool-or-None} for one (root, pattern) key."""
    return {m: mv[key]["exists"] for m, mv in votes_by_model.items() if key in mv}


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def run_agreement(data_dir, view=None, echo=print):
    """Pairwise raw agreement + Cohen kappa, Fleiss kappa, marginals.

    Writes data/analysis/agreement.json and disagreements.jsonl. Returns the
    agreement summary dict.
    """
    data_dir = Path(data_dir)
    out_dir = _analysis_dir(data_dir)
    votes_by_model = load_model_votes(data_dir)
    gold = load_gold(data_dir) or {}
    models = sorted(votes_by_model)

    summary = {"models": models, "pairwise": {}, "fleiss": None, "marginals": {}}
    echo(f"Populated models: {', '.join(models) if models else '(none)'}")

    # Per-model marginal exists-rate (answered votes only).
    for m in models:
        answered = [v["exists"] for v in votes_by_model[m].values()
                    if v["exists"] is not None]
        rate = fmt_rate(sum(answered), len(answered))
        summary["marginals"][m] = {"n_answered": len(answered),
                                   "n_yes": sum(answered), "rate": rate}
        echo(f"  {m} exists-rate: {rate}")

    # Pairwise raw % + Cohen kappa over answered-key intersections.
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            keys_a = {k for k, v in votes_by_model[a].items()
                      if v["exists"] is not None}
            keys_b = {k for k, v in votes_by_model[b].items()
                      if v["exists"] is not None}
            shared = sorted(keys_a & keys_b)
            n = len(shared)
            cell = {"n": n, "raw_agreement": None, "kappa": None, "note": None}
            if n:
                va = [votes_by_model[a][k]["exists"] for k in shared]
                vb = [votes_by_model[b][k]["exists"] for k in shared]
                agree = sum(1 for x, y in zip(va, vb) if x == y)
                cell["raw_agreement"] = fmt_rate(agree, n)
                if n >= MIN_KAPPA_N:
                    cell["kappa"] = round(cohen_kappa(va, vb), 4)
                else:
                    cell["note"] = "n-too-small"
            else:
                cell["note"] = "no shared answered keys"
            summary["pairwise"][f"{a}|{b}"] = cell
            kappa_str = (f"kappa={cell['kappa']}" if cell["kappa"] is not None
                         else f"kappa: {cell['note']} (n={n})")
            echo(f"  {a} vs {b}: n={n}  raw={cell['raw_agreement']}  {kappa_str}")
    if len(models) < 2:
        echo("  (fewer than 2 populated models — no pairwise agreement to compute)")

    # Fleiss kappa on the complete-case subset (all populated models answered).
    if len(models) >= 2:
        complete = None
        for m in models:
            keys = {k for k, v in votes_by_model[m].items()
                    if v["exists"] is not None}
            complete = keys if complete is None else (complete & keys)
        complete = sorted(complete or ())
        if complete:
            counts = []
            for k in complete:
                yes = sum(1 for m in models if votes_by_model[m][k]["exists"])
                counts.append((yes, len(models) - yes))
            summary["fleiss"] = {"n_items": len(complete),
                                 "n_raters": len(models),
                                 "kappa": round(fleiss_kappa(counts), 4)}
            echo(f"  Fleiss kappa (complete-case, n={len(complete)} items, "
                 f"{len(models)} raters): {summary['fleiss']['kappa']}")
        else:
            summary["fleiss"] = {"n_items": 0, "n_raters": len(models),
                                 "kappa": None, "note": "empty complete-case subset"}
            echo("  Fleiss kappa: empty complete-case subset")

    # Disagreements: keys with >= 2 votes containing both True and False.
    all_keys = set()
    for mv in votes_by_model.values():
        all_keys.update(mv)
    disagreement_keys = set()
    for key in all_keys:
        v = [x for x in _exists_votes(votes_by_model, key).values()
             if x is not None]
        if len(v) >= 2 and any(v) and not all(v):
            disagreement_keys.add(key)

    # Join single_token_bare from tokenizer_analysis.jsonl (streamed once).
    tok_join = {}
    tok_path = data_dir / "tokenizer_analysis.jsonl"
    if disagreement_keys and tok_path.exists():
        for rec in iter_tokenizer_records(data_dir):
            key = (rec["root"], rec["pattern"])
            if key in disagreement_keys:
                tok_join[key] = rec.get("single_token_bare")
                if len(tok_join) == len(disagreement_keys):
                    break

    dis_path = out_dir / "disagreements.jsonl"
    with open(dis_path, "w", encoding="utf-8") as f:
        for key in sorted(disagreement_keys):
            root, pattern = key
            g = gold.get(key)
            f.write(json.dumps({
                "root": root, "pattern": pattern,
                "votes": _exists_votes(votes_by_model, key),
                "gold": g["gold"] if g else None,
                "single_token_bare": tok_join.get(key),
            }, ensure_ascii=False) + "\n")
    summary["n_disagreements"] = len(disagreement_keys)
    echo(f"  Disagreements (>=2 votes, mixed): {len(disagreement_keys)} "
         f"-> {dis_path}")

    # --view filters for printed rows.
    if view:
        echo(f"\n--view {view}:")
        shown = 0
        for key in sorted(all_keys):
            v = _exists_votes(votes_by_model, key)
            answered = [x for x in v.values() if x is not None]
            g = gold.get(key)
            label = g["gold"] if g else None
            if view == "hallucinated":
                hit = label == "unattested" and any(x is True for x in answered)
            else:  # rare-real
                hit = (g is not None and g["real"] is True and answered
                       and 2 * sum(1 for x in answered if x is False)
                       > len(answered))
            if hit:
                echo(f"  {key[0]} {key[1]}  gold={label}  votes={v}")
                shown += 1
        echo(f"  ({shown} rows)")

    agr_path = out_dir / "agreement.json"
    with open(agr_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    echo(f"Wrote {agr_path}")
    return summary


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def _bump(table, group, hit):
    cell = table.setdefault(group, [0, 0])
    cell[1] += 1
    if hit:
        cell[0] += 1


def run_coverage(data_dir, echo=print):
    """Single-token-bare coverage of consensus-real verbs; tokenizer ghosts.

    Works with gold alone when no model CSVs are populated. Writes
    data/analysis/coverage.json and ghosts.jsonl. Returns the summary dict.
    """
    data_dir = Path(data_dir)
    out_dir = _analysis_dir(data_dir)
    votes_by_model = load_model_votes(data_dir)
    gold = load_gold(data_dir) or {}
    if not votes_by_model:
        echo("No populated model CSVs — consensus falls back to gold alone.")

    tok_names = list(TOKENIZER_NAMES) + ["any"]
    # per tokenizer -> breakdown kind -> group -> [k, n]
    tallies = {t: {"overall": {}, "by_root_type": {}, "by_weak_class": {},
                   "by_attestation": {}} for t in tok_names}
    n_real = n_not_real = n_ghosts = 0
    ghosts_path = out_dir / "ghosts.jsonl"

    with open(ghosts_path, "w", encoding="utf-8") as ghost_f:
        for rec in iter_tokenizer_records(data_dir):
            key = (rec["root"], rec["pattern"])
            g = gold.get(key)
            ev = _exists_votes(votes_by_model, key)
            c = consensus(ev, g["real"] if g else None)
            tb = rec["tokens_bare"]
            single_in = [t for t in TOKENIZER_NAMES if tb.get(t) == 1]

            if c["status"] == "real":
                n_real += 1
                classes = []
                if c["provenance"] == "gold":
                    classes.append("gold-dictionary")
                elif c["provenance"] in ("llm-majority", "llm-single"):
                    classes.append("llm-only")
                if any(mv[key].get("dialect_only") is True
                       for mv in votes_by_model.values() if key in mv):
                    classes.append("dialect_only")
                for t in tok_names:
                    if t == "any":
                        hit, known = bool(single_in), True
                    else:
                        known = tb.get(t) is not None
                        hit = tb.get(t) == 1
                    if not known:
                        continue
                    _bump(tallies[t]["overall"], "all", hit)
                    _bump(tallies[t]["by_root_type"], rec["root_type"], hit)
                    _bump(tallies[t]["by_weak_class"], rec["weak_class"], hit)
                    for cls in classes:
                        _bump(tallies[t]["by_attestation"], cls, hit)
            elif c["status"] == "not_real":
                n_not_real += 1
                if single_in:  # tokenizer ghost
                    n_ghosts += 1
                    ghost_f.write(json.dumps({
                        "verb": rec["verb"], "verb_bare": rec.get("verb_bare"),
                        "root": rec["root"], "root_type": rec["root_type"],
                        "pattern": rec["pattern"],
                        "weak_class": rec["weak_class"],
                        "plausibility": rec.get("plausibility"),
                        "single_token_in": single_in,
                        "gold": g["gold"] if g else None,
                        "provenance": c["provenance"],
                    }, ensure_ascii=False) + "\n")

    def fmt_table(table):
        return {grp: {"k": k, "n": n, "rate": fmt_rate(k, n)}
                for grp, (k, n) in sorted(table.items())}

    summary = {
        "n_consensus_real": n_real,
        "n_consensus_not_real": n_not_real,
        "per_tokenizer": {t: {kind: fmt_table(tbl)
                              for kind, tbl in tallies[t].items()}
                          for t in tok_names},
        "ghosts": {"n": n_ghosts, "of_not_real": fmt_rate(n_ghosts, n_not_real),
                   "file": str(ghosts_path)},
    }

    echo(f"Consensus-real verbs: {n_real:,} "
         f"(not-real: {n_not_real:,})")
    echo("\nSingle-token (bare) coverage of consensus-real verbs:")
    for t in tok_names:
        cell = tallies[t]["overall"].get("all")
        rate = fmt_rate(cell[0], cell[1]) if cell else fmt_rate(0, 0)
        echo(f"  {t:12s} {rate}")
    for kind, label in (("by_root_type", "root_type"),
                        ("by_weak_class", "weak_class"),
                        ("by_attestation", "attestation class")):
        echo(f"\nAny-tokenizer coverage by {label}:")
        for grp, (k, n) in sorted(tallies["any"][kind].items()):
            echo(f"  {grp:20s} {fmt_rate(k, n)}")
    echo(f"\nTokenizer ghosts (single-token but consensus not_real): "
         f"{fmt_rate(n_ghosts, n_not_real)} -> {ghosts_path}")

    cov_path = out_dir / "coverage.json"
    with open(cov_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    echo(f"Wrote {cov_path}")
    return summary


# ---------------------------------------------------------------------------
# neologisms
# ---------------------------------------------------------------------------

def run_neologisms(data_dir, include_unverified=False, tokenizer=None, top=50,
                   echo=print):
    """Unattested-but-single-token candidates ranked by plausibility.

    Writes data/analysis/neologisms.jsonl; returns the ranked row list.
    """
    if tokenizer and tokenizer not in TOKENIZER_NAMES:
        raise ValueError(f"unknown tokenizer {tokenizer!r} "
                         f"(choose from {', '.join(TOKENIZER_NAMES)})")
    data_dir = Path(data_dir)
    out_dir = _analysis_dir(data_dir)
    votes_by_model = load_model_votes(data_dir)
    gold = load_gold(data_dir) or {}

    rows = []
    for rec in iter_tokenizer_records(data_dir):
        key = (rec["root"], rec["pattern"])
        g = gold.get(key)
        label = g["gold"] if g else ""
        c = consensus(_exists_votes(votes_by_model, key),
                      g["real"] if g else None)
        core = label == "unattested" and c["status"] == "not_real"
        if not core:
            if not include_unverified:
                continue
            if c["status"] != "unverified" or label in GOLD_REAL_LABELS:
                continue
        tb = rec["tokens_bare"]
        single_in = [t for t in TOKENIZER_NAMES if tb.get(t) == 1]
        if tokenizer:
            if tb.get(tokenizer) != 1:
                continue
        elif not single_in:
            continue
        rows.append({
            "verb": rec["verb"], "verb_bare": rec.get("verb_bare"),
            "root": rec["root"], "root_type": rec["root_type"],
            "pattern": rec["pattern"], "weak_class": rec["weak_class"],
            "plausibility": rec.get("plausibility", 0.0),
            "single_token_in": single_in,
            "gold": label or None,
            "corpus_freq": g["corpus_freq"] if g else 0,
        })

    rows.sort(key=lambda r: (-(r["plausibility"] or 0.0),
                             r["root"], r["pattern"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    neo_path = out_dir / "neologisms.jsonl"
    with open(neo_path, "w", encoding="utf-8") as f:
        for row in rows:
            ordered = {"rank": row["rank"]}
            ordered.update({k: v for k, v in row.items() if k != "rank"})
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
    echo(f"{len(rows):,} neologism candidates -> {neo_path}")
    return rows


def print_neologism_table(rows, top=50):
    """Rich table of the top-N neologism candidates."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Top {min(top, len(rows))} single-token neologism "
                        f"candidates (of {len(rows):,})")
    for col in ("Rank", "Verb", "Bare", "Root", "Type", "Pattern",
                "Weak class", "Plaus.", "Single-token in", "Gold", "Freq"):
        table.add_column(col)
    for row in rows[:top]:
        table.add_row(
            str(row["rank"]), row["verb"], row["verb_bare"] or "",
            row["root"], row["root_type"], row["pattern"],
            row["weak_class"], f"{row['plausibility']:.2f}",
            ",".join(row["single_token_in"]), row["gold"] or "",
            str(row["corpus_freq"]),
        )
    Console().print(table)


# ---------------------------------------------------------------------------
# tokenizers
# ---------------------------------------------------------------------------

def run_tokenizers(data_dir, markdown=False, echo=print):
    """Per-tokenizer single-token counts (bare + diacriticized) and fertility.

    Writes data/analysis/tokenizers.json; returns the summary dict.
    """
    data_dir = Path(data_dir)
    out_dir = _analysis_dir(data_dir)
    stats = {t: {"bare_k": 0, "bare_n": 0, "diac_k": 0, "diac_n": 0,
                 "tok_sum": 0} for t in TOKENIZER_NAMES}
    any_bare = any_diac = total = 0
    # Which-tokenizers-hit-on-diacriticized combos, so any-over-a-subset
    # (e.g. tashkeel-preserving tokenizers only) is computable post-hoc.
    diac_combos = {}

    for rec in iter_tokenizer_records(data_dir):
        total += 1
        tb, td = rec["tokens_bare"], rec["tokens_diacriticized"]
        for t in TOKENIZER_NAMES:
            b, d = tb.get(t), td.get(t)
            if b is not None:
                stats[t]["bare_n"] += 1
                stats[t]["tok_sum"] += b
                if b == 1:
                    stats[t]["bare_k"] += 1
            if d is not None:
                stats[t]["diac_n"] += 1
                if d == 1:
                    stats[t]["diac_k"] += 1
        if any(tb.get(t) == 1 for t in TOKENIZER_NAMES):
            any_bare += 1
        diac_hits = frozenset(t for t in TOKENIZER_NAMES if td.get(t) == 1)
        if diac_hits:
            any_diac += 1
            diac_combos[diac_hits] = diac_combos.get(diac_hits, 0) + 1

    summary = {"n_candidates": total, "per_tokenizer": {}, "any_tokenizer": {}}
    for t in TOKENIZER_NAMES:
        s = stats[t]
        fertility = s["tok_sum"] / s["bare_n"] if s["bare_n"] else None
        summary["per_tokenizer"][t] = {
            "single_token_bare": {"k": s["bare_k"], "n": s["bare_n"],
                                  "rate": fmt_rate(s["bare_k"], s["bare_n"])},
            "single_token_diacriticized": {"k": s["diac_k"], "n": s["diac_n"],
                                           "rate": fmt_rate(s["diac_k"],
                                                            s["diac_n"])},
            "fertility_bare": round(fertility, 3) if fertility else None,
            "normalizer_strips_tashkeel": (s["diac_n"] > 0
                                           and s["diac_k"] == s["bare_k"]
                                           and s["bare_k"] > 0),
        }
    strippers = [t for t in TOKENIZER_NAMES
                 if summary["per_tokenizer"][t]["normalizer_strips_tashkeel"]]
    preserving = frozenset(TOKENIZER_NAMES) - frozenset(strippers)
    any_diac_preserving = sum(count for combo, count in diac_combos.items()
                              if combo & preserving)
    summary["any_tokenizer"] = {
        "single_token_bare": {"k": any_bare, "n": total,
                              "rate": fmt_rate(any_bare, total)},
        "single_token_diacriticized": {"k": any_diac, "n": total,
                                       "rate": fmt_rate(any_diac, total)},
        "single_token_diacriticized_tashkeel_preserving": {
            "k": any_diac_preserving, "n": total,
            "rate": fmt_rate(any_diac_preserving, total),
            "tokenizers": sorted(preserving),
        },
    }

    if markdown:
        echo("| Tokenizer | Single-token (bare) | % of n [95% CI] | "
             "Fertility | Notes |")
        echo("|---|---|---|---|---|")
        for t in TOKENIZER_NAMES:
            p = summary["per_tokenizer"][t]
            k, n = p["single_token_bare"]["k"], p["single_token_bare"]["n"]
            rate = p["single_token_bare"]["rate"]
            # fmt_rate = "k/n = p% (95% CI lo–hi%)" -> split for the columns
            pct = rate.split(" = ", 1)[1] if " = " in rate else rate
            notes = []
            if p["normalizer_strips_tashkeel"]:
                notes.append("normalizer-strips-tashkeel (diacriticized hits "
                             "equal bare hits)")
            if p["single_token_diacriticized"]["n"] == 0:
                notes.append("no diacriticized data")
            fert = (f"{p['fertility_bare']:.2f}"
                    if p["fertility_bare"] is not None else "n/a")
            echo(f"| {t} | {k:,}/{n:,} | {pct} | {fert} | "
                 f"{'; '.join(notes) or '—'} |")
        a = summary["any_tokenizer"]["single_token_bare"]
        pct = a["rate"].split(" = ", 1)[1] if " = " in a["rate"] else a["rate"]
        echo(f"| **any tokenizer** | {a['k']:,}/{a['n']:,} | {pct} | — | — |")
        echo("")
        echo(f"Note: the zero single-token rate on *diacriticized* forms "
             f"({fmt_rate(any_diac_preserving, total)} any-tokenizer) holds "
             f"only for tokenizers whose normalizer preserves tashkeel "
             f"({', '.join(sorted(preserving))}). "
             f"{', '.join(strippers) or 'None of the tokenizers'} strip(s) "
             f"diacritics during normalization, so its diacriticized hits "
             f"equal its bare hits and cannot be read as evidence about "
             f"tashkeel-bearing text.")
    else:
        echo(f"Candidates analyzed: {total:,}")
        for t in TOKENIZER_NAMES:
            p = summary["per_tokenizer"][t]
            fert = (f"{p['fertility_bare']:.2f}"
                    if p["fertility_bare"] is not None else "n/a")
            flag = (" [normalizer-strips-tashkeel]"
                    if p["normalizer_strips_tashkeel"] else "")
            echo(f"  {t:12s} bare: {p['single_token_bare']['rate']}  "
                 f"diac: {p['single_token_diacriticized']['rate']}  "
                 f"fertility(bare): {fert}{flag}")
        echo(f"  {'any':12s} bare: "
             f"{summary['any_tokenizer']['single_token_bare']['rate']}  "
             f"diac: {summary['any_tokenizer']['single_token_diacriticized']['rate']}")

    tok_path = out_dir / "tokenizers.json"
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    echo(f"Wrote {tok_path}")
    return summary
