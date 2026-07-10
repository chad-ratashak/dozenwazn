#!/usr/bin/env python3
"""Validate Arabic verb candidates using local LLMs via Ollama."""

import csv
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.generate_csvs import (
    CSV_HEADERS,
    assert_csv_schema,
    backup_csv,
    count_filled_rows,
    write_scaffold,
)
from src.models import ValidationResult

console = Console()

# ── Ollama Configuration ──

OLLAMA_BASE = "http://localhost:11434"

DEFAULT_OLLAMA_MODELS = {
    # Pinned tags (":latest" drifts). Both Meta and Google are US-origin,
    # compliant with the model-origin policy. The gemma choice is subject
    # to the spotcheck A/B (gemma4:e2b vs gemma3:4b) — see `spotcheck`.
    "llama": "llama3.2:3b",
    "gemma": "gemma4:e2b",
}

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]  # seconds

SYSTEM_PROMPT = """\
You are an expert in Arabic morphology (صرف). Your task is to classify Arabic Form I (فعل ثلاثي مجرد) verbs.

For each verb provided, determine:
1. **exists**: Does this verb exist in Arabic (classical, MSA, or any dialect)? true/false
2. **dialect_only**: Is this verb used ONLY in dialects and NOT in Modern Standard Arabic (MSA) or Classical Arabic? true/false
3. **list_of_dialects**: If dialect_only is true, list which dialects use it (e.g., ["Egyptian", "Levantine"]). Otherwise [].
4. **dialect_meanings**: If used in dialects, English meaning(s), at most 6 words. Otherwise "".
5. **msa**: Is this verb used in Modern Standard Arabic? true/false
6. **msa_meanings**: If used in MSA, English meaning(s), at most 6 words. Otherwise "".

Rules:
- Verbs are presented with full tashkeel (diacritics). The vowel pattern MATTERS.
- The three vocalizations of a root (فَعَلَ / فَعِلَ / فَعُلَ) are DIFFERENT verbs; usually at most
  one of them exists. NEVER give identical answers for all three vocalizations of a root,
  and never copy a meaning from one verb to another.
- Evaluate each verb independently. Most candidates in a batch do NOT exist —
  they are generated combinatorially.
- If a verb does not exist at all, set exists=false and leave all other fields at defaults.
- Be conservative: if unsure whether a verb exists, set exists=false.

Respond with a JSON object containing a single key "results" whose value is an array of objects, \
one per verb in the SAME ORDER as the input. Each object must have these exact keys:
  verb, exists, dialect_only, list_of_dialects, dialect_meanings, msa, msa_meanings

Example — note how the three vocalizations of كتب get DIFFERENT answers, and how
rare-but-real faʿila/faʿula verbs are recognized while invented roots are rejected:
{"results": [
  {"verb": "كَتَبَ", "exists": true, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": true, "msa_meanings": "to write"},
  {"verb": "كَتِبَ", "exists": false, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": false, "msa_meanings": ""},
  {"verb": "كَتُبَ", "exists": false, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": false, "msa_meanings": ""},
  {"verb": "عَلِمَ", "exists": true, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": true, "msa_meanings": "to know"},
  {"verb": "حَسُنَ", "exists": true, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": true, "msa_meanings": "to be good"},
  {"verb": "ثَذَغَ", "exists": false, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": false, "msa_meanings": ""},
  {"verb": "دَحْبَجَ", "exists": false, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": false, "msa_meanings": ""}
]}"""


def build_response_schema(n_verbs: int) -> dict:
    """JSON schema for Ollama structured outputs: exactly one result object
    per verb, exact field set — kills truncated arrays and free-form drift."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": n_verbs,
                "maxItems": n_verbs,
                "items": {
                    "type": "object",
                    "properties": {
                        "verb": {"type": "string"},
                        "exists": {"type": "boolean"},
                        "dialect_only": {"type": "boolean"},
                        "list_of_dialects": {"type": "array", "items": {"type": "string"}},
                        "dialect_meanings": {"type": "string"},
                        "msa": {"type": "boolean"},
                        "msa_meanings": {"type": "string"},
                    },
                    "required": ["verb", "exists", "dialect_only", "list_of_dialects",
                                 "dialect_meanings", "msa", "msa_meanings"],
                },
            },
        },
        "required": ["results"],
    }


def probe_schema_support(model_tag: str, client: httpx.Client, logger: logging.Logger) -> bool:
    """One tiny schema-constrained request; False → fall back to format=json."""
    try:
        resp = client.post("/api/chat", json={
            "model": model_tag,
            "messages": [{"role": "user", "content": 'Reply with JSON: {"ok": true}'}],
            "format": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                       "required": ["ok"]},
            "stream": False,
            "options": {"num_predict": 32},
        })
        resp.raise_for_status()
        json.loads(resp.json().get("message", {}).get("content", ""))
        return True
    except Exception as e:
        logger.warning(f"Structured-output probe failed ({e}); falling back to format=json")
        return False


# ── Ollama Health & Model Management ──


def check_ollama_health() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/", timeout=5.0)
        return resp.status_code == 200
    except httpx.ConnectError:
        return False


def pull_model(model_tag: str) -> bool:
    """Pull an Ollama model, showing progress."""
    console.print(f"  Pulling [bold]{model_tag}[/bold]...")
    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model_tag},
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                status = data.get("status", "")
                if "completed" in data and "total" in data:
                    pct = data["completed"] / data["total"] * 100 if data["total"] else 0
                    console.print(f"\r  {status}: {pct:.0f}%", end="")
                elif status:
                    console.print(f"  {status}")
        console.print()
        return True
    except Exception as e:
        console.print(f"  [red]Failed to pull {model_tag}: {e}[/red]")
        return False


# ── Batching ──


def build_batches(candidates: list[dict], batch_size: int = 25) -> list[list[dict]]:
    """Group candidates by root, then pack into batches without splitting roots."""
    # Group by root preserving order
    from collections import OrderedDict
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in candidates:
        groups.setdefault(c["root"], []).append(c)

    batches = []
    current_batch: list[dict] = []

    for root, verbs in groups.items():
        if len(current_batch) + len(verbs) > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(verbs)

    if current_batch:
        batches.append(current_batch)

    return batches


# ── LLM Communication ──


def classify_batch(
    model_tag: str,
    verbs: list[dict],
    client: httpx.Client,
    logger: logging.Logger,
    use_schema: bool = False,
) -> list[ValidationResult]:
    """Send a batch of verbs to Ollama and parse the response."""
    # Collapsed hollow/geminate triples share one surface — send each
    # distinct string once; update_csv fans the verdict back out to every
    # matching row.
    verb_list = list(dict.fromkeys(v["verb"] for v in verbs))
    user_msg = f"Classify these Arabic verbs:\n{json.dumps(verb_list, ensure_ascii=False)}"

    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "format": build_response_schema(len(verb_list)) if use_schema else "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            # Enough for ~120 tokens per verb plus wrapper — a hard cap on
            # degenerate repetition loops (a fixed 8192 let runaway outputs
            # burn a minute per batch).
            "num_predict": 120 * len(verb_list) + 256,
        },
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data.get("message", {}).get("content", "")
            logger.info(f"Batch verbs: {verb_list}")
            logger.info(f"Raw response: {raw_content}")

            results = parse_response(raw_content, verb_list, logger)
            if results is not None:
                return results

            # JSON parsed but missing verbs — retry with reinforcement
            last_error = "incomplete response"
            if attempt < MAX_RETRIES - 1:
                payload["messages"].append({
                    "role": "user",
                    "content": (
                        "Your response was missing some verbs or had invalid JSON. "
                        "Please output ONLY a valid JSON object with key 'results' "
                        "containing an array with one entry per verb, in the same order. "
                        "No markdown fences, no explanation."
                    ),
                })
                logger.warning(f"Retry {attempt + 1}: reinforced prompt")

        except httpx.ConnectError:
            last_error = "connection error"
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(f"Connection error, retrying in {wait}s...")
                time.sleep(wait)
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}"
            logger.error(f"HTTP error: {e}")
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error: {e}")
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)

    # All retries exhausted — return defaults for this batch
    logger.error(f"All retries failed ({last_error}), returning defaults for {len(verb_list)} verbs")
    return [
        ValidationResult(verb=v, exists=None)
        for v in verb_list
    ]


def parse_response(
    raw_content: str,
    expected_verbs: list[str],
    logger: logging.Logger,
) -> list[ValidationResult] | None:
    """Parse LLM JSON response into ValidationResult objects.

    Returns list of results if successful, None if should retry.
    """
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON even from fenced block")
                return None
        else:
            logger.error(f"Failed to parse JSON response")
            return None

    # Handle both {"results": [...]} and bare [...]
    if isinstance(parsed, dict):
        items = parsed.get("results", parsed.get("verbs", []))
    elif isinstance(parsed, list):
        items = parsed
    else:
        logger.error(f"Unexpected response type: {type(parsed)}")
        return None

    if not isinstance(items, list):
        logger.error(f"Results is not a list: {type(items)}")
        return None

    # Build lookup by verb
    result_map: dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and "verb" in item:
            result_map[item["verb"]] = item

    # Positional alignment fallback: some models echo verbs with dropped or
    # altered diacritics. When counts line up but strings drift, align by
    # index — sanity-checked on the consonant skeleton where possible.
    if len(items) == len(expected_verbs):
        strip = re.compile(r'[ً-ْٰ]')
        for idx, verb in enumerate(expected_verbs):
            if verb in result_map:
                continue
            item = items[idx]
            if not (isinstance(item, dict) and "verb" in item):
                continue
            got = item["verb"]
            if got in expected_verbs:
                continue  # it's a (possibly misordered) different verb, not a drift
            if strip.sub('', got) == strip.sub('', verb) or got == "":
                logger.warning(f"Positional alignment: {got!r} → {verb!r}")
                result_map[verb] = item

    # Match to expected verbs
    results = []
    missing = 0
    for verb in expected_verbs:
        if verb in result_map:
            item = result_map[verb]
            try:
                results.append(ValidationResult(
                    verb=verb,
                    exists=_to_bool(item.get("exists")),
                    dialect_only=_to_bool(item.get("dialect_only")),
                    list_of_dialects=item.get("list_of_dialects", []) or [],
                    dialect_meanings=str(item.get("dialect_meanings", "")),
                    msa=_to_bool(item.get("msa")),
                    msa_meanings=str(item.get("msa_meanings", "")),
                ))
            except Exception as e:
                logger.warning(f"Error parsing result for {verb}: {e}, using defaults")
                results.append(ValidationResult(verb=verb, exists=None))
        else:
            missing += 1
            results.append(ValidationResult(verb=verb, exists=None))

    if missing > len(expected_verbs) * 0.5:
        # More than half missing — worth retrying
        logger.warning(f"{missing}/{len(expected_verbs)} verbs missing from response")
        return None

    if missing > 0:
        logger.warning(f"{missing}/{len(expected_verbs)} verbs missing, using defaults for those")

    return results


def _to_bool(val) -> bool | None:
    """Coerce various LLM outputs to bool or None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.lower().strip()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


# ── CSV I/O ──


def load_csv_progress(csv_path: Path) -> set[tuple[str, str]]:
    """(ROOT, PATTERN) keys already validated (non-empty EXISTS? column).

    Keyed like the merge/import machinery — NOT by VERB: distinct roots can
    share a surface (قول/قيل → قَالَ), and keying done-ness by surface would
    permanently skip unfilled rows that happen to collide with a filled one.
    """
    done = set()
    if not csv_path.exists():
        return done
    assert_csv_schema(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("EXISTS?", "").strip():
                done.add((row["ROOT"], row["PATTERN"]))
    return done


def load_csv_rows(csv_path: Path) -> list[dict]:
    """Load all CSV rows as dicts."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_csv(csv_path: Path, results: list[ValidationResult]) -> int:
    """Update CSV rows matching the given validation results.

    Matches by VERB string: an LLM judgment is about the surface form, so
    every row sharing that surface (collapsed hollow/geminate triples, and
    cross-root collisions like قول/قيل → قَالَ) receives the same verdict.
    Returns the number of rows written.
    """
    assert_csv_schema(csv_path)
    rows = load_csv_rows(csv_path)
    result_map = {r.verb: r for r in results}

    updated = 0
    for row in rows:
        verb = row["VERB"]
        if verb in result_map:
            r = result_map[verb]
            if r.exists is not None:
                updated += 1
                row["EXISTS?"] = str(r.exists)
                row["DIALECT_ONLY?"] = str(r.dialect_only) if r.dialect_only is not None else ""
                row["LIST_OF_DIALECTS"] = "|".join(r.list_of_dialects) if r.list_of_dialects else ""
                row["DIALECT_MEANINGS"] = r.dialect_meanings
                row["MSA?"] = str(r.msa) if r.msa is not None else ""
                row["MSA_MEANINGS"] = r.msa_meanings

    # Atomic write
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.rename(csv_path)
    return updated


def ensure_csv_exists(csv_path: Path, candidates: list[dict]):
    """Create a CSV template if it doesn't exist."""
    if csv_path.exists():
        return
    console.print(f"  Creating {csv_path}...")
    write_scaffold(csv_path, candidates)


class RowStore:
    """In-memory model CSV with fan-out updates and atomic flush.

    The old path re-read and rewrote the whole 2.7MB CSV on every batch
    (~2,600 full rewrites per run — and a read-modify-write race once
    batches run in parallel). The store loads rows once, mutates in memory
    under a lock, and flushes atomically after each completed batch.
    """

    def __init__(self, csv_path: Path):
        assert_csv_schema(csv_path)
        self.csv_path = csv_path
        self._lock = threading.Lock()
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self._by_verb: dict[str, list[dict]] = {}
        for row in self.rows:
            self._by_verb.setdefault(row["VERB"], []).append(row)

    def filled_keys(self) -> set[tuple[str, str]]:
        with self._lock:
            return {
                (row["ROOT"], row["PATTERN"])
                for row in self.rows
                if (row.get("EXISTS?") or "").strip()
            }

    def update(self, results: list[ValidationResult]) -> int:
        """Apply verdicts (fan-out by VERB surface). Returns rows written.

        A result verb with no matching row is a bug upstream — raise, don't
        silently drop it.
        """
        updated = 0
        with self._lock:
            for r in results:
                if r.exists is None:
                    continue
                rows = self._by_verb.get(r.verb)
                if rows is None:
                    raise KeyError(f"result verb has no CSV row: {r.verb!r}")
                for row in rows:
                    updated += 1
                    row["EXISTS?"] = str(r.exists)
                    row["DIALECT_ONLY?"] = str(r.dialect_only) if r.dialect_only is not None else ""
                    row["LIST_OF_DIALECTS"] = "|".join(r.list_of_dialects) if r.list_of_dialects else ""
                    row["DIALECT_MEANINGS"] = r.dialect_meanings
                    row["MSA?"] = str(r.msa) if r.msa is not None else ""
                    row["MSA_MEANINGS"] = r.msa_meanings
        return updated

    def flush(self):
        with self._lock:
            tmp_path = self.csv_path.with_suffix(".csv.tmp")
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()
                writer.writerows(self.rows)
            tmp_path.rename(self.csv_path)


@dataclass
class RunStats:
    batches: int = 0
    verbs_requested: int = 0
    verbs_resolved_first_pass: int = 0
    verbs_missing_first_pass: int = 0
    verbs_resolved_retry: int = 0
    verbs_unresolved_final: int = 0
    rows_written: int = 0
    started: float = field(default_factory=time.monotonic)

    def report(self, logger: logging.Logger):
        from src.stats import fmt_rate
        elapsed = time.monotonic() - self.started
        rate = self.verbs_requested / elapsed if elapsed > 0 else 0.0
        lines = [
            f"Batches: {self.batches}",
            f"Verbs requested (distinct surfaces): {self.verbs_requested:,}",
            f"Missing after first pass: "
            f"{fmt_rate(self.verbs_missing_first_pass, max(self.verbs_requested, 1))}",
            f"Recovered by retry passes: {self.verbs_resolved_retry:,}",
            f"Unresolved (left blank, will resume): {self.verbs_unresolved_final:,}",
            f"CSV rows written: {self.rows_written:,}",
            f"Throughput: {rate:.2f} verbs/s ({elapsed:.0f}s elapsed)",
        ]
        for line in lines:
            console.print(f"  {line}")
            logger.info(line)


# ── Main Validation Pipeline ──


def setup_logger(model_name: str, data_dir: Path) -> logging.Logger:
    """Create a file logger for this validation run."""
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"validate_{model_name}_{ts}.log"

    logger = logging.getLogger(f"validate_{model_name}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    console.print(f"  Log: {log_path}")
    return logger


def validate_model(
    model_name: str,
    model_tag: str,
    candidates: list[dict],
    data_dir: Path,
    batch_size: int = 25,
    backup_every: int = 50,
    order: str = "plausibility",
    parallel: int = 1,
):
    """Run full validation pipeline for one model.

    parallel > 1 issues that many batches concurrently — the Ollama server
    must run with OLLAMA_NUM_PARALLEL >= parallel to actually serve them
    concurrently (requests queue harmlessly otherwise). Apple Silicon is
    compute-bound: expect ~1.3–1.8×, measure with `bench` before trusting.
    """
    csv_path = data_dir / f"{model_name}.csv"
    ensure_csv_exists(csv_path, candidates)

    # A multi-day run must survive crashes and overwrite bugs: snapshot any
    # existing progress before touching the file, then periodically below.
    if count_filled_rows(csv_path) > 0:
        console.print(f"  Backup: {backup_csv(csv_path, tag='run', keep=5)}")

    # Check what's already done
    done = load_csv_progress(csv_path)
    remaining = [c for c in candidates if (c["root"], c["pattern"]) not in done]

    if order == "plausibility":
        # Front-load the candidates most likely to be real so a partial or
        # interrupted run still yields a maximally useful attested sample.
        # Root-level score + root tie-break keeps whole-root grouping intact
        # for build_batches; key-set resume is order-independent.
        from src.plausibility import sort_candidates_for_validation
        remaining = sort_candidates_for_validation(remaining)

    # Run-level surface dedupe: an LLM judgment is per surface, and
    # update_csv fans it out to every matching row — sending قَالَ once
    # covers all six قول/قيل rows. Without this, cross-root shared surfaces
    # land in different batches and get re-queried (last writer wins).
    seen_surfaces: set[str] = set()
    deduped = []
    for c in remaining:
        if c["verb"] not in seen_surfaces:
            seen_surfaces.add(c["verb"])
            deduped.append(c)
    remaining = deduped

    if not remaining:
        console.print(f"  [green]All {len(candidates):,} verbs already validated![/green]")
        return

    console.print(f"  {len(done):,} already done, {len(remaining):,} remaining")

    logger = setup_logger(model_name, data_dir)
    logger.info(
        f"Starting validation: model={model_tag}, remaining={len(remaining)}, "
        f"batch_size={batch_size}, parallel={parallel}"
    )

    client = httpx.Client(
        base_url=OLLAMA_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )
    store = RowStore(csv_path)
    stats = RunStats()
    by_verb = {c["verb"]: c for c in remaining}
    use_schema = probe_schema_support(model_tag, client, logger)
    logger.info(f"Structured outputs: {'schema' if use_schema else 'format=json (fallback)'}")

    def run_pass(pass_candidates: list[dict], pass_batch_size: int,
                 progress, task, first_pass: bool) -> list[str]:
        """Run one pass over candidates; returns verbs still unresolved."""
        batches = build_batches(pass_candidates, pass_batch_size)
        unresolved: list[str] = []
        completed = 0

        def work(batch):
            return classify_batch(model_tag, batch, client, logger, use_schema=use_schema)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = {pool.submit(work, b): b for b in batches}
            try:
                for fut in as_completed(futures):
                    results = fut.result()
                    stats.batches += 1
                    resolved = [r for r in results if r.exists is not None]
                    missing = [r.verb for r in results if r.exists is None]
                    unresolved.extend(missing)
                    if first_pass:
                        stats.verbs_requested += len(results)
                        stats.verbs_resolved_first_pass += len(resolved)
                        stats.verbs_missing_first_pass += len(missing)
                    else:
                        stats.verbs_resolved_retry += len(resolved)
                    if resolved:
                        stats.rows_written += store.update(resolved)
                        store.flush()
                    completed += 1
                    if backup_every > 0 and completed % backup_every == 0:
                        backup_csv(csv_path, tag="run", keep=5)
                        logger.info(f"Periodic backup after batch {completed}")
                    progress.advance(task)
            except KeyboardInterrupt:
                # Cancel queued batches; in-flight ones finish and are lost —
                # their rows stay blank and the next run resumes them.
                for f in futures:
                    f.cancel()
                raise
        return unresolved

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} batches"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            n_batches = len(build_batches(remaining, batch_size))
            task = progress.add_task(f"[{model_name}]", total=n_batches)
            unresolved = run_pass(remaining, batch_size, progress, task, first_pass=True)

            # Retry queue: verbs the model dropped are re-batched smaller —
            # never written as defaults. Whatever survives both passes stays
            # blank and resumes next run.
            for retry_round in (1, 2):
                if not unresolved:
                    break
                retry_candidates = [by_verb[v] for v in dict.fromkeys(unresolved) if v in by_verb]
                retry_size = max(4, batch_size // 4)
                logger.info(f"Retry pass {retry_round}: {len(retry_candidates)} verbs "
                            f"at batch size {retry_size}")
                retry_task = progress.add_task(
                    f"[{model_name} retry {retry_round}]",
                    total=len(build_batches(retry_candidates, retry_size)),
                )
                unresolved = run_pass(retry_candidates, retry_size, progress,
                                      retry_task, first_pass=False)

            stats.verbs_unresolved_final = len(set(unresolved))

    except KeyboardInterrupt:
        console.print(f"\n  [yellow]Interrupted! {stats.rows_written:,} rows saved.[/yellow]")
        logger.info(f"Interrupted after {stats.rows_written} rows")
    finally:
        store.flush()
        client.close()

    console.print(f"  [green]Run summary → {csv_path}[/green]")
    stats.report(logger)


def validate_all(
    models: dict[str, str],
    candidates: list[dict],
    data_dir: Path,
    batch_size: int = 25,
    dry_run: bool = False,
    backup_every: int = 50,
    order: str = "plausibility",
    parallel: int = 1,
):
    """Run validation for all specified models sequentially."""
    for model_name, model_tag in models.items():
        console.print(f"\n[bold]{'═' * 50}[/bold]")
        console.print(f"[bold bright_blue]Model: {model_name}[/bold bright_blue] ({model_tag})")
        console.print(f"[bold]{'═' * 50}[/bold]")

        csv_path = data_dir / f"{model_name}.csv"
        ensure_csv_exists(csv_path, candidates)
        done = load_csv_progress(csv_path)
        remaining_list = [c for c in candidates if (c["root"], c["pattern"]) not in done]
        # Mirror validate_model's actual batching (sort + surface dedupe) so
        # the dry-run batch count and ETA match what a real run executes.
        if order == "plausibility":
            from src.plausibility import sort_candidates_for_validation
            remaining_list = sort_candidates_for_validation(remaining_list)
        seen: set[str] = set()
        deduped_list = []
        for c in remaining_list:
            if c["verb"] not in seen:
                seen.add(c["verb"])
                deduped_list.append(c)
        batches_needed = len(build_batches(deduped_list, batch_size))

        console.print(
            f"  Total: {len(candidates):,} | "
            f"Done: {len(candidates) - len(remaining_list):,} | "
            f"Remaining: {len(remaining_list):,}"
        )
        console.print(f"  Batches: {batches_needed:,} (batch size {batch_size})")
        est_minutes = batches_needed * 10 / 60  # ~10s per batch estimate
        console.print(f"  Estimated time: ~{est_minutes:.0f} minutes")

        if dry_run:
            continue

        validate_model(model_name, model_tag, candidates, data_dir, batch_size,
                       backup_every, order, parallel)


def load_candidates(candidates_path: Path) -> list[dict]:
    """Load candidates from JSONL."""
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates
