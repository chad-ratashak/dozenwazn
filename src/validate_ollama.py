#!/usr/bin/env python3
"""Validate Arabic verb candidates using local LLMs via Ollama."""

import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.generate_csvs import CSV_HEADERS
from src.models import ValidationResult

console = Console()

# ── Ollama Configuration ──

OLLAMA_BASE = "http://localhost:11434"

DEFAULT_OLLAMA_MODELS = {
    "llama": "llama3.2",
    "gemma": "gemma4:e2b",
}

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]  # seconds

SYSTEM_PROMPT = """\
You are an expert in Arabic morphology (صرف). Your task is to classify Arabic Form I (فعل ثلاثي مجرد) verbs.

For each verb provided, determine:
1. **exists**: Does this verb exist in Arabic (classical, MSA, or any dialect)? true/false
2. **single_token_word**: Is this a common, frequently-used word? true/false
3. **dialect_only**: Is this verb used ONLY in dialects and NOT in Modern Standard Arabic (MSA) or Classical Arabic? true/false
4. **list_of_dialects**: If dialect_only is true, list which dialects use it (e.g., ["Egyptian", "Levantine"]). Otherwise [].
5. **dialect_meanings**: If used in dialects, brief English meaning(s) in dialectal usage. Otherwise "".
6. **msa**: Is this verb used in Modern Standard Arabic? true/false
7. **msa_meanings**: If used in MSA, brief English meaning(s). Otherwise "".

Rules:
- Verbs are presented with full tashkeel (diacritics). Pay attention to the vowel pattern.
- A root may have some patterns that exist and others that don't. Evaluate each verb independently.
- If a verb does not exist at all, set exists=false and leave all other fields at defaults.
- Be conservative: if unsure whether a verb exists, set exists=false.

Respond with a JSON object containing a single key "results" whose value is an array of objects, \
one per verb in the SAME ORDER as the input. Each object must have these exact keys:
  verb, exists, single_token_word, dialect_only, list_of_dialects, dialect_meanings, msa, msa_meanings

Example for a batch of 2 verbs:
{"results": [
  {"verb": "كَتَبَ", "exists": true, "single_token_word": true, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": true, "msa_meanings": "to write"},
  {"verb": "ءَبُتَ", "exists": false, "single_token_word": false, "dialect_only": false, "list_of_dialects": [], "dialect_meanings": "", "msa": false, "msa_meanings": ""}
]}"""


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
) -> list[ValidationResult]:
    """Send a batch of verbs to Ollama and parse the response."""
    verb_list = [v["verb"] for v in verbs]
    user_msg = f"Classify these Arabic verbs:\n{json.dumps(verb_list, ensure_ascii=False)}"

    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 8192,
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
                    single_token_word=_to_bool(item.get("single_token_word")),
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


def load_csv_progress(csv_path: Path) -> set[str]:
    """Return set of verbs that have already been validated (non-empty EXISTS? column)."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("EXISTS?", "").strip():
                done.add(row["VERB"])
    return done


def load_csv_rows(csv_path: Path) -> list[dict]:
    """Load all CSV rows as dicts."""
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_csv(csv_path: Path, results: list[ValidationResult]):
    """Update CSV rows matching the given validation results."""
    rows = load_csv_rows(csv_path)
    result_map = {r.verb: r for r in results}

    for row in rows:
        verb = row["VERB"]
        if verb in result_map:
            r = result_map[verb]
            if r.exists is not None:
                row["EXISTS?"] = str(r.exists)
                row["SINGLE_TOKEN_WORD?"] = str(r.single_token_word) if r.single_token_word is not None else ""
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


def ensure_csv_exists(csv_path: Path, candidates: list[dict]):
    """Create a CSV template if it doesn't exist."""
    if csv_path.exists():
        return
    console.print(f"  Creating {csv_path}...")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for c in candidates:
            writer.writerow([
                c["root"], c["root_type"], c["arity"], c["pattern"], c["verb"],
                "", "", "", "", "", "", "",
            ])


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
):
    """Run full validation pipeline for one model."""
    csv_path = data_dir / f"{model_name}.csv"
    ensure_csv_exists(csv_path, candidates)

    # Check what's already done
    done = load_csv_progress(csv_path)
    remaining = [c for c in candidates if c["verb"] not in done]

    if not remaining:
        console.print(f"  [green]All {len(candidates):,} verbs already validated![/green]")
        return

    console.print(f"  {len(done):,} already done, {len(remaining):,} remaining")

    batches = build_batches(remaining, batch_size)
    logger = setup_logger(model_name, data_dir)
    logger.info(f"Starting validation: model={model_tag}, remaining={len(remaining)}, batches={len(batches)}, batch_size={batch_size}")

    client = httpx.Client(
        base_url=OLLAMA_BASE,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )

    total_validated = 0
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
            task = progress.add_task(f"[{model_name}]", total=len(batches))

            for batch in batches:
                results = classify_batch(model_tag, batch, client, logger)
                # Filter out results where we got no answer (exists=None) — don't save those
                valid_results = [r for r in results if r.exists is not None]
                if valid_results:
                    update_csv(csv_path, valid_results)
                    total_validated += len(valid_results)

                progress.advance(task)

    except KeyboardInterrupt:
        console.print(f"\n  [yellow]Interrupted! {total_validated:,} verbs saved.[/yellow]")
        logger.info(f"Interrupted after {total_validated} verbs")
    finally:
        client.close()

    console.print(f"  [green]Validated {total_validated:,} verbs → {csv_path}[/green]")
    logger.info(f"Completed: {total_validated} verbs validated")


def validate_all(
    models: dict[str, str],
    candidates: list[dict],
    data_dir: Path,
    batch_size: int = 25,
    dry_run: bool = False,
):
    """Run validation for all specified models sequentially."""
    for model_name, model_tag in models.items():
        console.print(f"\n[bold]{'═' * 50}[/bold]")
        console.print(f"[bold bright_blue]Model: {model_name}[/bold bright_blue] ({model_tag})")
        console.print(f"[bold]{'═' * 50}[/bold]")

        csv_path = data_dir / f"{model_name}.csv"
        ensure_csv_exists(csv_path, candidates)
        done = load_csv_progress(csv_path)
        remaining = len(candidates) - len(done)
        batches_needed = len(build_batches(
            [c for c in candidates if c["verb"] not in done],
            batch_size,
        ))

        console.print(f"  Total: {len(candidates):,} | Done: {len(done):,} | Remaining: {remaining:,}")
        console.print(f"  Batches: {batches_needed:,} (batch size {batch_size})")
        est_minutes = batches_needed * 10 / 60  # ~10s per batch estimate
        console.print(f"  Estimated time: ~{est_minutes:.0f} minutes")

        if dry_run:
            continue

        validate_model(model_name, model_tag, candidates, data_dir, batch_size)


def load_candidates(candidates_path: Path) -> list[dict]:
    """Load candidates from JSONL."""
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates
