#!/usr/bin/env python3
"""CLI interface for dozenwazn."""

import sys
from pathlib import Path

import click

DATA_DIR = Path('data')


@click.group()
def cli():
    """أوزان — Dozenwazn: Arabic verb form generator and validator."""
    pass


@cli.command()
@click.option('--include-xyzn', is_flag=True, help='Include XYZN quadrilateral roots (491K+ extra)')
def generate(include_xyzn):
    """Generate all verb candidates to data/candidates.jsonl."""
    from src.generate import write_candidates_jsonl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / 'candidates.jsonl'
    count = write_candidates_jsonl(output, include_xyzn=include_xyzn)
    click.echo(f"Generated {count:,} candidates → {output}", err=True)


@cli.command()
@click.option('--force', is_flag=True,
              help='Overwrite CSVs even if they contain filled validation rows (asks first)')
@click.option('--merge', is_flag=True,
              help='Rebuild CSVs carrying filled rows over by (ROOT, PATTERN)')
def csvs(force, merge):
    """Generate per-model CSV templates from candidates.

    Safe by default: CSVs with filled EXISTS? rows are skipped unless
    --merge (non-destructive migration) or --force (destructive, confirmed)
    is given.
    """
    from src.generate_csvs import MODELS, count_filled_rows, generate_csvs
    candidates = DATA_DIR / 'candidates.jsonl'
    if not candidates.exists():
        click.echo("Run 'generate' first.", err=True)
        sys.exit(1)
    if force:
        at_risk = {m: count_filled_rows(DATA_DIR / f'{m}.csv') for m in MODELS}
        at_risk = {m: n for m, n in at_risk.items() if n > 0}
        if at_risk:
            click.echo(f"--force will DESTROY filled rows: {at_risk}", err=True)
            click.confirm("Proceed?", abort=True)
    generate_csvs(candidates, DATA_DIR, force=force, merge=merge)


@cli.command('import-xlsx')
@click.argument('xlsx_path', type=click.Path(exists=True, path_type=Path),
                default=Path('data/haiku_filled.xlsx'))
@click.option('--model', default='haiku', help='Target model CSV (default: haiku)')
@click.option('--commit', is_flag=True,
              help='Actually write to the CSV (default is a dry-run report)')
def import_xlsx(xlsx_path, model, commit):
    """Import filled validation rows from an xlsx workbook into a model CSV.

    Dry-run by default: inspect the report, then re-run with --commit.
    """
    from src.import_xlsx import import_into_csv, read_filled_rows, report

    filled = read_filled_rows(xlsx_path)
    report(filled)
    if not commit:
        click.echo("\nDry-run only. Re-run with --commit to write.", err=True)
        return
    csv_path = DATA_DIR / f'{model}.csv'
    if not csv_path.exists():
        click.echo(f"{csv_path} does not exist — run 'csvs' first.", err=True)
        sys.exit(1)
    applied, unmatched = import_into_csv(filled, csv_path)
    click.echo(f"Imported {applied} rows into {csv_path} ({unmatched} unmatched).", err=True)


@cli.command()
@click.option('--refresh', is_flag=True,
              help='Re-download even if already vendored and verified')
@click.option('--only', default=None,
              help='Comma-separated tokenizer names (default: all in registry)')
def vendor(refresh, only):
    """Fetch tokenizer data files (vocab/merges only, no code) into vendor/.

    Downloads are pinned to a commit SHA and hashed into
    vendor/tokenizers/SHA256SUMS.json — commit that manifest. This is the
    only command that touches the network for tokenizer assets.
    """
    from src.vendor_tokenizers import vendor as vendor_tokenizers
    names = [n.strip() for n in only.split(',')] if only else None
    try:
        vendor_tokenizers(names, refresh=refresh)
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)


@cli.command()
def tokenize():
    """Run tokenizer analysis on candidates (with and without tashkeel).

    Uses tiktoken plus any vendored open-source tokenizers (JAIS, Qwen,
    Llama 3, Gemma, AraBERT, CAMeLBERT). Vendored files are hash-verified
    before loading; run 'vendor' first to fetch them.
    """
    from src.tokenizer_check import check_tokenization, summarize
    candidates = DATA_DIR / 'candidates.jsonl'
    output = DATA_DIR / 'tokenizer_analysis.jsonl'
    if not candidates.exists():
        click.echo("Run 'generate' first.", err=True)
        sys.exit(1)
    count = check_tokenization(candidates, output)
    click.echo(f"Analyzed {count:,} candidates → {output}", err=True)
    summarize(output)


@cli.command()
def display():
    """Display candidates and analysis in terminal + generate HTML."""
    from src.display import (
        console, load_candidates, load_tokenizer_data,
        display_stats, display_tier_summary, generate_html,
    )
    from rich.panel import Panel

    candidates_path = DATA_DIR / 'candidates.jsonl'
    tok_path = DATA_DIR / 'tokenizer_analysis.jsonl'

    if not candidates_path.exists():
        click.echo("Run 'generate' first.", err=True)
        sys.exit(1)

    candidates = load_candidates(candidates_path)
    console.print(Panel("[bold bright_blue]أوزان — Dozenwazn[/bold bright_blue]",
                        subtitle="Arabic Form I Verb Generator"))
    console.print()
    display_stats(candidates)

    if tok_path.exists():
        records = load_tokenizer_data(tok_path)
        console.print()
        display_tier_summary(records)

        html_path = DATA_DIR / 'dozenwazn.html'
        generate_html(records, html_path)
        console.print(f"\n[dim]Interactive HTML → {html_path}[/dim]")
    else:
        console.print("\n[dim]Run 'tokenize' to enable HTML export.[/dim]")


@cli.command()
@click.option('--models', '-m', default='llama,gemma',
              help='Comma-separated model names (default: llama,gemma)')
@click.option('--batch-size', '-b', default=25, type=int,
              help='Verbs per batch (default: 25)')
@click.option('--pull', 'do_pull', is_flag=True, help='Pull/update Ollama models before validating')
@click.option('--dry-run', is_flag=True, help='Show batch count and ETA without calling Ollama')
@click.option('--backup-every', default=50, type=int,
              help='Snapshot the CSV to data/backups/ every N batches (0 disables; default 50)')
def validate(models, batch_size, do_pull, dry_run, backup_every):
    """Validate verb candidates using local LLMs via Ollama."""
    from src.validate_ollama import (
        DEFAULT_OLLAMA_MODELS, check_ollama_health, pull_model,
        load_candidates, validate_all, console,
    )

    candidates_path = DATA_DIR / 'candidates.jsonl'
    if not candidates_path.exists():
        click.echo("Run 'generate' first.", err=True)
        sys.exit(1)

    # Parse model list
    model_names = [m.strip() for m in models.split(',')]
    model_map = {}
    for name in model_names:
        if ':' in name:
            # Allow explicit tag: llama:llama4-scout
            name, tag = name.split(':', 1)
            model_map[name] = tag
        elif name in DEFAULT_OLLAMA_MODELS:
            model_map[name] = DEFAULT_OLLAMA_MODELS[name]
        else:
            model_map[name] = name  # Use name as tag

    # Health check
    if not dry_run:
        if not check_ollama_health():
            click.echo("Ollama is not running. Install and start it:", err=True)
            click.echo("  curl -fsSL https://ollama.com/install.sh | sh", err=True)
            click.echo("  ollama serve", err=True)
            sys.exit(1)

    # Pull models if requested
    if do_pull:
        console.print("\n[bold]Pulling models...[/bold]")
        for name, tag in model_map.items():
            if not pull_model(tag):
                click.echo(f"Failed to pull {tag}. Skipping {name}.", err=True)

    candidates = load_candidates(candidates_path)
    validate_all(model_map, candidates, DATA_DIR, batch_size, dry_run, backup_every)


@cli.command()
def html():
    """Generate interactive HTML visualization only."""
    from src.display import load_tokenizer_data, generate_html

    tok_path = DATA_DIR / 'tokenizer_analysis.jsonl'
    if not tok_path.exists():
        click.echo("Run 'tokenize' first.", err=True)
        sys.exit(1)

    records = load_tokenizer_data(tok_path)
    html_path = DATA_DIR / 'dozenwazn.html'
    generate_html(records, html_path)
    click.echo(f"Interactive HTML → {html_path}", err=True)


if __name__ == '__main__':
    cli()
