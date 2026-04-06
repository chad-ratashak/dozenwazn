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
def csvs():
    """Generate per-model CSV templates from candidates."""
    from src.generate_csvs import generate_csvs
    candidates = DATA_DIR / 'candidates.jsonl'
    if not candidates.exists():
        click.echo("Run 'generate' first.", err=True)
        sys.exit(1)
    generate_csvs(candidates, DATA_DIR)


@cli.command()
def tokenize():
    """Run tiktoken + JAIS analysis on candidates (with and without tashkeel)."""
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
