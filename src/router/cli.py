"""Command line: `router run`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .config import Settings
from .loader import load_documents
from .pipeline import run_pipeline
from .report import print_report, summarize, write_outputs

app = typer.Typer(add_completion=False, help="Budget-aware cheap/expensive LLM router.")
console = Console()


@app.callback()
def main() -> None:
    """Budget-aware cheap/expensive LLM router."""


@app.command()
def run(
    documents: Path = typer.Option(None, help="JSONL file with {id, text} per line"),
    budget: float = typer.Option(None, help="Budget in units (cheap send = 1)"),
    policy: str = typer.Option(None, help="max_accuracy | target"),
    limit: int = typer.Option(0, help="Only process the first N documents (smoke test)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore cached model replies"),
) -> None:
    """Route every document, then print and save the report."""
    overrides = {"documents_path": documents, "budget": budget, "policy": policy}
    s = Settings(**{k: v for k, v in overrides.items() if v is not None})
    if no_cache:
        s.use_cache = False
    docs = load_documents(s.documents_path)
    if limit:
        docs = docs[:limit]
        s.budget = s.budget * limit / 300 if budget is None else s.budget
    console.print(f"{len(docs)} documents | cheap {s.cheap_provider}:{s.cheap_model} | "
                  f"expensive {s.expensive_provider}:{s.expensive_model} | budget {s.budget:g}")
    try:
        result = asyncio.run(run_pipeline(s, docs))
    except ValueError as e:  # missing key, budget too small, no documents
        console.print(f"[red]Cannot run:[/] {e}\n"
                      "Copy .env.example to .env and set ROUTER_GEMINI_API_KEY "
                      "(free key: https://aistudio.google.com). "
                      "`pytest` and `router validate` work without a key.")
        raise typer.Exit(code=1) from None
    sm = summarize(result)
    print_report(sm, console)
    write_outputs(result, sm, s.output_dir)
    console.print(f"Wrote {s.output_dir / 'output.jsonl'} and the report files.")


@app.command("validate")
def validate_cmd(
    output: Path = typer.Option(Path("output/output.jsonl"), help="File to check"),
    documents: Path = typer.Option(None, help="JSONL file with the input documents"),
) -> None:
    """Check output.jsonl (format, all ids, budget) and cross-check it. No API key needed."""
    from .validate import validate

    s = Settings(**({"documents_path": documents} if documents else {}))
    docs = load_documents(s.documents_path)
    v = validate(output, docs, s.budget, s.cost_cheap, s.cost_expensive)
    console.print(f"Rows: {v.rows} / {len(docs)} documents")
    console.print(f"Units from cheap_calls/expensive_calls: {v.units:g} / budget {s.budget:g}")
    for e in v.errors[:20]:
        console.print(f"[red]ERROR[/] {e}")
    console.print("Format and budget: " + ("[green]PASS[/]" if not v.errors else "[red]FAIL[/]"))

    if v.crosscheck:
        console.print("\nRule-based cross-check on clean (non-OCR) documents:")
        tm = tt = 0
        for fname, (m, t) in sorted(v.crosscheck.items()):
            console.print(f"  {fname:<13} {m}/{t} match ({m / t:.1%})")
            tm, tt = tm + m, tt + t
        console.print(f"  [bold]overall       {tm}/{tt} match ({tm / tt:.1%})[/]")
        for line in v.mismatches[:10]:
            console.print(f"  [yellow]mismatch[/] {line}")
    raise typer.Exit(code=1 if v.errors else 0)


if __name__ == "__main__":
    app()
