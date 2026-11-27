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
    result = asyncio.run(run_pipeline(s, docs))
    sm = summarize(result)
    print_report(sm, console)
    write_outputs(result, sm, s.output_dir)
    console.print(f"Wrote {s.output_dir / 'output.jsonl'} and the report files.")


if __name__ == "__main__":
    app()
