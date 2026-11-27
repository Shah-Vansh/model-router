"""Console report plus output files: output.jsonl (required format), details, summary, report.md."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .pipeline import RunResult


def summarize(run: RunResult) -> dict:
    res, s = run.results, run.settings
    n = len(res)
    expected = sum(r.expected_accuracy for r in res) / n
    cheap_only = sum(r.cheap_expected_accuracy for r in res) / n
    return {
        "documents": n,
        "models": {"cheap": s["cheap"], "expensive": s["expensive"]},
        "routes": dict(Counter(r.route for r in res)),
        "doc_types": dict(Counter(r.values.get("doc_type") for r in res)),
        "cost": {
            "units_spent": run.total_cost,
            "budget": s["budget"],
            "within_budget": run.total_cost <= s["budget"] + 1e-9,
            "cheap_calls": sum(r.cheap_calls for r in res),
            "expensive_calls": sum(r.expensive_calls for r in res),
            "cheap_only_cost": n * s["cost_cheap"],
            "expensive_only_cost": n * s["cost_expensive"],
        },
        "accuracy": {
            "target": s["target"],
            "expected": round(expected, 4),
            "expected_cheap_only": round(cheap_only, 4),
            "expected_expensive_only": run.calibration.expensive_accuracy,
        },
        "time": {"first_to_last_call_s": round(run.elapsed_s, 1),
                 "wall_s": round(run.wall_s, 1), "limit_s": s["time_limit_s"],
                 "within_limit": run.elapsed_s < s["time_limit_s"]},
        "requests": run.calls,
        "cache_hits": run.cache_hits,
        "failures": run.failures,
        "request_errors": run.errors,
        "policy": s["policy"],
        "plan_stop_reason": run.plan.stop_reason,
        "calibration": {
            "sample_size": run.calibration.sample_size,
            "reference": run.calibration.reference,
            "cell_accuracy": run.calibration.cell_accuracy,
            "cell_counts": run.calibration.cell_counts,
            "expensive_accuracy_assumed": run.calibration.expensive_accuracy,
        },
    }


def _ok(flag: bool) -> str:
    return "[green]PASS[/]" if flag else "[red]FAIL[/]"


def print_report(sm: dict, console: Console | None = None) -> None:
    c = console or Console()
    acc, cost, tm = sm["accuracy"], sm["cost"], sm["time"]
    t = Table(title=f"Model router: {sm['documents']} documents")
    for col, j in (("Metric", "left"), ("Value", "right"), ("Target", "right"), ("", "center")):
        t.add_column(col, justify=j)
    t.add_row("Expected accuracy", f"{acc['expected']:.1%}", f">= {acc['target']:.0%}",
              _ok(acc["expected"] >= acc["target"]))
    t.add_row("Units spent", f"{cost['units_spent']:g}", f"<= {cost['budget']:g}",
              _ok(cost["within_budget"]))
    t.add_row("Time (first to last call)", f"{tm['first_to_last_call_s']:.1f} s",
              f"< {tm['limit_s']:g} s", _ok(tm["within_limit"]))
    c.print(t)

    b = Table(title="Compared with using one model")
    for col in ("Strategy", "Units", "Expected accuracy"):
        b.add_column(col, justify="left" if col == "Strategy" else "right")
    b.add_row("Cheap only", f"{cost['cheap_only_cost']:g}", f"{acc['expected_cheap_only']:.1%}")
    b.add_row("Expensive only", f"{cost['expensive_only_cost']:g}",
              f"~{acc['expected_expensive_only']:.0%} (assumed)")
    b.add_row("[bold]Router[/]", f"[bold]{cost['units_spent']:g}[/]", f"{acc['expected']:.1%}")
    c.print(b)
    c.print(f"Models: cheap {sm['models']['cheap']}, expensive {sm['models']['expensive']}")
    c.print(f"Routes: {sm['routes']} | doc types: {sm['doc_types']}")
    c.print(f"Calls: {cost['cheap_calls']} cheap doc-sends, {cost['expensive_calls']} expensive "
            f"doc-sends; requests {sm['requests']}; cache hits {sm['cache_hits']}")
    c.print(f"Calibration ({sm['calibration']['sample_size']} docs, agreement with expensive "
            f"model): {sm['calibration']['cell_accuracy']}")
    c.print(f"Planner ({sm['policy']}) stopped: {sm['plan_stop_reason']}")
    if sm["failures"] or sm["request_errors"]:
        c.print(f"[yellow]Failures: {sm['failures']} | request errors: {sm['request_errors']}[/]")


def write_outputs(run: RunResult, sm: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "output.jsonl").open("w", encoding="utf-8") as f:
        for r in run.results:
            row = {"id": r.doc_id, **r.values, "cheap_calls": r.cheap_calls,
                   "expensive_calls": r.expensive_calls}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for r in run.results:
            f.write(r.model_dump_json() + "\n")
    (out_dir / "summary.json").write_text(json.dumps(sm, indent=2), encoding="utf-8")
    run.calibration.save(out_dir / "calibration.json")

    acc, cost, tm = sm["accuracy"], sm["cost"], sm["time"]
    lines = [
        "# Run report", "",
        f"- Documents: {sm['documents']}",
        f"- Models: cheap `{sm['models']['cheap']}`, expensive `{sm['models']['expensive']}`",
        f"- Routes: {sm['routes']}",
        f"- Expected accuracy: {acc['expected']:.1%} (target {acc['target']:.0%}; "
        f"cheap only {acc['expected_cheap_only']:.1%})",
        f"- Units spent: {cost['units_spent']:g} of {cost['budget']:g} "
        f"({cost['cheap_calls']} cheap + {cost['expensive_calls']} expensive document sends)",
        f"- Time, first call to last call: {tm['first_to_last_call_s']:.1f} s "
        f"(limit {tm['limit_s']:g} s)",
        f"- Planner stop reason: {sm['plan_stop_reason']}",
        "", "Per-document routing reasons are in `decisions.jsonl`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
