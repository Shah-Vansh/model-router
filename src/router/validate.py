"""Check a finished output.jsonl without calling any model.

1. Format: one line per input document, required keys, correct types/values.
2. Budget: units recomputed from cheap_calls / expensive_calls.
3. Cross-check: fields that a strict regex can read unambiguously from clean
   (non-OCR) documents are compared with the output. This gives a measured
   accuracy on that subset, independent of the models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import norm_amount, norm_date
from .schemas import DocType, Document

REQUIRED = ("id", "doc_type", "date", "total_amount", "counterparty", "reference",
            "cheap_calls", "expensive_calls")


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    units: float = 0.0
    rows: int = 0
    crosscheck: dict[str, list[int]] = field(default_factory=dict)  # field -> [match, total]
    mismatches: list[str] = field(default_factory=list)


def _expected_by_rule(text: str) -> dict:
    """Values readable with a strict pattern. Skips OCR scans and anything ambiguous."""
    if "scanned document" in text[:200].lower():
        return {}
    out: dict = {}
    if re.search(r"^\s*From:", text, re.MULTILINE):
        out["doc_type"] = DocType.EMAIL.value
        m = re.search(r"^Subject:.*?\b(PO-\d+)", text, re.MULTILINE)
        if m:
            out["reference"] = m.group(1)
        m = re.search(r"^Date:\s*\w{3}, (\d{1,2} \w{3} \d{4})", text, re.MULTILINE)
        if m:
            out["date"] = norm_date(m.group(1))
    elif re.search(r"^INVOICE\s*$", text, re.MULTILINE):
        out["doc_type"] = DocType.INVOICE.value
        if m := re.search(r"^Invoice No:\s*(INV-[\d-]+)", text, re.MULTILINE):
            out["reference"] = m.group(1)
        if m := re.search(r"^Invoice Date:\s*(.+)$", text, re.MULTILINE):
            try:
                out["date"] = norm_date(m.group(1))
            except ValueError:
                pass
        totals = re.findall(r"^Total Due:\s*[A-Z]{3}\s*([\d,]+\.\d{2})", text, re.MULTILINE)
        if len(totals) == 1 and "automated processing" not in text.lower():
            out["total_amount"] = norm_amount(totals[0])
    elif re.search(r"^SERVICE AGREEMENT", text, re.MULTILINE):
        out["doc_type"] = DocType.CONTRACT.value
        if m := re.search(r"Agreement No\.\s*(CTR-\d+)", text):
            out["reference"] = m.group(1)
    return out


def validate(output_path: Path, docs: list[Document], budget: float,
             cost_cheap: float, cost_expensive: float) -> Validation:
    v = Validation()
    if not output_path.exists():
        v.errors.append(f"{output_path} not found (run `router run` first)")
        return v
    rows = {}
    for n, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            v.errors.append(f"line {n}: not valid JSON")
            continue
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            v.errors.append(f"line {n}: missing {missing}")
            continue
        if row["id"] in rows:
            v.errors.append(f"line {n}: duplicate id {row['id']}")
        rows[row["id"]] = row
        if row["doc_type"] not in {t.value for t in DocType}:
            v.errors.append(f"{row['id']}: doc_type {row['doc_type']!r} not allowed")
        if row["date"] is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(row["date"])):
            v.errors.append(f"{row['id']}: date {row['date']!r} is not YYYY-MM-DD")
        if row["total_amount"] is not None and not isinstance(row["total_amount"], int | float):
            v.errors.append(f"{row['id']}: total_amount is not a number")
        for k in ("cheap_calls", "expensive_calls"):
            if not isinstance(row[k], int) or row[k] < 0:
                v.errors.append(f"{row['id']}: {k} must be a non-negative integer")
        v.units += row["cheap_calls"] * cost_cheap + row["expensive_calls"] * cost_expensive
    v.rows = len(rows)

    ids = {d.doc_id for d in docs}
    if absent := sorted(ids - rows.keys()):
        v.errors.append(f"{len(absent)} documents missing from output, e.g. {absent[:3]}")
    if extra := sorted(rows.keys() - ids):
        v.errors.append(f"{len(extra)} unknown ids in output, e.g. {extra[:3]}")
    if v.units > budget + 1e-9:
        v.errors.append(f"units {v.units:g} exceed budget {budget:g}")

    for d in docs:
        row = rows.get(d.doc_id)
        if row is None:
            continue
        for fname, want in _expected_by_rule(d.text).items():
            got = row.get(fname)
            ok = (abs(float(got) - want) < 0.01 if fname == "total_amount" and got is not None
                  else str(got).upper() == str(want).upper())
            cell = v.crosscheck.setdefault(fname, [0, 0])
            cell[0] += ok
            cell[1] += 1
            if not ok:
                v.mismatches.append(f"{d.doc_id} {fname}: output {got!r}, rule {want!r}")
    return v
