"""Free consistency checks on an extraction (no model calls).

Each check marks fields as suspicious. Calibration later learns how often
suspicious and clean fields are actually right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .extract import detect_type
from .normalize import norm_amount, norm_date, norm_text
from .schemas import FIELDS, OUR_COMPANY, Extraction, Kind

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b",
    rf"\b{_MONTH} \d{{1,2}},? \d{{4}}\b",
    rf"\b\d{{1,2}}[ -]{_MONTH}[ -]\d{{4}}\b",
]
_AMOUNT_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.-])\d+\.\d{2}\b")
_REF_PREFIX = {"invoice": "INV-", "contract": "CTR-", "email": "PO-"}
_SUSPICIOUS = re.compile(
    r"automated processing|disregard|ignore (?:all|the|previous)|instruction|record the total",
    re.IGNORECASE,
)


@dataclass
class TextIndex:
    norm: str
    compact: str  # alphanumerics only, for reference grounding
    dates: set[str]
    amounts: set[float]
    scanned: bool
    suspicious: bool


@lru_cache(maxsize=4096)
def index_text(text: str) -> TextIndex:
    dates: set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                dates.add(norm_date(m.group()))
            except ValueError:
                pass
    amounts: set[float] = set()
    for m in _AMOUNT_RE.finditer(text):
        try:
            amounts.add(norm_amount(m.group()))
        except ValueError:
            pass
    return TextIndex(
        norm=norm_text(text),
        compact=re.sub(r"[^A-Z0-9]", "", text.upper()),
        dates=dates,
        amounts=amounts,
        scanned="scanned document" in text[:200].lower(),
        suspicious=bool(_SUSPICIOUS.search(text)),
    )


def complexity(text: str) -> int:
    """Distinct dates + amounts, plus penalties for OCR scans and email threads."""
    idx = index_text(text)
    score = len(idx.dates) + len(idx.amounts)
    score += 6 * idx.scanned
    score += 3 * len(re.findall(r"^On .+ wrote:", text, re.MULTILINE))
    return score


def run_checks(text: str, ex: Extraction | None) -> dict[str, list[str]]:
    """{field: [reasons]} for every field; an empty list means 'looks fine'."""
    flags: dict[str, list[str]] = {f.name: [] for f in FIELDS}
    if ex is None or not ex.valid:
        return {k: ["no_answer"] for k in flags}
    idx = index_text(text)
    v = ex.values

    for f in FIELDS:
        val = v.get(f.name)
        if f.name in ex.format_errors:
            flags[f.name].append("bad_format")
        elif val is None:
            flags[f.name].append("missing")
        elif f.kind == Kind.DATE and val not in idx.dates:
            flags[f.name].append("not_in_text")
        elif f.kind == Kind.AMOUNT and float(val) not in idx.amounts:
            flags[f.name].append("not_in_text")
        elif f.kind == Kind.TEXT and norm_text(val) not in idx.norm:
            flags[f.name].append("not_in_text")
        elif f.kind == Kind.ID and re.sub(r"[^A-Z0-9]", "", val) not in idx.compact:
            flags[f.name].append("not_in_text")

    if v.get("doc_type") and v["doc_type"] != detect_type(text).value:
        flags["doc_type"].append("type_disagrees_with_keywords")
    ref, dt = v.get("reference"), v.get("doc_type")
    if ref and dt in _REF_PREFIX and not ref.startswith(_REF_PREFIX[dt]):
        flags["reference"].append("unexpected_prefix")
    cp = v.get("counterparty")
    if cp and "meridian" in norm_text(cp) and "meridian" in norm_text(OUR_COMPANY):
        flags["counterparty"].append("is_our_company")
    if idx.suspicious:
        for k in flags:
            flags[k].append("embedded_instructions")
    return flags
