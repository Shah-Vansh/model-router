"""Turn messy model output into comparable canonical values.

Each normaliser returns the canonical value, None for "empty", or raises
ValueError when the value is present but malformed.
"""

from __future__ import annotations

import re
from typing import Any

from dateutil import parser as dateparser

from .schemas import FieldSpec, Kind

EMPTY = {"", "null", "none", "n/a", "na", "-", "unknown"}


def is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in EMPTY)


def norm_text(v: Any) -> str:
    """Lower-case, drop punctuation, collapse spaces. Used for comparison and grounding."""
    s = str(v).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_date(v: Any) -> str:
    s = str(v).strip()
    dayfirst = bool(re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{4}", s))
    try:
        return dateparser.parse(s, dayfirst=dayfirst, fuzzy=False).date().isoformat()
    except (ValueError, OverflowError) as e:
        raise ValueError(f"bad date {s!r}") from e


def norm_amount(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError("bool is not an amount")
    if isinstance(v, int | float):
        return round(float(v), 2)
    s = re.sub(r"[^\d.,\-]", "", str(v)).replace(",", "")
    if not re.search(r"\d", s):
        raise ValueError(f"bad amount {v!r}")
    return round(float(s), 2)


def normalize(spec: FieldSpec, v: Any) -> Any:
    if is_empty(v):
        return None
    match spec.kind:
        case Kind.DATE:
            return norm_date(v)
        case Kind.AMOUNT:
            return norm_amount(v)
        case Kind.ENUM:
            s = str(v).strip().lower()
            if s not in spec.choices:
                raise ValueError(f"{v!r} not in {spec.choices}")
            return s
        case Kind.ID:
            return re.sub(r"\s+", "", str(v)).upper()
        case _:
            return re.sub(r"\s+", " ", str(v)).strip()


def values_equal(spec: FieldSpec, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if spec.kind == Kind.AMOUNT:
        return abs(float(a) - float(b)) < 0.01
    if spec.kind in (Kind.TEXT, Kind.ID):
        return norm_text(a) == norm_text(b)
    return a == b
