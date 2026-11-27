"""What we extract from each document, and the core data records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocType(StrEnum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    EMAIL = "email"


class Kind(StrEnum):
    TEXT = "text"
    ID = "id"
    DATE = "date"
    AMOUNT = "amount"
    ENUM = "enum"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Kind
    description: str
    choices: tuple[str, ...] = ()


OUR_COMPANY = "Meridian Works Ltd"

FIELDS: list[FieldSpec] = [
    FieldSpec("doc_type", Kind.ENUM, "invoice, contract or email",
              tuple(t.value for t in DocType)),
    FieldSpec(
        "date", Kind.DATE,
        "YYYY-MM-DD. Invoice: the invoice date. Contract: the date it was entered into/signed. "
        "Email: the date the latest (top) message was sent.",
    ),
    FieldSpec(
        "total_amount", Kind.AMOUNT,
        "Number only. Invoice: total due. Contract: total contract value. "
        "Email: the final agreed amount in the latest message.",
    ),
    FieldSpec(
        "counterparty", Kind.TEXT,
        f"The other organisation (never {OUR_COMPANY}). Invoice: the issuer. "
        "Contract: the other party. Email: the sender's company.",
    ),
    FieldSpec(
        "reference", Kind.ID,
        "Invoice number (INV-...), agreement number (CTR-...), or for emails the PO number "
        "(PO-...) the latest message is about.",
    ),
]
FIELD_BY_NAME = {f.name: f for f in FIELDS}


class Document(BaseModel):
    doc_id: str
    text: str


class Extraction(BaseModel):
    """Parsed + normalised output for one document from one model."""

    valid: bool
    values: dict[str, Any] = Field(default_factory=dict)
    format_errors: list[str] = Field(default_factory=list)


class DocResult(BaseModel):
    doc_id: str
    route: str  # "cheap" | "expensive" | "calibration"
    reason: str
    values: dict[str, Any]
    expected_accuracy: float
    cheap_expected_accuracy: float
    flags: dict[str, list[str]] = Field(default_factory=dict)
    complex_doc: bool = False
    cheap_calls: int = 0
    expensive_calls: int = 0
    cost: float = 0.0
