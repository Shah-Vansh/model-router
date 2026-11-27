"""Load documents from a JSONL file: one {"id": ..., "text": ...} per line."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Document


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {path}")
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                docs.append(Document(doc_id=str(row["id"]), text=row["text"]))
    return docs
