"""Offline tests: fake models stand in for the real LLMs (no key, no network)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from router import pipeline
from router.checks import run_checks
from router.config import Settings
from router.extract import parse_batch
from router.loader import load_documents
from router.models.base import ModelClient
from router.policy import Candidate, choose_escalations

DOCS = Path(__file__).resolve().parents[1] / "provided" / "candidate" / "documents.jsonl"


class FakeClient(ModelClient):
    """Answers with a fixed value per document; the cheap tier is wrong on some documents."""

    provider = "fake"

    def __init__(self, tier: str, wrong_every: int):
        super().__init__("fake", tier, concurrency=8, rpm=0, timeout_s=5, max_retries=1,
                         cache=None)
        self.wrong_every = wrong_every

    async def _generate(self, system: str, prompt: str) -> str:
        out = []
        for i, doc_id in enumerate(re.findall(r"### id: (\S+)", prompt)):
            n = int(re.sub(r"\D", "", doc_id) or 0)
            wrong = self.wrong_every and n % self.wrong_every == 0
            out.append({"id": doc_id, "doc_type": "invoice", "date": "2026-01-01",
                        "total_amount": 999999.0 if wrong else 100.0,
                        "counterparty": "Acme", "reference": f"INV-{i}"})
        return json.dumps({"results": out})


def test_parse_batch_handles_missing_and_bad_values():
    raw = json.dumps({"results": [
        {"id": "D-1", "doc_type": "Invoice", "date": "14 March 2026",
         "total_amount": "EUR 1,200.50", "counterparty": "X Ltd", "reference": "INV-1"},
        {"id": "D-2", "doc_type": "memo", "date": "not a date"},
    ]})
    out = parse_batch(raw, ["D-1", "D-2", "D-3"])
    assert out["D-1"].values["date"] == "2026-03-14"
    assert out["D-1"].values["total_amount"] == 1200.5
    assert out["D-1"].values["doc_type"] == "invoice"
    assert set(out["D-2"].format_errors) == {"doc_type", "date"}
    assert "D-3" not in out


def test_checks_flag_values_not_in_text_and_our_company():
    text = "INVOICE\nInvoice No: INV-2026-1\nInvoice Date: 2026-03-14\nTotal Due: EUR 1,200.50"
    ex = parse_batch(json.dumps({"results": [{"id": "a", "doc_type": "invoice",
        "date": "2026-03-15", "total_amount": 1200.5, "counterparty": "Meridian Works Ltd",
        "reference": "INV-2026-1"}]}), ["a"])["a"]
    flags = run_checks(text, ex)
    assert "not_in_text" in flags["date"]
    assert flags["total_amount"] == []
    assert "is_our_company" in flags["counterparty"]


def test_validate_catches_bad_output(tmp_path):
    from router.schemas import Document
    from router.validate import validate

    docs = [Document(doc_id="D-1", text="INVOICE\nInvoice No: INV-2026-1\n"),
            Document(doc_id="D-2", text="hello")]
    out = tmp_path / "output.jsonl"
    out.write_text(json.dumps({"id": "D-1", "doc_type": "memo", "date": "14/03/2026",
                               "total_amount": "12", "counterparty": "X",
                               "reference": "INV-2026-1", "cheap_calls": 1,
                               "expensive_calls": 200}) + "\n", encoding="utf-8")
    v = validate(out, docs, budget=2100, cost_cheap=1, cost_expensive=20)
    text = " ".join(v.errors)
    for problem in ("doc_type", "YYYY-MM-DD", "not a number", "missing from output",
                    "exceed budget"):
        assert problem in text
    assert v.crosscheck["reference"] == [1, 1]


def test_policy_never_exceeds_budget():
    cands = [Candidate(str(i), 5, 0.5) for i in range(100)]
    plan = choose_escalations(cands, expensive_accuracy=0.97, spent=300, budget=2100,
                              cost_expensive=20, target=0.99, policy="max_accuracy")
    assert plan.planned_cost <= 2100
    assert len(plan.escalate) == 90


class FlakyClient(FakeClient):
    """Expensive tier whose batch requests always fail; single-document requests work."""

    async def _generate(self, system: str, prompt: str) -> str:
        if len(re.findall(r"### id: ", prompt)) > 1:
            raise ValueError("simulated outage")
        return await super()._generate(system, prompt)


@pytest.mark.skipif(not DOCS.exists(), reason="provided documents not present")
async def test_failed_batches_are_retried_singly_within_budget(monkeypatch, tmp_path):
    clients = {"cheap": FakeClient("cheap", wrong_every=4),
               "expensive": FlakyClient("expensive", wrong_every=0)}
    monkeypatch.setattr(pipeline, "build_client", lambda tier, s, cache: clients[tier])
    s = Settings(use_cache=False, output_dir=tmp_path, cheap_rpm=0, expensive_rpm=0)
    run = await pipeline.run_pipeline(s, load_documents(DOCS))
    assert run.total_cost <= s.budget
    assert run.calls["retry_sends"] > 0
    # Every escalated/calibration document ends up with an expensive answer.
    assert "expensive_no_answer_kept_cheap" not in run.failures
    # Failed batch calls are not charged: each answered document is charged exactly once.
    assert all(r.expensive_calls <= 1 for r in run.results)


@pytest.mark.skipif(not DOCS.exists(), reason="provided documents not present")
async def test_pipeline_end_to_end_with_fakes(monkeypatch, tmp_path):
    clients = {"cheap": FakeClient("cheap", wrong_every=4),
               "expensive": FakeClient("expensive", wrong_every=0)}
    monkeypatch.setattr(pipeline, "build_client", lambda tier, s, cache: clients[tier])
    s = Settings(use_cache=False, output_dir=tmp_path, cheap_rpm=0, expensive_rpm=0)
    docs = load_documents(DOCS)
    run = await pipeline.run_pipeline(s, docs)
    assert len(run.results) == 300
    assert run.total_cost <= s.budget
    assert all(r.cheap_calls == 1 for r in run.results)
    assert sum(r.expensive_calls for r in run.results) == (run.total_cost - 300) / 20
