"""Decide which documents get the expensive model, without breaking the budget.

Every document already got one cheap call. Sending document d to the
expensive model costs `cost_expensive` and is expected to fix

    gain(d) = n_fields(d) * (expensive_accuracy - cheap_expected_accuracy(d))

fields. All escalations cost the same, so the best use of a fixed budget is
simply the documents with the largest gain first (greedy knapsack with equal
weights is optimal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    doc_id: str
    n_fields: int
    cheap_expected: float  # expected share of fields right with the cheap answer
    fixed: bool = False  # already on the expensive model (calibration sample)


@dataclass
class Plan:
    escalate: list[str]
    expected_accuracy: float
    planned_cost: float
    stop_reason: str


def choose_escalations(
    cands: list[Candidate],
    *,
    expensive_accuracy: float,
    spent: float,
    budget: float,
    cost_expensive: float,
    target: float,
    policy: str,
) -> Plan:
    total_fields = sum(c.n_fields for c in cands) or 1
    expected_correct = sum(
        c.n_fields * (expensive_accuracy if c.fixed else c.cheap_expected) for c in cands
    )
    ranked = sorted(
        (c for c in cands if not c.fixed),
        key=lambda c: c.n_fields * (expensive_accuracy - c.cheap_expected),
        reverse=True,
    )

    chosen: list[str] = []
    cost = spent
    reason = "no document left that the expensive model would improve"
    for c in ranked:
        if policy == "target" and expected_correct / total_fields >= target:
            reason = f"expected accuracy reached the target ({target:.1%})"
            break
        gain = c.n_fields * (expensive_accuracy - c.cheap_expected)
        if gain <= 0:
            break
        if cost + cost_expensive > budget + 1e-9:
            reason = "budget used up"
            break
        chosen.append(c.doc_id)
        cost += cost_expensive
        expected_correct += gain
    else:
        if ranked and len(chosen) == len(ranked):
            reason = "every document escalated"

    return Plan(
        escalate=chosen,
        expected_accuracy=expected_correct / total_fields,
        planned_cost=cost,
        stop_reason=reason,
    )
