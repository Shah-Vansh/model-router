"""End-to-end run: cheap pass -> checks -> calibrate -> plan -> expensive pass -> results."""

from __future__ import annotations

import asyncio
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import calibration as calib
from .cache import ResponseCache
from .checks import complexity, run_checks
from .config import Settings
from .extract import SYSTEM_PROMPT, build_prompt, detect_type, parse_batch
from .models import ModelCall, ModelClient, build_client
from .policy import Candidate, Plan, choose_escalations
from .schemas import FIELDS, DocResult, Document, Extraction


class BudgetError(ValueError):
    pass


@dataclass
class TierRun:
    answers: dict[str, Extraction] = field(default_factory=dict)
    calls_per_doc: Counter = field(default_factory=Counter)  # charged sends per document
    calls: list[ModelCall] = field(default_factory=list)
    first_call: float | None = None
    last_call: float | None = None
    retried: int = 0  # single-document retry sends


async def _send(client: ModelClient, batch: list[Document], s: Settings, run: TierRun) -> None:
    now = time.perf_counter()
    run.first_call = now if run.first_call is None else min(run.first_call, now)
    call = await client.generate(SYSTEM_PROMPT, build_prompt(batch, s.max_doc_chars))
    run.last_call = max(run.last_call or 0.0, time.perf_counter())
    run.calls.append(call)
    if call.error or not call.raw:
        return  # failed call, no answer: not charged (task rule)
    for d in batch:
        run.calls_per_doc[d.doc_id] += 1
    run.answers.update(parse_batch(call.raw, [d.doc_id for d in batch]))


async def run_tier(client: ModelClient, docs: list[Document], batch_size: int, s: Settings,
                   budget_units: float | None = None, unit_cost: float = 1.0,
                   retry_rounds: int = 2) -> TierRun:
    """Send documents in batches, then retry unanswered ones on their own.

    Failed calls are not charged, so the retry allowance is worked out from what
    was actually charged, never from the plan. Documents are retried in the
    order given (highest value first), and only while the budget still covers them.
    """
    run = TierRun()
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    await asyncio.gather(*(_send(client, b, s, run) for b in batches))
    for _ in range(retry_rounds):
        missing = [d for d in docs if d.doc_id not in run.answers]
        if budget_units is not None:
            charged = sum(run.calls_per_doc.values()) * unit_cost
            missing = missing[:max(0, int((budget_units - charged) // unit_cost))]
        if not missing:
            break
        run.retried += len(missing)
        await asyncio.gather(*(_send(client, [d], s, run) for d in missing))
    return run


def _calibration_sample(docs: list[Document], k: int, seed: int) -> list[str]:
    """Stratified by keyword-detected type so every type is represented."""
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        by_type[detect_type(d.text).value].append(d.doc_id)
    picked: list[str] = []
    for ids in by_type.values():
        share = max(1, round(k * len(ids) / len(docs))) if k else 0
        picked += rng.sample(sorted(ids), min(share, len(ids)))
    rng.shuffle(picked)
    return picked[:k]


@dataclass
class RunResult:
    results: list[DocResult]
    calibration: calib.Calibration
    plan: Plan
    elapsed_s: float  # first model call to last model call
    wall_s: float
    calls: dict[str, int]
    cache_hits: int
    failures: dict[str, int]
    settings: dict
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.results)


async def run_pipeline(s: Settings, docs: list[Document]) -> RunResult:
    n = len(docs)
    if n == 0:
        raise ValueError("No documents to process")
    if n * s.cost_cheap > s.budget:
        raise BudgetError(f"Budget {s.budget:g} cannot cover the cheap pass ({n} documents)")

    t_wall = time.perf_counter()
    cache = ResponseCache(s.cache_path) if s.use_cache else None
    cheap = build_client("cheap", s, cache)
    expensive = build_client("expensive", s, cache)
    by_id = {d.doc_id: d for d in docs}
    try:
        # 1. Cheap pass on every document.
        cheap_run = await run_tier(cheap, docs, s.cheap_batch_size, s,
                                   budget_units=s.budget, unit_cost=s.cost_cheap)

        # 2. Free checks and a complexity score per document.
        flags = {d.doc_id: run_checks(d.text, cheap_run.answers.get(d.doc_id)) for d in docs}
        comp = {d.doc_id: complexity(d.text) for d in docs}
        threshold = statistics.median(comp.values())
        is_complex = {i: c > threshold for i, c in comp.items()}

        # 3. Calibration: expensive model on a small random sample.
        spent = sum(cheap_run.calls_per_doc.values()) * s.cost_cheap
        room = int((s.budget - spent) // s.cost_expensive)
        cal_ids = _calibration_sample(docs, min(s.calibration_size, room), s.calibration_seed)
        exp_run = await run_tier(expensive, [by_id[i] for i in cal_ids],
                                 s.expensive_batch_size, s, budget_units=s.budget - spent,
                                 unit_cost=s.cost_expensive)
        samples = [
            {"cheap_values": cheap_run.answers[i].values if i in cheap_run.answers else {},
             "reference_values": exp_run.answers[i].values, "flags": flags[i],
             "complex_doc": is_complex[i]}
            for i in cal_ids if i in exp_run.answers
        ]
        cal = calib.fit(samples, prior_strength=s.calibration_prior_strength,
                        expensive_accuracy=s.expensive_accuracy_prior,
                        complexity_threshold=threshold)

        # 4. Expected accuracy of each cheap answer, then the escalation plan.
        def cheap_expected(i: str) -> float:
            f = flags[i]
            return sum(cal.p_correct(bool(v), is_complex[i]) for v in f.values()) / len(f)

        spent += sum(exp_run.calls_per_doc.values()) * s.cost_expensive
        plan = choose_escalations(
            [Candidate(d.doc_id, len(FIELDS), cheap_expected(d.doc_id),
                       fixed=d.doc_id in exp_run.answers) for d in docs],
            expensive_accuracy=cal.expensive_accuracy, spent=spent, budget=s.budget,
            cost_expensive=s.cost_expensive, target=s.target_accuracy + s.target_margin,
            policy=s.policy,
        )

        # 5. Expensive pass on the chosen documents. Unanswered documents are retried
        #    one by one, highest gain first, with whatever budget failed calls left unspent.
        esc_run = await run_tier(expensive, [by_id[i] for i in plan.escalate],
                                 s.expensive_batch_size, s, budget_units=s.budget - spent,
                                 unit_cost=s.cost_expensive)
    finally:
        await cheap.aclose()
        await expensive.aclose()
        if cache is not None:
            cache.close()

    # 6. Assemble final answers.
    exp_answers = {**exp_run.answers, **esc_run.answers}
    exp_calls = exp_run.calls_per_doc + esc_run.calls_per_doc
    escalated = set(plan.escalate)
    failures = Counter()
    results = []
    for d in docs:
        i = d.doc_id
        ce = cheap_expected(i)
        flagged = sorted({r for rs in flags[i].values() for r in rs})
        values = cheap_run.answers[i].values if i in cheap_run.answers else {}
        expected, route = ce, "cheap"
        reason = "cheap answer passed all checks" if not flagged else \
            "flags: " + ", ".join(flagged) + " (ranked below the budget cut)"
        if i not in cheap_run.answers:
            failures["cheap_no_answer"] += 1
        if i in cal_ids or i in escalated:
            route = "calibration" if i in cal_ids else "expensive"
            reason = "random calibration sample" if route == "calibration" else \
                f"escalated: cheap expected {ce:.0%}" + (f"; {', '.join(flagged)}" if flagged
                                                         else "; complex document")
            if i in exp_answers:
                values, expected = exp_answers[i].values, cal.expensive_accuracy
            else:
                failures["expensive_no_answer_kept_cheap"] += 1
                reason += "; expensive model gave no answer, kept cheap answer"
        cc, ec = cheap_run.calls_per_doc[i], exp_calls[i]
        results.append(DocResult(
            doc_id=i, route=route, reason=reason,
            values={f.name: values.get(f.name) for f in FIELDS},
            expected_accuracy=round(expected, 4), cheap_expected_accuracy=round(ce, 4),
            flags={k: v for k, v in flags[i].items() if v}, complex_doc=is_complex[i],
            cheap_calls=cc, expensive_calls=ec, cost=cc * s.cost_cheap + ec * s.cost_expensive,
        ))

    runs = [cheap_run, exp_run, esc_run]
    firsts = [r.first_call for r in runs if r.first_call is not None]
    lasts = [r.last_call for r in runs if r.last_call is not None]
    all_calls = [c for r in runs for c in r.calls]
    return RunResult(
        results=results, calibration=cal, plan=plan,
        elapsed_s=(max(lasts) - min(firsts)) if firsts else 0.0,
        wall_s=time.perf_counter() - t_wall,
        calls={"cheap_requests": len(cheap_run.calls),
               "expensive_requests": len(exp_run.calls) + len(esc_run.calls),
               "calibration_docs": len(cal_ids), "escalated_docs": len(plan.escalate),
               "failed_requests": sum(c.error is not None for c in all_calls),
               "retry_sends": sum(r.retried for r in runs)},
        errors=dict(Counter(c.error.split(":")[0] for c in all_calls if c.error)),
        cache_hits=sum(c.cached for c in all_calls),
        failures=dict(failures),
        settings={"cheap": cheap.model_id, "expensive": expensive.model_id, "budget": s.budget,
                  "cost_cheap": s.cost_cheap, "cost_expensive": s.cost_expensive,
                  "target": s.target_accuracy, "policy": s.policy,
                  "time_limit_s": s.time_limit_s},
    )
