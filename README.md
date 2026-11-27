# Model Router Under a Budget

Extracts `doc_type`, `date`, `total_amount`, `counterparty` and `reference` from 300
business documents (invoices, contracts, emails). It decides per document whether the
cheap model's answer is good enough or whether the document is worth sending to the
expensive model. It stays inside the cost budget and the 10-minute limit, and reports
the accuracy it expects.

| Rule (from the brief) | Value |
|---|---|
| Cheap model | 1 unit per document sent |
| Expensive model | 20 units per document sent |
| Budget | 2,100 units |
| Time limit | 10 minutes, first call to last call |
| Accuracy target | 92% of fields correct |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev]"
copy .env.example .env            # then put your Gemini key in ROUTER_GEMINI_API_KEY
router run                        # processes provided/candidate/documents.jsonl
router run --limit 20             # quick smoke test on 20 documents (budget scaled)
pytest                            # offline tests with fake models, no key needed
```

Outputs, written to `output/`:

| File | Contents |
|---|---|
| `output.jsonl` | The required format: one line per document with the five fields plus `cheap_calls` and `expensive_calls` |
| `decisions.jsonl` | Route, reason, check flags and expected accuracy for each document |
| `summary.json`, `report.md` | Units spent, time taken, expected accuracy, routing counts, calibration table |
| `calibration.json` | The learned cheap-model accuracy per risk cell |

## Models chosen

| Role | Model | Why |
|---|---|---|
| Cheap | `gemini-3.5-flash-lite` | Fast and good at JSON output. The free tier allows enough requests and tokens to batch all 300 documents in a couple of minutes |
| Expensive | `gemini-3.5-flash` | Clearly stronger at reading OCR noise, long email threads and picking the right one of several amounts. Same API, so the comparison is fair |

Both are behind one `ModelClient` interface. Groq, Ollama or any OpenAI-compatible API can be
swapped in through `.env` without code changes. I chose Gemini over Groq's free tier because
Groq's tokens-per-minute cap would not allow 300 documents in 10 minutes.

## Latest full run (300 documents)

| Metric | Result | Target |
|---|---|---|
| Expected accuracy | 96.5% (cheap only: 95.1%) | ≥ 92% |
| Units spent | 1,700 (300 cheap + 70 expensive sends) | ≤ 2,100 |
| Time, first to last call | 307 s | < 600 s |
| Routes | 210 cheap, 70 escalated, 20 calibration | |

Four expensive requests (20 documents) failed after all retries. Those calls were not
charged, and the router kept the cheap answers. Full details are in `output/`.

## How it works

```
documents.jsonl
   │
   ▼
1. CHEAP PASS ─ every document, 10 per request ─────────────────── 300 units
   │
   ▼
2. FREE CHECKS ─ no model calls
   • is each value actually present in the text? (date, amount, reference, name)
   • doc_type agrees with a keyword detector?
   • reference prefix fits the type (INV- / CTR- / PO-)?
   • counterparty is not our own company (Meridian Works Ltd)?
   • text contains embedded instructions aimed at automated systems?
   • complexity: number of dates/amounts, OCR scan, length of the email thread
   │
   ▼
3. CALIBRATION ─ expensive model on 20 random documents ────────── 400 units
   learns P(cheap field correct) for each cell:
   {clean, flagged} × {simple, complex}
   │
   ▼
4. PLAN ─ expected gain per document = fields × (P_expensive − P_cheap(doc))
   escalate the highest-gain documents until the budget is used up
   │
   ▼
5. EXPENSIVE PASS ─ chosen documents, 5 per request ────────────── ≤ 1,400 units
   │
   ▼
output.jsonl + report (expected accuracy, units, time)
```

**Why a cascade rather than a classifier before any model call:** the most useful signal
about whether a document is hard is the cheap model's own answer, checked against the text.
A value that does not appear in the document, or a counterparty that is our own company,
is a much stronger warning than anything visible before extraction.

**Why the expected accuracy can be trusted:** no labels are provided, so the router estimates
accuracy itself. It runs the expensive model on a random, stratified sample and measures how
often the cheap answer agrees with it in each risk cell. Beta smoothing stops a cell with a
few samples from claiming 100%. The expected accuracy is then the average over all fields
of P(correct), using the cell accuracy for cheap answers and the expensive model's accuracy
for escalated ones. The expensive model's accuracy is an assumption (97%, set in config).
This is stated in the report.

**Budget guarantees:**
- The planner never plans past the budget.
- Retries of documents an expensive batch left out are capped by the remaining budget.
- Failed calls with no answer are not charged, as the rules allow.
- Each document's `cheap_calls` and `expensive_calls` are counted when it is actually sent.

**Reliability:**
- Timeouts, and retries with exponential backoff on 429 and 5xx responses.
- A requests-per-minute limiter for free-tier quotas.
- A document missing from a batch reply is retried on its own.
- If the expensive model gives no answer, the cheap answer is kept, and the reason says so.
- Document text is always treated as data, never as instructions.
- Every reply is cached in SQLite, so re-running costs nothing.

**Policies:**
- `max_accuracy` (default) spends the budget on the riskiest documents.
- `target` stops as soon as expected accuracy clears 92% plus a 2-point margin. Use it when cost matters more than extra accuracy.

## Project layout

```
src/router/
  config.py        settings and task rules (env / .env)
  schemas.py       fields to extract, records
  extract.py       batched prompt, JSON parsing, keyword type detector
  normalize.py     dates, amounts, IDs to canonical form
  checks.py        free consistency checks and complexity score
  calibration.py   per-cell cheap accuracy, Beta smoothing
  policy.py        budget-aware greedy escalation
  pipeline.py      the run: cheap → checks → calibrate → plan → expensive
  models/          ModelClient base (retry, rate limit, cache), Gemini/Groq/OpenAI-compatible/Ollama
  report.py        console report and output files
  cli.py           `router run`
tests/             offline tests with fake models
provided/          the task files as given
```

## Design note: running this for 5 million documents a month

- **Throughput and batching:** 5M a month is about 115 documents a minute on average, with peaks several times higher. Put a queue (SQS, Pub/Sub or Kafka) in front of stateless workers that scale on queue depth. Send non-urgent work to providers' batch APIs, which cost about 50% less, and keep the real-time path only for documents someone is waiting on.
- **Model registry, not hard-coded models:** prices, cost ratios, rate limits and measured accuracy per document type live in a config registry. The router re-solves its escalation threshold from those numbers every day. A price cut or a better cheap model then changes the routing automatically, with no code release.
- **Continuous calibration:** keep sending a small random slice (about 1%) of traffic to the expensive model, plus human review of a smaller slice. Use these to re-fit the risk-to-accuracy table each day per document type and source. Alert when measured accuracy drops below 92% or the share of escalations jumps. Either usually means the incoming documents have changed (a new vendor layout, a new scanner).
- **Golden set and shadow testing for model changes:** a fixed, labelled set of documents across all types and difficulty levels. Every new model or prompt is scored on it and run in shadow on live traffic, and it is promoted only if it beats the current setup on accuracy per unit of cost. Providers update models silently, so this runs on a schedule too.
- **Cost control as a hard limit:** monthly and daily budgets are enforced by the planner, not checked afterwards. Under pressure the system raises the escalation threshold rather than overspending. Exact and near-duplicate documents are served from cache, and prompt caching is used for the fixed instructions.
- **Resilience and safety:** each tier has a fallback provider behind the same interface, with circuit breakers. If the expensive tier is down, high-risk documents are held for later rather than accepted at low quality. Document text is always treated as data, and documents with embedded instructions or signs of tampering are flagged for review.

## Limitations and next steps

- No ground truth was provided, so accuracy is estimated, not measured. Labelling about 50 documents would let the report show measured accuracy as well.
- The calibration sample costs 400 units. In production, calibration would be shared across runs instead.
- Possible improvements: field-level escalation (re-ask only for doubtful fields), self-consistency on the cheap model as a further risk signal, and a budget sweep chart.
