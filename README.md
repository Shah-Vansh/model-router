# Model Router Under a Budget

A router that pulls five key details from 300 business documents (invoices, contracts
and emails). For each document it decides whether a **cheap** LLM's answer is good
enough, or whether the document is worth sending to an **expensive** LLM. It stays
within the cost budget and the time limit, and reports the accuracy it expects.

## Results (full run on the 300 provided documents)

| Metric | Result | Target | |
|---|---|---|---|
| Expected accuracy | **96.5%** | ≥ 92% | ✅ |
| Units spent | **1,700** (300 cheap + 70 expensive) | ≤ 2,100 | ✅ |
| Time, first call to last call | **307 s** | < 600 s | ✅ |
| Rule-based cross-check (measured) | **866 / 866 fields match** | – | ✅ |

| Strategy | Units | Expected accuracy |
|---|---|---|
| Cheap model only | 300 | 95.1% |
| Expensive model only | 6,000 (breaks budget) | ~97% |
| **This router** | **1,700** | **96.5%** |

Routing: 210 documents kept the cheap answer, 70 were escalated, and 20 were the
calibration sample. Every per-document decision and its reason is in
`output/decisions.jsonl`.

---

## The task

| Rule (from `provided/candidate/README.txt`) | Value |
|---|---|
| Input | `documents.jsonl`, 300 × `{"id", "text"}`. Our company is Meridian Works Ltd |
| Fields to extract | `doc_type`, `date`, `total_amount`, `counterparty`, `reference` |
| Cheap model | 1 unit per document sent |
| Expensive model | 20 units per document sent |
| Failed call with no answer | not charged |
| Batching | allowed; every document in a batch still counts |
| Budget | 2,100 units |
| Time limit | 10 minutes, first call to last call |
| Accuracy target | 92% of details correct |

The data is deliberately difficult:
- 45 documents are OCR scans with character errors ("Ju1y", "Agrecment").
- Email threads mention several offers, and only the final agreed amount is correct.
- Some documents contain text telling automated systems to record wrong values.

---

## How to test it

### 1. Install (Python 3.11+)

```bash
git clone https://github.com/Shah-Vansh/model-router.git
cd model-router
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

### 2. Checks that need no API key

```bash
pytest -q             # 6 offline tests with fake models
router validate       # checks the submitted output/output.jsonl
```

`pytest` covers:
- parsing of messy model output
- the consistency checks
- the budget planner never overspending
- a full 300-document pipeline run with fake models
- failed batches being retried one by one within the budget
- the validator catching bad output

`router validate` reads `output/output.jsonl` and checks:
- every one of the 300 ids is present, exactly once
- `doc_type`, `date` and `total_amount` have the required format
- the units recomputed from `cheap_calls` and `expensive_calls` are within 2,100

It then cross-checks the output against values a strict regex can read from clean
(non-OCR) documents: dates, types, references and invoice totals. Expected result:

```
Rows: 300 / 300 documents
Units from cheap_calls/expensive_calls: 1700 / budget 2100
Format and budget: PASS
Rule-based cross-check on clean (non-OCR) documents:
  date          225/225 match (100.0%)
  doc_type      255/255 match (100.0%)
  reference     255/255 match (100.0%)
  total_amount  131/131 match (100.0%)
```

### 3. Full run with real models (free Gemini key)

1. Create a free key at https://aistudio.google.com → **Get API key**.
2. Copy the example settings and put the key in `ROUTER_GEMINI_API_KEY`:
   ```bash
   copy .env.example .env      # Windows  (cp on macOS / Linux)
   ```
3. Run:
   ```bash
   router run --limit 20       # smoke test: 20 docs, budget scaled to 140, ~20 s
   router run                  # all 300 docs, about 5 minutes
   router validate             # check what was just produced
   ```

`router run` prints PASS or FAIL for accuracy, budget and time, compares the result
with the single-model baselines, and writes the files below. Model replies are cached in
`.cache/`, so a second run is instant and costs nothing. Use `--no-cache` for a fresh,
timed run.

Note: the free tier has a **daily** request quota. One full run fits comfortably. Several
full runs on the same day can use up the expensive model's quota (HTTP 429). When that
happens the router keeps the cheap answers and says so in the report.

### Output files (`output/`)

| File | Contents |
|---|---|
| `output.jsonl` | **The required deliverable**: one line per document, `id, doc_type, date, total_amount, counterparty, reference, cheap_calls, expensive_calls` |
| `decisions.jsonl` | Route, reason, check flags, expected accuracy and cost for each document |
| `summary.json` | Units, time, expected accuracy, routing counts, request errors, calibration |
| `report.md` | A short human-readable version of the summary |
| `calibration.json` | The learned cheap-model accuracy for each risk group |
| `run_log.txt` | Console output of the submitted run |

---

## Models chosen

| Role | Model | Why |
|---|---|---|
| Cheap | `gemini-3.5-flash-lite` | Fast, reliable JSON output, and a free tier generous enough to batch all 300 documents in about 30 requests |
| Expensive | `gemini-3.5-flash` | Noticeably better at OCR noise, long email threads, and choosing the right amount among several. Same API and prompt, so the comparison is fair |

I also considered:
- **Groq** (Llama 8B / 70B): its free tier's tokens-per-minute cap makes 300 documents in 10 minutes unrealistic.
- **Local Ollama:** a 7B model on a laptop CPU is too slow for the time limit.

Both are still supported through `.env`, as is any OpenAI-compatible API, with no code
changes.

---

## How it works

```
documents.jsonl (300)
   │
   ▼
① CHEAP PASS ─ every document, 10 per request ──────────────────────── 300 units
   │
   ▼
② FREE CHECKS ─ plain Python, no model calls. Flags a field when:
   • its value does not appear in the text (date, amount, reference, name)
   • doc_type disagrees with a keyword detector
   • the reference prefix does not fit the type (INV- / CTR- / PO-)
   • the counterparty is our own company
   • the document contains instructions aimed at automated systems
   + a complexity score: number of dates/amounts, OCR scan, email-thread depth
   │
   ▼
③ CALIBRATION ─ expensive model on 20 random documents (stratified by type) ── 400 units
   measures how often the cheap answer is right in each group:
                    simple doc   complex doc
      clean field     ~97%          ~96%
      flagged field   ~44%          ~34%
   │
   ▼
④ PLAN ─ gain(doc) = fields × (P_expensive − P_cheap(doc))
   escalate the highest-gain documents until the budget is used
   │
   ▼
⑤ EXPENSIVE PASS ─ chosen documents, 5 per request ─────────────────── ≤ 1,400 units
   unanswered documents are retried one at a time with the units left over
   │
   ▼
output.jsonl + report (expected accuracy, units spent, time taken)
```

### Key decisions

**A cascade, not a classifier before any model call.** The best warning that a document
is hard is the cheap model's own answer, checked against the text. A value that doesn't
appear in the document, or a counterparty that is our own company, says much more than
anything visible before extraction.

**Greedy escalation is optimal here.** Every escalation costs exactly 20 units. Taking
the documents with the largest expected gain first therefore gives the best expected
accuracy for any budget.

**Expected accuracy without labels.** No answer key is provided, so the router
calibrates itself:
- It measures how often the cheap answer agrees with the expensive model on a random sample, separately for each of the four groups.
- It smooths each group with a Beta prior, so a group with few samples can't claim 100%.
- The reference is itself imperfect, so P(cheap correct) = agreement × the expensive model's accuracy (assumed 97% in config). This keeps the estimate conservative.
- Expected accuracy is the average P(correct) over all 1,500 fields.

The rule-based cross-check in `router validate` is an independent, measured check on
the subset that can be read without a model.

**The budget can't be exceeded.**
- The planner never plans past it.
- Retries are paid only from units actually left unspent.
- Failed calls are not charged, as the rules allow.
- `cheap_calls` and `expensive_calls` count real sends per document.

**Time.**
- Batching cuts 300 documents to about 30 cheap requests.
- Requests run concurrently under a requests-per-minute limiter.
- Time is measured from the first model call to the last, as the rules specify.

**Reliability and safety.**
- Timeouts, and retries with exponential backoff on 429 and 5xx responses.
- Documents left out of a batch reply are re-sent on their own.
- If the expensive model fails, the cheap answer is kept and the reason says so.
- Document text is always treated as data: the prompt says so, and a check escalates documents containing embedded instructions.

**Two policies.**
- `max_accuracy` (default) spends the budget on the riskiest documents.
- `target` stops once expected accuracy clears 92% plus a 2-point margin. Use it when saving units matters more than the last few points of accuracy.

---

## Configuration

All settings live in `src/router/config.py` and can be overridden in `.env` (prefix
`ROUTER_`). See `.env.example`.

| Setting | Default | Meaning |
|---|---|---|
| `CHEAP_PROVIDER` / `CHEAP_MODEL` | `gemini` / `gemini-3.5-flash-lite` | Provider: `gemini`, `groq`, `ollama` or `openai_compat` |
| `EXPENSIVE_PROVIDER` / `EXPENSIVE_MODEL` | `gemini` / `gemini-3.5-flash` | |
| `BUDGET` | 2100 | Units |
| `COST_CHEAP` / `COST_EXPENSIVE` | 1 / 20 | Units per document sent |
| `CHEAP_BATCH_SIZE` / `EXPENSIVE_BATCH_SIZE` | 10 / 5 | Documents per request |
| `CHEAP_RPM` / `EXPENSIVE_RPM` | 15 / 10 | Requests per minute (0 = unlimited) |
| `POLICY` | `max_accuracy` | Or `target` |
| `TARGET_ACCURACY` / `TARGET_MARGIN` | 0.92 / 0.02 | Used by the `target` policy |
| `CALIBRATION_SIZE` | 20 | Documents in the calibration sample |
| `EXPENSIVE_ACCURACY_PRIOR` | 0.97 | Assumed accuracy of the expensive model |

## Tech stack

| Tool | Used for |
|---|---|
| Python 3.11 | Main language |
| httpx + asyncio | Concurrent API calls |
| tenacity | Retries with exponential backoff |
| Pydantic, pydantic-settings | Data records, validation, `.env` config |
| python-dateutil | Parsing dates in any format |
| SQLite | Cache of model replies |
| Typer, Rich | Command line and report tables |
| pytest, Ruff | Tests and linting |

## Project layout

```
src/router/
  cli.py           router run / router validate
  config.py        settings and task rules (.env)
  loader.py        reads documents.jsonl
  schemas.py       the five fields and data records
  extract.py       batched prompt, JSON parsing, keyword type detector
  normalize.py     dates, amounts and IDs to canonical form
  checks.py        free consistency checks and complexity score
  calibration.py   per-group cheap accuracy with Beta smoothing
  policy.py        budget-aware greedy escalation
  pipeline.py      cheap → checks → calibrate → plan → expensive → results
  report.py        console report and output files
  validate.py      output format, budget and rule-based cross-check
  models/          ModelClient base (retry, rate limit, cache) + Gemini/Groq/OpenAI-compatible/Ollama
tests/             offline tests with fake models
provided/          the task files exactly as given
output/            results of the submitted run
```

---

## Design note: running this for 5 million documents a month

- **Throughput and batching:** 5M a month is about 115 documents a minute on average, with peaks several times higher. Put a queue (SQS, Pub/Sub or Kafka) in front of stateless workers that scale on queue depth. Send non-urgent work to providers' batch APIs, which cost about 50% less, and keep the real-time path only for documents someone is waiting on.
- **A model registry, not hard-coded models:** prices, cost ratios, rate limits and measured accuracy per document type live in a config registry. The router re-solves its escalation threshold from those numbers every day. A price cut or a better cheap model then changes the routing automatically, with no code release.
- **Continuous calibration:** keep sending a small random slice (about 1%) of traffic to the expensive model, plus human review of a smaller slice. Use these to re-fit the risk-to-accuracy table daily per document type and source. Alert when measured accuracy drops below 92% or the share of escalations jumps. Either usually means the incoming documents changed (a new vendor layout, a new scanner).
- **Golden set and shadow testing:** a fixed, labelled set of documents across all types and difficulty levels. Every new model or prompt is scored on it and run in shadow on live traffic, and it is promoted only if it beats the current setup on accuracy per unit of cost. Providers change models silently, so this also runs on a schedule.
- **Cost as a hard limit:** monthly and daily budgets are enforced by the planner, not checked afterwards. Under pressure the system raises the escalation threshold rather than overspending. Exact and near-duplicate documents are served from cache, and prompt caching is used for the fixed instructions.
- **Resilience and safety:** each tier has a fallback provider behind the same interface, with circuit breakers. If the expensive tier is down, high-risk documents are held for later rather than accepted at low quality. Document text is always treated as data, and documents with embedded instructions or signs of tampering are flagged for review.

## Limitations and next steps

- **Accuracy is estimated.** No labels were provided. The cross-check measures part of the data (100% match on 866 fields), but counterparties and the OCR documents are covered only by the estimate. Labelling about 50 documents would give a measured figure for everything.
- **The retry pass was added after the submitted run.** In that run, 4 expensive requests (20 documents) failed and those documents kept their cheap answers. The retry pass would now re-send them with the 400 unspent units. It is covered by a test, but hasn't been re-run on all 300 documents because the free daily quota was used up.
- **Calibration costs 400 units** on random documents. In production it would be shared across runs.
- **Next steps:**
  - Field-level escalation: re-ask only for the doubtful field.
  - Self-consistency on the cheap model as a further risk signal.
  - A budget sweep (accuracy against budget).
