MODEL ROUTER UNDER A BUDGET: DATA AND RULES

DOCUMENTS
documents.jsonl contains 300 documents, one JSON object per line:
  {"id": "D-001", "text": "..."}
Our company is Meridian Works Ltd.

THE TWO MODELS
You can use any LLM. Pick two models yourself:
  cheap      a small, fast model
  expensive  a larger, more accurate model
Say in your README which two you picked and why.

COST AND TIME
Cost is counted in units, not in real money.
  Each time a document is sent to your cheap model:      1 unit
  Each time a document is sent to your expensive model: 20 units
  A call that fails and returns no answer is not charged.
  Sending several documents in one call is allowed. Each document still counts.
Budget for the full run: 2,100 units.
Time limit for the full run: 10 minutes, measured from your first call to your last.
You keep the count yourself. Your report must show the units spent and the time taken.

OUTPUT
Produce a file named output.jsonl with one line per document:
  {"id": "D-001", "doc_type": "invoice", "date": "2026-03-14", "total_amount": 12450.0,
   "counterparty": "Northgate Logistics Ltd", "reference": "INV-2026-04512",
   "cheap_calls": 1, "expensive_calls": 0}
  doc_type is invoice, contract, or email. date is YYYY-MM-DD. total_amount is a number.
  cheap_calls and expensive_calls are how many times that document was sent to each model.
