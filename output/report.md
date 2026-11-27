# Run report

- Documents: 300
- Models: cheap `gemini:gemini-3.5-flash-lite`, expensive `gemini:gemini-3.5-flash`
- Routes: {'expensive': 70, 'cheap': 210, 'calibration': 20}
- Expected accuracy: 96.5% (target 92%; cheap only 95.1%)
- Units spent: 1700 of 2100 (300 cheap + 70 expensive document sends)
- Time, first call to last call: 306.8 s (limit 600 s)
- Planner stop reason: budget used up

Per-document routing reasons are in `decisions.jsonl`.
