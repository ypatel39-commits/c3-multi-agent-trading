# STATE — C3 Multi-Agent Trading

## Status
v0.1.0 — initial build complete. End-to-end pipeline runs offline (mock LLM)
and live (Ollama qwen2.5:7b). Backtest, debate, transcript logging, Streamlit
UI, and 12+ pytest tests are all in place.

## What works
- 3 distinct system prompts (Quant / Risk / Compliance) on one Ollama model
- 2-round debate with peer-aware Round 2
- Majority-rule verdict with risk-averse tie-break (rejects beat approvals)
- JSONL transcript + equity-curve PNG written to `docs/`
- Streamlit dashboard: live mode + replay mode
- pytest suite mocks Ollama end-to-end (no network, no model required)

## Known follow-ups (not blocking v0.1)
- Add walk-forward parameter validation for the EMA grid
- Expand committee with a "Portfolio Manager" tie-breaker role
- Wire `plotly` for an interactive equity curve in Streamlit
- CSV-cache invalidation policy (currently never refreshes once cached)
- Confidence-weighted voting (currently 1 agent = 1 vote)

## Ops notes
- Model: `qwen2.5:7b` at `http://localhost:11434`
- Seed: `random_state=42` (numpy + Ollama options.seed=42)
- All source files under 300 lines

## Author
Yash Patel — yashpatel06050@gmail.com
