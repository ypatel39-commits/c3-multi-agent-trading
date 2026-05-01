"""Streamlit dashboard: backtest a strategy, watch the committee debate.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from c3_multi_agent_trading.agents import build_committee  # noqa: E402
from c3_multi_agent_trading.runner import build_proposal, run_pipeline  # noqa: E402
from c3_multi_agent_trading.strategy import load_spy, run_backtest  # noqa: E402


st.set_page_config(page_title="C3 Multi-Agent Trading", layout="wide")
st.title("C3 — Multi-Agent Trading Research")
st.caption("Quant / Risk / Compliance LLM committee debating a momentum strategy on SPY.")

with st.sidebar:
    st.header("Strategy Params")
    start = st.text_input("Start", "2020-01-01")
    end = st.text_input("End", "2024-12-31")
    fast = st.number_input("Fast EMA", 5, 200, 50, step=1)
    slow = st.number_input("Slow EMA", 20, 400, 200, step=1)
    mode = st.radio("LLM mode", ["Live Ollama (qwen2.5:7b)", "Replay last transcript"])
    run_button = st.button("Run committee review")

docs = REPO / "docs"
docs.mkdir(exist_ok=True)
transcript_path = docs / "debate-transcript.jsonl"
plot_path = docs / "equity-curve.png"
data_cache = REPO / "data" / "spy.csv"


def _render_backtest(bt_dict: dict) -> None:
    m = bt_dict["metrics"]
    cols = st.columns(4)
    cols[0].metric("Total Return", f"{m['total_return_pct']}%")
    cols[1].metric("Sharpe", f"{m['sharpe']}")
    cols[2].metric("Max Drawdown", f"{m['max_drawdown_pct']}%")
    cols[3].metric("# Trades", m["num_trades"])
    if plot_path.exists():
        st.image(str(plot_path), caption="Equity curve", use_column_width=True)


def _render_debate(turns: list, verdict: str, vote: dict, dissent: list) -> None:
    st.subheader(f"Verdict: {verdict}")
    st.write(f"Vote tally: `{vote}`")
    if dissent:
        st.warning(f"Dissent: {', '.join(dissent)}")
    for r in (1, 2):
        st.markdown(f"### Round {r}")
        round_turns = [t for t in turns if t["round"] == r]
        for t in round_turns:
            with st.expander(f"{t['agent']} — {t['position'].upper()} (conf {t['confidence']:.2f})"):
                st.write(t["reasoning"])
                if t["concerns"]:
                    st.markdown("**Concerns:**")
                    for c in t["concerns"]:
                        st.markdown(f"- {c}")


if run_button and mode.startswith("Live"):
    with st.spinner("Running backtest + committee debate (this calls Ollama)…"):
        if transcript_path.exists():
            transcript_path.unlink()
        verdict = run_pipeline(
            start=start,
            end=end,
            fast=int(fast),
            slow=int(slow),
            transcript_path=transcript_path,
            plot_path=plot_path,
            data_cache=data_cache,
        )
        st.success("Committee review complete.")
        _render_backtest(verdict.backtest.to_dict())
        debate = verdict.debate.to_dict()
        _render_debate(
            [t for t in debate["turns"]],
            debate["verdict"],
            debate["vote"],
            debate["dissent"],
        )

elif run_button and mode.startswith("Replay"):
    if not transcript_path.exists():
        st.error("No transcript found. Run a live debate first or run scripts/run_demo.py --offline.")
    else:
        with st.spinner("Re-running backtest (LLM not called)…"):
            df = load_spy(start=start, end=end, cache_path=data_cache)
            bt = run_backtest(df, fast=int(fast), slow=int(slow), plot_path=plot_path)
        _render_backtest(bt.to_dict())
        turns: list = []
        verdict = "?"
        vote: dict = {}
        dissent: list = []
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event") == "turn":
                turns.append(ev)
            elif ev.get("event") == "verdict":
                verdict = ev.get("verdict", "?")
                vote = ev.get("vote", {})
                dissent = ev.get("dissent", [])
        _render_debate(turns, verdict, vote, dissent)

else:
    st.info("Set parameters in the sidebar and click **Run committee review**.")
    if transcript_path.exists():
        st.caption(f"Last transcript: `{transcript_path.relative_to(REPO)}`")
