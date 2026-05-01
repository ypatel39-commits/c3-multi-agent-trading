"""Demo driver: full pipeline -> writes transcript JSONL + equity-curve PNG.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --offline   # mock LLM, no Ollama needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from c3_multi_agent_trading.agents import build_committee  # noqa: E402
from c3_multi_agent_trading.runner import run_pipeline  # noqa: E402


class _OfflineClient:
    """Deterministic stand-in for Ollama (used with --offline)."""

    _SCRIPT = {
        "Quant": {
            "position": "approve",
            "concerns": ["Sharpe modest", "trend-only edge", "fast/slow grid not validated"],
            "reasoning": (
                "Total return positive and Sharpe above 0.6 on 5y SPY, but the edge "
                "is purely trend-following with limited statistical power. Acceptable "
                "as a low-turnover sleeve; not a standalone alpha engine."
            ),
            "confidence": 0.62,
        },
        "Risk": {
            "position": "modify",
            "concerns": [
                "drawdown not capped",
                "no stop-loss",
                "single-asset concentration",
                "tail risk in 2020 COVID gap",
            ],
            "reasoning": (
                "Max drawdown is acceptable but unbounded by design. I require a "
                "volatility-targeted overlay and a hard stop at -10% peak-to-trough "
                "before approval. Single-name SPY concentration is fine for a sleeve "
                "but unacceptable as the only book."
            ),
            "confidence": 0.78,
        },
        "Compliance": {
            "position": "approve",
            "concerns": [
                "ensure best-execution venue routing",
                "verify suitability vs client mandate",
            ],
            "reasoning": (
                "Strategy trades a regulated ETF on lit venues with no manipulation "
                "vectors, no MNPI exposure, and no position-limit concerns at this "
                "size. Standard 17a-4 recordkeeping covers it. Compliant."
            ),
            "confidence": 0.9,
        },
    }

    def chat(self, model, messages, format=None, options=None):  # noqa: D401
        sys_prompt = messages[0]["content"]
        if "QUANTITATIVE" in sys_prompt:
            agent = "Quant"
        elif "RISK OFFICER" in sys_prompt:
            agent = "Risk"
        else:
            agent = "Compliance"
        payload = self._SCRIPT[agent]
        return {"message": {"content": json.dumps(payload)}}


@click.command()
@click.option("--start", default="2020-01-01", show_default=True)
@click.option("--end", default="2024-12-31", show_default=True)
@click.option("--fast", default=50, show_default=True, type=int)
@click.option("--slow", default=200, show_default=True, type=int)
@click.option(
    "--offline",
    is_flag=True,
    help="Use a deterministic mock LLM (no Ollama required).",
)
def main(start: str, end: str, fast: int, slow: int, offline: bool) -> None:
    docs = REPO / "docs"
    docs.mkdir(exist_ok=True)
    transcript = docs / "debate-transcript.jsonl"
    plot = docs / "equity-curve.png"
    data_cache = REPO / "data" / "spy.csv"

    if transcript.exists():
        transcript.unlink()

    agents = None
    if offline:
        agents = build_committee(client=_OfflineClient())

    verdict = run_pipeline(
        start=start,
        end=end,
        fast=fast,
        slow=slow,
        transcript_path=transcript,
        plot_path=plot,
        data_cache=data_cache,
        agents=agents,
    )

    bt = verdict.backtest.to_dict()["metrics"]
    click.echo("=" * 60)
    click.echo(f"Backtest: SPY {start} -> {end}, EMA({fast}/{slow})")
    click.echo(f"  total_return : {bt['total_return_pct']}%")
    click.echo(f"  buy & hold   : {bt['buy_hold_return_pct']}%")
    click.echo(f"  sharpe       : {bt['sharpe']}")
    click.echo(f"  max drawdown : {bt['max_drawdown_pct']}%")
    click.echo(f"  num trades   : {bt['num_trades']}")
    click.echo("-" * 60)
    click.echo(f"Verdict      : {verdict.debate.verdict}")
    click.echo(f"Vote tally   : {verdict.debate.vote}")
    click.echo(f"Dissent      : {verdict.debate.dissent}")
    click.echo(f"Transcript   : {transcript.relative_to(REPO)}")
    click.echo(f"Equity curve : {plot.relative_to(REPO)}")


if __name__ == "__main__":
    main()
