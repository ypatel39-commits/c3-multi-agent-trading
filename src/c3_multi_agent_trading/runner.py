"""End-to-end driver: backtest -> debate -> committee verdict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import Agent, build_committee
from .debate import DebateResult, run_debate
from .strategy import BacktestResult, load_spy, run_backtest


@dataclass
class CommitteeVerdict:
    """Bundles backtest stats + debate result for downstream consumers."""

    backtest: BacktestResult
    debate: DebateResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backtest": self.backtest.to_dict(),
            "debate": self.debate.to_dict(),
        }


def build_proposal(bt: BacktestResult) -> Dict[str, Any]:
    """Turn raw backtest stats into a proposal payload for the agents."""
    return {
        "name": "SPY momentum (50/200 EMA crossover)",
        "asset_class": "US equity ETF",
        "instrument": "SPY",
        "rules": "Long when EMA50 crosses above EMA200; flat otherwise.",
        "leverage": "1x cash",
        "position_sizing": "Full notional on signal",
        "transaction_cost": "10 bps per fill",
        "data_window": bt.to_dict()["period"],
        "backtest_metrics": bt.to_dict()["metrics"],
        "intended_capital": "$100,000",
        "client_mandate": "Discretionary US equity, daily liquidity",
    }


def run_pipeline(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    fast: int = 50,
    slow: int = 200,
    transcript_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
    data_cache: Optional[Path] = None,
    agents: Optional[List[Agent]] = None,
) -> CommitteeVerdict:
    """Full flow: load -> backtest -> debate."""
    df = load_spy(start=start, end=end, cache_path=data_cache)
    bt = run_backtest(df, fast=fast, slow=slow, plot_path=plot_path)
    proposal = build_proposal(bt)
    if agents is None:
        agents = build_committee()
    debate = run_debate(proposal, agents=agents, transcript_path=transcript_path)
    return CommitteeVerdict(backtest=bt, debate=debate)
