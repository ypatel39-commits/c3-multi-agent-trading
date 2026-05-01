"""Strategy/backtest tests using a synthetic OHLC frame (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c3_multi_agent_trading.strategy import run_backtest


def _synthetic_spy(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic trending random-walk for testing."""
    rng = np.random.default_rng(seed)
    drift = 0.0004
    vol = 0.011
    rets = rng.normal(drift, vol, n_days)
    price = 300.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    high = price * (1 + rng.uniform(0, 0.005, n_days))
    low = price * (1 - rng.uniform(0, 0.005, n_days))
    open_ = price * (1 + rng.uniform(-0.002, 0.002, n_days))
    vol_ = rng.integers(50_000_000, 150_000_000, n_days)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": price,
            "Volume": vol_,
        },
        index=idx,
    )


@pytest.mark.skipif(
    pytest.importorskip("backtesting", reason="backtesting.py not installed") is None,
    reason="backtesting.py not installed",
)
def test_backtest_returns_compact_metrics():
    df = _synthetic_spy()
    result = run_backtest(df, fast=50, slow=200)
    metrics = result.to_dict()["metrics"]
    assert {"sharpe", "max_drawdown_pct", "total_return_pct", "num_trades"} <= set(
        metrics
    )
    assert result.num_trades >= 0
    assert -100.0 <= result.max_drawdown_pct <= 0.0
    assert result.fast == 50 and result.slow == 200
