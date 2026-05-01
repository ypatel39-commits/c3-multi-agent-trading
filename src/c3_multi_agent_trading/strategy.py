"""SPY 50/200 EMA-crossover momentum strategy on backtesting.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:  # heavy deps — optional at import time for tests
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except Exception:  # pragma: no cover
    Backtest = None  # type: ignore[assignment]
    Strategy = object  # type: ignore[assignment]

    def crossover(a: Any, b: Any) -> bool:  # type: ignore[misc]
        return False


try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore[assignment]


SEED = 42


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


class EmaCross(Strategy):  # type: ignore[misc]
    """Long-only 50/200 EMA crossover. Long on golden cross, flat on death cross."""

    fast = 50
    slow = 200

    def init(self) -> None:
        close = pd.Series(self.data.Close)
        self.ema_fast = self.I(_ema, close, self.fast)
        self.ema_slow = self.I(_ema, close, self.slow)

    def next(self) -> None:
        if crossover(self.ema_fast, self.ema_slow):
            self.position.close()
            self.buy()
        elif crossover(self.ema_slow, self.ema_fast):
            self.position.close()


@dataclass
class BacktestResult:
    """Trimmed backtest summary suitable for sending to LLM agents."""

    symbol: str
    start: str
    end: str
    fast: int
    slow: int
    total_return_pct: float
    buy_hold_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    final_equity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "period": f"{self.start} to {self.end}",
            "params": {"fast_ema": self.fast, "slow_ema": self.slow},
            "metrics": {
                "total_return_pct": round(self.total_return_pct, 2),
                "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
                "sharpe": round(self.sharpe, 3),
                "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                "win_rate_pct": round(self.win_rate_pct, 2),
                "num_trades": self.num_trades,
                "final_equity": round(self.final_equity, 2),
            },
        }


def load_spy(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Pull SPY OHLCV from yfinance (cached on disk if `cache_path` given)."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return _prep(df)
    if yf is None:
        raise RuntimeError("yfinance not installed; cannot fetch SPY")
    df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = _prep(df)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    return df


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in needed if c in df.columns]]
    return df.dropna()


def run_backtest(
    df: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    cash: float = 100_000.0,
    commission: float = 0.001,
    plot_path: Optional[Path] = None,
) -> BacktestResult:
    """Run the EMA-cross backtest and return a compact result."""
    if Backtest is None:
        raise RuntimeError("backtesting.py not installed")

    np.random.seed(SEED)

    class _Strat(EmaCross):
        pass

    _Strat.fast = fast
    _Strat.slow = slow

    bt = Backtest(df, _Strat, cash=cash, commission=commission)
    stats = bt.run()

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _save_plot(df, stats, plot_path)
        except Exception:  # pragma: no cover - matplotlib edge cases
            pass

    return BacktestResult(
        symbol="SPY",
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        fast=fast,
        slow=slow,
        total_return_pct=float(stats.get("Return [%]", 0.0)),
        buy_hold_return_pct=float(stats.get("Buy & Hold Return [%]", 0.0)),
        sharpe=float(stats.get("Sharpe Ratio", 0.0) or 0.0),
        max_drawdown_pct=float(stats.get("Max. Drawdown [%]", 0.0)),
        win_rate_pct=float(stats.get("Win Rate [%]", 0.0) or 0.0),
        num_trades=int(stats.get("# Trades", 0)),
        final_equity=float(stats.get("Equity Final [$]", cash)),
    )


def _save_plot(df: pd.DataFrame, stats: Any, path: Path) -> None:
    """Render an equity curve PNG via matplotlib (no GUI required)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    equity = stats.get("_equity_curve")
    if equity is None or len(equity) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity["Equity"], label="Strategy Equity", linewidth=1.5)
    ax.set_title("C3 Momentum Strategy — SPY 50/200 EMA Crossover")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
