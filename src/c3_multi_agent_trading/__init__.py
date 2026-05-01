"""C3 Multi-Agent Trading Research System.

Three local LLM agents (Quant, Risk, Compliance) debate trading strategy
proposals, vote, and log dissent + reasoning chain. Backtest powered by
backtesting.py on real SPY data via yfinance. LLM inference uses local
Ollama (qwen2.5:7b). No API keys required.
"""

__version__ = "0.1.0"
__all__ = ["agents", "debate", "strategy", "runner"]
