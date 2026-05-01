"""Shared fixtures: deterministic mock Ollama client."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


class MockOllamaClient:
    """Stand-in for ollama.Client. Returns canned JSON keyed by system prompt."""

    def __init__(self, scripts: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.scripts: Dict[str, Dict[str, Any]] = scripts or _DEFAULT_SCRIPTS

    def chat(self, model, messages, format=None, options=None):  # noqa: D401
        self.calls.append(
            {"model": model, "messages": messages, "format": format, "options": options}
        )
        sys_prompt = messages[0]["content"]
        agent = _classify(sys_prompt)
        payload = self.scripts[agent]
        return {"message": {"content": json.dumps(payload)}}


def _classify(system_prompt: str) -> str:
    if "QUANTITATIVE" in system_prompt:
        return "Quant"
    if "RISK OFFICER" in system_prompt:
        return "Risk"
    return "Compliance"


_DEFAULT_SCRIPTS: Dict[str, Dict[str, Any]] = {
    "Quant": {
        "position": "approve",
        "concerns": ["modest sharpe"],
        "reasoning": "Statistical edge is real but small.",
        "confidence": 0.65,
    },
    "Risk": {
        "position": "modify",
        "concerns": ["unbounded drawdown"],
        "reasoning": "Need vol-target overlay and hard stop.",
        "confidence": 0.8,
    },
    "Compliance": {
        "position": "approve",
        "concerns": [],
        "reasoning": "Liquid ETF, no manipulation vectors.",
        "confidence": 0.95,
    },
}


@pytest.fixture
def mock_client() -> MockOllamaClient:
    return MockOllamaClient()


@pytest.fixture
def reject_client() -> MockOllamaClient:
    """All three agents reject — used to test risk-averse verdict."""
    scripts = {
        name: {
            "position": "reject",
            "concerns": ["fatal"],
            "reasoning": "no",
            "confidence": 0.9,
        }
        for name in ("Quant", "Risk", "Compliance")
    }
    return MockOllamaClient(scripts=scripts)
