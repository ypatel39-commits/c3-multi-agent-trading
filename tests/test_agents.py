"""Agent-level tests with mocked Ollama."""

from __future__ import annotations

import json

from c3_multi_agent_trading.agents import (
    AGENT_PROMPTS,
    Agent,
    build_committee,
)


def test_three_distinct_system_prompts():
    """Each agent role must carry a unique system prompt."""
    prompts = list(AGENT_PROMPTS.values())
    assert len(set(prompts)) == 3
    assert "QUANTITATIVE" in AGENT_PROMPTS["Quant"]
    assert "RISK OFFICER" in AGENT_PROMPTS["Risk"]
    assert "COMPLIANCE" in AGENT_PROMPTS["Compliance"]


def test_agent_parses_clean_json(mock_client):
    agent = Agent(
        "Quant", AGENT_PROMPTS["Quant"], client=mock_client
    )
    reply = agent.respond("Test proposal")
    assert reply.agent == "Quant"
    assert reply.position == "approve"
    assert 0.0 <= reply.confidence <= 1.0
    assert isinstance(reply.concerns, list)


def test_agent_handles_malformed_json():
    class BadClient:
        def chat(self, **_):
            return {"message": {"content": "not json at all"}}

    agent = Agent("Risk", AGENT_PROMPTS["Risk"], client=BadClient())
    reply = agent.respond("x")
    # Falls back to safe default
    assert reply.position in {"approve", "reject", "modify"}
    assert reply.confidence == 0.0


def test_agent_clamps_confidence():
    payload = {
        "position": "approve",
        "concerns": [],
        "reasoning": "ok",
        "confidence": 5.0,  # absurd, must clamp to 1.0
    }

    class ClampClient:
        def chat(self, **_):
            return {"message": {"content": json.dumps(payload)}}

    agent = Agent("Quant", AGENT_PROMPTS["Quant"], client=ClampClient())
    assert agent.respond("x").confidence == 1.0


def test_build_committee_size_and_roles(mock_client):
    committee = build_committee(client=mock_client)
    names = [a.name for a in committee]
    assert names == ["Quant", "Risk", "Compliance"]
    assert all(isinstance(a, Agent) for a in committee)
