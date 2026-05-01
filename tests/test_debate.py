"""Debate orchestrator tests with mocked Ollama."""

from __future__ import annotations

import json
from pathlib import Path

from c3_multi_agent_trading.agents import build_committee
from c3_multi_agent_trading.debate import run_debate


PROPOSAL = {"name": "test strategy", "rules": "buy and hold"}


def test_debate_runs_two_rounds_and_logs_jsonl(tmp_path: Path, mock_client):
    transcript = tmp_path / "debate.jsonl"
    committee = build_committee(client=mock_client)
    result = run_debate(PROPOSAL, agents=committee, transcript_path=transcript)

    # 3 agents x 2 rounds = 6 turns
    assert len(result.turns) == 6
    assert {t.round for t in result.turns} == {1, 2}
    assert {t.agent for t in result.turns} == {"Quant", "Risk", "Compliance"}

    lines = transcript.read_text().strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert events[-1]["event"] == "verdict"
    assert sum(1 for e in events if e["event"] == "turn") == 6


def test_debate_majority_vote_records_dissent(mock_client):
    """Default fixture: 2 approve (Quant, Compliance) + 1 modify (Risk).
    Verdict = APPROVED, Risk dissents."""
    committee = build_committee(client=mock_client)
    result = run_debate(PROPOSAL, agents=committee)
    assert result.verdict == "APPROVED"
    assert result.vote["approve"] == 2
    assert "Risk" in result.dissent


def test_debate_unanimous_reject(reject_client):
    committee = build_committee(client=reject_client)
    result = run_debate(PROPOSAL, agents=committee)
    assert result.verdict == "REJECTED"
    assert result.vote["reject"] == 3
    assert result.dissent == []


def test_debate_round2_includes_peer_concerns(mock_client):
    committee = build_committee(client=mock_client)
    run_debate(PROPOSAL, agents=committee)
    # 6 calls total: 3 round-1 + 3 round-2
    assert len(mock_client.calls) == 6
    # Round-2 user prompts should mention ROUND 2
    round2_user_prompts = [c["messages"][1]["content"] for c in mock_client.calls[3:]]
    for p in round2_user_prompts:
        assert "ROUND 2" in p
