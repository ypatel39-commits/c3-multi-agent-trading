"""Two-round debate orchestrator with vote tally and JSONL transcript."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents import Agent, AgentResponse, build_committee


@dataclass
class DebateTurn:
    """One agent's contribution at a given round."""

    round: int
    agent: str
    position: str
    concerns: List[str]
    reasoning: str
    confidence: float


@dataclass
class DebateResult:
    """Full debate output: transcript, vote tally, final verdict."""

    proposal: Dict[str, Any]
    turns: List[DebateTurn] = field(default_factory=list)
    verdict: str = "pending"
    vote: Dict[str, int] = field(default_factory=dict)
    dissent: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal,
            "turns": [asdict(t) for t in self.turns],
            "verdict": self.verdict,
            "vote": self.vote,
            "dissent": self.dissent,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _format_proposal(proposal: Dict[str, Any]) -> str:
    return (
        "STRATEGY PROPOSAL UNDER REVIEW\n"
        "==============================\n"
        f"{json.dumps(proposal, indent=2, default=str)}\n"
    )


def _format_round1_prompt(proposal: Dict[str, Any]) -> str:
    return (
        _format_proposal(proposal)
        + "\nROUND 1: From your role's mandate, list concerns and take an "
        "initial position (approve/reject/modify). Reply in JSON."
    )


def _format_round2_prompt(
    proposal: Dict[str, Any],
    others: List[AgentResponse],
    self_name: str,
) -> str:
    peer_block = "\n\n".join(
        f"-- {r.agent} (round 1) --\n"
        f"position={r.position}, confidence={r.confidence:.2f}\n"
        f"concerns={r.concerns}\n"
        f"reasoning={r.reasoning}"
        for r in others
        if r.agent != self_name
    )
    return (
        _format_proposal(proposal)
        + "\nROUND 2: Your peers have spoken. Read their concerns below, "
        "then issue your FINAL vote (approve/reject/modify). You may revise.\n\n"
        f"{peer_block}\n\nReply in JSON."
    )


def _tally(round2: List[AgentResponse]) -> Dict[str, int]:
    tally = {"approve": 0, "reject": 0, "modify": 0}
    for r in round2:
        tally[r.position] = tally.get(r.position, 0) + 1
    return tally


def _verdict(tally: Dict[str, int]) -> str:
    """Majority rule. Reject wins ties over approve (risk-averse)."""
    if tally.get("reject", 0) >= 2:
        return "REJECTED"
    if tally.get("approve", 0) >= 2:
        return "APPROVED"
    if tally.get("modify", 0) >= 2:
        return "MODIFY"
    # No majority — fall back to risk-averse
    if tally.get("reject", 0) >= 1:
        return "REJECTED"
    return "MODIFY"


def run_debate(
    proposal: Dict[str, Any],
    agents: Optional[List[Agent]] = None,
    transcript_path: Optional[Path] = None,
) -> DebateResult:
    """Run a two-round committee debate and return the full result.

    If `transcript_path` is given, every turn plus the final verdict is
    appended as one JSON object per line (JSONL).
    """
    if agents is None:
        agents = build_committee()

    result = DebateResult(proposal=proposal)
    sink = _open_sink(transcript_path)

    # Round 1
    r1_prompt = _format_round1_prompt(proposal)
    round1: List[AgentResponse] = []
    for agent in agents:
        reply = agent.respond(r1_prompt)
        round1.append(reply)
        turn = DebateTurn(
            round=1,
            agent=reply.agent,
            position=reply.position,
            concerns=reply.concerns,
            reasoning=reply.reasoning,
            confidence=reply.confidence,
        )
        result.turns.append(turn)
        _write(sink, {"event": "turn", **asdict(turn)})

    # Round 2 (each agent sees peers' round-1 statements)
    round2: List[AgentResponse] = []
    for agent in agents:
        r2_prompt = _format_round2_prompt(proposal, round1, agent.name)
        reply = agent.respond(r2_prompt)
        round2.append(reply)
        turn = DebateTurn(
            round=2,
            agent=reply.agent,
            position=reply.position,
            concerns=reply.concerns,
            reasoning=reply.reasoning,
            confidence=reply.confidence,
        )
        result.turns.append(turn)
        _write(sink, {"event": "turn", **asdict(turn)})

    tally = _tally(round2)
    verdict = _verdict(tally)
    majority = verdict.lower().rstrip("ed")  # APPROVED -> approv (rough)
    # Identify dissent against the verdict
    winning_position = {
        "APPROVED": "approve",
        "REJECTED": "reject",
        "MODIFY": "modify",
    }.get(verdict, "modify")
    dissent = [r.agent for r in round2 if r.position != winning_position]

    result.vote = tally
    result.verdict = verdict
    result.dissent = dissent
    result.finished_at = time.time()

    _write(
        sink,
        {
            "event": "verdict",
            "verdict": verdict,
            "vote": tally,
            "dissent": dissent,
            "duration_sec": result.finished_at - result.started_at,
        },
    )
    _close(sink)
    return result


# --- transcript helpers ----------------------------------------------------

def _open_sink(path: Optional[Path]):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def _write(sink, payload: Dict[str, Any]) -> None:
    if sink is None:
        return
    sink.write(json.dumps(payload, default=str) + "\n")
    sink.flush()


def _close(sink) -> None:
    if sink is not None:
        sink.close()
