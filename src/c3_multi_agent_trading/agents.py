"""Three LLM agents (Quant, Risk, Compliance) backed by local Ollama.

Each agent is the same model (qwen2.5:7b) but carries a distinct system
prompt that shapes its mandate. Every call is requested in JSON format so
downstream orchestration can vote/parse deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import ollama
except Exception:  # pragma: no cover - import guard
    ollama = None  # type: ignore[assignment]


DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_HOST = "http://localhost:11434"


QUANT_PROMPT = """You are a senior QUANTITATIVE RESEARCHER on a trading committee.
Mandate: judge proposals on statistical edge ONLY.
Focus areas: Sharpe ratio, alpha vs benchmark, total return, hit rate,
information coefficient, regime stability, capacity, and overfit risk.
Be skeptical of in-sample-only results. Reward strategies with Sharpe > 1.0
and statistically significant edge. Punish data-mined parameter sweeps.
You MUST respond in valid JSON with keys:
  position: "approve" | "reject" | "modify"
  concerns: array of short strings
  reasoning: one paragraph
  confidence: float 0..1
"""

RISK_PROMPT = """You are the CHIEF RISK OFFICER on a trading committee.
Mandate: protect capital. Judge proposals on downside ONLY.
Focus areas: max drawdown, daily VaR (95/99%), CVaR / tail risk, volatility
clustering, leverage, concentration risk, liquidity in stress, gap risk,
correlation breakdowns. Veto strategies with max drawdown > 25% or
unbounded tail exposure. Demand position sizing and stop-loss discipline.
You MUST respond in valid JSON with keys:
  position: "approve" | "reject" | "modify"
  concerns: array of short strings
  reasoning: one paragraph
  confidence: float 0..1
"""

COMPLIANCE_PROMPT = """You are the HEAD OF COMPLIANCE on a trading committee.
Mandate: enforce regulatory boundaries. Judge proposals on legality and
fiduciary fitness ONLY.
Focus areas: SEC Rule 10b-5 anti-fraud, market manipulation (spoofing,
layering, marking the close), CFTC position limits for futures, Reg SHO
short-sale rules, Reg NMS, wash-trade prohibitions, suitability vs client
mandate, MNPI / insider information, recordkeeping (17a-4), best execution.
Veto any strategy that resembles manipulation or breaches position limits.
You MUST respond in valid JSON with keys:
  position: "approve" | "reject" | "modify"
  concerns: array of short strings
  reasoning: one paragraph
  confidence: float 0..1
"""


AGENT_PROMPTS: Dict[str, str] = {
    "Quant": QUANT_PROMPT,
    "Risk": RISK_PROMPT,
    "Compliance": COMPLIANCE_PROMPT,
}


@dataclass
class AgentResponse:
    """Structured agent reply from one debate turn."""

    agent: str
    position: str  # approve | reject | modify
    concerns: List[str]
    reasoning: str
    confidence: float
    raw: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "position": self.position,
            "concerns": self.concerns,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


class Agent:
    """One committee member backed by Ollama."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        client: Any = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.host = host
        if client is not None:
            self.client = client
        elif ollama is not None:
            self.client = ollama.Client(host=host)
        else:
            self.client = None

    def respond(self, user_message: str) -> AgentResponse:
        """Send `user_message` to Ollama and parse the JSON reply."""
        if self.client is None:
            raise RuntimeError(
                "ollama package not installed; pass a mock client for testing"
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        result = self.client.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={"temperature": 0.4, "seed": 42},
        )
        content = _extract_content(result)
        return self._parse(content)

    def _parse(self, content: str) -> AgentResponse:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {
                "position": "modify",
                "concerns": ["malformed JSON from model"],
                "reasoning": content[:500],
                "confidence": 0.0,
            }
        position = str(data.get("position", "modify")).lower().strip()
        if position not in {"approve", "reject", "modify"}:
            position = "modify"
        concerns = data.get("concerns", []) or []
        if isinstance(concerns, str):
            concerns = [concerns]
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return AgentResponse(
            agent=self.name,
            position=position,
            concerns=[str(c) for c in concerns],
            reasoning=str(data.get("reasoning", "")),
            confidence=max(0.0, min(1.0, confidence)),
            raw=content,
        )


def _extract_content(result: Any) -> str:
    """Best-effort extraction of message content from an Ollama reply."""
    if isinstance(result, dict):
        msg = result.get("message") or {}
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        if "response" in result:
            return str(result["response"])
    msg = getattr(result, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is not None:
            return str(content)
    return str(result)


def build_committee(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    client: Optional[Any] = None,
) -> List[Agent]:
    """Construct the canonical 3-agent committee."""
    return [
        Agent("Quant", QUANT_PROMPT, model=model, host=host, client=client),
        Agent("Risk", RISK_PROMPT, model=model, host=host, client=client),
        Agent(
            "Compliance",
            COMPLIANCE_PROMPT,
            model=model,
            host=host,
            client=client,
        ),
    ]
