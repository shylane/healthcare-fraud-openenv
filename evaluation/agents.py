"""
Agent implementations for Option D evaluation experiments.

Four agents covering the spectrum from rule-based to context-aware LLM:

1. RandomAgent       — random valid decision, no reasoning
2. ThresholdAgent    — rule-based heuristics on claim features, no LLM
3. NaiveLLMAgent     — LLM with minimal prompt, no budget/memory awareness
4. BudgetAwareAgent  — LLM with explicit budget+memory reasoning in system prompt

All agents implement the Agent protocol from harness.py:
    agent.name: str
    agent.act(prompt: str) -> str
    agent.reset() -> None
"""

from __future__ import annotations

import random
import re
import time
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Random Agent — pure baseline, expected to score near-zero
# ---------------------------------------------------------------------------

class RandomAgent:
    """
    Picks a random valid decision each step. No reasoning, no API calls.
    Establishes the floor: a fair environment should score this poorly.
    """

    name = "RandomAgent"
    model = "none"

    DECISIONS = ["APPROVE", "FLAG_REVIEW", "INVESTIGATE", "DENY", "REQUEST_INFO"]
    # Bias toward APPROVE to roughly match fraud_rate=5% base rate
    WEIGHTED = (
        ["APPROVE"] * 50 + ["FLAG_REVIEW"] * 25 +
        ["INVESTIGATE"] * 15 + ["DENY"] * 5 + ["REQUEST_INFO"] * 5
    )

    def reset(self) -> None:
        pass

    def act(self, prompt: str) -> str:
        decision = random.choice(self.WEIGHTED)
        return (
            f"Decision: {decision}\n"
            f"Rationale: Random selection.\n"
            f"Evidence: N/A"
        )


# ---------------------------------------------------------------------------
# 2. Threshold Agent — rule-based, uses claim features visible in prompt
# ---------------------------------------------------------------------------

class ThresholdAgent:
    """
    Rule-based fraud detection using simple heuristics extracted from the
    plain-text prompt. No LLM — regex parses the formatted prompt fields.

    Rules:
    - Known fraud provider in memory → FLAG_REVIEW (don't waste investigation)
    - Risk level HIGH → INVESTIGATE (if budget allows) else FLAG_REVIEW
    - Risk level MODERATE → FLAG_REVIEW
    - Prior fraud flag rate > 10% → FLAG_REVIEW
    - Claim amount > 5× avg for provider → FLAG_REVIEW
    - Budget < 3 remaining → never INVESTIGATE, use FLAG_REVIEW
    - Default → APPROVE

    This agent CAN read budget and memory from the prompt but uses simple
    threshold logic rather than learned policy. It represents a well-tuned
    rule engine — the "strong heuristic" baseline.
    """

    name = "ThresholdAgent"
    model = "none"

    def reset(self) -> None:
        pass

    def act(self, prompt: str) -> str:
        decision, rationale, evidence = self._decide(prompt)
        return (
            f"Decision: {decision}\n"
            f"Rationale: {rationale}\n"
            f"Evidence: {evidence}"
        )

    def _decide(self, prompt: str) -> tuple[str, str, str]:
        # Parse budget remaining
        budget_remaining = self._extract_budget(prompt)
        # Parse risk level
        risk_level = self._extract_risk(prompt)
        # Check if current provider is a known fraud provider
        known_fraud = "KNOWN PROVIDER" in prompt and "FRAUD" in prompt
        known_legit = "KNOWN PROVIDER" in prompt and "LEGIT" in prompt
        # Parse fraud flag rate
        fraud_flag_rate = self._extract_fraud_flag_rate(prompt)
        # Parse claim amount vs avg
        amount_ratio = self._extract_amount_ratio(prompt)

        if known_legit:
            return (
                "APPROVE",
                "Provider previously investigated and found legitimate.",
                "Memory: provider flagged as LEGIT in investigation memory.",
            )

        if known_fraud:
            return (
                "FLAG_REVIEW",
                "Provider previously investigated and found fraudulent. "
                "Using FLAG_REVIEW to conserve investigation budget.",
                "Memory: provider flagged as FRAUD in investigation memory.",
            )

        if budget_remaining is not None and budget_remaining < 3:
            # Conserve budget — use FLAG_REVIEW for anything suspicious
            if risk_level in ("HIGH", "MODERATE") or fraud_flag_rate > 0.05:
                return (
                    "FLAG_REVIEW",
                    "Suspicious claim but investigation budget critically low — using FLAG_REVIEW.",
                    f"Budget remaining: {budget_remaining}. Risk: {risk_level}.",
                )
            return (
                "APPROVE",
                "Low risk. Budget conservation mode.",
                f"Budget remaining: {budget_remaining}. Risk: {risk_level}.",
            )

        if risk_level == "HIGH":
            action = "INVESTIGATE" if (budget_remaining is None or budget_remaining >= 5) else "FLAG_REVIEW"
            return (
                action,
                f"High risk claim. {action} warranted.",
                f"Risk assessment: HIGH. Fraud flag rate: {fraud_flag_rate:.1%}.",
            )

        if risk_level == "MODERATE" or fraud_flag_rate > 0.08:
            return (
                "FLAG_REVIEW",
                "Moderate risk. Flagging for manual review.",
                f"Risk assessment: {risk_level}. Fraud flag rate: {fraud_flag_rate:.1%}.",
            )

        if amount_ratio is not None and amount_ratio > 4.0:
            return (
                "FLAG_REVIEW",
                f"Claim amount is {amount_ratio:.1f}× provider average — unusual.",
                f"Amount ratio vs provider avg: {amount_ratio:.1f}×.",
            )

        return (
            "APPROVE",
            "Low risk. No significant fraud indicators detected.",
            f"Risk: LOW. Fraud flag rate: {fraud_flag_rate:.1%}.",
        )

    @staticmethod
    def _extract_budget(prompt: str) -> Optional[int]:
        m = re.search(r"Budget:\s*(\d+)/\d+\s+remaining", prompt)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_risk(prompt: str) -> str:
        if "Risk Level: HIGH" in prompt or "Risk Assessment**: HIGH" in prompt:
            return "HIGH"
        if "Risk Level: Moderate" in prompt or "Risk Assessment**: MODERATE" in prompt:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _extract_fraud_flag_rate(prompt: str) -> float:
        m = re.search(r"Prior Fraud Flags\*\*:\s*([\d.]+)%", prompt)
        if m:
            return float(m.group(1)) / 100
        return 0.0

    @staticmethod
    def _extract_amount_ratio(prompt: str) -> Optional[float]:
        """Compute claim_amount / avg_claim_amount from prompt."""
        claim_m = re.search(r"Billed Amount\*\*:\s*\$([\d,]+\.?\d*)", prompt)
        avg_m = re.search(r"Average Claim\*\*:\s*\$([\d,]+\.?\d*)", prompt)
        if claim_m and avg_m:
            claim = float(claim_m.group(1).replace(",", ""))
            avg = float(avg_m.group(1).replace(",", ""))
            if avg > 0:
                return claim / avg
        return None


# ---------------------------------------------------------------------------
# OpenRouter base — shared HTTP logic for LLM agents
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error
import json as _json


class OpenRouterBase:
    """
    Shared OpenRouter API logic with exponential-backoff retry on 429.
    Subclasses set system_prompt and model.

    Recommended models (Apr 2026):
      Free:
        qwen/qwen3.6-plus-preview:free          <- DEFAULT; Mar 30 2026, SOTA, 1M ctx
        minimax/minimax-m2.5:free               <- 197K ctx, BUT: mandatory <think> blocks
                                                   (see _strip_think) + OpenInference provider
                                                   (unreliable). Use with caution.
        openai/gpt-oss-120b:free                <- OpenInference provider; may be intermittent
      Skip:
        nvidia/llama-3.1-nemotron-ultra-253b-v1:free  <- knowledge cutoff Mar 2024, too old
      Paid (~$0.80 total for full experiment):
        deepseek/deepseek-v3.2                  <- $0.26/$0.38 per 1M, newest DeepSeek
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3.6-plus-preview:free"

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3.6-plus-preview:free",
        max_retries: int = 4,
        temperature: float = 0.3,
        max_tokens: int = 512,
        site_url: str = "https://github.com/shylane/healthcare-fraud-openenv",
        site_name: str = "healthcare-fraud-openenv",
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.site_url = site_url
        self.site_name = site_name
        self._call_count = 0

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    @staticmethod
    def _strip_think(text: str) -> str:
        """
        Remove <think>...</think> blocks injected by reasoning models (MiniMax M2.5,
        DeepSeek R1-style models). Prevents think-block content from being
        mistakenly parsed as the actual Decision/Rationale/Evidence output.
        """
        import re as _re
        return _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    def act(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = self._call_api(messages)
        return self._strip_think(raw)

    def _call_api(self, messages: list[dict]) -> str:
        payload = _json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.BASE_URL, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = _json.loads(resp.read())
                    self._call_count += 1
                    return data["choices"][0]["message"]["content"]

            except urllib.error.HTTPError as e:
                body = b""
                try:
                    body = e.read()
                except Exception:
                    pass
                body_str = body.decode(errors="replace")[:300]

                if e.code == 429:
                    wait = 2 ** (attempt + 1)   # 2, 4, 8, 16s
                    print(f"    [429] Rate limited. Waiting {wait}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                elif e.code == 503:
                    wait = 5 * (attempt + 1)
                    print(f"    [503] Unavailable. Waiting {wait}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                elif e.code == 404:
                    # Provider-side 404 ("Model not found") — no point retrying,
                    # model/endpoint is broken. Fail immediately.
                    print(f"    [404] Model not found on provider — switch model. {body_str}")
                    break
                elif e.code in (401, 403):
                    print(f"    [HTTP {e.code}] Auth error — check API key. {body_str}")
                    break
                else:
                    print(f"    [HTTP {e.code}] {body_str}")
                    if attempt < self.max_retries - 1:
                        time.sleep(3)
                        continue
                    break
            except Exception as ex:
                print(f"    [Error] {ex}")
                if attempt < self.max_retries - 1:
                    time.sleep(3)
                    continue
                break

        # Fallback after all retries exhausted
        return "Decision: FLAG_REVIEW\nRationale: API error — defaulting to flag for safety.\nEvidence: N/A"


# ---------------------------------------------------------------------------
# 3. Naive LLM Agent — capable model, no budget/memory awareness in prompt
# ---------------------------------------------------------------------------

class NaiveLLMAgent(OpenRouterBase):
    """
    LLM agent with a minimal system prompt — knows the task but NOT told
    to reason about budget or memory. Represents a well-prompted but
    context-blind agent.

    This is the control condition: same model as BudgetAwareAgent,
    different system prompt. Any score gap = value of budget reasoning.
    """

    def __init__(self, api_key: str, model: str = "qwen/qwen3.6-plus-preview:free", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.name = f"NaiveLLM({model.split('/')[-1]})"

    @property
    def system_prompt(self) -> str:
        return """You are a healthcare fraud detection specialist reviewing insurance claims.

For each claim, analyze the data and respond with EXACTLY this format:
Decision: [APPROVE|FLAG_REVIEW|INVESTIGATE|DENY|REQUEST_INFO]
Rationale: [2-3 sentences explaining your reasoning]
Evidence: [specific data points from the claim that influenced your decision]

Decision guidelines:
- APPROVE: claim appears legitimate, no suspicious patterns
- FLAG_REVIEW: some indicators warrant manual review
- INVESTIGATE: strong fraud indicators, formal investigation needed
- DENY: clear fraud, deny the claim
- REQUEST_INFO: need more documentation before deciding

Be concise. Always include all three sections."""


# ---------------------------------------------------------------------------
# 4. Budget-Aware Agent — same model, explicit budget+memory reasoning
# ---------------------------------------------------------------------------

class BudgetAwareAgent(OpenRouterBase):
    """
    LLM agent with a system prompt that explicitly instructs the model to:
    1. Track and conserve investigation budget
    2. Leverage investigation memory to avoid re-investigating known providers
    3. Reason about the tradeoff between budget use and fraud detection

    This is the experimental condition. Score gap vs NaiveLLMAgent measures
    whether the model can exploit the multi-step structure when told to.
    """

    def __init__(self, api_key: str, model: str = "qwen/qwen3.6-plus-preview:free", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.name = f"BudgetAware({model.split('/')[-1]})"

    @property
    def system_prompt(self) -> str:
        return """You are a healthcare fraud detection specialist reviewing insurance claims.
You have LIMITED investigation resources: typically 15 investigations per 100 claims.

CRITICAL RULES FOR RESOURCE MANAGEMENT:
1. INVESTIGATE is expensive — use it only for high-confidence fraud with no prior investigation
2. If a provider is in Active Memory as FRAUD → use FLAG_REVIEW (saves your investigation budget)
3. If a provider is in Active Memory as LEGIT → use APPROVE (no need to re-investigate)
4. When Budget remaining < 4, NEVER use INVESTIGATE — use FLAG_REVIEW instead
5. When Budget remaining < 8, prefer FLAG_REVIEW over INVESTIGATE for moderate-risk claims

DECISION STRATEGY:
- Budget > 8 AND Risk HIGH AND provider NOT in memory → INVESTIGATE
- Provider in memory as FRAUD → FLAG_REVIEW (budget-conserving)
- Provider in memory as LEGIT → APPROVE
- Budget ≤ 4 AND any suspicion → FLAG_REVIEW
- Risk MODERATE OR fraud flags > 5% → FLAG_REVIEW
- Risk LOW AND no flags → APPROVE

Always respond with EXACTLY this format:
Decision: [APPROVE|FLAG_REVIEW|INVESTIGATE|DENY|REQUEST_INFO]
Rationale: [2-3 sentences — reference budget level and memory if relevant]
Evidence: [specific data points: amount, provider ID, risk score, budget status]"""


# ---------------------------------------------------------------------------
# 5. DeepSeek V3 agents (paid — gold standard comparison)
# ---------------------------------------------------------------------------

class DeepSeekNaiveAgent(NaiveLLMAgent):
    """Naive LLM agent using DeepSeek V3.2 as the backbone."""
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            api_key=api_key,
            model="deepseek/deepseek-v3.2",
            **kwargs
        )
        self.name = "NaiveLLM(deepseek-v3.2)"


class DeepSeekBudgetAwareAgent(BudgetAwareAgent):
    """Budget-aware agent using DeepSeek V3.2 as the backbone."""
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            api_key=api_key,
            model="deepseek/deepseek-v3.2",
            **kwargs
        )
        self.name = "BudgetAware(deepseek-v3.2)"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_agents(api_key: str, model: str = "openai/gpt-oss-120b:free") -> list:
    """Return all 4 agents for a full experiment run."""
    return [
        RandomAgent(),
        ThresholdAgent(),
        NaiveLLMAgent(api_key=api_key, model=model),
        BudgetAwareAgent(api_key=api_key, model=model),
    ]
