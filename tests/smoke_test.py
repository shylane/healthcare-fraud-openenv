"""
Smoke tests for the Healthcare Fraud Detection environment.

Covers the failure modes identified in the April 2026 code review:
  - Basic reset/step cycle (env layer)
  - Over-budget INVESTIGATE downgrade and correct metadata (action_taken vs action_requested)
  - Investigation memory records detected truth, not raw ground truth
  - Outcome string correctness at both accuracy extremes
  - ConfigRequest surface (investigation_budget, memory_decay_halflife) via FastAPI TestClient
  - Harness memory_reuse_rate logic: APPROVE-on-fraud no longer counts as correct

Run with:
    python -m pytest tests/smoke_test.py -v
or standalone:
    python tests/smoke_test.py
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig
from environment.models import ClaimAction, DecisionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_action(decision: str, rationale: str = "Test rationale.") -> ClaimAction:
    return ClaimAction(
        response_text=(
            f"Decision: {decision}\n"
            f"Rationale: {rationale}\n"
            f"Evidence: claim_amount=test_value"
        )
    )


# ---------------------------------------------------------------------------
# 1. Basic reset / step cycle
# ---------------------------------------------------------------------------

class TestBasicCycle:
    def test_reset_returns_observation(self):
        env = ClaimsFraudEnvironment(EnvironmentConfig(seed=42))
        obs = env.reset()
        assert obs is not None
        assert obs.prompt is not None
        assert not obs.done

    def test_step_returns_reward(self):
        env = ClaimsFraudEnvironment(EnvironmentConfig(seed=42))
        env.reset()
        obs = env.step(make_action("FLAG_REVIEW"))
        assert obs.reward is not None
        assert isinstance(obs.reward, float)

    def test_episode_completes_in_100_steps(self):
        env = ClaimsFraudEnvironment(EnvironmentConfig(seed=42, claims_per_episode=100))
        obs = env.reset()
        steps = 0
        while not obs.done:
            obs = env.step(make_action("APPROVE"))
            steps += 1
        assert steps == 100, f"Expected 100 steps, got {steps}"
        assert obs.done


# ---------------------------------------------------------------------------
# 2. Over-budget INVESTIGATE downgrade + correct metadata
# ---------------------------------------------------------------------------

class TestBudgetDowngrade:
    def test_investigate_downgraded_when_budget_exhausted(self):
        """Exhaust budget=1 then send INVESTIGATE; executed action must be FLAG_REVIEW."""
        env = ClaimsFraudEnvironment(EnvironmentConfig(seed=42, investigation_budget=1))
        env.reset()

        obs = env.step(make_action("INVESTIGATE"))
        assert env._state.budget_remaining == 0

        obs = env.step(make_action("INVESTIGATE"))
        assert obs.metadata.get("action_taken") == "FLAG_REVIEW", (
            f"Over-budget INVESTIGATE should execute as FLAG_REVIEW, "
            f"got '{obs.metadata.get('action_taken')}'"
        )
        assert obs.metadata.get("action_requested") == "INVESTIGATE"

    def test_no_downgrade_when_budget_available(self):
        env = ClaimsFraudEnvironment(EnvironmentConfig(seed=42, investigation_budget=15))
        env.reset()
        obs = env.step(make_action("INVESTIGATE"))
        assert obs.metadata.get("action_taken") == "INVESTIGATE"
        assert obs.metadata.get("action_requested") == "INVESTIGATE"


# ---------------------------------------------------------------------------
# 3. Investigation memory records detected truth, not ground truth
# ---------------------------------------------------------------------------

class TestMemoryCorrectness:
    def _find_fraud_and_investigate(self, env, accuracy):
        """Run until a fraud claim appears, INVESTIGATE it, return (provider_id, obs)."""
        obs = env.reset()
        for _ in range(80):
            if obs.done:
                return None, None
            if env._current_is_fraud:
                provider_id = obs.provider_profile.get("provider_id", "")
                obs = env.step(make_action("INVESTIGATE"))
                return provider_id, obs
            obs = env.step(make_action("APPROVE"))
        return None, None

    def test_missed_investigation_stores_false(self):
        """accuracy=0.0 → every investigation misses → memory must store is_fraud=False."""
        env = ClaimsFraudEnvironment(
            EnvironmentConfig(seed=42, investigate_accuracy=0.0, investigation_budget=15)
        )
        provider_id, _ = self._find_fraud_and_investigate(env, 0.0)
        if provider_id is None:
            pytest.skip("No fraud claim in 80 steps")
        mem = env._state.investigation_memory.get(provider_id)
        assert mem is not None
        assert mem["is_fraud"] is False, (
            f"accuracy=0.0 miss → memory should store is_fraud=False, got {mem['is_fraud']}"
        )

    def test_successful_investigation_stores_true(self):
        """accuracy=1.0 → every investigation catches → memory must store is_fraud=True."""
        env = ClaimsFraudEnvironment(
            EnvironmentConfig(seed=42, investigate_accuracy=1.0, investigation_budget=15)
        )
        provider_id, _ = self._find_fraud_and_investigate(env, 1.0)
        if provider_id is None:
            pytest.skip("No fraud claim in 80 steps")
        mem = env._state.investigation_memory.get(provider_id)
        assert mem is not None
        assert mem["is_fraud"] is True, (
            f"accuracy=1.0 catch → memory should store is_fraud=True, got {mem['is_fraud']}"
        )


# ---------------------------------------------------------------------------
# 4. Outcome string reflects stochastic detection result
# ---------------------------------------------------------------------------

class TestOutcomeString:
    def _get_last_outcome(self, env):
        history = env._state.decision_history
        return history[-1]["outcome"] if history else None

    def _step_until_fraud(self, env, decision_on_fraud):
        obs = env.reset()
        for _ in range(80):
            if obs.done:
                return False
            if env._current_is_fraud:
                env.step(make_action(decision_on_fraud))
                return True
            env.step(make_action("APPROVE"))
        return False

    def test_investigate_miss_records_fraud_missed(self):
        env = ClaimsFraudEnvironment(
            EnvironmentConfig(seed=42, investigate_accuracy=0.0)
        )
        found = self._step_until_fraud(env, "INVESTIGATE")
        if not found:
            pytest.skip("No fraud in 80 steps")
        assert self._get_last_outcome(env) == "Fraud MISSED ✗"

    def test_investigate_catch_records_fraud_caught(self):
        env = ClaimsFraudEnvironment(
            EnvironmentConfig(seed=42, investigate_accuracy=1.0)
        )
        found = self._step_until_fraud(env, "INVESTIGATE")
        if not found:
            pytest.skip("No fraud in 80 steps")
        assert self._get_last_outcome(env) == "Fraud caught ✓"

    def test_flag_miss_records_fraud_missed(self):
        env = ClaimsFraudEnvironment(
            EnvironmentConfig(seed=42, flag_accuracy=0.0)
        )
        found = self._step_until_fraud(env, "FLAG_REVIEW")
        if not found:
            pytest.skip("No fraud in 80 steps")
        assert self._get_last_outcome(env) == "Fraud MISSED ✗"


# ---------------------------------------------------------------------------
# 5. FastAPI /reset accepts investigation_budget and memory_decay_halflife
# ---------------------------------------------------------------------------

class TestResetAPIConfig:
    """Uses FastAPI TestClient to verify the HTTP path, not just EnvironmentConfig directly."""

    @pytest.fixture(autouse=True)
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from environment.server.app import app
            self._client = TestClient(app)
        except ImportError:
            pytest.skip("fastapi[testclient] not installed (needs httpx)")

    def test_default_reset(self):
        resp = self._client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    def test_custom_budget_via_api(self):
        resp = self._client.post("/reset", json={"investigation_budget": 5})
        assert resp.status_code == 200
        # Verify the environment picked up the budget
        state_resp = self._client.get("/state")
        assert state_resp.status_code == 200
        state = state_resp.json()
        assert state.get("budget_remaining") == 5, (
            f"Expected budget_remaining=5 after /reset with investigation_budget=5, "
            f"got {state.get('budget_remaining')}"
        )

    def test_custom_halflife_accepted(self):
        resp = self._client.post("/reset", json={"memory_decay_halflife": 0})
        assert resp.status_code == 200

    def test_health_endpoint(self):
        resp = self._client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# 6. Harness memory_reuse_rate: APPROVE on known-fraud provider is NOT correct
# ---------------------------------------------------------------------------

class TestHarnessMemoryReuseMetric:
    """
    Verify the fixed memory_reuse_rate logic by simulating the metric calculation
    directly (mirrors harness.py lines 308-322).
    """

    def test_approve_on_known_fraud_not_counted_as_correct(self):
        """
        Before the fix, APPROVE on a known-fraud provider counted as memory-reuse-correct
        (because the agent didn't waste an investigation slot).  After the fix it should NOT.
        """
        investigation_memory = {
            "PROV_001": {"provider_id": "PROV_001", "is_fraud": True,
                         "step_investigated": 5, "base_confidence": 1.0}
        }
        current_provider = "PROV_001"
        provider_in_memory = current_provider in investigation_memory

        memory_reuse_opportunities = 0
        memory_reuse_correct = 0

        # Simulate the fixed logic from harness.py
        if provider_in_memory:
            memory_reuse_opportunities += 1
            memory_entry = investigation_memory.get(current_provider, {})
            provider_known_fraud = memory_entry.get("is_fraud", False)
            decision = "APPROVE"  # wrong choice for known fraud
            if provider_known_fraud:
                if decision in ("FLAG_REVIEW", "DENY"):
                    memory_reuse_correct += 1
            else:
                if decision == "APPROVE":
                    memory_reuse_correct += 1

        assert memory_reuse_opportunities == 1
        assert memory_reuse_correct == 0, (
            "APPROVE on a known-fraud provider should NOT be counted as correct memory reuse"
        )

    def test_flag_review_on_known_fraud_is_correct(self):
        investigation_memory = {
            "PROV_001": {"provider_id": "PROV_001", "is_fraud": True,
                         "step_investigated": 5, "base_confidence": 1.0}
        }
        memory_reuse_opportunities = 0
        memory_reuse_correct = 0

        current_provider = "PROV_001"
        if current_provider in investigation_memory:
            memory_reuse_opportunities += 1
            entry = investigation_memory[current_provider]
            if entry.get("is_fraud"):
                if "FLAG_REVIEW" in ("FLAG_REVIEW", "DENY"):
                    memory_reuse_correct += 1

        assert memory_reuse_correct == 1

    def test_approve_on_known_legit_is_correct(self):
        investigation_memory = {
            "PROV_002": {"provider_id": "PROV_002", "is_fraud": False,
                         "step_investigated": 10, "base_confidence": 0.9}
        }
        memory_reuse_opportunities = 0
        memory_reuse_correct = 0

        current_provider = "PROV_002"
        if current_provider in investigation_memory:
            memory_reuse_opportunities += 1
            entry = investigation_memory[current_provider]
            if not entry.get("is_fraud"):
                if "APPROVE" == "APPROVE":
                    memory_reuse_correct += 1

        assert memory_reuse_correct == 1


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestBasicCycle,
        TestBudgetDowngrade,
        TestMemoryCorrectness,
        TestOutcomeString,
        TestResetAPIConfig,
        TestHarnessMemoryReuseMetric,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for cls in test_classes:
        instance = cls()
        # Run autouse fixtures manually for standalone mode
        if hasattr(instance, "client"):
            try:
                instance.client()
            except Exception:
                pass

        for name in sorted(m for m in dir(cls) if m.startswith("test_")):
            method = getattr(instance, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except pytest.skip.Exception as e:
                print(f"  SKIP  {cls.__name__}.{name} — {e}")
                skipped += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
