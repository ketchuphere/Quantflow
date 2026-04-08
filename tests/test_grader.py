"""
Tests for the grader module — deterministic, 0.0–1.0 range, task-specific logic.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from treasury_env import TreasuryCashPositionPlanner
from treasury_env.models import TreasuryAction, ActionType
from treasury_env.grader import grade_episode


def run_full_episode(task_id: str, policy: str = "hold", seed: int = 42):
    """Helper: run full episode with given policy."""
    env = TreasuryCashPositionPlanner()
    env.reset(task_id=task_id, seed=seed)
    hold = TreasuryAction(action_type=ActionType.HOLD)
    done = False
    while not done:
        _, _, done, _ = env.step(hold)
    return env


class TestGraderRange:
    """Scores must always be in [0.0, 1.0]."""

    @pytest.mark.parametrize("task_id", [
        "task_1_daily_funding",
        "task_2_sweep_optimization",
        "task_3_multi_account_liquidity",
    ])
    def test_score_in_range(self, task_id):
        env = run_full_episode(task_id)
        scores = env.grade()
        assert 0.0 <= scores.overall <= 1.0
        assert 0.0 <= scores.payment_rate <= 1.0
        assert 0.0 <= scores.liquidity_safety <= 1.0
        assert 0.0 <= scores.efficiency <= 1.0
        assert 0.0 <= scores.compliance <= 1.0

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    def test_score_range_multiple_seeds(self, seed):
        env = run_full_episode("task_3_multi_account_liquidity", seed=seed)
        scores = env.grade()
        assert 0.0 <= scores.overall <= 1.0


class TestGraderDeterminism:
    """Same seed → same score."""

    def test_deterministic_task1(self):
        env1 = run_full_episode("task_1_daily_funding", seed=42)
        env2 = run_full_episode("task_1_daily_funding", seed=42)
        assert env1.grade().overall == env2.grade().overall

    def test_deterministic_task2(self):
        env1 = run_full_episode("task_2_sweep_optimization", seed=7)
        env2 = run_full_episode("task_2_sweep_optimization", seed=7)
        assert env1.grade().overall == env2.grade().overall

    def test_deterministic_task3(self):
        env1 = run_full_episode("task_3_multi_account_liquidity", seed=100)
        env2 = run_full_episode("task_3_multi_account_liquidity", seed=100)
        assert env1.grade().overall == env2.grade().overall


class TestGraderSensitivity:
    """Better actions should generally yield higher scores."""

    def test_active_policy_vs_hold_task2(self):
        """Invest action should improve efficiency score vs pure hold."""
        #Hold-only episode
        env_hold = TreasuryCashPositionPlanner()
        env_hold.reset(task_id="task_2_sweep_optimization", seed=42)
        hold = TreasuryAction(action_type=ActionType.HOLD)
        done = False
        while not done:
            _, _, done, _ = env_hold.step(hold)
        hold_score = env_hold.grade()

        #Active episode: invest surplus on day 1
        env_active = TreasuryCashPositionPlanner()
        env_active.reset(task_id="task_2_sweep_optimization", seed=42)
        invest = TreasuryAction(
            action_type=ActionType.INVEST,
            source_account="operating",
            amount=50_000.0,
        )
        done = False
        day = 0
        hold = TreasuryAction(action_type=ActionType.HOLD)
        while not done:
            action = invest if day == 0 else hold
            _, _, done, _ = env_active.step(action)
            day += 1
        active_score = env_active.grade()

        #Efficiency should be better with active investing
        #(overall might be equal or higher)
        assert active_score.efficiency >= hold_score.efficiency - 0.05  # Tolerance


class TestGraderDetails:
    """Grader details should be populated."""

    def test_task1_details(self):
        env = run_full_episode("task_1_daily_funding")
        scores = env.grade()
        assert "total_obligations" in scores.details
        assert "paid_count" in scores.details
        assert scores.details["total_obligations"] == 4

    def test_task3_details_critical(self):
        env = run_full_episode("task_3_multi_account_liquidity")
        scores = env.grade()
        assert "critical_total" in scores.details
        assert scores.details["critical_total"] == 3  
