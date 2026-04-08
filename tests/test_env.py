"""
Tests for the Treasury Cash Position Planner environment.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from treasury_env import TreasuryCashPositionPlanner
from treasury_env.models import TreasuryAction, ActionType


@pytest.fixture
def env():
    return TreasuryCashPositionPlanner()


@pytest.fixture
def env_task1(env):
    env.reset(task_id="task_1_daily_funding", seed=42)
    return env


@pytest.fixture
def env_task2(env):
    env.reset(task_id="task_2_sweep_optimization", seed=42)
    return env


@pytest.fixture
def env_task3(env):
    env.reset(task_id="task_3_multi_account_liquidity", seed=42)
    return env


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="task_1_daily_funding")
        assert obs is not None
        assert obs.day == 0
        assert "operating" in obs.balances
        assert obs.balances["operating"] > 0

    def test_reset_initializes_state(self, env):
        env.reset(task_id="task_1_daily_funding", seed=42)
        assert env._sim is not None
        assert env._task_id == "task_1_daily_funding"
        assert env.done is False

    def test_reset_task2(self, env):
        obs = env.reset(task_id="task_2_sweep_optimization", seed=0)
        assert "operating" in obs.balances
        assert "sweep" in obs.balances

    def test_reset_task3(self, env):
        obs = env.reset(task_id="task_3_multi_account_liquidity", seed=0)
        assert len(obs.balances) == 4
        assert "payroll" in obs.balances
        assert "reserve" in obs.balances

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent_task")

    def test_reset_reproducibility(self, env):
        obs1 = env.reset(task_id="task_3_multi_account_liquidity", seed=99)
        b1 = dict(obs1.balances)
        obs2 = env.reset(task_id="task_3_multi_account_liquidity", seed=99)
        b2 = dict(obs2.balances)
        assert b1 == b2


class TestStep:
    def test_step_returns_tuple(self, env_task1):
        action = TreasuryAction(action_type=ActionType.HOLD)
        result = env_task1.step(action)
        assert len(result) == 4
        obs, reward, done, info = result

    def test_step_advances_day(self, env_task1):
        assert env_task1._sim.day == 0
        action = TreasuryAction(action_type=ActionType.HOLD)
        obs, _, _, _ = env_task1.step(action)
        assert obs.day == 1

    def test_step_reward_has_components(self, env_task1):
        action = TreasuryAction(action_type=ActionType.HOLD)
        _, reward, _, _ = env_task1.step(action)
        assert hasattr(reward, "total")
        assert hasattr(reward, "on_time_payment")
        assert hasattr(reward, "overdraft_penalty")

    def test_hold_action_no_transfer(self, env_task1):
        obs_before = env_task1._sim.get_all_balances()
        action = TreasuryAction(action_type=ActionType.HOLD)
        env_task1.step(action)
        # After hold, any balance change is only from inflows/outflows, not agent action

    def test_invest_action(self, env):
        obs = env.reset(task_id="task_2_sweep_optimization", seed=42)
        initial_inv = obs.investment_balance
        action = TreasuryAction(
            action_type=ActionType.INVEST,
            source_account="operating",
            amount=20_000.0,
        )
        obs, reward, done, info = env.step(action)
        # Investment balance may increase (if sweep succeeded)

    def test_done_after_horizon(self, env):
        env.reset(task_id="task_1_daily_funding", seed=42)
        hold = TreasuryAction(action_type=ActionType.HOLD)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(hold)
            steps += 1
        assert steps == 7  # Task 1 horizon
        assert env.done is True

    def test_step_after_done_raises(self, env):
        env.reset(task_id="task_1_daily_funding", seed=42)
        hold = TreasuryAction(action_type=ActionType.HOLD)
        for _ in range(7):
            env.step(hold)
        with pytest.raises(RuntimeError):
            env.step(hold)

    def test_step_before_reset_raises(self, env):
        hold = TreasuryAction(action_type=ActionType.HOLD)
        with pytest.raises(RuntimeError):
            env.step(hold)

    def test_info_has_grader_scores(self, env_task1):
        action = TreasuryAction(action_type=ActionType.HOLD)
        _, _, _, info = env_task1.step(action)
        assert "grader_scores" in info
        gs = info["grader_scores"]
        assert "overall" in gs
        assert 0.0 <= gs["overall"] <= 1.0


class TestState:
    def test_state_returns_serializable(self, env_task1):
        state = env_task1.state()
        assert state is not None
        d = state.dict()
        assert "task_id" in d
        assert "day" in d
        assert "accounts" in d
        assert "obligations" in d

    def test_state_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.state()

    def test_state_day_matches_sim(self, env_task1):
        action = TreasuryAction(action_type=ActionType.HOLD)
        env_task1.step(action)
        env_task1.step(action)
        state = env_task1.state()
        assert state.day == 2



class TestTasks:
    def test_tasks_returns_list(self, env):
        tasks = env.tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 3

    def test_tasks_have_required_fields(self, env):
        tasks = env.tasks()
        for t in tasks:
            assert "task_id" in t
            assert "name" in t
            assert "difficulty" in t
            assert "action_schema" in t

    def test_task_ids_valid(self, env):
        tasks = env.tasks()
        ids = {t["task_id"] for t in tasks}
        assert "task_1_daily_funding" in ids
        assert "task_2_sweep_optimization" in ids
        assert "task_3_multi_account_liquidity" in ids



class TestGrader:
    def test_grade_returns_scores(self, env_task1):
        scores = env_task1.grade()
        assert 0.0 <= scores.overall <= 1.0
        assert 0.0 <= scores.payment_rate <= 1.0
        assert 0.0 <= scores.liquidity_safety <= 1.0
        assert 0.0 <= scores.efficiency <= 1.0
        assert 0.0 <= scores.compliance <= 1.0

    def test_grade_all_tasks(self, env):
        for task_id in TreasuryCashPositionPlanner.VALID_TASK_IDS:
            env.reset(task_id=task_id, seed=42)
            hold = TreasuryAction(action_type=ActionType.HOLD)
            done = False
            while not done:
                _, _, done, _ = env.step(hold)
            scores = env.grade()
            assert 0.0 <= scores.overall <= 1.0, f"Score out of range for {task_id}"

    def test_grade_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.grade()
