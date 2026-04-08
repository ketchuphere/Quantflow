"""
Main TreasuryCashPositionPlanner environment.
Implements the OpenEnv interface: reset() / step() / state()
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

from .models import (
    TreasuryAction, Observation, StepReward, EnvState, GraderScores
)
from .simulator import TreasurySimulator
from .tasks import get_task, list_tasks
from .grader import grade_episode


class TreasuryCashPositionPlanner:
    """
    OpenEnv-compliant treasury simulation environment.

    Manages cash across multiple bank accounts, forecasts inflows/outflows,
    and requires the agent to make daily funding, sweep, and liquidity decisions.

    Usage:
        env = TreasuryCashPositionPlanner()
        obs = env.reset(task_id="task_1_daily_funding", seed=42)
        obs, reward, done, info = env.step(action)
        state = env.state()
        scores = env.grade()
    """

    VALID_TASK_IDS = [
        "task_1_daily_funding",
        "task_2_sweep_optimization",
        "task_3_multi_account_liquidity",
    ]

    def __init__(self):
        self._sim: Optional[TreasurySimulator] = None
        self._task_id: Optional[str] = None
        self._seed: Optional[int] = None
        self._done: bool = False


    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Reset the environment to initial state for the given task.

        Args:
            task_id: One of task_1_daily_funding, task_2_sweep_optimization,
                     task_3_multi_account_liquidity. Defaults to task_1.
            seed: Random seed for reproducible episodes.

        Returns:
            Initial Observation (day=0, no actions taken).
        """
        task_id = task_id or "task_1_daily_funding"
        if task_id not in self.VALID_TASK_IDS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid options: {self.VALID_TASK_IDS}"
            )

        task_config = get_task(task_id)
        self._task_id = task_id
        self._seed = seed
        self._done = False

        self._sim = TreasurySimulator(
            accounts=task_config["accounts"],
            inflows=task_config["inflows"],
            obligations=task_config["obligations"],
            rates_and_fees=task_config["rates_and_fees"],
            liquidity_buffer_target=task_config["liquidity_buffer_target"],
            horizon_days=task_config["horizon_days"],
            task_id=task_id,
            seed=seed,
            seed_randomness=task_config.get("seed_randomness", False),
        )

        return self._sim.build_observation(done=False)

    def step(
        self,
        action: TreasuryAction,
    ) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        """
        Execute one action and advance the simulation by one day.

        Args:
            action: A TreasuryAction (transfer, invest, redeem, hold, emergency_fund).

        Returns:
            (observation, reward, done, info)
            - observation: Updated Observation after day advance.
            - reward: StepReward with component breakdown.
            - done: True when horizon_days reached.
            - info: Dict with grader sub-scores and event log.
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        step_reward, done, step_info = self._sim.step(action)
        self._done = done

        #Compute grader scores at every step (for info)
        grader = grade_episode(self._sim)

        obs = self._sim.build_observation(done=done)

        info = {
            **step_info,
            "grader_scores": grader.dict(),
            "cumulative_reward": round(self._sim.cumulative_reward, 4),
            "day": self._sim.day,
            "done": done,
        }

        return obs, step_reward, done, info

    def state(self) -> EnvState:
        """
        Return a fully serializable snapshot of the current environment state.
        Useful for debugging, checkpointing, and reproducibility.
        """
        if self._sim is None:
            raise RuntimeError("Call reset() before state()")

        grader = grade_episode(self._sim)

        return EnvState(
            task_id=self._task_id,
            day=self._sim.day,
            horizon_days=self._sim.horizon_days,
            accounts=list(self._sim.accounts.values()),
            obligations=self._sim.obligations,
            inflows=self._sim.inflows,
            pending_transfers=self._sim.pending_transfers,
            investment_balance=self._sim.get_investment_balance(),
            liquidity_buffer_target=self._sim.buffer_target,
            rates_and_fees=self._sim.rates,
            realized_pnl=round(self._sim.realized_pnl, 4),
            total_fees_paid=round(self._sim.total_fees_paid, 4),
            total_overdraft_fees=round(self._sim.total_overdraft_fees, 4),
            emergency_funds_used=round(self._sim.emergency_funds_used, 4),
            action_history=self._sim.action_history,
            reward_history=[round(r, 4) for r in self._sim.reward_history],
            cumulative_reward=round(self._sim.cumulative_reward, 4),
            seed=self._seed,
            done=self._done,
            grader_scores=grader,
        )

    def grade(self) -> GraderScores:
        #Grader scores are computed at the end of the episode based on the full trajectory.
        if self._sim is None:
            raise RuntimeError("Call reset() before grade()")
        return grade_episode(self._sim)

    def tasks(self) -> list:
        #Return list of available tasks with metadata. Useful for dynamic UIs and validation.
        return list_tasks()

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    @property
    def done(self) -> bool:
        return self._done
