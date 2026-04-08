"""
Typed Pydantic models for the Treasury Cash Position Planner environment.
All models are strict and serializable for OpenEnv compliance.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator

#Enum definitions for action types, payment priorities, and transfer statuses. These provide clear semantics for the types of actions the agent can take, the importance of obligations, and the state of transfers in the simulation.
class ActionType(str, Enum):
    TRANSFER = "transfer"
    INVEST = "invest"
    REDEEM = "redeem"
    HOLD = "hold"
    EMERGENCY_FUND = "emergency_fund"


class PaymentPriority(str, Enum):
    CRITICAL = "critical"    # Must pay — penalties/legal risk
    HIGH = "high"            # Should pay — relationship risk
    NORMAL = "normal"        # Routine — deferrable 1 day
    LOW = "low"              # Optional — can defer 2+ days


class TransferStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    FAILED = "failed"

#Core data models for accounts, cash flow events, obligations, pending transfers, and rates/fees. These models capture the essential attributes of each entity in the simulation and are designed to be easily serialized and validated.
class AccountBalance(BaseModel):
    account_id: str
    name: str
    balance: float
    min_balance: float = 0.0
    is_investment: bool = False

    class Config:
        frozen = False


class CashFlowEvent(BaseModel):
    event_id: str
    account_id: str
    amount: float                    # Positive = inflow
    due_day: int
    description: str
    probability: float = 1.0         # 1.0 = certain; <1.0 = uncertain
    realized: bool = False


class Obligation(BaseModel):
    obligation_id: str
    account_id: str
    amount: float                    # Always positive (outflow)
    due_day: int
    priority: PaymentPriority
    description: str
    paid: bool = False
    paid_day: Optional[int] = None
    overdue_penalty_per_day: float = 0.0


class PendingTransfer(BaseModel):
    transfer_id: str
    source_account: str
    destination_account: str
    amount: float
    initiated_day: int
    settlement_day: int              # Day funds arrive
    status: TransferStatus = TransferStatus.PENDING
    fee_charged: float = 0.0


class RatesAndFees(BaseModel):
    transfer_fee_flat: float = 25.0          # $ per transfer
    transfer_fee_pct: float = 0.0001         # 0.01% of amount
    overdraft_fee_daily: float = 150.0       # $ per day overdrawn
    investment_yield_daily: float = 0.00015  # ~5.5% annualized
    emergency_fund_fee_pct: float = 0.015    # 1.5% of amount drawn
    transfer_settlement_days: int = 1        # T+1 default

#Action model for the agent to specify its decisions. This model includes validation to ensure that the amount is non-negative and that the action type is valid. It also allows for optional fields depending on the action type, such as source/destination accounts for transfers and priority overrides for payments.
class TreasuryAction(BaseModel):
    action_type: ActionType
    source_account: Optional[str] = None
    destination_account: Optional[str] = None
    amount: float = Field(default=0.0, ge=0.0)
    reference_id: Optional[str] = None
    priority_override: Optional[PaymentPriority] = None
    partial_payment_ok: bool = False

    @validator("amount")
    def amount_non_negative(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return round(v, 2)

    class Config:
        use_enum_values = True

#Observation and reward models to capture the state of the environment and the feedback to the agent. The Observation model includes all relevant information about the current day, account balances, scheduled cash flows, and other contextual data. The StepReward model breaks down the reward into components for more granular feedback on the agent's performance.
class Observation(BaseModel):
    day: int
    balances: Dict[str, float]
    account_metadata: Dict[str, Dict[str, Any]]  # id -> {name, min_balance, is_investment}
    scheduled_inflows: List[CashFlowEvent]
    scheduled_outflows: List[Obligation]
    pending_transfers: List[PendingTransfer]
    liquidity_buffer_target: float
    rates_and_fees: RatesAndFees
    risk_flags: List[str]
    investment_balance: float
    horizon_days: int
    task_id: str
    done: bool = False

    class Config:
        use_enum_values = True

#Reward model to capture the reward components for each step. This allows for detailed feedback to the agent on its performance across multiple dimensions, such as payment timeliness, liquidity management, and cost efficiency.
class StepReward(BaseModel):
    total: float
    on_time_payment: float = 0.0
    buffer_compliance: float = 0.0
    sweep_efficiency: float = 0.0
    overdraft_penalty: float = 0.0
    missed_payment_penalty: float = 0.0
    idle_cash_penalty: float = 0.0
    transfer_cost_penalty: float = 0.0
    emergency_fund_penalty: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)

#GraderScores and EnvState models to capture the overall performance of the agent and the complete state of the environment. The GraderScores model provides a structured way to report scores across multiple dimensions, while the EnvState model captures all relevant information about the current state of the environment for debugging, analysis, and potential replay.
class GraderScores(BaseModel):
    overall: float
    payment_rate: float
    liquidity_safety: float
    efficiency: float
    compliance: float
    details: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    task_id: str
    day: int
    horizon_days: int
    accounts: List[AccountBalance]
    obligations: List[Obligation]
    inflows: List[CashFlowEvent]
    pending_transfers: List[PendingTransfer]
    investment_balance: float
    liquidity_buffer_target: float
    rates_and_fees: RatesAndFees
    realized_pnl: float
    total_fees_paid: float
    total_overdraft_fees: float
    emergency_funds_used: float
    action_history: List[Dict[str, Any]]
    reward_history: List[float]
    cumulative_reward: float
    seed: Optional[int]
    done: bool
    grader_scores: Optional[GraderScores] = None

    class Config:
        use_enum_values = True
