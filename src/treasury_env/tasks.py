"""
Task definitions for the Treasury Cash Position Planner.
Each task defines accounts, inflows, obligations, and environment parameters.
"""
from __future__ import annotations
from typing import Dict, Any, List
from .models import (
    AccountBalance, CashFlowEvent, Obligation, RatesAndFees,
    PaymentPriority
)


def get_task(task_id: str) -> Dict[str, Any]:
    tasks = {
        "task_1_daily_funding": _task_1(),
        "task_2_sweep_optimization": _task_2(),
        "task_3_multi_account_liquidity": _task_3(),
    }
    if task_id not in tasks:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(tasks.keys())}")
    return tasks[task_id]


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "task_id": "task_1_daily_funding",
            "name": "Daily Funding",
            "difficulty": "easy",
            "description": (
                "Keep the operating account above a $50,000 minimum buffer "
                "and pay all due obligations over 7 days. Single account, "
                "deterministic cash flows, no transfer fees."
            ),
            "max_steps": 7,
            "horizon_days": 7,
            "success_threshold": 0.70,
            "action_schema": _action_schema(),
        },
        {
            "task_id": "task_2_sweep_optimization",
            "name": "Sweep Optimization",
            "difficulty": "medium",
            "description": (
                "Manage two accounts: an operating account and an investment sweep account. "
                "Pay all obligations on time while maximizing yield on surplus cash over 7 days. "
                "Transfer fees apply. Penalty for keeping too much idle cash."
            ),
            "max_steps": 7,
            "horizon_days": 7,
            "success_threshold": 0.65,
            "action_schema": _action_schema(),
        },
        {
            "task_id": "task_3_multi_account_liquidity",
            "name": "Multi-Account Liquidity Planning",
            "difficulty": "hard",
            "description": (
                "Manage 4 accounts (operating, payroll, reserve, investment) over 14 days. "
                "Inflows are uncertain (probabilistic), obligations have priority tiers, "
                "transfers settle T+1, and emergency funding costs 1.5%. "
                "Minimize overdrafts, meet all critical payments, maximize efficiency."
            ),
            "max_steps": 14,
            "horizon_days": 14,
            "success_threshold": 0.55,
            "action_schema": _action_schema(),
        },
    ]


def _action_schema() -> Dict[str, Any]:
    return {
        "action_type": {
            "type": "string",
            "enum": ["transfer", "invest", "redeem", "hold", "emergency_fund"],
            "required": True,
        },
        "source_account": {"type": "string", "nullable": True},
        "destination_account": {"type": "string", "nullable": True},
        "amount": {"type": "float", "minimum": 0.0, "default": 0.0},
        "reference_id": {"type": "string", "nullable": True},
        "partial_payment_ok": {"type": "boolean", "default": False},
    }

#D
def _task_1() -> Dict[str, Any]:
    accounts = [
        AccountBalance(
            account_id="operating",
            name="Operating Account",
            balance=120_000.0,
            min_balance=50_000.0,
            is_investment=False,
        )
    ]

    inflows = [
        CashFlowEvent(
            event_id="inf_1_d2",
            account_id="operating",
            amount=80_000.0,
            due_day=2,
            description="Customer receivable batch A",
            probability=1.0,
        ),
        CashFlowEvent(
            event_id="inf_2_d4",
            account_id="operating",
            amount=60_000.0,
            due_day=4,
            description="Customer receivable batch B",
            probability=1.0,
        ),
        CashFlowEvent(
            event_id="inf_3_d6",
            account_id="operating",
            amount=40_000.0,
            due_day=6,
            description="Government grant disbursement",
            probability=1.0,
        ),
    ]

    obligations = [
        Obligation(
            obligation_id="obl_1_d1",
            account_id="operating",
            amount=30_000.0,
            due_day=1,
            priority=PaymentPriority.CRITICAL,
            description="Supplier payment — raw materials",
            overdue_penalty_per_day=500.0,
        ),
        Obligation(
            obligation_id="obl_2_d3",
            account_id="operating",
            amount=45_000.0,
            due_day=3,
            priority=PaymentPriority.HIGH,
            description="Payroll disbursement",
            overdue_penalty_per_day=200.0,
        ),
        Obligation(
            obligation_id="obl_3_d5",
            account_id="operating",
            amount=25_000.0,
            due_day=5,
            priority=PaymentPriority.NORMAL,
            description="Utilities and lease",
            overdue_penalty_per_day=50.0,
        ),
        Obligation(
            obligation_id="obl_4_d7",
            account_id="operating",
            amount=35_000.0,
            due_day=7,
            priority=PaymentPriority.HIGH,
            description="Tax installment",
            overdue_penalty_per_day=300.0,
        ),
    ]

    rates = RatesAndFees(
        transfer_fee_flat=0.0,
        transfer_fee_pct=0.0,
        overdraft_fee_daily=150.0,
        investment_yield_daily=0.0,
        emergency_fund_fee_pct=0.015,
        transfer_settlement_days=0,
    )

    return {
        "task_id": "task_1_daily_funding",
        "name": "Daily Funding",
        "difficulty": "easy",
        "max_steps": 7,
        "horizon_days": 7,
        "accounts": accounts,
        "inflows": inflows,
        "obligations": obligations,
        "rates_and_fees": rates,
        "liquidity_buffer_target": 50_000.0,
        "investment_balance": 0.0,
        "seed_randomness": False,
    }


def _task_2() -> Dict[str, Any]:
    accounts = [
        AccountBalance(
            account_id="operating",
            name="Operating Account",
            balance=200_000.0,
            min_balance=75_000.0,
            is_investment=False,
        ),
        AccountBalance(
            account_id="sweep",
            name="Money Market Sweep",
            balance=50_000.0,
            min_balance=0.0,
            is_investment=True,
        ),
    ]

    inflows = [
        CashFlowEvent(event_id="inf_1_d1", account_id="operating", amount=100_000.0,
                      due_day=1, description="ACH receivables batch", probability=1.0),
        CashFlowEvent(event_id="inf_2_d3", account_id="operating", amount=75_000.0,
                      due_day=3, description="Wire transfer — large client", probability=1.0),
        CashFlowEvent(event_id="inf_3_d5", account_id="operating", amount=50_000.0,
                      due_day=5, description="Check clearing", probability=1.0),
        CashFlowEvent(event_id="inf_4_d7", account_id="operating", amount=90_000.0,
                      due_day=7, description="End-of-week receivables", probability=1.0),
    ]

    obligations = [
        Obligation(obligation_id="obl_1_d2", account_id="operating", amount=60_000.0,
                   due_day=2, priority=PaymentPriority.CRITICAL,
                   description="Debt service payment", overdue_penalty_per_day=1000.0),
        Obligation(obligation_id="obl_2_d2", account_id="operating", amount=85_000.0,
                   due_day=2, priority=PaymentPriority.HIGH,
                   description="Biweekly payroll", overdue_penalty_per_day=300.0),
        Obligation(obligation_id="obl_3_d4", account_id="operating", amount=40_000.0,
                   due_day=4, priority=PaymentPriority.NORMAL,
                   description="Vendor payments batch", overdue_penalty_per_day=100.0),
        Obligation(obligation_id="obl_4_d6", account_id="operating", amount=55_000.0,
                   due_day=6, priority=PaymentPriority.HIGH,
                   description="Insurance premium", overdue_penalty_per_day=500.0),
        Obligation(obligation_id="obl_5_d7", account_id="operating", amount=30_000.0,
                   due_day=7, priority=PaymentPriority.NORMAL,
                   description="Office & facilities", overdue_penalty_per_day=50.0),
    ]

    rates = RatesAndFees(
        transfer_fee_flat=25.0,
        transfer_fee_pct=0.0001,
        overdraft_fee_daily=200.0,
        investment_yield_daily=0.00015,  # ~5.5% APY
        emergency_fund_fee_pct=0.015,
        transfer_settlement_days=1,
    )

    return {
        "task_id": "task_2_sweep_optimization",
        "name": "Sweep Optimization",
        "difficulty": "medium",
        "max_steps": 7,
        "horizon_days": 7,
        "accounts": accounts,
        "inflows": inflows,
        "obligations": obligations,
        "rates_and_fees": rates,
        "liquidity_buffer_target": 75_000.0,
        "investment_balance": 50_000.0,
        "seed_randomness": False,
    }


def _task_3() -> Dict[str, Any]:
    accounts = [
        AccountBalance(account_id="operating", name="Operating Account",
                       balance=300_000.0, min_balance=100_000.0, is_investment=False),
        AccountBalance(account_id="payroll", name="Payroll Account",
                       balance=50_000.0, min_balance=10_000.0, is_investment=False),
        AccountBalance(account_id="reserve", name="Reserve/Liquidity Fund",
                       balance=150_000.0, min_balance=50_000.0, is_investment=False),
        AccountBalance(account_id="investment", name="Short-Term Investment Account",
                       balance=200_000.0, min_balance=0.0, is_investment=True),
    ]

    inflows = [
        # Certain inflows
        CashFlowEvent(event_id="inf_c1_d1", account_id="operating", amount=120_000.0,
                      due_day=1, description="Wire: Client Alpha Q4 payment", probability=1.0),
        CashFlowEvent(event_id="inf_c2_d5", account_id="operating", amount=200_000.0,
                      due_day=5, description="Bond maturity proceeds", probability=1.0),
        CashFlowEvent(event_id="inf_c3_d10", account_id="reserve", amount=100_000.0,
                      due_day=10, description="CD maturity", probability=1.0),

        # Uncertain inflows
        CashFlowEvent(event_id="inf_u1_d3", account_id="operating", amount=80_000.0,
                      due_day=3, description="Receivable: Client Beta (expected)", probability=0.7),
        CashFlowEvent(event_id="inf_u2_d7", account_id="operating", amount=150_000.0,
                      due_day=7, description="Receivable: Client Gamma (likely)", probability=0.8),
        CashFlowEvent(event_id="inf_u3_d9", account_id="operating", amount=60_000.0,
                      due_day=9, description="Receivable: Client Delta (uncertain)", probability=0.5),
        CashFlowEvent(event_id="inf_u4_d12", account_id="operating", amount=90_000.0,
                      due_day=12, description="Govt grant (pending approval)", probability=0.6),
    ]

    obligations = [
        # Critical — must pay
        Obligation(obligation_id="obl_crit_d1", account_id="operating", amount=80_000.0,
                   due_day=1, priority=PaymentPriority.CRITICAL,
                   description="Bond coupon payment", overdue_penalty_per_day=2000.0),
        Obligation(obligation_id="obl_crit_d7", account_id="operating", amount=120_000.0,
                   due_day=7, priority=PaymentPriority.CRITICAL,
                   description="Term loan installment", overdue_penalty_per_day=3000.0),
        Obligation(obligation_id="obl_crit_d14", account_id="operating", amount=100_000.0,
                   due_day=14, priority=PaymentPriority.CRITICAL,
                   description="Trade finance settlement", overdue_penalty_per_day=2500.0),

        # High priority
        Obligation(obligation_id="obl_high_d3", account_id="payroll", amount=95_000.0,
                   due_day=3, priority=PaymentPriority.HIGH,
                   description="Biweekly payroll", overdue_penalty_per_day=500.0),
        Obligation(obligation_id="obl_high_d6", account_id="operating", amount=70_000.0,
                   due_day=6, priority=PaymentPriority.HIGH,
                   description="Insurance & benefits", overdue_penalty_per_day=400.0),
        Obligation(obligation_id="obl_high_d10", account_id="payroll", amount=95_000.0,
                   due_day=10, priority=PaymentPriority.HIGH,
                   description="Biweekly payroll (cycle 2)", overdue_penalty_per_day=500.0),

        # Normal priority
        Obligation(obligation_id="obl_norm_d2", account_id="operating", amount=45_000.0,
                   due_day=2, priority=PaymentPriority.NORMAL,
                   description="Vendor batch A", overdue_penalty_per_day=100.0),
        Obligation(obligation_id="obl_norm_d5", account_id="operating", amount=35_000.0,
                   due_day=5, priority=PaymentPriority.NORMAL,
                   description="Utilities & telecom", overdue_penalty_per_day=75.0),
        Obligation(obligation_id="obl_norm_d8", account_id="operating", amount=55_000.0,
                   due_day=8, priority=PaymentPriority.NORMAL,
                   description="Vendor batch B", overdue_penalty_per_day=100.0),
        Obligation(obligation_id="obl_norm_d11", account_id="operating", amount=40_000.0,
                   due_day=11, priority=PaymentPriority.NORMAL,
                   description="Facilities & maintenance", overdue_penalty_per_day=50.0),

        # Low priority
        Obligation(obligation_id="obl_low_d4", account_id="operating", amount=20_000.0,
                   due_day=4, priority=PaymentPriority.LOW,
                   description="Discretionary spend batch", overdue_penalty_per_day=0.0),
        Obligation(obligation_id="obl_low_d13", account_id="operating", amount=25_000.0,
                   due_day=13, priority=PaymentPriority.LOW,
                   description="Office supplies & misc", overdue_penalty_per_day=0.0),
    ]

    rates = RatesAndFees(
        transfer_fee_flat=30.0,
        transfer_fee_pct=0.00008,
        overdraft_fee_daily=250.0,
        investment_yield_daily=0.00015,
        emergency_fund_fee_pct=0.015,
        transfer_settlement_days=1,
    )

    return {
        "task_id": "task_3_multi_account_liquidity",
        "name": "Multi-Account Liquidity Planning",
        "difficulty": "hard",
        "max_steps": 14,
        "horizon_days": 14,
        "accounts": accounts,
        "inflows": inflows,
        "obligations": obligations,
        "rates_and_fees": rates,
        "liquidity_buffer_target": 100_000.0,
        "investment_balance": 200_000.0,
        "seed_randomness": True,  # uncertain inflows use probability
    }
