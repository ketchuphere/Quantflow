"""
Deterministic grader for each task.
Returns a score from 0.0 to 1.0 based on:
  - Payment rate (35%)
  - Liquidity safety (25%)
  - Cash efficiency / yield (20%)
  - Compliance / cost control (20%)
"""
from __future__ import annotations
from typing import List, Dict, Any, TYPE_CHECKING

from .models import Obligation, AccountBalance, GraderScores, PaymentPriority

if TYPE_CHECKING:
    from .simulator import TreasurySimulator


def grade_episode(sim: "TreasurySimulator") -> GraderScores:
    """Run the grader on a completed (or in-progress) episode."""
    grader_map = {
        "task_1_daily_funding": _grade_task_1,
        "task_2_sweep_optimization": _grade_task_2,
        "task_3_multi_account_liquidity": _grade_task_3,
    }
    fn = grader_map.get(sim.task_id, _grade_generic)
    return fn(sim)

#task 1 grader focuses on payment rate and liquidity safety, with a simple efficiency metric based on avoiding overdrafts and emergency funds. Task 2 adds a more nuanced efficiency component based on yield generation and fee management. Task 3 has stricter compliance requirements and also considers the priority of paid obligations in the payment rate calculation. The generic grader provides a baseline scoring for any task without specific criteria.
def _grade_task_1(sim: "TreasurySimulator") -> GraderScores:
    payment_rate = _compute_payment_rate(sim.obligations, priority_weight=True)
    liquidity_safety = _compute_liquidity_safety(
        sim.accounts, sim.buffer_target, sim.total_overdraft_fees
    )
    efficiency = _compute_efficiency_task1(sim)
    compliance = _compute_compliance(sim)

    overall = (
        0.35 * payment_rate
        + 0.25 * liquidity_safety
        + 0.20 * efficiency
        + 0.20 * compliance
    )
    return GraderScores(
        overall=round(min(1.0, max(0.0, overall)), 4),
        payment_rate=round(payment_rate, 4),
        liquidity_safety=round(liquidity_safety, 4),
        efficiency=round(efficiency, 4),
        compliance=round(compliance, 4),
        details={
            "total_obligations": len(sim.obligations),
            "paid_count": sum(1 for o in sim.obligations if o.paid),
            "overdraft_fees": sim.total_overdraft_fees,
            "emergency_used": sim.emergency_funds_used,
        },
    )


def _compute_efficiency_task1(sim: "TreasurySimulator") -> float:
    """For task 1: efficiency is simply not going overdraft and not using emergency funds."""
    score = 1.0
    if sim.total_overdraft_fees > 0:
        score -= min(0.5, sim.total_overdraft_fees / 1000.0)
    if sim.emergency_funds_used > 0:
        score -= 0.3
    return max(0.0, score)


#Task2 grader adds a more nuanced efficiency component based on yield generation and fee management. The yield efficiency considers the actual yield generated relative to a theoretical maximum based on investable surplus and penalizes for fees that eat into the yield. The compliance score also considers fee efficiency relative to transaction volume.
def _grade_task_2(sim: "TreasurySimulator") -> GraderScores:
    payment_rate = _compute_payment_rate(sim.obligations, priority_weight=True)
    liquidity_safety = _compute_liquidity_safety(
        sim.accounts, sim.buffer_target, sim.total_overdraft_fees
    )
    efficiency = _compute_yield_efficiency(sim)
    compliance = _compute_compliance(sim)

    overall = (
        0.35 * payment_rate
        + 0.25 * liquidity_safety
        + 0.20 * efficiency
        + 0.20 * compliance
    )
    return GraderScores(
        overall=round(min(1.0, max(0.0, overall)), 4),
        payment_rate=round(payment_rate, 4),
        liquidity_safety=round(liquidity_safety, 4),
        efficiency=round(efficiency, 4),
        compliance=round(compliance, 4),
        details={
            "total_obligations": len(sim.obligations),
            "paid_count": sum(1 for o in sim.obligations if o.paid),
            "realized_pnl": round(sim.realized_pnl, 2),
            "total_fees": round(sim.total_fees_paid, 2),
            "overdraft_fees": sim.total_overdraft_fees,
            "investment_balance": sim.get_investment_balance(),
        },
    )


def _compute_yield_efficiency(sim: "TreasurySimulator") -> float:
    """Reward for generating investment yield relative to opportunity."""
    #Theoretical max: all surplus invested for full period
    max_investable = sum(
        max(0, acc.balance - sim.buffer_target)
        for acc in sim.accounts.values()
        if not acc.is_investment
    )
    theoretical_yield = max_investable * sim.rates.investment_yield_daily * sim.horizon_days
    if theoretical_yield <= 0:
        return 0.8  #No investment opportunity - neutral score

    actual_yield = sim.realized_pnl
    ratio = min(1.0, actual_yield / max(1.0, theoretical_yield))

    #Penalize fees eating into yield
    fee_drag = sim.total_fees_paid / max(1.0, actual_yield + sim.total_fees_paid)
    score = ratio * (1.0 - fee_drag * 0.5)
    return max(0.0, min(1.0, score))

#Task 3 grader has stricter compliance requirements and also considers the priority of paid obligations in the payment rate calculation. The compliance score penalizes emergency fund usage and overdrafts more heavily, and also considers excessive fees relative to transaction volume.
def _grade_task_3(sim: "TreasurySimulator") -> GraderScores:
    payment_rate = _compute_payment_rate(sim.obligations, priority_weight=True)
    liquidity_safety = _compute_liquidity_safety(
        sim.accounts, sim.buffer_target, sim.total_overdraft_fees
    )
    efficiency = _compute_yield_efficiency(sim)
    compliance = _compute_compliance_task3(sim)

    overall = (
        0.35 * payment_rate
        + 0.25 * liquidity_safety
        + 0.20 * efficiency
        + 0.20 * compliance
    )
    return GraderScores(
        overall=round(min(1.0, max(0.0, overall)), 4),
        payment_rate=round(payment_rate, 4),
        liquidity_safety=round(liquidity_safety, 4),
        efficiency=round(efficiency, 4),
        compliance=round(compliance, 4),
        details={
            "total_obligations": len(sim.obligations),
            "paid_count": sum(1 for o in sim.obligations if o.paid),
            "critical_paid": sum(1 for o in sim.obligations
                                 if o.paid and _is_priority(o, "critical")),
            "critical_total": sum(1 for o in sim.obligations
                                  if _is_priority(o, "critical")),
            "realized_pnl": round(sim.realized_pnl, 2),
            "total_fees": round(sim.total_fees_paid, 2),
            "overdraft_fees": sim.total_overdraft_fees,
            "emergency_used": round(sim.emergency_funds_used, 2),
            "investment_balance": round(sim.get_investment_balance(), 2),
        },
    )


def _compute_compliance_task3(sim: "TreasurySimulator") -> float:
    """Stricter compliance for hard task: penalize emergency funds and overdrafts heavily."""
    score = 1.0
    if sim.emergency_funds_used > 0:
        score -= min(0.4, sim.emergency_funds_used / 500_000.0)
    if sim.total_overdraft_fees > 0:
        score -= min(0.4, sim.total_overdraft_fees / 5000.0)
    #Penalize excessive fee drag
    if sim.total_fees_paid > 5000:
        score -= min(0.2, (sim.total_fees_paid - 5000) / 50_000.0)
    return max(0.0, score)

# Generic grader provides a baseline scoring for any task without specific criteria, using the same underlying metrics but with more lenient thresholds and no priority weighting.
def _is_priority(obl: Obligation, priority_str: str) -> bool:
    p = obl.priority.value if hasattr(obl.priority, 'value') else str(obl.priority)
    return p == priority_str


def _compute_payment_rate(
    obligations: List[Obligation],
    priority_weight: bool = True
) -> float:
    if not obligations:
        return 1.0

    if not priority_weight:
        paid = sum(1 for o in obligations if o.paid)
        return paid / len(obligations)

    #Weighted by priority
    weights = {"critical": 4.0, "high": 2.0, "normal": 1.0, "low": 0.5}
    total_weight = 0.0
    paid_weight = 0.0
    for obl in obligations:
        p = obl.priority.value if hasattr(obl.priority, 'value') else str(obl.priority)
        w = weights.get(p, 1.0)
        total_weight += w
        if obl.paid:
            paid_weight += w

    return paid_weight / total_weight if total_weight > 0 else 1.0


def _compute_liquidity_safety(
    accounts: Dict[str, "AccountBalance"],
    buffer_target: float,
    total_overdraft_fees: float,
) -> float:
    #Severe penalty for any overdraft
    if total_overdraft_fees > 0:
        base = max(0.0, 0.5 - (total_overdraft_fees / 10_000.0))
    else:
        base = 1.0

    #Check if buffer is maintained in operating account
    primary = accounts.get("operating")
    if primary:
        if primary.balance < buffer_target:
            shortfall_pct = (buffer_target - primary.balance) / buffer_target
            base -= min(0.3, shortfall_pct * 0.5)

    return max(0.0, min(1.0, base))


def _compute_compliance(sim: "TreasurySimulator") -> float:
    score = 1.0
    if sim.emergency_funds_used > 0:
        score -= 0.3
    if sim.total_overdraft_fees > 0:
        score -= min(0.3, sim.total_overdraft_fees / 3000.0)
    #Penalize excessive fees relative to transaction volume
    total_volume = sum(
        o.amount for o in sim.obligations if o.paid
    )
    if total_volume > 0 and sim.total_fees_paid / total_volume > 0.01:
        score -= 0.1
    return max(0.0, score)


def _grade_generic(sim: "TreasurySimulator") -> GraderScores:
    payment_rate = _compute_payment_rate(sim.obligations)
    liquidity_safety = _compute_liquidity_safety(
        sim.accounts, sim.buffer_target, sim.total_overdraft_fees
    )
    efficiency = 0.5
    compliance = _compute_compliance(sim)
    overall = (
        0.35 * payment_rate + 0.25 * liquidity_safety
        + 0.20 * efficiency + 0.20 * compliance
    )
    return GraderScores(
        overall=round(min(1.0, max(0.0, overall)), 4),
        payment_rate=round(payment_rate, 4),
        liquidity_safety=round(liquidity_safety, 4),
        efficiency=round(efficiency, 4),
        compliance=round(compliance, 4),
    )
