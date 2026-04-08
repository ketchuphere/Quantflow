"""
Treasury simulation engine.
Handles: inflow application, obligation payment, action execution,
fee charging, day advancement, and reward computation.
All transitions are deterministic given a seed.
"""
from __future__ import annotations
import random
import uuid
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy

from .models import (
    AccountBalance, CashFlowEvent, Obligation, PendingTransfer,
    RatesAndFees, TreasuryAction, ActionType, PaymentPriority,
    StepReward, Observation, TransferStatus
)


class TreasurySimulator:
    """
    Core simulation engine. Manages account state, processes actions,
    applies cash flows, charges fees, and computes step rewards.
    """

    def __init__(
        self,
        accounts: List[AccountBalance],
        inflows: List[CashFlowEvent],
        obligations: List[Obligation],
        rates_and_fees: RatesAndFees,
        liquidity_buffer_target: float,
        horizon_days: int,
        task_id: str,
        seed: Optional[int] = None,
        seed_randomness: bool = False,
    ):
        self.accounts: Dict[str, AccountBalance] = {a.account_id: deepcopy(a) for a in accounts}
        self.inflows: List[CashFlowEvent] = deepcopy(inflows)
        self.obligations: List[Obligation] = deepcopy(obligations)
        self.rates = deepcopy(rates_and_fees)
        self.buffer_target = liquidity_buffer_target
        self.horizon_days = horizon_days
        self.task_id = task_id
        self.seed = seed
        self.seed_randomness = seed_randomness
        self.rng = random.Random(seed)

        self.day: int = 0
        self.pending_transfers: List[PendingTransfer] = []
        self.realized_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.total_overdraft_fees: float = 0.0
        self.emergency_funds_used: float = 0.0
        self.action_history: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        self.cumulative_reward: float = 0.0
        self._transfer_counter: int = 0

        # Resolve probabilistic inflows upfront for reproducibility
        if self.seed_randomness:
            self._resolve_uncertain_inflows()

    def _resolve_uncertain_inflows(self):
        """Pre-resolve probabilistic inflows using seeded RNG."""
        for inflow in self.inflows:
            if inflow.probability < 1.0:
                inflow.realized = self.rng.random() < inflow.probability
            else:
                inflow.realized = True

    def get_investment_balance(self) -> float:
        total = 0.0
        for acc in self.accounts.values():
            if acc.is_investment:
                total += acc.balance
        return total

    def get_non_investment_balances(self) -> Dict[str, float]:
        return {
            acc_id: acc.balance
            for acc_id, acc in self.accounts.items()
            if not acc.is_investment
        }

    def get_all_balances(self) -> Dict[str, float]:
        return {acc_id: acc.balance for acc_id, acc in self.accounts.items()}

    def step(self, action: TreasuryAction) -> Tuple[StepReward, bool, Dict[str, Any]]:
        """
        Advance one day. Sequence:
        1. Settle pending transfers that are due
        2. Apply scheduled inflows
        3. Pay due obligations
        4. Execute agent's action
        5. Apply investment yield
        6. Check overdrafts
        7. Advance day counter
        8. Compute reward
        """
        self.day += 1
        info: Dict[str, Any] = {"day": self.day, "events": []}
        reward_components = {
            "on_time_payment": 0.0,
            "buffer_compliance": 0.0,
            "sweep_efficiency": 0.0,
            "overdraft_penalty": 0.0,
            "missed_payment_penalty": 0.0,
            "idle_cash_penalty": 0.0,
            "transfer_cost_penalty": 0.0,
            "emergency_fund_penalty": 0.0,
        }

        # 1. Settle pending transfers
        self._settle_transfers(info)

        # 2. Apply scheduled inflows for today
        self._apply_inflows(info)

        # 3. Auto-pay due obligations (by priority)
        self._pay_obligations(info, reward_components)

        # 4. Execute agent action
        action_result = self._execute_action(action, info, reward_components)
        self.action_history.append({
            "day": self.day,
            "action": action.dict(),
            "result": action_result,
        })

        # 5. Apply investment yield
        self._apply_investment_yield(reward_components)

        # 6. Check overdrafts on all accounts
        self._check_overdrafts(info, reward_components)

        # 7. Reward for buffer compliance
        self._check_buffer_compliance(reward_components)

        # 8. Penalize idle cash
        self._check_idle_cash(reward_components)

        # Compute total reward
        total_reward = sum(reward_components.values())
        self.cumulative_reward += total_reward
        self.reward_history.append(total_reward)

        step_reward = StepReward(
            total=round(total_reward, 4),
            **{k: round(v, 4) for k, v in reward_components.items()},
            details=info,
        )

        done = (self.day >= self.horizon_days)
        return step_reward, done, info

    def _settle_transfers(self, info: Dict):
        for t in self.pending_transfers:
            if t.status == TransferStatus.PENDING and t.settlement_day <= self.day:
                dest = self.accounts.get(t.destination_account)
                if dest:
                    dest.balance += t.amount
                    t.status = TransferStatus.SETTLED
                    info["events"].append(
                        f"Transfer {t.transfer_id} settled: ${t.amount:,.0f} → {t.destination_account}"
                    )

    def _apply_inflows(self, info: Dict):
        for inflow in self.inflows:
            if inflow.due_day == self.day and not inflow.realized:
                # Mark as realized (certain inflows always arrive; uncertain were pre-resolved)
                if not self.seed_randomness or inflow.probability >= 1.0:
                    inflow.realized = True
                # For seed_randomness tasks, `realized` was already set in __init__

                if inflow.realized:
                    acc = self.accounts.get(inflow.account_id)
                    if acc:
                        acc.balance += inflow.amount
                        info["events"].append(
                            f"Inflow received: ${inflow.amount:,.0f} → {inflow.account_id} ({inflow.description})"
                        )

    def _pay_obligations(self, info: Dict, rc: Dict):
        """Auto-pay obligations due today, sorted by priority."""
        priority_order = {
            PaymentPriority.CRITICAL: 0,
            PaymentPriority.HIGH: 1,
            PaymentPriority.NORMAL: 2,
            PaymentPriority.LOW: 3,
        }
        due_today = [
            o for o in self.obligations
            if o.due_day == self.day and not o.paid
        ]
        due_today.sort(key=lambda o: priority_order.get(o.priority, 99))

        for obl in due_today:
            acc = self.accounts.get(obl.account_id)
            if acc and acc.balance >= obl.amount:
                acc.balance -= obl.amount
                obl.paid = True
                obl.paid_day = self.day
                rc["on_time_payment"] += 1.0
                info["events"].append(
                    f"✓ Paid {obl.description}: ${obl.amount:,.0f} from {obl.account_id}"
                )
            else:
                # Missed payment — will be picked up in overdraft/missed logic
                priority_str = obl.priority.value if hasattr(obl.priority, 'value') else str(obl.priority)
                if priority_str == "critical":
                    rc["missed_payment_penalty"] -= 2.0
                elif priority_str == "high":
                    rc["missed_payment_penalty"] -= 1.0
                else:
                    rc["missed_payment_penalty"] -= 0.5
                info["events"].append(
                    f"✗ MISSED {obl.description}: ${obl.amount:,.0f} — insufficient funds in {obl.account_id}"
                )

    def _execute_action(self, action: TreasuryAction, info: Dict, rc: Dict) -> Dict:
        action_type = action.action_type
        if hasattr(action_type, 'value'):
            action_type = action_type.value

        result = {"success": False, "message": ""}

        if action_type == ActionType.HOLD or action_type == "hold":
            result = {"success": True, "message": "Hold — no action taken"}

        elif action_type in (ActionType.TRANSFER, "transfer"):
            result = self._do_transfer(action, rc)

        elif action_type in (ActionType.INVEST, "invest"):
            result = self._do_invest(action, rc)

        elif action_type in (ActionType.REDEEM, "redeem"):
            result = self._do_redeem(action, rc)

        elif action_type in (ActionType.EMERGENCY_FUND, "emergency_fund"):
            result = self._do_emergency_fund(action, rc)

        info["action_result"] = result
        return result

    def _do_transfer(self, action: TreasuryAction, rc: Dict) -> Dict:
        src_id = action.source_account
        dst_id = action.destination_account
        amount = action.amount

        src = self.accounts.get(src_id)
        dst = self.accounts.get(dst_id)

        if not src or not dst:
            return {"success": False, "message": f"Invalid account(s): {src_id} → {dst_id}"}
        if amount <= 0:
            return {"success": False, "message": "Amount must be positive"}
        if src.balance < amount:
            amount = src.balance if action.partial_payment_ok else 0
            if amount == 0:
                return {"success": False, "message": "Insufficient funds for transfer"}

        # Charge fee
        fee = self.rates.transfer_fee_flat + (amount * self.rates.transfer_fee_pct)
        if src.balance < amount + fee:
            fee = max(0, src.balance - amount)
        src.balance -= (amount + fee)
        self.total_fees_paid += fee
        rc["transfer_cost_penalty"] -= (fee / 1000.0)  # Small penalty for fee drag

        # Create pending transfer (T+N settlement)
        self._transfer_counter += 1
        transfer_id = f"TXN-{self.day:02d}-{self._transfer_counter:03d}"
        settlement_day = self.day + self.rates.transfer_settlement_days

        pending = PendingTransfer(
            transfer_id=transfer_id,
            source_account=src_id,
            destination_account=dst_id,
            amount=amount,
            initiated_day=self.day,
            settlement_day=settlement_day,
            fee_charged=fee,
        )
        self.pending_transfers.append(pending)

        return {
            "success": True,
            "message": f"Transfer initiated: ${amount:,.0f} from {src_id} to {dst_id}. Fee: ${fee:.2f}. Settles day {settlement_day}.",
            "transfer_id": transfer_id,
            "fee": fee,
        }

    def _do_invest(self, action: TreasuryAction, rc: Dict) -> Dict:
        """Move funds from operating account to investment account."""
        src_id = action.source_account or "operating"
        amount = action.amount

        # Find an investment account
        inv_accounts = [a for a in self.accounts.values() if a.is_investment]
        if not inv_accounts:
            return {"success": False, "message": "No investment account available"}

        inv_acc = inv_accounts[0]
        src = self.accounts.get(src_id)
        if not src:
            return {"success": False, "message": f"Source account {src_id} not found"}

        # Keep buffer
        available = max(0, src.balance - self.buffer_target)
        amount = min(amount, available)
        if amount <= 0:
            return {"success": False, "message": "No surplus available to invest after buffer"}

        fee = self.rates.transfer_fee_flat + (amount * self.rates.transfer_fee_pct)
        src.balance -= (amount + fee)
        inv_acc.balance += amount
        self.total_fees_paid += fee
        rc["sweep_efficiency"] += 0.1  # Reward for proactive investing

        return {
            "success": True,
            "message": f"Invested ${amount:,.0f} into {inv_acc.account_id}. Fee: ${fee:.2f}",
            "fee": fee,
        }

    def _do_redeem(self, action: TreasuryAction, rc: Dict) -> Dict:
        """Redeem from investment account back to operating."""
        dst_id = action.destination_account or "operating"
        amount = action.amount

        inv_accounts = [a for a in self.accounts.values() if a.is_investment]
        if not inv_accounts:
            return {"success": False, "message": "No investment account available"}

        inv_acc = inv_accounts[0]
        dst = self.accounts.get(dst_id)
        if not dst:
            return {"success": False, "message": f"Destination account {dst_id} not found"}

        amount = min(amount, inv_acc.balance)
        if amount <= 0:
            return {"success": False, "message": "No investment balance to redeem"}

        fee = self.rates.transfer_fee_flat + (amount * self.rates.transfer_fee_pct)
        inv_acc.balance -= amount
        dst.balance += (amount - fee)
        self.total_fees_paid += fee

        return {
            "success": True,
            "message": f"Redeemed ${amount:,.0f} from {inv_acc.account_id}. Fee: ${fee:.2f}",
            "fee": fee,
        }

    def _do_emergency_fund(self, action: TreasuryAction, rc: Dict) -> Dict:
        """Draw emergency credit line — high cost, instant."""
        dst_id = action.destination_account or "operating"
        amount = action.amount

        dst = self.accounts.get(dst_id)
        if not dst:
            return {"success": False, "message": f"Account {dst_id} not found"}
        if amount <= 0:
            return {"success": False, "message": "Amount must be positive"}

        fee = amount * self.rates.emergency_fund_fee_pct
        dst.balance += (amount - fee)
        self.emergency_funds_used += amount
        self.total_fees_paid += fee
        rc["emergency_fund_penalty"] -= 0.5  # Penalty for using expensive line

        return {
            "success": True,
            "message": f"Emergency fund drawn: ${amount:,.0f}. Fee: ${fee:,.0f} ({self.rates.emergency_fund_fee_pct*100:.1f}%)",
            "fee": fee,
        }

    def _apply_investment_yield(self, rc: Dict):
        """Daily yield on investment accounts."""
        total_yield = 0.0
        for acc in self.accounts.values():
            if acc.is_investment and acc.balance > 0:
                daily_yield = acc.balance * self.rates.investment_yield_daily
                acc.balance += daily_yield
                self.realized_pnl += daily_yield
                total_yield += daily_yield
        if total_yield > 0:
            rc["sweep_efficiency"] += 0.1  # Reward for yield generation

    def _check_overdrafts(self, info: Dict, rc: Dict):
        for acc in self.accounts.values():
            if not acc.is_investment and acc.balance < 0:
                penalty = self.rates.overdraft_fee_daily
                acc.balance -= penalty
                self.total_overdraft_fees += penalty
                rc["overdraft_penalty"] -= 1.5
                info["events"].append(
                    f"⚠ OVERDRAFT: {acc.account_id} balance ${acc.balance:,.0f}. Fee: ${penalty:.0f}"
                )

    def _check_buffer_compliance(self, rc: Dict):
        primary = self.accounts.get("operating")
        if primary:
            if primary.balance >= self.buffer_target:
                rc["buffer_compliance"] += 0.2
            # Partial credit for being close to buffer
            elif primary.balance >= self.buffer_target * 0.75:
                rc["buffer_compliance"] += 0.05

    def _check_idle_cash(self, rc: Dict):
        """Penalize operating cash significantly above buffer when investment account exists."""
        has_investment = any(a.is_investment for a in self.accounts.values())
        if not has_investment:
            return
        primary = self.accounts.get("operating")
        if primary:
            idle_threshold = self.buffer_target * 1.5
            if primary.balance > idle_threshold:
                excess = primary.balance - idle_threshold
                rc["idle_cash_penalty"] -= min(0.2, excess / 1_000_000)

    def build_observation(self, done: bool = False) -> Observation:
        risk_flags = self._compute_risk_flags()
        return Observation(
            day=self.day,
            balances=self.get_all_balances(),
            account_metadata={
                acc_id: {
                    "name": acc.name,
                    "min_balance": acc.min_balance,
                    "is_investment": acc.is_investment,
                }
                for acc_id, acc in self.accounts.items()
            },
            scheduled_inflows=[
                inf for inf in self.inflows if not inf.realized and inf.due_day > self.day
            ],
            scheduled_outflows=[
                obl for obl in self.obligations if not obl.paid
            ],
            pending_transfers=[
                t for t in self.pending_transfers if t.status == TransferStatus.PENDING
            ],
            liquidity_buffer_target=self.buffer_target,
            rates_and_fees=self.rates,
            risk_flags=risk_flags,
            investment_balance=self.get_investment_balance(),
            horizon_days=self.horizon_days,
            task_id=self.task_id,
            done=done,
        )

    def _compute_risk_flags(self) -> List[str]:
        flags = []
        primary = self.accounts.get("operating")
        if primary:
            if primary.balance < self.buffer_target:
                flags.append(f"BELOW_BUFFER: operating at ${primary.balance:,.0f}, target ${self.buffer_target:,.0f}")
            if primary.balance < 0:
                flags.append(f"OVERDRAFT: operating at ${primary.balance:,.0f}")

        # Check upcoming critical obligations
        for obl in self.obligations:
            if not obl.paid and obl.due_day <= self.day + 2:
                priority_str = obl.priority.value if hasattr(obl.priority, 'value') else str(obl.priority)
                if priority_str in ("critical", "high"):
                    acc = self.accounts.get(obl.account_id)
                    if acc and acc.balance < obl.amount:
                        flags.append(
                            f"SHORTFALL: {obl.description} due day {obl.due_day}, "
                            f"need ${obl.amount:,.0f}, have ${acc.balance:,.0f}"
                        )

        return flags
