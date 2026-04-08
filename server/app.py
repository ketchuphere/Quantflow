"""
FastAPI server exposing the OpenEnv HTTP interface.
Endpoints:
  POST /reset         → Initial observation
  POST /step          → Step the environment
  GET  /state         → Full serializable state
  GET  /tasks         → Task list + action schema
  GET  /grader        → Grader score for current episode
  POST /baseline      → Run baseline inference on all tasks
  GET  /health        → Health check
"""
from __future__ import annotations
from fastapi import Body
import os
import json
import time
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from treasury_env import TreasuryCashPositionPlanner
from treasury_env.models import TreasuryAction, ActionType


class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1_daily_funding"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str = "hold"
    source_account: Optional[str] = None
    destination_account: Optional[str] = None
    amount: float = 0.0
    reference_id: Optional[str] = None
    partial_payment_ok: bool = False


class BaselineRequest(BaseModel):
    seed: int = 42
    model: str = "gpt-4o-mini"

#
_env = TreasuryCashPositionPlanner()
_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env, _initialized
    _env = TreasuryCashPositionPlanner()
    _initialized = True
    yield


app = FastAPI(
    title="Quantflow — Treasury Cash Position Planner",
    description=(
        "OpenEnv-compliant treasury operations simulation. "
        "Agents manage cash across multiple bank accounts, forecast inflows/outflows, "
        "sweep surplus to investments, and avoid overdrafts while meeting all obligations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/reset")
async def reset(req: ResetRequest = Body(default=ResetRequest())):
    """Reset the environment and return the initial observation."""
    try:
        obs = _env.reset(task_id=req.task_id, seed=req.seed)
        return obs.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "env": "treasury_cash_position_planner", "version": "1.0.0"}


@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset the environment and return the initial observation."""
    try:
        obs = _env.reset(task_id=req.task_id, seed=req.seed)
        return obs.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step(req: StepRequest):
    """Execute one action and advance by one day."""
    if _env._sim is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    if _env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )

    try:
        action = TreasuryAction(
            action_type=ActionType(req.action_type),
            source_account=req.source_account,
            destination_account=req.destination_account,
            amount=req.amount,
            reference_id=req.reference_id,
            partial_payment_ok=req.partial_payment_ok,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state():
    """Return full serializable environment state."""
    if _env._sim is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state().dict()


@app.get("/tasks")
async def tasks():
    """List all available tasks with action schema."""
    return {"tasks": _env.tasks()}


@app.get("/grader")
async def grader():
    """Return grader score for the current episode."""
    if _env._sim is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    scores = _env.grade()
    return scores.dict()


@app.post("/baseline")
async def baseline(req: BaselineRequest):
    """
    Run the rule-based baseline policy on all 3 tasks.
    Returns per-task scores and summary. Does NOT require an LLM API key.
    """
    results = run_rule_based_baseline(seed=req.seed)
    return results


def rule_based_policy(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Greedy rule-based policy:
    1. If any account is below buffer → redeem from investment if available.
    2. If operating is above 150% of buffer and investment account exists → invest surplus.
    3. Otherwise hold.
    """
    balances = obs_dict.get("balances", {})
    account_meta = obs_dict.get("account_metadata", {})
    buffer = obs_dict.get("liquidity_buffer_target", 50_000)
    investment_balance = obs_dict.get("investment_balance", 0)

    #Find investment account
    inv_account = None
    for acc_id, meta in account_meta.items():
        if meta.get("is_investment"):
            inv_account = acc_id
            break

    operating_balance = balances.get("operating", 0)

    #Rule 1: Below buffer — redeem from investment
    if operating_balance < buffer * 1.1 and investment_balance > 0 and inv_account:
        needed = buffer * 1.3 - operating_balance
        redeem_amount = min(needed, investment_balance)
        return {
            "action_type": "redeem",
            "source_account": inv_account,
            "destination_account": "operating",
            "amount": round(redeem_amount, 2),
        }

    #Rule 2: Well above buffer — sweep surplus to investment
    if operating_balance > buffer * 1.5 and inv_account:
        surplus = operating_balance - buffer * 1.2
        if surplus > 10_000:
            return {
                "action_type": "invest",
                "source_account": "operating",
                "destination_account": inv_account,
                "amount": round(surplus * 0.8, 2),
            }

    #Rule 3: Check payroll account needs funding
    payroll_balance = balances.get("payroll", None)
    if payroll_balance is not None:
        payroll_meta = account_meta.get("payroll", {})
        payroll_min = payroll_meta.get("min_balance", 10_000)
        # Look for upcoming payroll obligations
        outflows = obs_dict.get("scheduled_outflows", [])
        upcoming_payroll = sum(
            o.get("amount", 0)
            for o in outflows
            if o.get("account_id") == "payroll"
            and o.get("due_day", 99) <= obs_dict.get("day", 0) + 2
            and not o.get("paid", False)
        )
        if upcoming_payroll > 0 and payroll_balance < upcoming_payroll + payroll_min:
            needed = upcoming_payroll + payroll_min - payroll_balance
            if operating_balance > buffer + needed:
                return {
                    "action_type": "transfer",
                    "source_account": "operating",
                    "destination_account": "payroll",
                    "amount": round(needed, 2),
                }

    return {"action_type": "hold", "amount": 0.0}


def run_rule_based_baseline(seed: int = 42) -> Dict[str, Any]:
    """Run rule-based baseline on all 3 tasks."""
    task_ids = [
        "task_1_daily_funding",
        "task_2_sweep_optimization",
        "task_3_multi_account_liquidity",
    ]
    results = {}
    summary = []

    for task_id in task_ids:
        env = TreasuryCashPositionPlanner()
        obs = env.reset(task_id=task_id, seed=seed)
        obs_dict = obs.dict()

        episode_rewards = []
        step_count = 0
        done = False

        while not done:
            action_dict = rule_based_policy(obs_dict)
            try:
                action = TreasuryAction(**{
                    "action_type": ActionType(action_dict.get("action_type", "hold")),
                    "source_account": action_dict.get("source_account"),
                    "destination_account": action_dict.get("destination_account"),
                    "amount": action_dict.get("amount", 0.0),
                })
                obs, reward, done, info = env.step(action)
                obs_dict = obs.dict()
                episode_rewards.append(reward.total)
                step_count += 1
            except Exception as e:
                done = True
                break

        final_score = env.grade()
        task_result = {
            "task_id": task_id,
            "score": final_score.overall,
            "payment_rate": final_score.payment_rate,
            "liquidity_safety": final_score.liquidity_safety,
            "efficiency": final_score.efficiency,
            "compliance": final_score.compliance,
            "cumulative_reward": round(sum(episode_rewards), 4),
            "steps": step_count,
            "details": final_score.details,
        }
        results[task_id] = task_result
        summary.append({
            "task": task_id,
            "score": final_score.overall,
            "steps": step_count,
        })

    return {
        "policy": "rule_based_greedy",
        "seed": seed,
        "task_results": results,
        "summary": summary,
        "mean_score": round(
            sum(r["score"] for r in results.values()) / len(results), 4
        ),
    }

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/")
def home():
    return {"message": "Quantflow API running "}

import uvicorn

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )

if __name__ == "__main__":
    main()