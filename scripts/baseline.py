"""
Baseline inference script for the Treasury Cash Position Planner.
Runs a rule-based policy OR an LLM-guided policy (via OpenAI API) against
all 3 tasks and prints/saves reproducible baseline scores.

Usage:
    # Rule-based baseline (no API key needed):
    python scripts/baseline.py --policy rule_based

    # LLM-guided baseline:
    OPENAI_API_KEY=sk-... python scripts/baseline.py --policy llm --model gpt-4o-mini

    # Save output:
    python scripts/baseline.py --policy rule_based --output results.json
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from treasury_env import TreasuryCashPositionPlanner
from treasury_env.models import TreasuryAction, ActionType

TASK_IDS = [
    "task_1_daily_funding",
    "task_2_sweep_optimization",
    "task_3_multi_account_liquidity",
]

FIXED_SEED = 42
FIXED_TEMPERATURE = 0.0  # Deterministic LLM output


def rule_based_policy(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Greedy rule-based policy:
    1. Fund payroll if upcoming payroll obligation detected.
    2. Redeem from investment if below buffer.
    3. Invest surplus above 150% buffer.
    4. Otherwise hold.
    """
    balances = obs_dict.get("balances", {})
    account_meta = obs_dict.get("account_metadata", {})
    buffer = obs_dict.get("liquidity_buffer_target", 50_000)
    investment_balance = obs_dict.get("investment_balance", 0)
    day = obs_dict.get("day", 0)

    inv_account = next(
        (acc_id for acc_id, meta in account_meta.items() if meta.get("is_investment")),
        None,
    )
    operating_balance = balances.get("operating", 0)

    # Rule 1: Pre-fund payroll
    payroll_balance = balances.get("payroll", None)
    if payroll_balance is not None:
        outflows = obs_dict.get("scheduled_outflows", [])
        upcoming_payroll = sum(
            o.get("amount", 0)
            for o in outflows
            if o.get("account_id") == "payroll"
            and o.get("due_day", 99) <= day + 2
            and not o.get("paid", False)
        )
        if upcoming_payroll > 0 and payroll_balance < upcoming_payroll:
            needed = upcoming_payroll - payroll_balance + 10_000
            if operating_balance > buffer + needed:
                return {
                    "action_type": "transfer",
                    "source_account": "operating",
                    "destination_account": "payroll",
                    "amount": round(needed, 2),
                }

    # Rule 2: Below buffer → redeem
    if operating_balance < buffer * 1.05 and investment_balance > 0 and inv_account:
        needed = buffer * 1.3 - operating_balance
        return {
            "action_type": "redeem",
            "source_account": inv_account,
            "destination_account": "operating",
            "amount": round(min(needed, investment_balance), 2),
        }

    # Rule 3: Well above buffer → sweep to investment
    if operating_balance > buffer * 1.6 and inv_account:
        surplus = operating_balance - buffer * 1.3
        if surplus > 15_000:
            return {
                "action_type": "invest",
                "source_account": "operating",
                "destination_account": inv_account,
                "amount": round(surplus * 0.75, 2),
            }

    return {"action_type": "hold", "amount": 0.0}


SYSTEM_PROMPT = """You are a treasury operations AI assistant managing cash for a company.
Your goal is to:
1. Always keep the operating account above the liquidity_buffer_target.
2. Pay all due obligations on time (especially CRITICAL and HIGH priority).
3. Sweep excess cash to investment accounts to earn yield.
4. Minimize overdrafts, missed payments, and unnecessary fees.

You must respond with ONLY a valid JSON action object (no markdown, no explanation):
{
  "action_type": "hold|transfer|invest|redeem|emergency_fund",
  "source_account": "<account_id or null>",
  "destination_account": "<account_id or null>",
  "amount": <float>,
  "reference_id": "<optional string>"
}

Rules:
- action_type "hold": Do nothing. Use when no action is needed.
- action_type "transfer": Move cash between accounts (T+1 settlement unless noted).
- action_type "invest": Sweep surplus from operating to investment account.
- action_type "redeem": Pull funds back from investment to operating.
- action_type "emergency_fund": Draw expensive credit line (1.5% fee) — last resort only.

Always maintain the buffer. Never let critical obligations go unpaid.
"""


def llm_policy(obs_dict: Dict[str, Any], client, model: str) -> Dict[str, Any]:
    """LLM-guided policy using OpenAI-compatible API."""
    obs_summary = json.dumps(obs_dict, indent=2, default=str)
    user_msg = f"""Current treasury state (day {obs_dict.get('day', 0)}):

{obs_summary}

What action should the treasury desk take today? Respond with only a JSON action object."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=FIXED_TEMPERATURE,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        #Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        action_dict = json.loads(content)
        return action_dict
    except Exception as e:
        print(f"  [LLM error]: {e} — falling back to hold")
        return {"action_type": "hold", "amount": 0.0}


def run_episode(
    task_id: str,
    policy: str,
    seed: int,
    client=None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    env = TreasuryCashPositionPlanner()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.dict()

    episode_rewards = []
    step_count = 0
    done = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id} | Policy: {policy} | Seed: {seed}")
        print(f"Initial balances: { {k: f'${v:,.0f}' for k, v in obs_dict['balances'].items()} }")
        print(f"{'='*60}")

    while not done:
        # Get action from policy
        if policy == "rule_based":
            action_dict = rule_based_policy(obs_dict)
        else:
            action_dict = llm_policy(obs_dict, client, model)

        # Execute action
        try:
            action = TreasuryAction(
                action_type=ActionType(action_dict.get("action_type", "hold")),
                source_account=action_dict.get("source_account"),
                destination_account=action_dict.get("destination_account"),
                amount=float(action_dict.get("amount", 0.0)),
                reference_id=action_dict.get("reference_id"),
            )
            obs, reward, done, info = env.step(action)
            obs_dict = obs.dict()
            episode_rewards.append(reward.total)
            step_count += 1

            if verbose:
                print(f"\nDay {info['day']}: Action={action_dict.get('action_type')} "
                      f"amount=${action_dict.get('amount', 0):,.0f}")
                for event in info.get("events", []):
                    print(f"  {event}")
                print(f"  Reward: {reward.total:+.3f} | "
                      f"Cumulative: {info['cumulative_reward']:+.3f}")
                gs = info.get("grader_scores", {})
                print(f"  Score so far: {gs.get('overall', 0):.3f} | "
                      f"Payment rate: {gs.get('payment_rate', 0):.3f}")

        except Exception as e:
            if verbose:
                print(f"  [Step error]: {e}")
            done = True

    final_score = env.grade()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"FINAL SCORE: {final_score.overall:.4f}")
        print(f"  payment_rate:     {final_score.payment_rate:.4f}")
        print(f"  liquidity_safety: {final_score.liquidity_safety:.4f}")
        print(f"  efficiency:       {final_score.efficiency:.4f}")
        print(f"  compliance:       {final_score.compliance:.4f}")
        print(f"  cumulative_reward: {sum(episode_rewards):.4f}")
        print(f"  steps: {step_count}")
        for k, v in (final_score.details or {}).items():
            print(f"  {k}: {v}")

    return {
        "task_id": task_id,
        "policy": policy,
        "seed": seed,
        "score": final_score.overall,
        "payment_rate": final_score.payment_rate,
        "liquidity_safety": final_score.liquidity_safety,
        "efficiency": final_score.efficiency,
        "compliance": final_score.compliance,
        "cumulative_reward": round(sum(episode_rewards), 4),
        "steps": step_count,
        "details": final_score.details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Treasury Cash Position Planner — Baseline Inference Script"
    )
    parser.add_argument(
        "--policy", choices=["rule_based", "llm"], default="rule_based",
        help="Policy to use: rule_based (no API key) or llm (requires OPENAI_API_KEY)"
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--seed", type=int, default=FIXED_SEED, help="Random seed")
    parser.add_argument("--task", default=None, help="Run a single task (optional)")
    parser.add_argument("--output", default=None, help="Save results to JSON/CSV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    # Initialize LLM client if needed
    client = None
    if args.policy == "llm":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except ImportError:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)

    task_ids = [args.task] if args.task else TASK_IDS
    all_results = []

    print(f"\n{'#'*60}")
    print(f"  Quantflow — Treasury Cash Position Planner")
    print(f"  Baseline Policy: {args.policy.upper()}")
    if args.policy == "llm":
        print(f"  Model: {args.model}")
    print(f"  Seed: {args.seed}")
    print(f"{'#'*60}")

    for task_id in task_ids:
        result = run_episode(
            task_id=task_id,
            policy=args.policy,
            seed=args.seed,
            client=client,
            model=args.model,
            verbose=not args.quiet,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<40} {'Score':>8}")
    print(f"{'-'*50}")
    for r in all_results:
        name = r["task_id"].replace("task_", "T").replace("_", " ")
        print(f"{name:<40} {r['score']:>8.4f}")
    mean = sum(r["score"] for r in all_results) / len(all_results)
    print(f"{'-'*50}")
    print(f"{'Mean score':<40} {mean:>8.4f}")
    print(f"{'='*60}\n")

    # Save output
    if args.output:
        output_data = {
            "policy": args.policy,
            "model": args.model if args.policy == "llm" else "n/a",
            "seed": args.seed,
            "results": all_results,
            "mean_score": round(mean, 4),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")

    return all_results


if __name__ == "__main__":
    main()
