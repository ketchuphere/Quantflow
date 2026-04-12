"""
inference.py — Treasury Cash Position Planner

Environment variables required:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face API token (used as API key)

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \
    MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
    HF_TOKEN=hf_xxx \
    python inference.py

    # Override seed or run only one task:
    python inference.py --seed 0 --task task_1_daily_funding
    python inference.py --output results.json

"""
from __future__ import annotations

import os
import sys
import json
import re
import textwrap
import argparse
from typing import List, Dict, Any, Optional, final
from openai import OpenAI

#Treasury environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from treasury_env import TreasuryCashPositionPlanner
from treasury_env.models import TreasuryAction, ActionType

#Config from environment variables (mandatory)
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

#Inference hyperparameters
MAX_STEPS   = 14        # Upper bound; tasks end at their own horizon
TEMPERATURE = 0.0       # Deterministic — reproducible baseline
MAX_TOKENS  = 256
FIXED_SEED  = 42
FALLBACK_ACTION: Dict[str, Any] = {"action_type": "hold", "amount": 0.0}

TASK_IDS = [
    "task_1_daily_funding",
    "task_2_sweep_optimization",
    "task_3_multi_account_liquidity",
]

#JSON extraction regex
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_BRACE_RE      = re.compile(r"\{[\s\S]*\}", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert corporate treasury manager controlling a cash position simulation.

    Your objectives (in priority order):
    1. NEVER miss a CRITICAL-priority payment — these carry huge penalties.
    2. Keep the operating account above the liquidity_buffer_target at all times.
    3. Pre-fund payroll account before payroll obligations fall due (check due_day).
    4. Sweep surplus cash (above 1.5× buffer) to investment accounts for yield.
    5. Redeem from investment when operating balance falls near the buffer.
    6. Use emergency_fund ONLY as an absolute last resort (1.5% fee).

    ACCOUNT IDs you may see:
      "operating"  — main account, must stay above buffer
      "sweep" / "investment" — investment account (earns daily yield ~5.5% APY)
      "payroll"    — must be funded before payroll due_day
      "reserve"    — secondary liquidity pool

    TRANSFER SETTLEMENT: transfers settle T+1 (next day). Plan one day ahead.

    RESPOND with ONLY a valid JSON object — no markdown, no explanation:
    {
      "action_type": "hold|transfer|invest|redeem|emergency_fund",
      "source_account": "<account_id or null>",
      "destination_account": "<account_id or null>",
      "amount": <float>,
      "reference_id": "<optional string or null>"
    }
""").strip()


def _build_user_prompt(step: int, obs: Dict[str, Any]) -> str:
    """Construct a concise but information-rich prompt for the LLM."""
    day       = obs.get("day", step)
    horizon   = obs.get("horizon_days", "?")
    buffer    = obs.get("liquidity_buffer_target", 0)
    balances  = obs.get("balances", {})
    inv_bal   = obs.get("investment_balance", 0)
    risk      = obs.get("risk_flags", [])
    pending   = obs.get("pending_transfers", [])

    #Summarise upcoming obligations (unpaid, due soon)
    outflows = obs.get("scheduled_outflows", [])
    upcoming = [
        o for o in outflows
        if not o.get("paid", False) and o.get("due_day", 999) <= day + 3
    ]
    upcoming.sort(key=lambda o: (o.get("due_day", 999),
                                  {"critical":0,"high":1,"normal":2,"low":3}
                                   .get(o.get("priority","low"), 3)))

    #Format balances
    bal_lines = "\n".join(
        f"  {acct}: ${bal:>12,.2f}" for acct, bal in balances.items()
    )

    #Format upcoming obligations
    obl_lines = "\n".join(
        f"  [{o.get('priority','?').upper():8s}] Day {o.get('due_day'):>2d} — "
        f"${o.get('amount',0):>10,.2f} from '{o.get('account_id')}' "
        f"({o.get('description','')})"
        for o in upcoming
    ) or "  (none in next 3 days)"

    #Format pending transfers
    pend_lines = "\n".join(
        f"  ${t.get('amount',0):>10,.2f} → {t.get('destination_account')} "
        f"(settles day {t.get('settlement_day')})"
        for t in pending
    ) or "  (none)"

    #Format expected inflows (certain and likely)
    inflows = obs.get("scheduled_inflows", [])
    inflow_soon = [
        i for i in inflows
        if not i.get("realized", False) and i.get("due_day", 999) <= day + 4
    ]
    inflow_lines = "\n".join(
        f"  Day {i.get('due_day'):>2d} — ${i.get('amount',0):>10,.2f} → "
        f"'{i.get('account_id')}' (p={i.get('probability',1.0):.0%}) "
        f"{i.get('description','')}"
        for i in inflow_soon
    ) or "  (none in next 4 days)"

    rates = obs.get("rates_and_fees", {})

    prompt = textwrap.dedent(f"""
        === Treasury Desk — Day {day}/{horizon} ===

        CURRENT BALANCES:
        {bal_lines}
          investment_total: ${inv_bal:>12,.2f}

        LIQUIDITY BUFFER TARGET: ${buffer:,.2f}

        RISK FLAGS: {risk if risk else "none"}

        UPCOMING OBLIGATIONS (next 3 days):
        {obl_lines}

        EXPECTED INFLOWS (next 4 days):
        {inflow_lines}

        PENDING IN-FLIGHT TRANSFERS (T+1):
        {pend_lines}

        RATES & FEES:
          transfer_fee_flat:     ${rates.get('transfer_fee_flat', 0):.2f}
          transfer_fee_pct:      {rates.get('transfer_fee_pct', 0)*100:.4f}%
          overdraft_fee_daily:   ${rates.get('overdraft_fee_daily', 0):.2f}/day
          investment_yield_daily:{rates.get('investment_yield_daily', 0)*100:.4f}%/day
          emergency_fund_fee:    {rates.get('emergency_fund_fee_pct', 0)*100:.2f}%
          settlement_days:       T+{rates.get('transfer_settlement_days', 1)}

        Decide the single best treasury action for Day {day}.
        Respond with ONLY the JSON action object.
    """).strip()
    return prompt


#LLM interaction utilities: call and response parsing
def _call_llm(client: OpenAI, step: int, obs: Dict[str, Any]) -> str:
    """Call the LLM and return raw response text."""
    user_prompt = _build_user_prompt(step, obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return completion.choices[0].message.content or ""


def _parse_action(response_text: str) -> Dict[str, Any]:
    """
    Extract a JSON action dict from the LLM response.
    Handles: raw JSON, ```json ... ```, stray text before/after braces.
    Falls back to hold on any parse failure.
    """
    if not response_text.strip():
        return FALLBACK_ACTION.copy()

    # 1. Try fenced code block
    m = _JSON_BLOCK_RE.search(response_text)
    candidate = m.group(1) if m else None

    # 2. Try bare braces
    if candidate is None:
        m2 = _BRACE_RE.search(response_text)
        candidate = m2.group(0) if m2 else None

    if candidate is None:
        return FALLBACK_ACTION.copy()

    try:
        action = json.loads(candidate)
        # Normalise: ensure action_type is a valid string
        if "action_type" not in action:
            return FALLBACK_ACTION.copy()
        # Ensure amount is float
        action["amount"] = float(action.get("amount", 0.0))
        return action
    except (json.JSONDecodeError, ValueError):
        return FALLBACK_ACTION.copy()

#episode runner
def run_episode(
    task_id: str,
    client: OpenAI,
    seed: int = FIXED_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one full episode of the given task using the LLM policy.
    Returns a result dict with scores and metadata.
    """
    env = TreasuryCashPositionPlanner()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.dict()
    print(f"[START] task={task_id}", flush=True)
    step_rewards: List[float] = []
    action_log:   List[Dict[str, Any]] = []
    done = False
    step = 0

    if verbose:
        print(f"\n")
        print(f"  TASK: {task_id}  |  seed={seed}  |  model={MODEL_NAME}")
        print(f"  Initial balances: { {k: f'${v:,.0f}' for k,v in obs_dict['balances'].items()} }")
        print(f"\n")

    while not done and step < MAX_STEPS:
        step += 1

        
        try:
            raw_response = _call_llm(client, step, obs_dict)
        except Exception as exc:
            if verbose:
                print(f"  [Step {step}] LLM call failed: {exc} — using fallback")
            raw_response = ""

        action_dict = _parse_action(raw_response)

        if verbose:
            print(f"\n  Day {step}: action={action_dict.get('action_type')} "
                  f"amount=${action_dict.get('amount', 0):,.0f} "
                  f"src={action_dict.get('source_account')} "
                  f"dst={action_dict.get('destination_account')}")

        
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
        except Exception as exc:
            if verbose:
                print(f"    [Step error]: {exc} — using hold fallback")
            action = TreasuryAction(action_type=ActionType.HOLD)
            obs, reward, done, info = env.step(action)
            obs_dict = obs.dict()

        step_rewards.append(reward.total)
        print(f"[STEP] step={step} reward={reward.total}", flush=True)
        action_log.append({
            "day": step,
            "action": action_dict,
            "reward": reward.total,
            "events": info.get("events", []),
        })

        if verbose:
            for event in info.get("events", []):
                print(f"    {event}")
            gs = info.get("grader_scores", {})
            print(f"    reward={reward.total:+.3f} | "
                  f"score_so_far={gs.get('overall', 0):.3f} | "
                  f"payment_rate={gs.get('payment_rate', 0):.3f}")

    
    final = env.grade()
    epsilon = 1e-3
    safe_score = max(min(final.overall, 1 - epsilon), epsilon)
    print(f"[END] task={task_id} score={safe_score} steps={step}", flush=True)

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  FINAL SCORE:      {safe_score:.4f}")
        print(f"  payment_rate:     {final.payment_rate:.4f}")
        print(f"  liquidity_safety: {final.liquidity_safety:.4f}")
        print(f"  efficiency:       {final.efficiency:.4f}")
        print(f"  compliance:       {final.compliance:.4f}")
        print(f"  cumulative_reward:{sum(step_rewards):.4f}")
        print(f"  steps:            {step}")

    return {
        "task_id": task_id,
        "model": MODEL_NAME,
        "seed": seed,
        "score": safe_score,
        "payment_rate": final.payment_rate,
        "liquidity_safety": final.liquidity_safety,
        "efficiency": final.efficiency,
        "compliance": final.compliance,
        "cumulative_reward": round(sum(step_rewards), 4),
        "steps": step,
        "action_log": action_log,
        "details": final.details
    }



#main inference loop with argument parsing and result saving
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantflow — LLM inference script (OpenEnv hackathon)"
    )
    parser.add_argument("--seed",   type=int, default=FIXED_SEED,
                        help="Random seed (default: 42)")
    parser.add_argument("--task",   default=None,
                        help="Run a single task ID (default: all 3 tasks)")
    parser.add_argument("--output", default=None,
                        help="Save JSON results to this file path")
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    # Sanity check environment variables
    if not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY) environment variable is not set.")
        print("  Export your Hugging Face token:  export HF_TOKEN=hf_xxx")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is not set.")
        sys.exit(1)

    # Initialize LLM client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = [args.task] if args.task else TASK_IDS

    print(f"\n")
    print(f"  Quantflow — Treasury Cash Position Planner")
    print(f"  Inference script  |  model: {MODEL_NAME}")
    print(f"  API base: {API_BASE_URL}")
    print(f"  Seed: {args.seed}  |  Tasks: {len(task_ids)}")
    print(f"{'-'*30}")

    all_results: List[Dict[str, Any]] = []

    for task_id in task_ids:
        result = run_episode(
            task_id=task_id,
            client=client,
            seed=args.seed,
            verbose=not args.quiet,
        )
        all_results.append(result)

   
    print(f"\n")
    print("  INFERENCE SUMMARY")
    print(f"{'-'*40}")
    print(f"  {'Task':<42} {'Score':>8}")
    print(f"\n")
    for r in all_results:
        name = r["task_id"].replace("task_", "T").replace("_", " ")
        print(f"  {name:<42} {r['score']:>8.4f}")
    mean = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n")
    print(f"  {'Mean score':<42} {mean:>8.4f}")
    print(f"\n")

    
    output_path = args.output or "inference_results.json"
    payload = {
        "model":       MODEL_NAME,
        "api_base":    API_BASE_URL,
        "seed":        args.seed,
        "mean_score":  round(mean, 4),
        "results":     all_results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()
