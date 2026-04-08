
# Quantflow(Treasury Cash Position Planner)

> *An AI agent that thinks like a CFO managing millions in cash, juggling payments, and maximizing returns. Every. Single. Day.*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen)](https://python.org)
---

## The Problem We're Solving

Every morning at a real company, a treasury team sits down and answers the same stressful questions:

- *Do we have enough cash to make payroll on Friday?*
- *We have $500K sitting idle should we sweep it into the money market?*
- *A $120K supplier payment is due today but our operating account is at $80K — what do we do?*

This is **cash positioning** one of the most consequential operational tasks in corporate finance. Get it wrong and you miss payroll. Get it right and you earn millions in yield on idle cash.

**Quantflow** turns this into a learning environment where AI agents can master treasury operations through trial, reward, and iteration.

---

## What Makes This Different

Most finance AI demos are just sentiment classifiers or stock price predictors. Quantflow models an actual *operational workflow* with real constraints:

| Real Treasury Problem | How Quantflow Models It |
|---|---|
| Multiple bank accounts | 4 accounts: operating, payroll, reserve, investment |
| Payments have deadlines & priorities | Critical → High → Normal → Low with penalty tiers |
| Cash does not move instantly | T+1 transfer settlement — plan ahead or fail |
| Some income is uncertain | Probabilistic inflows (50%–80% confidence) |
| Idle cash has opportunity cost | Investment account earns ~5.5% APY daily yield |
| Emergency credit is expensive | Draw fee: 1.5% of amount — last resort only |

---

## The Three Challenges

### Task 1 — Daily Funding `(Easy)`

**Scenario:** Single operating account. $120K balance. 4 payments due over 7 days totalling $135K. Inflows arrive in between.

**Agent must:** Keep balance above $50K buffer. Pay every obligation on time. No tricks needed — just don't run dry.

**Scoring threshold:** 0.70

---

### Task 2 — Sweep Optimization `(Medium)`

**Scenario:** Two accounts : operating and a money market sweep. Transfer fees apply. Funds take 1 day to settle (T+1).

**Agent must:** Pay all 5 obligations across 7 days *and* maximize yield by sweeping surplus into the investment account. Timing matters — sweep too late and you earn nothing; sweep too much and you cannot pay tomorrow's bills.

**Scoring threshold:** 0.65

---

### Task 3 — Multi-Account Liquidity Planning `(Hard)`

**Scenario:** 4 accounts. 14 days. 12 obligations across priority tiers. 7 inflows, 4 of which are probabilistic (might not arrive). T+1 settlement throughout.

**Agent must:** Prefund payroll before the deadline, protect critical debt service payments, invest surplus for yield, and never trigger an overdraft — all while uncertain whether three key client payments will actually arrive.

**Scoring threshold:** 0.55

---

## How Scoring Works

Every episode is graded 0.0 to 1.0 on four dimensions:

```
Final Score = 0.35 × payment_rate
            + 0.25 × liquidity_safety
            + 0.20 × efficiency
            + 0.20 × compliance
```

| Dimension | What It Measures |
|---|---|
| `payment_rate` | Obligations paid on time, weighted by priority |
| `liquidity_safety` | Did any account go into overdraft? Was buffer maintained? |
| `efficiency` | How much yield did the agent earn vs theoretical maximum? |
| `compliance` | Did the agent avoid expensive emergency credit draws? |

The agent also gets **shaped rewards every step** — not just at the end:

| Event | Reward |
|---|---|
| Critical payment made on time | +1.0 |
| Buffer maintained above target | +0.2 |
| Investment yield earned | +0.1 |
| Overdraft triggered | -1.5 |
| Critical payment missed | -2.0 |
| Emergency fund drawn | -0.5 |

---

## Baseline Results

Reproducible at `seed=42`.

| Task | Hold-Only | Rule-Based Agent |
|---|---|---|
| Task 1: Daily Funding | 1.00 | **1.00** |
| Task 2: Sweep Optimization | 0.80 | **0.87** |
| Task 3: Multi-Account | 0.64 | **0.72** |

A frontier LLM agent with good treasury reasoning should push Task 3 above **0.85**.

---

## Quickstart (Local - No Docker)

### Step 1 — Install dependencies

```powershell
pip install pydantic fastapi uvicorn openai pytest
```

### Step 2 — Start the API server

```powershell
python app.py
```

You will see:
```
INFO:     Uvicorn running on http://0.0.0.0:7860
```

### Step 3 — Confirm it is running

```powershell
curl.exe http://localhost:7860/health
```

Expected response:
```json
{"status": "ok", "env": "treasury_cash_position_planner", "version": "1.0.0"}
```

---

## API Walkthrough (PowerShell)

### List all available tasks

```powershell
curl.exe http://localhost:7860/tasks
```

---

### Reset environment — Task 1 (Easy)

```powershell
curl.exe --% -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{ \"task_id\": \"task_1_daily_funding\", \"seed\": 42 }"
```

---

### Reset environment — Task 2 (Medium)

```powershell
curl.exe --% -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{ \"task_id\": \"task_2_sweep_optimization\", \"seed\": 42 }"
```

---

### Reset environment — Task 3 (Hard)

```powershell
curl.exe --% -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{ \"task_id\": \"task_3_multi_account_liquidity\", \"seed\": 42 }"
```

---

### Action — Hold (do nothing today)

```powershell
curl.exe --% -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{ \"action_type\": \"hold\", \"amount\": 0 }"
```

---

### Action — Invest surplus cash

```powershell
curl.exe --% -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{ \"action_type\": \"invest\", \"source_account\": \"operating\", \"amount\": 50000 }"
```

---

### Action — Transfer cash to payroll account

```powershell
curl.exe --% -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{ \"action_type\": \"transfer\", \"source_account\": \"operating\", \"destination_account\": \"payroll\", \"amount\": 95000 }"
```

---

### Action — Redeem from investment back to operating

```powershell
curl.exe --% -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{ \"action_type\": \"redeem\", \"source_account\": \"investment\", \"destination_account\": \"operating\", \"amount\": 30000 }"
```

---

### Action — Emergency fund (last resort, 1.5% fee)

```powershell
curl.exe --% -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{ \"action_type\": \"emergency_fund\", \"destination_account\": \"operating\", \"amount\": 20000 }"
```

---

### Check your score at any point

```powershell
curl.exe http://localhost:7860/grader
```

Returns:
```json
{
  "overall": 0.8750,
  "payment_rate": 1.0,
  "liquidity_safety": 1.0,
  "efficiency": 0.52,
  "compliance": 1.0
}
```

---

### Check full environment state

```powershell
curl.exe http://localhost:7860/state
```

---

### Run rule-based baseline — no API key needed

```powershell
curl.exe --% -X POST http://localhost:7860/baseline -H "Content-Type: application/json" -d "{ \"seed\": 42 }"
```

Runs all 3 tasks with a greedy policy and returns scores in ~2 seconds.

---

## Docker

### Build the image

```powershell
docker build -t quantflow .
```

### Run the container

```powershell
docker run -p 7860:7860 `
  -e HF_TOKEN=$env:HF_TOKEN `
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" `
  quantflow
```

### Verify it is running

```powershell
curl.exe http://localhost:7860/health
```

### Reset inside Docker

```powershell
curl.exe --% -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{ \"task_id\": \"task_1_daily_funding\", \"seed\": 42 }"
```

---

## LLM Inference Script

This runs a real LLM agent through all 3 tasks and saves results.

### Set environment variables

```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN     = "hf_your_token_here"
```

### Run all 3 tasks

```powershell
python inference.py
```

### Run a single task

```powershell
python inference.py --task task_1_daily_funding
```

### Save results to file

```powershell
python inference.py --output results.json
```

### Reproduce rule-based baseline scores

```powershell
python scripts/baseline.py --policy rule_based --seed 42 --output baseline_scores.json
```

---

## Run Tests

```powershell
pytest tests/ -v
```

Runs 35+ tests — reset/step/state correctness, grader range, determinism, difficulty progression.

---

## Full Pre-Submission Validation

```powershell
python run_checks.py
```

Automatically checks: dependencies, all 3 tasks, all 7 HTTP endpoints, grader scores in [0,1], inference.py requirements, and the full test suite.

---

## Observation Space

What the agent sees each step:

```python
{
    "day": int,                        # Today (0 = start)
    "balances": {                      # Live account balances in USD
        "operating":  float,
        "payroll":    float,
        "reserve":    float,
        "investment": float
    },
    "scheduled_inflows": [{            # Cash arriving soon
        "amount":      float,
        "due_day":     int,
        "probability": float,          # 1.0 = guaranteed, 0.5 = uncertain
        "account_id":  str,
        "description": str
    }],
    "scheduled_outflows": [{           # Bills that must be paid
        "amount":    float,
        "due_day":   int,
        "priority":  "critical | high | normal | low",
        "account_id": str,
        "paid":      bool
    }],
    "pending_transfers": [{            # In-flight transfers (T+1 settlement)
        "amount":              float,
        "destination_account": str,
        "settlement_day":      int
    }],
    "liquidity_buffer_target": float,  # Minimum operating balance required
    "investment_balance":      float,  # Total currently earning yield
    "risk_flags":              list,   # Warnings — shortfalls, overdrafts
    "rates_and_fees": {
        "transfer_fee_flat":       float,
        "overdraft_fee_daily":     float,
        "investment_yield_daily":  float,
        "emergency_fund_fee_pct":  float,
        "transfer_settlement_days": int
    }
}
```

---

## Action Space

```json
{
  "action_type":        "hold | transfer | invest | redeem | emergency_fund",
  "source_account":     "operating | payroll | reserve | investment | null",
  "destination_account":"operating | payroll | reserve | investment | null",
  "amount":             50000.00,
  "reference_id":       "optional-label"
}
```

| Action | When to use |
|---|---|
| `hold` | Nothing needs doing today |
| `transfer` | Move cash between accounts — T+1 settlement, fee applies |
| `invest` | Sweep surplus into investment account to earn yield |
| `redeem` | Pull cash back from investment to operating |
| `emergency_fund` | Draw expensive credit line — 1.5% fee, crisis only |

---

## Project Structure

```
treasury_cash_position_planner/
│
├── inference.py              ← LLM agent script (start here for AI runs)
├── app.py                    ← FastAPI server — all 7 HTTP endpoints
├── run_checks.py             ← Full pre-submission validation
│
├── src/treasury_env/
│   ├── env.py                ← reset() / step() / state()
│   ├── simulator.py          ← Day logic, fees, rewards, T+1 settlement
│   ├── grader.py             ← Deterministic scoring 0.0 to 1.0
│   ├── tasks.py              ← 3 scenario definitions
│   └── models.py             ← All Pydantic typed models
│
├── scripts/
│   └── baseline.py           ← Rule-based agent (no API key needed)
│
├── tests/
│   ├── test_env.py
│   └── test_grader.py
│
├── openenv.yaml              ← OpenEnv metadata
├── Dockerfile
└── pyproject.toml
```

---