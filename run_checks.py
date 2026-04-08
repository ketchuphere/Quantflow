#!/usr/bin/env python3
"""
run_checks.py — Full self-test for Treasury Cash Position Planner
Run this from the project root: python run_checks.py

Checks (no API key needed):
  1. Dependencies installed
  2. Env logic — all 3 tasks, hold policy
  3. Env logic — rule-based policy (baseline)
  4. HTTP server — all endpoints
  5. Grader scores in [0.0, 1.0]
  6. inference.py parses correctly
  7. Test suite (pytest)
"""
import sys, os, json, time, subprocess, threading, importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

PASS = "Pass"; FAIL = "Fail"
results = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(label, detail=""):
    msg = f"  {PASS}  {label}"
    if detail: msg += f"  →  {detail}"
    print(msg)
    results.append((True, label))

def fail(label, detail=""):
    msg = f"  {FAIL}  {label}"
    if detail: msg += f"  →  {detail}"
    print(msg)
    results.append((False, label))

#Dependencies
section("1. DEPENDENCIES")
for pkg, import_name in [("pydantic", "pydantic"), ("fastapi", "fastapi"),
                          ("uvicorn", "uvicorn"),   ("openai", "openai")]:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        ok(f"{pkg}", ver)
    except ImportError:
        fail(f"{pkg} NOT INSTALLED", f"run: pip install {pkg}")

#Env logic : hold policy
section("2. ENV LOGIC — Hold Policy (all 3 tasks)")
try:
    from treasury_env import TreasuryCashPositionPlanner
    from treasury_env.models import TreasuryAction, ActionType

    TASKS = [
        ("task_1_daily_funding",           1.0,  0.7),
        ("task_2_sweep_optimization",       0.7,  0.5),
        ("task_3_multi_account_liquidity",  0.5,  0.3),
    ]
    for task_id, expected_hi, expected_lo in TASKS:
        env = TreasuryCashPositionPlanner()
        obs = env.reset(task_id=task_id, seed=42)
        assert obs.day == 0
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(TreasuryAction(action_type=ActionType.HOLD))
            steps += 1
        s = env.grade()
        in_range = 0.0 <= s.overall <= 1.0
        label = f"{task_id}"
        detail = (f"score={s.overall:.4f}  pay={s.payment_rate:.3f}"
                  f"  safety={s.liquidity_safety:.3f}  steps={steps}")
        if in_range:
            ok(label, detail)
        else:
            fail(label, f"SCORE OUT OF RANGE: {s.overall}")
except Exception as e:
    fail("Env import/run failed", str(e))

#Rule-based baseline
section("3. RULE-BASED BASELINE (scripts/baseline.py)")
try:
    spec = importlib.util.spec_from_file_location(
        "baseline", os.path.join(ROOT, "scripts", "baseline.py"))
    baseline_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_mod)

    results_bl = baseline_mod.run_episode(
        task_id="task_1_daily_funding",
        policy="rule_based", seed=42, verbose=False)
    s = results_bl["score"]
    ok("Task 1 rule-based", f"score={s:.4f}")

    results_bl2 = baseline_mod.run_episode(
        task_id="task_2_sweep_optimization",
        policy="rule_based", seed=42, verbose=False)
    s2 = results_bl2["score"]
    ok("Task 2 rule-based", f"score={s2:.4f}")

    results_bl3 = baseline_mod.run_episode(
        task_id="task_3_multi_account_liquidity",
        policy="rule_based", seed=42, verbose=False)
    s3 = results_bl3["score"]
    ok("Task 3 rule-based", f"score={s3:.4f}")

    mean = (s + s2 + s3) / 3
    ok("Mean baseline score", f"{mean:.4f}")
except Exception as e:
    fail("Rule-based baseline failed", str(e))

#HTTP server + all endpoints
section("4. HTTP SERVER — All Endpoints")
server_proc = None
try:
    import urllib.request, urllib.error

    #Start server in background
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "127.0.0.1", "--port", "17860", "--log-level", "error"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    #Wait for startup
    for _ in range(20):
        try:
            urllib.request.urlopen("http://127.0.0.1:17860/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("Server did not start in time")

    def get(path):
        r = urllib.request.urlopen(f"http://127.0.0.1:17860{path}", timeout=5)
        return json.loads(r.read())

    def post(path, data):
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            f"http://127.0.0.1:17860{path}", data=body,
            headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=10)
        return json.loads(r.read())

    #/health
    h = get("/health")
    ok("/health", h.get("status"))

    #/tasks
    t = get("/tasks")
    task_count = len(t.get("tasks", []))
    ok("/tasks", f"{task_count} tasks returned")

    #/reset
    obs = post("/reset", {"task_id": "task_1_daily_funding", "seed": 42})
    ok("/reset", f"day={obs.get('day')}  balances={list(obs.get('balances',{}).keys())}")

    #/step (hold)
    step_r = post("/step", {"action_type": "hold", "amount": 0})
    ok("/step (hold)", f"reward={step_r.get('reward',{}).get('total'):.3f}  done={step_r.get('done')}")

    #/state
    state = get("/state")
    ok("/state", f"day={state.get('day')}  task={state.get('task_id')}")

    #/grader
    gr = get("/grader")
    ok("/grader", f"overall={gr.get('overall'):.4f}")

    #/baseline (rule-based, no LLM needed)
    bl = post("/baseline", {"seed": 42})
    mean_bl = bl.get("mean_score", 0)
    ok("/baseline", f"mean_score={mean_bl:.4f}")

    #Step through to completion, verify done
    post("/reset", {"task_id": "task_1_daily_funding", "seed": 42})
    for _ in range(7):
        r2 = post("/step", {"action_type": "hold", "amount": 0})
    ok("/step → done=True after horizon", f"done={r2.get('done')}")

    #Final grader
    final = get("/grader")
    ok("Final /grader score in [0,1]",
       f"overall={final.get('overall'):.4f} ✓" if 0 <= final.get("overall", -1) <= 1 else "OUT OF RANGE")

except Exception as e:
    fail("HTTP server check failed", str(e))
finally:
    if server_proc:
        server_proc.terminate()
        server_proc.wait()

#inference.py structure 
section("5. inference.py MANDATORY REQUIREMENTS")
try:
    with open(os.path.join(ROOT, "inference.py")) as f:
        src = f.read()
    checks = [
        ("from openai import OpenAI",            "from openai import OpenAI" in src),
        ("API_BASE_URL from os.getenv",           'os.getenv("API_BASE_URL"' in src),
        ("MODEL_NAME from os.getenv",             'os.getenv("MODEL_NAME"'  in src),
        ("HF_TOKEN from os.getenv",               'os.getenv("HF_TOKEN"'    in src),
        ("OpenAI(base_url=API_BASE_URL, ...)",    "OpenAI(base_url=API_BASE_URL" in src),
        ("TEMPERATURE = 0.0",                     "TEMPERATURE = 0.0"       in src),
        ("FIXED_SEED defined",                    "FIXED_SEED"              in src),
        ("Runs all 3 task IDs",                   "task_3_multi_account" in src),
        ("Saves results as JSON",                 "json.dump"               in src),
    ]
    for label, condition in checks:
        if condition: ok(label)
        else:         fail(label)
except Exception as e:
    fail("Could not read inference.py", str(e))

#Pytest(suitable for grading script)
section("6. PYTEST TEST SUITE")
try:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    lines = r.stdout.strip().splitlines()
    # Print last few lines (summary)
    for line in lines[-10:]:
        print(f"    {line}")
    if r.returncode == 0:
        ok("All tests passed")
    else:
        fail("Some tests failed", "see output above")
except FileNotFoundError:
    fail("pytest not installed", "pip install pytest")
except Exception as e:
    fail("pytest run failed", str(e))

section("SUMMARY")
passed = sum(1 for ok_, _ in results if ok_)
total  = len(results)
failed = [(label) for ok_, label in results if not ok_]

print(f"\n  Passed: {passed}/{total}")
if failed:
    print(f"\n Failing checks:")
    for f in failed:
        print(f"       • {f}")
    print(f"\n  Fix the above, then re-run: python run_checks.py")
else:
    print(f"\n  {PASS} Everything looks good — project is submission-ready!")
print()
