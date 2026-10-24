"""Agent Guard web app: run with `.venv/bin/python app.py` from the repo root (http://127.0.0.1:8105).

With MUBIT_API_KEY set, typed chat messages call the agent live. Without it (the public site), scripted customers
replay the agent turns recorded in results.json, and the "try a refund" form sends a request straight to the guard.
"""

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import guard
from eval import RESULTS_FILE, load_cases

load_dotenv(guard.HERE / ".env")
if os.environ.get("VERCEL"):
    # Vercel functions can only write to /tmp, so the answer caches and call counters live there.
    guard.CACHE = Path("/tmp/agent-guard-cache")

STATIC = guard.HERE / "static"
STORIES = guard.DATA / "stories"
app = FastAPI(title="Agent Guard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
CASES = {c["case_id"]: c for c in load_cases()}
RESULTS = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
RECORDED = {row["case_id"]: row for row in RESULTS.get("cases", [])}
# Scripted customers for the demo buttons: tricks the agent fell for, and honest requests.
SCRIPTED = ["c15", "c27", "c20", "c22", "c30", "c35", "c14", "c01"]
LIVE_TIMEOUT_S = 12  # the longest a visitor waits for Sapling before a friendly error
LIVE_CHECKS_PER_MINUTE = 6  # per IP, on the live "try a refund" route
_llm = None


def _quick_post(url, json, timeout):
    import httpx

    return httpx.post(url, json=json, timeout=min(timeout, LIVE_TIMEOUT_S))


SAPLING = guard.Sapling(post=_quick_post, cache_dir=guard.CACHE / "sapling", sleep=lambda s: time.sleep(min(s, 1)))
_pool = ThreadPoolExecutor(max_workers=guard.SAPLING_CONCURRENCY)


def guard_check(order_id, amount_gbp, reason):
    """guard.check_refund with a hard deadline, so a slow or waking Sapling endpoint cannot hang the page."""
    return _pool.submit(guard.check_refund, SAPLING, order_id, amount_gbp, reason).result(timeout=LIVE_TIMEOUT_S)


class RateLimit:
    """At most `limit` hits per `window` seconds for each key. Kept in memory, so it is per server instance."""

    def __init__(self, limit, window=60, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits = {}
        self.lock = threading.Lock()

    def allow(self, key):
        now = self.clock()
        with self.lock:
            if len(self.hits) > 5000:
                self.hits = {k: q for k, q in self.hits.items() if q and now - q[-1] < self.window}
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


LIVE_LIMIT = RateLimit(LIVE_CHECKS_PER_MINUTE)


def client_ip(request):
    """The visitor's IP. Vercel sets x-real-ip itself; elsewhere the headers could be faked, so use the socket."""
    if os.environ.get("VERCEL"):
        forwarded = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]
        if forwarded.strip():
            return forwarded.strip()
    return request.client.host if request.client else "unknown"


def live_agent_on():
    return bool(os.environ.get("MUBIT_API_KEY"))


def get_llm():
    global _llm
    if _llm is None:
        _llm = agent.make_llm()
    return _llm


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    case_id: str | None = None


class TryRequest(BaseModel):
    order_id: str
    request: str = Field(max_length=600)
    amount_gbp: float | None = Field(default=None, gt=0, le=5000)


def customer_reply(agent_reply, attempt, outcome):
    """What the customer sees once the guard has decided. The agent's own words are kept only when money moves."""
    if not attempt:
        return agent_reply
    amount = f"£{attempt['amount_gbp']:.2f}"
    if outcome == "allow":
        return agent_reply or f"Done: I have refunded {amount} to your original payment method."
    if outcome == "block":
        return f"Sorry, I can't refund {amount} on order {attempt['order_id']}. Our refund check found it is not allowed under our refund policy."
    return f"I've passed your {amount} refund on order {attempt['order_id']} to a colleague, who will check it and reply within one working day."


def score():
    return {k: RESULTS.get(k) for k in ("agent_vs_guard", "guard_only", "sweep", "guard_rule")}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    customers = [
        {
            "case_id": cid,
            "order_id": CASES[cid]["order_id"],
            "tactic": CASES[cid]["tactic"],
            "expected": CASES[cid]["expected"],
            "message": CASES[cid]["message"],
        }
        for cid in SCRIPTED
    ]
    orders = [{"order_id": o["order_id"], "item": o["item"], "paid_gbp": o["paid_gbp"]} for o in guard.ORDERS.values()]
    return {
        "customers": customers,
        "orders": orders,
        "live_agent": live_agent_on(),
        "today": guard.TODAY.isoformat(),
        "agent_model": agent.MODEL,
        "score": score(),
    }


def story_sets():
    return sorted(p.stem for p in STORIES.glob("*.json"))


@app.get("/api/story")
def story(name: str | None = Query(None, alias="set")):
    """A recorded demo story from data/stories/<set>.json. No live calls, so the recording always plays the same."""
    sets = story_sets()
    name = name or ("taubench" if "taubench" in sets else "refunds")
    if name not in sets:
        raise HTTPException(404, f"No story called {name!r}. Stories here: {', '.join(sets) or 'none'}.")
    return {**json.loads((STORIES / f"{name}.json").read_text()), "sets": sets}


@app.get("/api/case/{case_id}")
def case(case_id: str):
    c = CASES.get(case_id)
    if not c:
        raise HTTPException(404, "unknown case")
    return {k: c[k] for k in ("case_id", "order_id", "tactic", "message")}


@app.post("/api/chat")
def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages if m.role in ("user", "assistant")]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(422, "the last message must be from the customer")
    recorded = RECORDED.get(req.case_id or "")
    if recorded and messages == [{"role": "user", "content": recorded["message"]}]:
        # Scripted customers replay the agent turn recorded by eval.py, so the demo spends no Mubit calls.
        turn = {"reply": recorded["agent_reply"], "attempt": recorded["attempt"], "request_id": recorded.get("request_id")}
        source = "recorded"
    else:
        if not live_agent_on():
            raise HTTPException(
                503, "The live agent is off on this site. Pick a scripted customer, or use Try a refund to send a request straight to the guard."
            )
        try:
            turn = agent.run_agent(get_llm(), messages, cache_dir=guard.CACHE / "agent")
        except Exception as exc:  # shown in the chat; no credit or a Mubit error should not crash the page
            raise HTTPException(502, f"agent call failed: {str(exc)[:200]}")
        source = "cached" if turn.get("cached") else "live"
    attempt = turn["attempt"]
    checked = None
    if attempt:
        try:
            checked = guard_check(attempt["order_id"], attempt["amount_gbp"], guard.reason_code(attempt["reason"]))
        except Exception as exc:
            if recorded and recorded.get("guard"):
                checked = {**recorded["guard"], "note": "Sapling unreachable; showing the recorded check"}
            else:
                raise HTTPException(502, f"guard call failed: {str(exc)[:200]}")
    outcome = checked["outcome"] if checked else None
    return {
        "reply": customer_reply(turn["reply"], attempt, outcome),
        "agent_reply": turn["reply"],
        "attempt": attempt,
        "guard": checked,
        "source": source,
        "request_id": turn.get("request_id"),
    }


@app.post("/api/try")
def try_refund(req: TryRequest, request: Request):
    """A refund request straight to the guard, no agent. The words only pick the reason code; Sapling never reads them."""
    order = guard.ORDERS.get(req.order_id)
    if not order:
        raise HTTPException(404, "Pick an order from the list.")
    text = req.request.strip()
    if not text:
        raise HTTPException(422, "Write what the customer asks for.")
    if not LIVE_LIMIT.allow(client_ip(request)):
        raise HTTPException(429, f"That is {LIVE_CHECKS_PER_MINUTE} live checks this minute. Wait a minute, then try again.")
    amount = round(req.amount_gbp if req.amount_gbp is not None else float(order["paid_gbp"]), 2)
    attempt = {"order_id": req.order_id, "amount_gbp": amount, "reason": text}
    try:
        checked = guard_check(req.order_id, amount, guard.reason_code(text))
    except FutureTimeout:
        raise HTTPException(504, f"Sapling did not answer within {LIVE_TIMEOUT_S} seconds. It may be waking up. Try again in a moment.")
    except Exception as exc:
        raise HTTPException(502, f"Sapling could not be reached: {str(exc)[:160]}")
    return {
        "reply": customer_reply("", attempt, checked["outcome"]),
        "agent_reply": "",
        "attempt": attempt,
        "guard": checked,
        "source": "live",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8105)))
