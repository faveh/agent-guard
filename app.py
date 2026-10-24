"""Agent Guard web app: run with `.venv/bin/python app.py` from the repo root (http://127.0.0.1:8105)."""

import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agent
import guard
from eval import RESULTS_FILE, load_cases

load_dotenv(guard.HERE / ".env")

app = FastAPI(title="Agent Guard")
CASES = {c["case_id"]: c for c in load_cases()}
RESULTS = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
RECORDED = {row["case_id"]: row for row in RESULTS.get("cases", [])}
# Scripted customers for the demo buttons: tricks the agent fell for, and honest requests.
SCRIPTED = ["c15", "c27", "c20", "c22", "c30", "c35", "c14", "c01"]
SAPLING = guard.Sapling()
_llm = None


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


@app.get("/")
def index():
    return FileResponse(guard.HERE / "static" / "index.html")


@app.get("/api/meta")
def meta():
    customers = [
        {"case_id": cid, "order_id": CASES[cid]["order_id"], "tactic": CASES[cid]["tactic"], "message": CASES[cid]["message"]}
        for cid in SCRIPTED
    ]
    return {
        "customers": customers,
        "today": guard.TODAY.isoformat(),
        "agent_model": agent.MODEL,
        "score": {k: RESULTS.get(k) for k in ("agent_vs_guard", "guard_only", "sweep", "guard_rule")},
    }


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
        try:
            turn = agent.run_agent(get_llm(), messages)
        except Exception as exc:  # shown in the chat; no key or no credit should not crash the page
            raise HTTPException(502, f"agent call failed: {str(exc)[:200]}")
        source = "cached" if turn.get("cached") else "live"
    attempt = turn["attempt"]
    checked = None
    if attempt:
        try:
            checked = guard.check_refund(SAPLING, attempt["order_id"], attempt["amount_gbp"], guard.reason_code(attempt["reason"]))
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8105)
