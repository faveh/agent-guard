"""Agent Guard core: the refund policy, evidence built from the order record, the Sapling call and the decision.

The guard never reads the chat. The claim is filled from the agent's tool call (order id, amount, reason code)
and the evidence comes only from the order record, with every date and money comparison worked out here.
"""

import hashlib
import json
import re
import threading
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / ".cache"

SAPLING_URL = "https://6adbee0669a00298cd4774c9.endpoints.huggingface.cloud"
SAPLING_CAP = 400  # real Sapling calls across every run; cached answers do not count
SAPLING_CONCURRENCY = 4
LABELS = ("supported", "violation", "insufficient_information", "inconsistent_context")

POLICY_LINES = [
    line.strip() for line in (DATA / "policy.md").read_text().splitlines() if re.match(r"R\d+\. ", line.strip())
]
RULES = {line.split(".")[0]: line for line in POLICY_LINES}

_orders = json.loads((DATA / "orders.json").read_text())
TODAY = date.fromisoformat(_orders["today"])
ORDERS = {o["order_id"]: o for o in _orders["orders"]}

CHANGE_OF_MIND_DAYS = 14
FAULTY_DAYS = 30
APPROVAL_LIMIT_GBP = 200
REASONS = ("change_of_mind", "faulty", "not_delivered")
COURIER = {"delivered": "delivered", "lost": "lost", "in_transit": "in transit"}


def money(amount):
    return f"{float(amount):.2f}"


def reason_code(text):
    """Map the agent's free-text reason to a fixed code, so no chat words reach Sapling."""
    t = (text or "").lower()
    if re.search(r"never (came|arrived)|not (arrived|delivered|received)|didn.t arrive|hasn.t arrived|lost|missing|\blate\b", t):
        return "not_delivered"
    if re.search(r"fault|broke|broken|defect|damag|crack|smash|stopped working|not working|flicker|chip|burn|sinking", t):
        return "faulty"
    return "change_of_mind"


def build_claim(order_id, amount_gbp, reason):
    if reason == "faulty":
        what = "faulty-item refund"
    elif reason == "not_delivered":
        what = "refund for a parcel that did not arrive"
    else:
        what = "change-of-mind refund"
    return f"The agent may issue a {money(amount_gbp)} GBP {what} on order {order_id}."


def _yes_no(flag):
    return "yes" if flag else "no"


def build_evidence(order, amount_gbp, today=TODAY):
    return " ".join(evidence_facts(order, amount_gbp, today))


def evidence_facts(order, amount_gbp, today=TODAY):
    """Facts from the order record only. Dates and money are compared here, not by Sapling."""
    if order is None:
        return ["No order with this id is on record."]
    paid = float(order["paid_gbp"])
    amount = float(amount_gbp)
    facts = [f"Order {order['order_id']} record. Item: {order['item']}. Amount paid: {money(paid)} GBP."]
    if order.get("delivered_on"):
        days = (today - date.fromisoformat(order["delivered_on"])).days
        facts.append(
            f"Delivered on {order['delivered_on']}, {days} days before the refund request on {today.isoformat()}. "
            f"The change-of-mind limit is {CHANGE_OF_MIND_DAYS} days and the faulty-item limit is {FAULTY_DAYS} days."
        )
    elif order.get("shipped_on"):
        days = (today - date.fromisoformat(order["shipped_on"])).days
        facts.append(f"Not delivered. Shipped on {order['shipped_on']}, {days} days before the refund request.")
    else:
        facts.append("No shipping or delivery date on record.")
    facts.append(f"Courier status: {COURIER.get(order.get('courier_status'), 'unknown')}.")
    facts.append(
        f"Final-sale item: {_yes_no(order.get('final_sale'))}. Personalised item: {_yes_no(order.get('personalised'))}. "
        f"Gift card: {_yes_no(order.get('gift_card'))}."
    )
    fault = order.get("fault_confirmed")
    facts.append(f"Fault confirmed on record: yes, on {fault['on']} ({fault['note']})." if fault else "Fault confirmed on record: no.")
    refunds = order.get("refunds") or []
    if refunds:
        done = "; ".join(f"{money(r['amount_gbp'])} GBP on {r['on']}" for r in refunds)
        facts.append(f"Refunds already on record for this order: {done}.")
    else:
        facts.append("Refunds already on record for this order: none.")
    approval = order.get("manager_approval")
    facts.append(
        f"Manager approval on record: yes, up to {money(approval['up_to_gbp'])} GBP." if approval else "Manager approval on record: none."
    )
    compare = "more than" if amount > paid else "equal to" if amount == paid else "less than"
    over = "over" if amount > APPROVAL_LIMIT_GBP else "not over"
    facts.append(
        f"Refund amount: {money(amount)} GBP, which is {compare} the {money(paid)} GBP paid "
        f"and {over} the {APPROVAL_LIMIT_GBP} GBP manager-approval limit."
    )
    for note in order.get("notes") or []:
        facts.append(f"Record note: {note}")
    return facts


def relevant_rules(order, reason):
    """The rule ids that apply to this kind of refund. Chosen from the reason code and the record, never the verdict."""
    ids = {"change_of_mind": ["R1", "R4"], "faulty": ["R2", "R4", "R9"], "not_delivered": ["R3", "R9"]}[reason]
    if order and order.get("gift_card"):
        ids.append("R5")
    ids += ["R6", "R7", "R8"]
    return list(dict.fromkeys(ids))


def policy_for(rule_ids):
    """Sapling gets only the rules that apply: on our 40 cases the whole policy blurred its verdicts."""
    return "\n".join(["Quillmoor refund rules for the support agent.", *(RULES[i] for i in rule_ids)])


def decide(raw_label):
    """The guard rule: block on violation or inconsistent context, allow on supported, else ask a person."""
    if raw_label in ("violation", "inconsistent_context"):
        return "block"
    if raw_label == "supported":
        return "allow"
    return "person"


def decide_by_threshold(probabilities, threshold):
    """A simpler two-way rule used in the threshold sweep: block when p(violation) >= threshold, else allow."""
    return "block" if probabilities.get("violation", 0.0) >= threshold else "allow"


def bump_counter(name, cap):
    """Count real paid or rate-limited calls across runs and refuse once the cap is reached."""
    path = CACHE / f"{name}_calls.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    used = int(path.read_text() or 0) if path.exists() else 0
    if used >= cap:
        raise RuntimeError(f"{name} call cap reached ({cap})")
    path.write_text(str(used + 1))
    return used + 1


def input_hash(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


class Sapling:
    """Banyan Sapling client: at most 4 calls at once, backoff on 429/5xx, answers cached by input hash."""

    def __init__(self, post=None, cache_dir=CACHE / "sapling", cap=SAPLING_CAP, sleep=time.sleep):
        if post is None:
            import httpx

            post = httpx.post
        self.post = post
        self.cache_dir = Path(cache_dir)
        self.cap = cap
        self.sleep = sleep
        self.sem = threading.Semaphore(SAPLING_CONCURRENCY)
        self.lock = threading.Lock()

    def check(self, policy, evidence, claim):
        key = input_hash(policy, evidence, claim)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return {**json.loads(path.read_text()), "cached": True}
        with self.sem:
            with self.lock:
                bump_counter("sapling", self.cap)
            body = {"inputs": {"policy": policy, "evidence": evidence, "claim": claim}}
            for attempt in range(5):
                resp = self.post(SAPLING_URL, json=body, timeout=60)
                if resp.status_code == 429 or resp.status_code >= 500:
                    self.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                break
            else:
                raise RuntimeError(f"Sapling kept failing (HTTP {resp.status_code})")
        data = resp.json()
        result = {
            "raw_label": data["raw_label"],
            "probabilities": {k: round(float(data["probabilities"][k]), 4) for k in LABELS},
            "status": data.get("status"),
            "model_ms": round((data.get("timing_ms") or {}).get("model", 0)),
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        return {**result, "cached": False}


def check_refund(sapling, order_id, amount_gbp, reason):
    """One guard check for one refund attempt. Returns everything the app panel and eval need."""
    order = ORDERS.get(str(order_id))
    claim = build_claim(order_id, amount_gbp, reason)
    facts = evidence_facts(order, amount_gbp)
    evidence = " ".join(facts)
    rule_ids = relevant_rules(order, reason)
    result = sapling.check(policy_for(rule_ids), evidence, claim)
    return {
        "order_id": str(order_id),
        "amount_gbp": float(amount_gbp),
        "reason": reason,
        "claim": claim,
        "evidence": evidence,
        "facts": facts,
        "rules": [{"id": i, "text": RULES[i]} for i in rule_ids],
        "sapling": result,
        "outcome": decide(result["raw_label"]),
    }
