"""Policy Check web app: serves the page, the packs, the measured score and live Sapling checks."""

import json
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import check

app = FastAPI(title="Policy Check")
LIVE_CACHE = check.APP_DIR / ".cache" / "live_cache.json"


class CheckIn(BaseModel):
    pack: str
    claim_id: str


class OwnClaimIn(BaseModel):
    pack: str
    category: str = ""
    cost_type: str = ""
    details: str
    amount: float


class OwnPolicyIn(BaseModel):
    policy: str
    evidence: str
    claim: str = "This expense claim can be reimbursed under the policy."


@lru_cache(maxsize=1)
def get_sapling():
    return check.Sapling(LIVE_CACHE)


def get_pack(pack_id):
    try:
        return check.load_pack(pack_id)
    except KeyError:
        raise HTTPException(404, "No such policy pack.") from None


def read_claims(pack):
    path = check.PACKS_DIR / pack["id"] / "claims.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_check(payload):
    try:
        return get_sapling().check(payload)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


def decision_of(claim):
    if claim.get("label") == "example":
        return {"kind": "example", "expected": claim.get("expected")}
    return {"kind": "ipsa", "status": claim["status"], "reason": claim.get("reason", ""),
            "amount_paid": claim["amount_paid"], "label": claim["label"]}


@app.get("/")
def index():
    return FileResponse(check.APP_DIR / "static" / "index.html")


@app.get("/api/packs")
def packs():
    out = []
    for p in check.list_packs():
        types = sorted({k.split("|", 1)[1] for k in p["rules"]["cost_types"]})
        out.append({k: p[k] for k in ("id", "name", "measured", "summary", "data_note", "claim_sentence", "claimant")}
                   | {"cost_types": types})
    return out


@app.get("/api/packs/{pack_id}/claims")
def claims(pack_id: str):
    pack = get_pack(pack_id)
    return [{k: c.get(k) for k in ("id", "category", "cost_type", "short_description", "details", "amount_claimed")}
            for c in read_claims(pack)]


@app.get("/api/packs/{pack_id}/results")
def results(pack_id: str):
    path = check.PACKS_DIR / get_pack(pack_id)["id"] / "results.json"
    if not path.exists():
        raise HTTPException(404, "This pack is not measured.")
    data = json.loads(path.read_text())
    data.pop("rows", None)
    return data


@app.post("/api/check")
def check_claim(body: CheckIn):
    pack = get_pack(body.pack)
    claim = next((c for c in read_claims(pack) if c["id"] == body.claim_id), None)
    if claim is None:
        raise HTTPException(404, "No claim with that id.")
    payload, passages = check.payload_for_claim(pack, claim)
    return {"result": run_check(payload), "passages": passages, "inputs": payload["inputs"],
            "decision": decision_of(claim)}


@app.post("/api/check-own-claim")
def check_own_claim(body: OwnClaimIn):
    pack = get_pack(body.pack)
    details = body.details.strip()
    if not details:
        raise HTTPException(400, "Describe the expense first.")
    if len(details) > 600:
        raise HTTPException(400, "Keep the description under 600 characters.")
    if body.amount <= 0:
        raise HTTPException(400, "The amount must be more than zero.")
    claim = {"category": body.category or "Office Costs", "cost_type": body.cost_type or "Other",
             "details": details, "amount_claimed": body.amount}
    payload, passages = check.payload_for_claim(pack, claim)
    return {"result": run_check(payload), "passages": passages, "inputs": payload["inputs"], "decision": None}


@app.post("/api/check-own-policy")
def check_own_policy(body: OwnPolicyIn):
    if not body.evidence.strip():
        raise HTTPException(400, "Describe the claim first.")
    if len(body.evidence) > 1000 or len(body.claim) > 300:
        raise HTTPException(400, "Keep the claim facts under 1000 characters.")
    try:
        payload = check.build_payload(body.policy, body.evidence, body.claim)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    passages = [{"id": "pasted", "ref": "Your pasted policy", "text": payload["inputs"]["policy"]}]
    return {"result": run_check(payload), "passages": passages, "inputs": payload["inputs"], "decision": None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8104)
