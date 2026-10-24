"""API routes with the Sapling client faked. No network calls."""

from fastapi.testclient import TestClient

import app as web
import check
from fakes import FakeHttp


def client(monkeypatch, tmp_path):
    fake = check.Sapling(tmp_path / "live.json", http=FakeHttp())
    monkeypatch.setattr(web, "get_sapling", lambda: fake)
    return TestClient(web.app), fake


def test_pages_packs_and_score(monkeypatch, tmp_path):
    http, _ = client(monkeypatch, tmp_path)
    assert http.get("/").status_code == 200
    packs = http.get("/api/packs").json()
    assert [p["id"] for p in packs][:1] == ["ipsa"] and packs[0]["measured"] is True
    claims = http.get("/api/packs/ipsa/claims").json()
    assert claims and "status" not in claims[0] and "label" not in claims[0]  # the decision stays hidden
    assert http.get("/api/packs/nope/claims").status_code == 404
    assert http.get("/api/packs/travel-example/results").status_code == 404


def test_check_a_real_claim_reveals_the_decision(monkeypatch, tmp_path):
    http, fake = client(monkeypatch, tmp_path)
    claim_id = http.get("/api/packs/ipsa/claims").json()[0]["id"]
    out = http.post("/api/check", json={"pack": "ipsa", "claim_id": claim_id}).json()
    assert out["result"]["raw_label"] == "violation"
    assert out["passages"][0]["ref"].startswith("IPSA")
    assert out["decision"]["kind"] == "ipsa" and out["decision"]["status"] in {"Paid", "Not Paid"}
    assert len(fake.http.posts) == 1
    assert http.post("/api/check", json={"pack": "ipsa", "claim_id": "nope"}).status_code == 404


def test_own_claim_and_own_policy(monkeypatch, tmp_path):
    http, fake = client(monkeypatch, tmp_path)
    res = http.post("/api/check-own-claim", json={"pack": "ipsa", "category": "Office Costs", "cost_type": "Hospitality",
                                                  "details": "TV licence for the office", "amount": 169.5})
    assert res.status_code == 200 and "TV licence" in res.json()["inputs"]["evidence"]
    assert http.post("/api/check-own-claim", json={"pack": "ipsa", "details": " ", "amount": 5}).status_code == 400
    res = http.post("/api/check-own-policy", json={"policy": "Refunds within 30 days only.",
                                                   "evidence": "The customer asked for a refund after 45 days.",
                                                   "claim": "This refund can be paid."})
    assert res.status_code == 200 and fake.http.posts[-1]["inputs"]["policy"] == "Refunds within 30 days only."
    assert http.post("/api/check-own-policy", json={"policy": "", "evidence": "x"}).status_code == 400
