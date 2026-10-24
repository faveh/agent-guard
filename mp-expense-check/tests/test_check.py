"""Rule lookup, payload building, the cached Sapling client and scoring. No network calls."""

import json

import pytest

import check
from fakes import VIOLATION, FakeHttp

RULES = {
    "passages": {"a": {"ref": "Rulebook para 1", "label": "Para 1", "text": "No alcohol."},
                 "b": {"ref": "Rulebook para 2", "text": "Office kit is fine."},
                 "c": {"ref": "Rulebook para 3", "text": "Be reasonable."}},
    "cost_types": {"Office Costs|Hospitality": ["a", "c"]},
    "defaults": {"Office Costs": ["b"], "*": ["c"]},
}


def test_rule_lookup_uses_exact_type_then_category_then_star():
    assert check.rule_ids(RULES, "Office Costs", "Hospitality") == ["a", "c"]
    assert check.rule_ids(RULES, "Office Costs", "Rent") == ["b"]
    assert check.rule_ids(RULES, "Travel", "Taxi") == ["c"]


def test_policy_text_quotes_passages_with_their_labels():
    text, passages = check.policy_for(RULES, "Office Costs", "Hospitality")
    assert text == "[Para 1] No alcohol.\n\n[Rulebook para 3] Be reasonable."
    assert [p["id"] for p in passages] == ["a", "c"]


def test_evidence_is_one_plain_sentence():
    claim = {"category": "Office Costs", "cost_type": "Equipment - purchase", "short_description": "Office furniture",
             "details": "Desk ", "amount_claimed": 1210.5}
    assert check.evidence_for(claim, "An MP") == (
        'An MP claimed £1,210.50 from the Office Costs budget under the cost type Equipment - purchase '
        '(Office furniture), with the details: "Desk".')
    assert 'no details given' in check.evidence_for(dict(claim, details=""))


def test_build_payload_checks_the_policy():
    p = check.build_payload(" rule ", " fact ", " claim ")
    assert p == {"inputs": {"policy": "rule", "evidence": "fact", "claim": "claim"}}
    with pytest.raises(ValueError):
        check.build_payload("   ", "fact", "claim")
    with pytest.raises(ValueError):
        check.build_payload("x" * (check.MAX_POLICY_CHARS + 1), "fact", "claim")


def test_ipsa_pack_rules_are_complete_and_fit_sapling():
    pack = check.load_pack("ipsa")
    rules = pack["rules"]
    for key, ids in list(rules["cost_types"].items()) + list(rules["defaults"].items()):
        for i in ids:
            assert rules["passages"][i]["text"], (key, i)
        text, _ = check.policy_for(rules, *key.split("|", 1)) if "|" in key else check.policy_for(rules, key, "?")
        assert len(text) < check.MAX_POLICY_CHARS
    for p in rules["passages"].values():
        assert p["ref"].startswith("IPSA")


def test_every_ipsa_claim_builds_a_payload():
    pack = check.load_pack("ipsa")
    claims = [json.loads(line) for line in (check.PACKS_DIR / "ipsa" / "claims.jsonl").read_text().splitlines()]
    assert {c["label"] for c in claims} == {"refused", "paid"}
    for c in claims:
        payload, passages = check.payload_for_claim(pack, c)
        assert payload["inputs"]["evidence"].startswith("An MP claimed")
        assert passages


def test_example_pack_uses_its_own_claimant():
    pack = check.load_pack("travel-example")
    claim = {"category": "Travel", "cost_type": "Taxi", "details": "Airport", "amount_claimed": 30}
    payload, _ = check.payload_for_claim(pack, claim)
    assert payload["inputs"]["evidence"].startswith("An employee claimed")
    assert [p["id"] for p in check.list_packs()][0] == "ipsa"


def test_sapling_caches_by_input_hash(tmp_path):
    http = FakeHttp()
    s = check.Sapling(tmp_path / "cache.json", http=http)
    payload = check.build_payload("rule", "fact", "claim")
    first = s.check(payload)
    second = s.check(payload)
    assert first["raw_label"] == "violation" and first["cached"] is False and second["cached"] is True
    assert len(http.posts) == 1
    again = check.Sapling(tmp_path / "cache.json", http=FakeHttp())
    assert again.check(payload)["cached"] is True


def test_sapling_retries_429_and_5xx_with_backoff(tmp_path):
    waits = []
    http = FakeHttp([(429, {}), (503, {}), (200, VIOLATION)])
    s = check.Sapling(tmp_path / "c.json", http=http, pause=waits.append)
    assert s.check(check.build_payload("r", "f", "c"))["raw_label"] == "violation"
    assert waits == [2.0, 4.0] and s.calls == 3


def test_sapling_gives_up_on_a_client_error(tmp_path):
    s = check.Sapling(tmp_path / "c.json", http=FakeHttp([(400, {"error": "bad"})]), pause=lambda _: None)
    with pytest.raises(RuntimeError):
        s.check(check.build_payload("r", "f", "c"))
    assert s.calls == 1


def result(label, violation):
    return {"raw_label": label, "probabilities": {"violation": violation}}


ROWS = [
    {"label": "refused", "cost_type": "A", "result": result("violation", 0.9)},
    {"label": "refused", "cost_type": "A", "result": result("insufficient_information", 0.4)},
    {"label": "refused", "cost_type": "B", "result": result("violation", 0.6)},
    {"label": "paid", "cost_type": "A", "result": result("supported", 0.1)},
    {"label": "paid", "cost_type": "B", "result": result("violation", 0.7)},
    {"label": "paid", "cost_type": "B", "result": result("insufficient_information", 0.2)},
]


def test_score_counts_flags_against_ipsa_decisions():
    s = check.score(ROWS)
    assert (s["refused"], s["flagged_refused"], s["paid"], s["flagged_paid"]) == (3, 2, 3, 1)
    assert s["recall"] == 0.6667 and s["false_flag_rate"] == 0.3333 and s["precision"] == 0.6667


def test_threshold_sweep_uses_the_violation_probability():
    s = check.score(ROWS, threshold=0.65)
    assert (s["flagged_refused"], s["flagged_paid"]) == (1, 1)
    assert check.score([], threshold=0.5)["recall"] is None


def test_score_by_cost_type():
    by = check.score_by(ROWS, lambda r: r["cost_type"])
    assert by["A"]["flagged_refused"] == 1 and by["B"]["flagged_paid"] == 1


def test_ranking_auc_is_half_for_a_coin_toss():
    tie = [{"label": "refused", "result": result("violation", 0.5)}, {"label": "paid", "result": result("violation", 0.5)}]
    assert check.ranking_auc(tie) == 0.5
    assert check.ranking_auc(ROWS) == 0.7778
    assert check.ranking_auc([]) is None


def test_merchant_only_details_are_card_lines():
    assert check.is_merchant_only("AMAZON [***]")
    assert check.is_merchant_only("THE LILY PAD FLORIST")
    assert not check.is_merchant_only("Office desk and chair for home working")
