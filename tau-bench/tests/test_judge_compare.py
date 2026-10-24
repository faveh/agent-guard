"""Tests for the Sapling vs general model comparison: label parsing, groups, cache, duplicate inputs and the credit guard. No network."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge_compare as jc  # noqa: E402


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def answer(label, tokens=600):
    return Resp(200, {"model": "gpt-6-luna", "choices": [{"message": {"content": json.dumps({"label": label, "reason": "r"})}}], "usage": {"total_tokens": tokens}})


@pytest.fixture(autouse=True)
def fresh_stop():
    jc.Judge.stop.clear()
    yield
    jc.Judge.stop.clear()


def judge(tmp_path, post, **kw):
    return jc.Judge("gpt-6-luna", post=post, key="k", cache_dir=tmp_path / "c", ledger=tmp_path / "u.json", sleep=lambda s: None, **kw)


def test_parse_label():
    assert jc.parse_label('```json\n{"label": "Violation", "reason": "x"}\n```') == ("violation", "x")
    assert jc.parse_label('{"label": "maybe"}')[0] == "unparsed"
    assert jc.parse_label("no json")[0] == "unparsed"


def test_group():
    assert jc.group({"label": "right", "rules_broken": ["x"], "api_error": False}) == "right"
    assert jc.group({"label": "wrong", "rules_broken": ["x"], "api_error": False}) == "rule_breaking_ran"
    assert jc.group({"label": "wrong", "rules_broken": ["x"], "api_error": True}) == "wrong_other"
    assert jc.group({"label": "wrong", "rules_broken": [], "api_error": False}) == "wrong_other"


def test_duplicates_asked_once_and_cached(tmp_path):
    calls = []

    def post(url, json, headers, timeout):
        calls.append(json)
        assert "max_completion_tokens" in json and "store" not in json and "metadata" not in json
        return answer("violation")

    rows = [{"id": i, "policy": "p", "evidence": "e", "claim": c} for i, c in (("a", "x"), ("b", "x"), ("c", "y"))]
    j = judge(tmp_path, post)
    res = jc.judge_all(j, rows)
    assert len(calls) == 2 and res["a"] == res["b"]
    jc.judge_all(j, rows)
    assert len(calls) == 2
    assert json.loads((tmp_path / "u.json").read_text())["tokens"] == 1200


def test_credit_error_stops(tmp_path):
    j = judge(tmp_path, lambda *a, **k: Resp(402, {"error": "payment required"}))
    with pytest.raises(jc.CreditStop):
        j.check("p", "e", "c")
    assert jc.Judge.stop.is_set()


def test_reason_mentioning_credit_is_not_a_credit_error(tmp_path):
    j = judge(tmp_path, lambda *a, **k: Resp(200, {"choices": [{"message": {"content": '{"label": "supported", "reason": "paid by credit card"}'}}], "usage": {}}))
    assert j.check("p", "e", "c")["raw_label"] == "supported"


def test_model_not_on_starter_credit_is_refused(tmp_path):
    j = judge(tmp_path, lambda *a, **k: Resp(400, {"error": {"code": "starter_credit_model", "message": "The starter credit serves these OpenAI models"}}))
    with pytest.raises(jc.Refused):
        j.check("p", "e", "c")
    assert not jc.Judge.stop.is_set()


def test_token_cap(tmp_path):
    j = judge(tmp_path, lambda *a, **k: answer("supported", tokens=700), token_cap=1000)
    j.check("p", "e", "1")
    j.check("p", "e", "2")
    with pytest.raises(jc.CreditStop):
        j.check("p", "e", "3")
