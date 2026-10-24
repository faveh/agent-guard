"""Policy check engine: one claim, one policy pack, one Banyan Sapling verdict.

A policy pack is a folder under packs/ with pack.json (name, claim sentence, notes) and rules.json
(exact rule passages and which passages apply to which cost type). The engine knows nothing about
MPs: the IPSA pack is just the one pack we can score, because IPSA publishes its own decisions.
"""

import hashlib
import json
import re
import threading
import time
from pathlib import Path

import httpx

APP_DIR = Path(__file__).resolve().parent
PACKS_DIR = APP_DIR / "packs"
SAPLING_URL = "https://6adbee0669a00298cd4774c9.endpoints.huggingface.cloud"
LABELS = ("supported", "violation", "insufficient_information", "inconsistent_context")
MAX_POLICY_CHARS = 8000  # Sapling read a 9,000-character policy (2,173 tokens) without cutting it


# ---------- packs and rules ----------

def list_packs():
    """All packs, measured ones first."""
    packs = [load_pack(p.name) for p in sorted(PACKS_DIR.iterdir()) if (p / "pack.json").exists()]
    return sorted(packs, key=lambda p: (not p["measured"], p["name"]))


def load_pack(pack_id):
    folder = PACKS_DIR / pack_id
    if not (folder / "pack.json").exists() or "/" in pack_id or pack_id.startswith("."):
        raise KeyError(pack_id)
    pack = json.loads((folder / "pack.json").read_text())
    pack["id"] = pack_id
    pack["rules"] = json.loads((folder / "rules.json").read_text())
    return pack


def rule_ids(rules, category, cost_type):
    """Passage ids for a claim: exact category|cost type, then the category default, then '*'."""
    key = f"{category}|{cost_type}"
    if key in rules["cost_types"]:
        return list(rules["cost_types"][key])
    defaults = rules.get("defaults", {})
    return list(defaults.get(category) or defaults.get("*", []))


def policy_for(rules, category, cost_type):
    """Policy text sent to Sapling, plus the passages used (ref + text) for the page."""
    passages = [dict(rules["passages"][i], id=i) for i in rule_ids(rules, category, cost_type)]
    text = "\n\n".join(f"[{p.get('label', p['ref'])}] {p['text']}" for p in passages)
    return text, passages


# ---------- payload ----------

def money(value):
    return f"£{float(value):,.2f}"


def evidence_for(claim, claimant="Someone"):
    """One plain sentence of the claim's facts."""
    kind = claim["cost_type"]
    if claim.get("short_description"):
        kind += f" ({claim['short_description']})"
    details = (claim.get("details") or "").strip() or "no details given"
    return (f"{claimant} claimed {money(claim['amount_claimed'])} from the {claim['category']} budget "
            f"under the cost type {kind}, with the details: \"{details}\".")


def build_payload(policy, evidence, claim_sentence):
    policy = policy.strip()
    if not policy:
        raise ValueError("The policy is empty.")
    if len(policy) > MAX_POLICY_CHARS:
        raise ValueError(f"Keep the policy under {MAX_POLICY_CHARS} characters.")
    return {"inputs": {"policy": policy, "evidence": evidence.strip(), "claim": claim_sentence.strip()}}


def payload_for_claim(pack, claim):
    policy, passages = policy_for(pack["rules"], claim["category"], claim["cost_type"])
    evidence = evidence_for(claim, pack.get("claimant", "Someone"))
    return build_payload(policy, evidence, pack["claim_sentence"]), passages


# ---------- Sapling client with a file cache ----------

def cache_key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Sapling:
    """Calls the shared Sapling endpoint politely: cached by input hash, retried with backoff."""

    def __init__(self, cache_file, http=None, retries=5, pause=time.sleep):
        self.cache_file = Path(cache_file)
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.http = http or httpx.Client(timeout=120)
        self.retries = retries
        self.pause = pause
        self.calls = 0
        self.lock = threading.Lock()

    def check(self, payload):
        key = cache_key(payload)
        with self.lock:
            if key in self.cache:
                return dict(self.cache[key], cached=True)
        result = self._post(payload)
        keep = {k: result.get(k) for k in ("raw_label", "probabilities", "status", "prediction_set", "timing_ms", "usage")}
        with self.lock:
            self.cache[key] = keep
            self.save()
        return dict(keep, cached=False)

    def _post(self, payload):
        delay = 2.0
        for attempt in range(self.retries + 1):
            with self.lock:
                self.calls += 1
            try:
                resp = self.http.post(SAPLING_URL, json=payload)
            except httpx.TransportError:
                resp = None
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                if data.get("raw_label") in LABELS:
                    return data
                raise RuntimeError(f"Sapling reply had no label: {str(data)[:200]}")
            retryable = resp is None or resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == self.retries:
                code = "no response" if resp is None else resp.status_code
                raise RuntimeError(f"Sapling call failed ({code})")
            self.pause(delay)
            delay = min(delay * 2, 60)

    def save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=0, sort_keys=True))
        tmp.replace(self.cache_file)


# ---------- scoring ----------

def is_flag(result, threshold=None):
    """Flag for review: raw label is violation, or (sweep) violation probability >= threshold."""
    if threshold is None:
        return result["raw_label"] == "violation"
    return result["probabilities"]["violation"] >= threshold


def rate(num, den):
    return round(num / den, 4) if den else None


def score(rows, threshold=None):
    """rows: dicts with 'label' ('refused' or 'paid') and 'result'. Returns the confusion counts and rates."""
    tp = sum(1 for r in rows if r["label"] == "refused" and is_flag(r["result"], threshold))
    fn = sum(1 for r in rows if r["label"] == "refused" and not is_flag(r["result"], threshold))
    fp = sum(1 for r in rows if r["label"] == "paid" and is_flag(r["result"], threshold))
    tn = sum(1 for r in rows if r["label"] == "paid" and not is_flag(r["result"], threshold))
    return {"refused": tp + fn, "paid": fp + tn, "flagged_refused": tp, "flagged_paid": fp,
            "recall": rate(tp, tp + fn), "false_flag_rate": rate(fp, fp + tn), "precision": rate(tp, tp + fp)}


def score_by(rows, key):
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    return {k: score(v) for k, v in sorted(groups.items())}


def ranking_auc(rows):
    """Chance that a refused claim gets a higher violation probability than a paid one (0.5 = coin toss)."""
    pos = [r["result"]["probabilities"]["violation"] for r in rows if r["label"] == "refused"]
    neg = [r["result"]["probabilities"]["violation"] for r in rows if r["label"] == "paid"]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def is_merchant_only(details):
    """True when the details are only a card statement line, such as 'AMAZON [***]', with no words about the purpose."""
    letters = re.sub(r"[^A-Za-z]", "", details or "")
    return "[***]" in (details or "") or not letters or letters.upper() == letters


def label_counts(rows):
    out = {}
    for r in rows:
        k = (r["label"], r["result"]["raw_label"])
        out[k] = out.get(k, 0) + 1
    return {f"{label}/{raw}": n for (label, raw), n in sorted(out.items())}
