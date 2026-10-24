"""Build the small anonymised evaluation set from IPSA's 2024-25 CSV.

Positives: every "Not Paid" claim with reason "Not claimable".
Negatives: a fixed-seed random sample of "Paid" claims, split across the same category and cost type
pairs in proportion to the positives. Late submission and insufficient evidence refusals are left out.
Name, constituency, parliamentary id and claim number are dropped, and bracketed reference numbers are cut
from the details. Rows whose text holds a person's name, an MP's web address, a street address, a
postcode, a constituency name or another claim's number are left out.

    python fetch_data.py && python build_eval_set.py
"""

import csv
import json
import random
import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
RAW_CSV = APP_DIR / "raw" / "ipsa_24_25.csv"
OUT = APP_DIR / "packs" / "ipsa" / "claims.jsonl"
SEED = 2410
NEGATIVES = 300
MIN_PER_TYPE = 3

KEEP = {"Date": "date", "Category": "category", "Cost Type": "cost_type", "Short Description": "short_description",
        "Details": "details", "Amount Claimed": "amount_claimed", "Amount Paid": "amount_paid",
        "Status": "status", "Reason If Not Paid": "reason"}

TITLES = re.compile(r"\b(Mr|Mrs|Ms|Miss|Mx|Dr|Cllr|Councillor|Sir|Dame|Lord|Lady|Baroness)\.?\s+[A-Z][a-z]")
INITIAL_SURNAME = re.compile(r"\b[A-Z]\.?\s[A-Z][a-z]{2,}")
INITIALS_AT_WORK = re.compile(r"\b[A-Z]{2}\s+work\b")
NAMED_MP = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z'-]+\s+MP\b")
POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)
STREET = re.compile(r"\b\d+[a-z]?\s+[a-z]+\s+(road|rd|street|st|avenue|ave|lane|way|place|close|drive|square)\b", re.IGNORECASE)
MP_DOMAIN = re.compile(r"[a-z]+mp\.(co\.uk|org\.uk|com|uk)\b", re.IGNORECASE)
POSSESSIVE = re.compile(r"\b([A-Z][a-z]+)[\u00b4'\u2019]s\b")
REF = re.compile(r"\s*\[\d{6,}-[\d:]+\]")
CLAIM_NUMBER = re.compile(r"\b\d{8}\b")
# Details a hand review of the sample found to point at a person by their initials.
HAND_EXCLUDED = {"Meeting with VW"}
# Card lines a hand review found to be a business named after a person. Dropped after sampling (see build).
OWNER_NAMED = {"H CARTER SONS", "WB WHITLEY - FORECOURT"}
FIRST_NAMES = set("""
adam adrian alan alex alexander alice alison amanda amy andrew angela ann anna anne anthony ben benjamin beth
bill bob brian carl caroline catherine charles charlie charlotte chris christine christopher claire colin
craig daniel david dawn deborah dennis diane donna ed edward elizabeth emily emma eric fiona frank gary
gavin george graham hannah harry helen ian jack james jamie jane janet jason jeff jennifer jenny jessica jim
joanne joe john jonathan joseph julia julie karen kate katherine kathryn kelly kevin kim laura lee lisa liz
louise lucy luke margaret maria mark martin mary matt matthew michael michelle mike natalie neil nicholas
nick nicola oliver patrick paul peter philip rachel ray rebecca richard rob robert ruth ryan sally sam samuel sean
sandra sarah scott sean sharon simon sophie stephen steve steven stuart susan tom thomas tony tracey vicky
victoria william jenny owen
""".split())
FIRST_NAME_WORD = re.compile(r"\b([A-Z][a-z]+)\b")
CAPS_NAME = re.compile(r"\b([A-Z]{3,})\s+[A-Z]{2,}")  # card lines in capitals, such as a sole trader's name


def has_person_name(text, known_names=()):
    """True if text looks like it holds a person's name (titles, 'First Last MP', a common first name written
    as a name or at the start of a line in capitals, an MP web address, or any MP name from the known list)."""
    if not text:
        return False
    if any(p.search(text) for p in (TITLES, NAMED_MP, INITIAL_SURNAME, INITIALS_AT_WORK)):
        return True
    for pattern in (FIRST_NAME_WORD, POSSESSIVE):
        for m in pattern.finditer(text):
            if m.group(1).lower() in FIRST_NAMES:
                return True
    for m in CAPS_NAME.finditer(text):
        word = m.group(1).lower()
        if word in FIRST_NAMES or word.rstrip("s") in FIRST_NAMES:
            return True
    if MP_DOMAIN.search(text):
        return True
    low = text.lower()
    squashed = low.replace(" ", "")
    return any(name in low or name.replace(" ", "") in squashed for name in known_names)


def has_address(text):
    return bool(text and (POSTCODE.search(text) or STREET.search(text)))


def anonymise(row):
    """Keep only the claim fields; drop who claimed it."""
    out = {new: (row.get(old) or "").strip() for old, new in KEEP.items()}
    out["details"] = REF.sub("", out["details"]).strip()
    out["amount_claimed"] = float(out["amount_claimed"] or 0)
    out["amount_paid"] = float(out["amount_paid"] or 0)
    return out


def usable(row, known_names):
    text = " ".join([row.get("Short Description", ""), row.get("Details", "")])
    if has_person_name(text, known_names) or has_address(text):
        return False
    if CLAIM_NUMBER.search(text) or REF.sub("", row.get("Details", "")).strip() in HAND_EXCLUDED:
        return False
    if float(row.get("Amount Claimed") or 0) <= 0:
        return False
    return not row.get("Details", "").startswith("Aggregated figure")


def constituencies(rows):
    """Constituency names from the CSV, without the ' CC' / ' BC' suffix."""
    names = {re.sub(r"\s+(CC|BC)$", "", r.get("Constituency", "").strip()) for r in rows}
    return {n.lower() for n in names if n}


def names_place_or_owner(row, places):
    """True if the text names a constituency (which points at one MP) or a business named after a person."""
    text = " ".join([row.get("Short Description", ""), row.get("Details", "")]).lower()
    if REF.sub("", row.get("Details", "")).strip() in OWNER_NAMED:
        return True
    return any(re.search(r"\b" + re.escape(p) + r"\b", text) for p in places)


def type_key(row):
    return f"{row['Category']}|{row['Cost Type']}"


def select(rows, seed=SEED, negatives=NEGATIVES, min_per_type=MIN_PER_TYPE):
    """Return (positives, negatives) as raw rows."""
    known = {r["Name"].strip().lower() for r in rows if r.get("Name", "").strip()}
    pos = [r for r in rows if r["Status"] == "Not Paid" and r["Reason If Not Paid"] == "Not claimable" and usable(r, known)]
    counts = {}
    for r in pos:
        counts[type_key(r)] = counts.get(type_key(r), 0) + 1
    pool = {}
    for r in rows:
        if r["Status"] == "Paid" and type_key(r) in counts and usable(r, known):
            pool.setdefault(type_key(r), []).append(r)
    rng = random.Random(seed)
    neg = []
    for key in sorted(counts):
        want = max(min_per_type, round(negatives * counts[key] / len(pos)))
        candidates = pool.get(key, [])
        neg.extend(rng.sample(candidates, min(want, len(candidates))))
    return pos, neg


def build(rows, seed=SEED, **kw):
    pos, neg = select(rows, seed=seed, **kw)
    places = constituencies(rows)
    claims = []
    for label, group in (("refused", pos), ("paid", neg)):
        for r in group:
            # Filtered after sampling, so the random sample of every other row stays the same.
            if not names_place_or_owner(r, places):
                claims.append(dict(anonymise(r), label=label))
    random.Random(seed).shuffle(claims)  # ids and order must not give the decision away
    for i, c in enumerate(claims, 1):
        c["id"] = f"ipsa-{i:04d}"
    return claims


def main():
    with RAW_CSV.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    claims = build(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for c in claims:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    refused = sum(c["label"] == "refused" for c in claims)
    print(f"{len(rows):,} claims read; wrote {refused} refused + {len(claims) - refused} paid to {OUT.relative_to(APP_DIR)}")


if __name__ == "__main__":
    main()
