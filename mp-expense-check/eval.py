"""Score Banyan Sapling on the IPSA pack: real claims, real rulebook, IPSA's own decisions.

Every Sapling reply is cached in packs/ipsa/sapling_cache.json, so a re-run makes no new calls.
Writes packs/ipsa/results.json and prints the headline numbers.

    python eval.py            # uses the cache, calls Sapling only for rows not cached yet
    python eval.py --limit 20 # first 20 rows only (a quick look)
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import check

PACK = "ipsa"
PACK_DIR = check.PACKS_DIR / PACK
CLAIMS_FILE = PACK_DIR / "claims.jsonl"
CACHE_FILE = PACK_DIR / "sapling_cache.json"
RESULTS_FILE = PACK_DIR / "results.json"
WORKERS = 4  # shared free endpoint: at most 4 calls at once
CALL_CAP = 800
THRESHOLDS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def read_claims(path=CLAIMS_FILE):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(pack, claims, sapling):
    def one(claim):
        payload, passages = check.payload_for_claim(pack, claim)
        return {"id": claim["id"], "label": claim["label"], "category": claim["category"],
                "cost_type": claim["cost_type"], "rule_ids": [p["id"] for p in passages],
                "merchant_only": check.is_merchant_only(claim["details"]),
                "result": sapling.check(payload)}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(one, claims))


def summarise(rows, calls):
    headline = check.score(rows)
    sweep = [dict(check.score(rows, t), threshold=t) for t in THRESHOLDS]
    by_type = check.score_by(rows, lambda r: r["cost_type"])
    statuses = {}
    for r in rows:
        statuses[r["result"]["status"]] = statuses.get(r["result"]["status"], 0) + 1
    return {
        "pack": PACK,
        "rule": "flag for review when Sapling's raw_label is 'violation'",
        "headline": headline,
        "threshold_sweep": sweep,
        "by_cost_type": by_type,
        "by_detail_kind": check.score_by(rows, lambda r: "card line only" if r["merchant_only"] else "words about the purpose"),
        "ranking_auc": check.ranking_auc(rows),
        "raw_labels": check.label_counts(rows),
        "conformal_status": statuses,
        "sapling_calls_this_run": calls,
        "rows": [{"id": r["id"], "label": r["label"], "raw_label": r["result"]["raw_label"],
                  "violation": round(r["result"]["probabilities"]["violation"], 4),
                  "status": r["result"]["status"]} for r in rows],
    }


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    args = parser.parse_args()

    pack = check.load_pack(PACK)
    claims = read_claims()
    if args.limit:
        claims = claims[: args.limit]
    sapling = check.Sapling(CACHE_FILE)
    todo = sum(1 for c in claims if check.cache_key(check.payload_for_claim(pack, c)[0]) not in sapling.cache)
    if todo > CALL_CAP:
        raise SystemExit(f"{todo} uncached rows is over the {CALL_CAP}-call cap.")
    print(f"{len(claims)} claims, {todo} not cached yet")

    rows = run(pack, claims, sapling)
    out = summarise(rows, sapling.calls)
    if not args.limit:
        RESULTS_FILE.write_text(json.dumps(out, indent=1) + "\n")

    h = out["headline"]
    print()
    print(f"Refused as 'Not claimable': {h['refused']}   flagged: {h['flagged_refused']}  (recall {pct(h['recall'])})")
    print(f"Paid (random sample):       {h['paid']}   flagged: {h['flagged_paid']}  (false-flag rate {pct(h['false_flag_rate'])})")
    print(f"Precision of a flag:        {pct(h['precision'])}")
    print(f"Sapling calls this run:     {sapling.calls}")
    print(f"Ranking AUC (0.5 = chance): {out['ranking_auc']}")
    for kind, s in out["by_detail_kind"].items():
        print(f"  details are {kind}: refused {s['flagged_refused']}/{s['refused']} flagged, paid {s['flagged_paid']}/{s['paid']} flagged")
    print()
    print("violation prob >= t   recall  false-flag  precision")
    for s in out["threshold_sweep"]:
        print(f"  {s['threshold']:.2f}              {pct(s['recall']):>5}   {pct(s['false_flag_rate']):>5}       {pct(s['precision']):>5}")
    print()
    print(f"{'cost type':45} refused flagged  paid flagged")
    for name, s in out["by_cost_type"].items():
        print(f"{name[:45]:45} {s['refused']:4} {s['flagged_refused']:6}  {s['paid']:5} {s['flagged_paid']:6}")


if __name__ == "__main__":
    main()
