# Policy Check (MP expense check)

Paste an expense policy as written, and Banyan Sapling checks each claim against it in about a second.
It flags the few claims that need a human look, each with the rule passage quoted. No rule set-up, no training.
We measured it on UK MP expenses, because IPSA, the body that pays them, publishes its own decisions.

**Who it is for.** A finance lead at a 300-person company gets about 1,500 expense claims a month and can only
spot-check a sample against a long policy. This tool checks every claim against the policy text and flags
the ones to look at. AI expense-audit tools already exist. The difference here is that you paste the policy as
written, instead of turning it into rules first.

This is the second experiment in [Agent Guard](../README.md).

Research preview. A flag means "review this", not a finding of wrongdoing. No MP names are shown anywhere.

## Measured result (IPSA 2024-25)

IPSA refused 117 claims in 2024-25 as "Not claimable" (after we removed rows whose text names a person).
We gave Sapling the real rulebook and the real claims. **It flagged 6 of them, with no training, and flagged
3 of 297 claims IPSA paid.**

| | Claims | Flagged for review | Rate |
|---|---:|---:|---:|
| Refused by IPSA as "Not claimable" | 117 | 6 | 5% (recall) |
| Paid by IPSA (fixed-seed random sample, same cost types) | 297 | 3 | 1% (false-flag rate) |
| All flags | 9 | 6 were refused claims | 67% (precision) |

Flag rule: Sapling's `raw_label` is `violation`. One Sapling call per claim, 414 claims.

**What this means.** On this public data, Sapling is a careful flagger, not a full auditor. It misses most
refused claims, and it rarely flags a paid one.

- **Most refused claims do not say why.** 68 of the 117 refused claims show only a card statement line, such as
  `AMAZON [***]` or a shop name. IPSA saw receipts and context that the public file does not hold. Sapling
  flagged 1 of those 68.
- **Where the details say what the money was for, it does better.** It flagged 5 of 49 refused claims (10%) and
  3 of 142 paid claims (2%).
- **Its violation score does not rank claims well here.** Sorting all 414 claims by the violation probability puts
  a refused claim above a paid one 54% of the time (AUC 0.535; 0.5 is a coin toss).
- **It never said "fits the policy".** Every claim came back `violation` (9) or `insufficient_information` (405).
  Its conformal status was `unsure` on all 414.

The 6 refused claims it flagged were 2 accountancy fees for an MP's own tax return, 3 staff or team meals and
1 advertising order. The 3 paid claims it flagged were 2 accountancy fees and 1 staff away-day lunch.
These are real grey areas in the rules.

Threshold sweep (flag when the violation probability is at least t):

| t | Refused flagged | Paid flagged | Precision |
|---:|---:|---:|---:|
| 0.10 | 23 (20%) | 53 (18%) | 30% |
| 0.15 | 11 (9%) | 11 (4%) | 50% |
| 0.20 | 10 (9%) | 8 (3%) | 56% |
| 0.30 | 8 (7%) | 5 (2%) | 62% |
| 0.50 | 5 (4%) | 2 (1%) | 71% |
| 0.80 | 2 (2%) | 0 (0%) | 100% |

The table by cost type is in `packs/ipsa/results.json` (`by_cost_type`) and on the page.

**Example pack (not measured).** We also wrote a short company travel policy and 10 claims by hand, to show the
same engine on another industry. Sapling flagged all 6 claims we wrote to break the policy and none of the 4 we
wrote to fit it. We wrote these claims ourselves, so this is a demo, not a measurement.

## How it works

1. **Data.** `fetch_data.py` downloads IPSA's 2024-25 claims CSV (105,874 claims) into `raw/`, which git ignores.
   `build_eval_set.py` takes every "Not Paid / Not claimable" claim, plus about 300 paid claims picked at random
   (seed 2410) across the same category and cost type pairs. It drops Name, Constituency, Parliamentary ID and
   Claim Number. It leaves out any row whose text holds a person's name, an MP's web address, a street address,
   a postcode or another claim's number (3 refused rows were left out this way). After sampling, it also drops
   rows that name a constituency or a business named after a person (6 paid rows). The result is
   `packs/ipsa/claims.jsonl` (117 refused, 297 paid), shuffled so the ids do not give the answer away.
2. **Rules.** `packs/ipsa/rules.json` quotes the IPSA Scheme of MPs' Staffing and Business Costs 2024-25
   (17th edition, the one in force for these claims) word for word, with the paragraph number on each passage.
   It adds some passages from IPSA's current office costs guidance. Each cost type maps to the passages that
   apply to it (for example, Hospitality gets paras 6.1, 6.4, 6.5, 3.27 and the hospitality guidance).
3. **Sapling.** For each claim, `check.py` sends one call:
   - `policy`: the rule passages for that cost type, each starting with its paragraph label;
   - `evidence`: one plain sentence, for example *An MP claimed £600.00 from the Office Costs budget under the
     cost type Bought-in services (Parliamentary accountancy), with the details: "Fee in relation to prep of tax
     return for TYE 2024/25".*;
   - `claim`: *This expense claim can be paid under the IPSA Scheme.*
   Calls are cached in `packs/ipsa/sapling_cache.json` by a hash of the inputs, at most 4 run at once, and 429 and
   5xx replies are retried with backoff.
4. **Score.** `eval.py` compares each flag with IPSA's real decision, writes `packs/ipsa/results.json` and prints
   the headline numbers. A re-run uses the cache and makes no new calls.

The engine knows nothing about MPs. A policy pack is a folder under `packs/` with `pack.json` (name, claim
sentence, notes), `rules.json` (passages and which apply to which cost type) and `claims.jsonl`. The page also has
a "Paste your own policy" box that sends any policy text (up to 8,000 characters) with your own claim facts.

## Limits

- **Small and public.** 117 refused claims is a small set. The public CSV holds a short description, not the
  receipts IPSA saw, so many refusals cannot be explained from the text at all.
- **Some refusals are left out.** The 10 "Insufficient evidence" and 3 "Late submission" refusals are not scored:
  they depend on receipts and dates of submission that Sapling cannot see. No claims were "Partpaid".
- **Rule text.** The Scheme passages are from the 2024-25 edition. The guidance passages are from IPSA's current
  page (updated April 2026), not the version in force in 2024-25. We left out the guidance list of
  communications IPSA now funds, because it allows newsletters and the 2024-25 Scheme (para 6.5c) did not.
- **Name filtering is by pattern.** It uses titles, common first names, known MP names, MP web addresses and
  initials, plus constituency names. We also checked the refused rows and a sample of paid rows by hand. Business and newspaper names stay
  in, as IPSA publishes them.
- **Sapling.** It returned `insufficient_information` for almost everything that was not a clear break, and never
  `supported`. Its conformal status was always `unsure`, so we score its raw label and probabilities.
- **Paid is not the same as fine.** A paid claim can still be one a reviewer would ask about. The false-flag rate
  counts every flag on a paid claim as wrong.

## Run it

```sh
cd mp-expense-check                     # from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                 # http://127.0.0.1:8104
```

The page works from the committed files. Live checks call the public Sapling endpoint (no key needed).

To rebuild the data and the score:

```sh
.venv/bin/python fetch_data.py          # downloads the IPSA CSV and the Scheme PDF into raw/
.venv/bin/python build_eval_set.py      # writes packs/ipsa/claims.jsonl
.venv/bin/python eval.py                # cached: 0 new Sapling calls; uncached: about 420
.venv/bin/python -m pytest -q           # no network: Sapling is faked
```

Demo links: `?pack=ipsa&check=ipsa-0210&reveal=1&focus=1` opens a claim already checked, with IPSA's decision shown.
`focus=1` hides the score card.

## 2-minute demo

1. **(0:00)** Open http://127.0.0.1:8104. Read the top line: "IPSA refused 117 claims ... It flagged 6 of them, and
   3 of 297 claims IPSA paid." Point at the research-preview note.
2. **(0:20)** In the score card, open "What this means". Say: most refused claims in the public file are only a card
   line, so nobody could tell why from the text. Where the claim says what the money was for, Sapling flags 1 in 10
   refused claims and 1 in 50 paid ones.
3. **(0:45)** In the claim list, find `ipsa-0210` (an accountancy fee for a tax return). Press **Check with Sapling**.
   Show the "Flag for review" verdict, the four bars and the paragraph labels. Open the rule passages. Press
   **Reveal IPSA's real decision**: Not Paid, Not claimable.
4. **(1:10)** Check `ipsa-0005` (an office desk and chair for home working). Sapling does not flag it. Reveal: IPSA
   refused it. Say this plainly: it misses most refusals.
5. **(1:30)** Click **Company travel and expenses (example)**. Check `travel-04` (a business-class upgrade). It is
   flagged, with the travel section quoted. Say: same engine, a different policy, nothing to set up.
6. **(1:45)** Open **Paste your own policy**. Paste one rule and one claim, and press **Check with Sapling**.

## Data

Contains public sector information published by IPSA ([theipsa.org.uk](https://theipsa.org.uk)), licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Sources: the [2024-25 claims data](https://www.theipsa.org.uk/mp-costs/other-published-data), the
[Scheme, 17th edition](https://theipsa.org.uk/publications/the-scheme) and the
[office costs guidance](https://information.ipsaonline.org.uk/guidance/office-costs). Names, constituencies,
parliamentary ids and claim numbers are removed. The raw CSV is not committed.
