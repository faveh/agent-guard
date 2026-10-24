"""Data filtering and anonymising for the IPSA evaluation set."""

import build_eval_set as ds


def row(status="Paid", reason="", details="Printer paper", cost_type="Stationery & printing", amount="12.5", **kw):
    base = {"Parliamentary ID": "1", "Year": "24_25", "Date": "01/05/2024", "Claim Number": "60000001-1",
            "Name": "Pat Example", "Constituency": "Somewhere", "Category": "Office Costs", "Cost Type": cost_type,
            "Short Description": "", "Details": details, "Amount Claimed": amount, "Amount Paid": amount,
            "Status": status, "Reason If Not Paid": reason}
    base.update(kw)
    return base


def test_person_names_are_caught():
    assert ds.has_person_name("Small posters to advertise Sean´s surgeries")
    assert ds.has_person_name("Window graphics, Sarah imag and poster holders")
    assert ds.has_person_name("J Reynolds work mobile - August")
    assert ds.has_person_name("Dinner with Mr Smith")
    assert ds.has_person_name("Goweb services for janedoemp.co.uk website")
    assert ds.has_person_name("OWEN KERR SIGNS")
    assert ds.has_person_name("Leaflet about pat example's surgery", known_names={"pat example"})


def test_ordinary_claims_pass():
    for text in ["AMAZON [***]", "Laptop bought for Westminster office", "Festive team meal", "VISTAPRINT",
                 "Tea, coffee, biscuits for community coffee morning", "HOC CRS"]:
        assert not ds.has_person_name(text), text


def test_addresses_are_caught():
    assert ds.has_address("BT bill for new office at 9 Example Road")
    assert ds.has_address("office SW1A 0AA")
    assert not ds.has_address("Printer ink")


def test_anonymise_drops_who_claimed_and_reference_numbers():
    out = ds.anonymise(row(details="BT BUSINESS BILL [200011725-7744]"))
    assert set(out) == {"date", "category", "cost_type", "short_description", "details", "amount_claimed",
                        "amount_paid", "status", "reason"}
    assert out["details"] == "BT BUSINESS BILL"
    assert out["amount_claimed"] == 12.5


def test_select_keeps_only_not_claimable_and_stratifies_negatives():
    rows = [
        row("Not Paid", "Not claimable", "Festive team meal", "Hospitality", amount="200"),
        row("Not Paid", "Not claimable", "Mugs for staff kitchen"),
        row("Not Paid", "Late submission", "Printer paper"),
        row("Not Paid", "Insufficient evidence", "Printer paper"),
        row("Not Paid", "Not claimable", "Dinner with Mr Smith"),
        row("Repaid", "", "Printer paper"),
    ] + [row(details=f"Paper box {i}") for i in range(10)] + [row(details=f"Coffee {i}", cost_type="Hospitality") for i in range(10)] \
      + [row(details="Toner", cost_type="Rent")] + [row(details="Aggregated figure for this business cost")]
    pos, neg = ds.select(rows, negatives=4, min_per_type=1)
    assert [r["Details"] for r in pos] == ["Festive team meal", "Mugs for staff kitchen"]
    assert all(r["Status"] == "Paid" for r in neg)
    assert {r["Cost Type"] for r in neg} == {"Hospitality", "Stationery & printing"}
    assert len(neg) == 4
    assert ds.select(rows, negatives=4, min_per_type=1) == (pos, neg)  # fixed seed


def test_build_shuffles_ids_and_carries_no_names():
    rows = [row("Not Paid", "Not claimable", f"Item {i}") for i in range(5)] + [row(details=f"Paper {i}") for i in range(20)]
    claims = ds.build(rows, negatives=10, min_per_type=1)
    assert [c["id"] for c in claims] == [f"ipsa-{i:04d}" for i in range(1, len(claims) + 1)]
    assert [c["label"] for c in claims] != sorted(c["label"] for c in claims) or len(claims) < 3
    for c in claims:
        assert "Pat Example" not in str(c) and "Somewhere" not in str(c) and "60000001" not in str(c)


def test_build_drops_rows_that_name_a_constituency_or_an_owner():
    rows = [row("Not Paid", "Not claimable", f"Item {i}") for i in range(3)]
    rows += [row(details="Thanks to north antrim"), row(details="H CARTER SONS"), row(details="Paper"),
             row(Constituency="North Antrim CC", details="Pens")]
    claims = ds.build(rows, negatives=10, min_per_type=1)
    assert sorted(c["details"] for c in claims if c["label"] == "paid") == ["Paper", "Pens"]
