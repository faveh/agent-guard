"""Download IPSA's published 2024-25 business costs CSV and the 2024-25 Scheme PDF into raw/ (not in git).

    python fetch_data.py
"""

from pathlib import Path

import httpx

RAW_DIR = Path(__file__).resolve().parent / "raw"
FILES = {
    "ipsa_24_25.csv": "https://www.theipsa.org.uk/api/download?type=individualBusinessCosts&year=24_25",
    "scheme_2024_25.pdf": (
        "https://assets.ctfassets.net/nc7h1cs4q6ic/5j9dlpZsDtyFgqi43bBB91/7eaea5bcc329496878b420bf2e511586/"
        "Seventeenth_edition_of_the_Scheme_2024-25.pdf"
    ),
}


def main():
    RAW_DIR.mkdir(exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as http:
        for name, url in FILES.items():
            target = RAW_DIR / name
            if target.exists():
                print(f"have {target.name} ({target.stat().st_size:,} bytes)")
                continue
            resp = http.get(url)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            print(f"saved {target.name} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    main()
