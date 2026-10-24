"""Download the public tau-bench trajectories and policies into tau-bench/raw/ (gitignored).

Source: https://github.com/sierra-research/tau-bench (MIT licence), pinned to one commit so the numbers can be rerun.
"""

import sys
from pathlib import Path

import httpx

COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
BASE = f"https://raw.githubusercontent.com/sierra-research/tau-bench/{COMMIT}/"
RAW = Path(__file__).resolve().parent / "raw"

FILES = [
    "historical_trajectories/gpt-4o-airline.json",
    "historical_trajectories/gpt-4o-retail.json",
    "historical_trajectories/sonnet-35-new-airline.json",
    "historical_trajectories/sonnet-35-new-retail.json",
    "tau_bench/envs/airline/wiki.md",
    "tau_bench/envs/retail/wiki.md",
    "LICENSE",
]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = RAW / name.split("/")[-1] if name.startswith("historical") else RAW / name.replace("tau_bench/envs/", "").replace("/", "-")
        if out.exists() and out.stat().st_size:
            print(f"have {out.name}")
            continue
        resp = httpx.get(BASE + name, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        print(f"got  {out.name} ({len(resp.content):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
