"""Vercel entry point: Vercel's Python runtime serves the FastAPI app in app.py from here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402,F401
