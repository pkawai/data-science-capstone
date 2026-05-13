"""Streamlit Cloud entry point.

Streamlit Cloud auto-discovers `streamlit_app.py` at the repo root.
This file simply delegates to the real dashboard in bot/dashboard.py.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = os.path.join(ROOT, "bot")

if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)

runpy.run_path(os.path.join(BOT_DIR, "dashboard.py"), run_name="__main__")
