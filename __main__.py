"""
GSM Pipeline — python -m gsm support (fallback entry point).

The primary entry point is gsm/__main__.py.
This file exists as a fallback if run from the project root.

Usage:
    python -m gsm                  # Interactive menu
    python -m gsm train            # Train pipeline
    python -m gsm infer            # Clinical inference
    python -m gsm bundle-info      # Inspect a model bundle
    python -m gsm ui               # Launch Streamlit dashboard
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.cli import main  # noqa: E402

main()
