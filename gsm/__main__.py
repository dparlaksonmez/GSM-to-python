"""
GSM Pipeline — python -m gsm support.

Usage:
    python -m gsm                  # Interactive menu
    python -m gsm train            # Train pipeline
    python -m gsm infer            # Clinical inference
    python -m gsm bundle-info      # Inspect a model bundle
"""

import sys
from pathlib import Path

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.cli import main  # noqa: E402

main()
