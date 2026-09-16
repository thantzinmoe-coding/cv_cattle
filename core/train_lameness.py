"""Compatibility entry point for the real-video lameness trainer.

Run ``python scripts/extract_features.py`` first when the raw videos change.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_lameness import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
