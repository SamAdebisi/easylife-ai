"""
Utility script to ensure the expected DVC data directories exist.

This lets new contributors run `dvc repro` without manually
creating `data/raw`, `data/processed`, and `data/external`.
"""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SUBDIRS = ("raw", "processed", "external")


def main() -> None:
    DATA_ROOT.mkdir(exist_ok=True)
    for name in SUBDIRS:
        subdir = DATA_ROOT / name
        subdir.mkdir(parents=True, exist_ok=True)
        gitkeep = subdir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


if __name__ == "__main__":
    main()
