"""Generate synthetic interaction data for the recommendation service."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from recsys_service.data_utils import generate_interactions, save_datasets

    train, test, items = generate_interactions()
    save_datasets(train, test, items)


if __name__ == "__main__":
    main()
