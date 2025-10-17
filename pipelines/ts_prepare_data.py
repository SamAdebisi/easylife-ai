"""Bootstrap synthetic data for the time-series forecasting phase."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from ts_forecasting.data_utils import generate_synthetic_series, save_dataset

    df = generate_synthetic_series()
    save_dataset(df)


if __name__ == "__main__":
    main()
