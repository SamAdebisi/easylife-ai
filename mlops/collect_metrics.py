"""Collect Prometheus metrics and export them as a CSV summary."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import requests

DEFAULT_METRICS = [
    "nlp_predictions_total",
    "ts_forecast_requests_total",
    "recsys_recommendation_requests_total",
]


def _query_metric(endpoint: str, metric: str) -> float:
    response = requests.get(
        f"{endpoint}/api/v1/query",
        params={"query": metric},
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed for {metric!r}: {payload}")

    results = payload.get("data", {}).get("result", [])
    total = 0.0
    for item in results:
        try:
            value = float(item["value"][1])
        except (KeyError, ValueError, IndexError) as exc:
            raise RuntimeError(f"Invalid value for metric {metric!r}: {item}") from exc
        total += value
    return total


def collect_metrics(
    endpoint: str,
    metrics: Iterable[str],
    output: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []
    for metric in metrics:
        value = _query_metric(endpoint, metric)
        rows.append([timestamp, metric, f"{value:.6f}"])

    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["timestamp", "metric", "value"])
        writer.writerows(rows)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect metrics from Prometheus and store them as CSV.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:9090",
        help="Prometheus base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Space separated list of Prometheus metric names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlops") / "reports" / "metrics_summary.csv",
        help="Output CSV file (default: %(default)s).",
    )
    args = parser.parse_args()

    collect_metrics(
        endpoint=args.endpoint,
        metrics=args.metrics,
        output=args.output,
    )


if __name__ == "__main__":
    main()
