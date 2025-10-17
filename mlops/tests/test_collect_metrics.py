from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mlops import collect_metrics


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.mark.parametrize(
    "values",
    [
        {"metric_a": 3.2, "metric_b": 0.0},
        {"metric_a": 1.0, "metric_b": 2.5, "metric_c": 7.9},
    ],
)
def test_collect_metrics_writes_csv(tmp_path: Path, monkeypatch, values: dict) -> None:
    def fake_get(url: str, params: dict, timeout: int) -> DummyResponse:  # noqa: ARG001
        metric = params["query"]
        value = values[metric]
        payload = {
            "status": "success",
            "data": {
                "result": [
                    {"value": [0, str(value)]},
                ],
            },
        }
        return DummyResponse(payload)

    monkeypatch.setattr(collect_metrics.requests, "get", fake_get)
    output = tmp_path / "report.csv"
    collect_metrics.collect_metrics("http://localhost:9090", values.keys(), output)

    assert output.exists()
    with output.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    assert len(rows) == len(values)
    for row in rows:
        metric = row["metric"]
        assert pytest.approx(float(row["value"])) == values[metric]
