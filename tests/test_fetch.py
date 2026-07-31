"""Tests for data fetchers using mocked HTTP, ccxt, and yfinance calls (no real network)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import responses

from btcpred.data.fetch import (
    _fetch_ohlcv_batch,
    _request_with_backoff,
    fetch_fear_greed,
    fetch_macro,
    fetch_ohlcv,
    fetch_onchain,
)


@responses.activate
def test_fetch_fear_greed_caches_and_records_manifest(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        "https://api.alternative.me/fng/",
        json={
            "data": [
                {"value": "50", "value_classification": "Neutral", "timestamp": "1700000000"},
                {"value": "70", "value_classification": "Greed", "timestamp": "1700086400"},
            ]
        },
        status=200,
    )
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"

    df = fetch_fear_greed(raw_data_dir=raw_dir, manifest_path=manifest_path)

    assert len(df) == 2
    assert (raw_dir / "fear_greed.parquet").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest[0]["dataset"] == "fear_greed"
    assert manifest[0]["rows"] == 2


@responses.activate
def test_fetch_onchain_caches_multiple_charts(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        "https://api.blockchain.info/charts/hash-rate",
        json={"values": [{"x": 1700000000, "y": 1.23}, {"x": 1700086400, "y": 1.30}]},
        status=200,
    )
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"

    result = fetch_onchain(
        {"hash_rate": "hash-rate"}, raw_data_dir=raw_dir, manifest_path=manifest_path
    )

    assert "hash_rate" in result
    assert len(result["hash_rate"]) == 2
    assert (raw_dir / "onchain_hash_rate.parquet").exists()


def test_fetch_macro_uses_yfinance(tmp_path: Path) -> None:
    fake_df = pd.DataFrame(
        {"Open": [1.0, 2.0], "Close": [1.1, 2.1]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"

    with patch("btcpred.data.fetch.yf.download", return_value=fake_df) as mock_download:
        result = fetch_macro(
            {"dxy": "DX-Y.NYB"}, "2024-01-01", raw_data_dir=raw_dir, manifest_path=manifest_path
        )

    mock_download.assert_called_once()
    assert "dxy" in result
    assert (raw_dir / "macro_dxy.parquet").exists()


def test_fetch_ohlcv_paginates_and_caches(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.json"

    batch_1 = [[1700000000000 + i * 3_600_000, 1.0, 1.1, 0.9, 1.05, 100.0] for i in range(3)]
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = [batch_1, []]

    with patch("btcpred.data.fetch.ccxt.binance", return_value=mock_exchange):
        df = fetch_ohlcv(
            "BTC/USDT", "1h", "2023-11-14", raw_data_dir=raw_dir, manifest_path=manifest_path
        )

    assert len(df) == 3
    assert (raw_dir / "ohlcv_BTCUSDT_1h.parquet").exists()


def test_fetch_ohlcv_resumes_from_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"

    existing = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [100.0]},
        index=pd.to_datetime(["2023-11-14 00:00:00"], utc=True),
    )
    existing.index.name = "timestamp"
    existing.to_parquet(raw_dir / "ohlcv_BTCUSDT_1h.parquet")

    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = [[]]

    with patch("btcpred.data.fetch.ccxt.binance", return_value=mock_exchange):
        df = fetch_ohlcv(
            "BTC/USDT", "1h", "2023-11-14", raw_data_dir=raw_dir, manifest_path=manifest_path
        )

    assert len(df) == 1
    called_since = mock_exchange.fetch_ohlcv.call_args.kwargs["since"]
    assert called_since > int(pd.Timestamp("2023-11-14", tz="UTC").timestamp() * 1000)


def test_request_with_backoff_retries_then_succeeds() -> None:
    ok_response = MagicMock()
    ok_response.raise_for_status.return_value = None

    with (
        patch(
            "btcpred.data.fetch.requests.get",
            side_effect=[requests.ConnectionError("boom"), ok_response],
        ),
        patch("btcpred.data.fetch.time.sleep"),
    ):
        result = _request_with_backoff("https://example.com")

    assert result is ok_response


def test_request_with_backoff_raises_after_exhausting_retries() -> None:
    with (
        patch(
            "btcpred.data.fetch.requests.get",
            side_effect=requests.ConnectionError("boom"),
        ),
        patch("btcpred.data.fetch.time.sleep"),
        pytest.raises(RuntimeError, match="Failed to fetch"),
    ):
        _request_with_backoff("https://example.com")


def test_fetch_ohlcv_batch_retries_then_succeeds() -> None:
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = [Exception("boom"), [[1, 1, 1, 1, 1, 1]]]

    with patch("btcpred.data.fetch.time.sleep"):
        result = _fetch_ohlcv_batch(mock_exchange, "BTC/USDT", "1h", 0, 1000)

    assert result == [[1, 1, 1, 1, 1, 1]]
