"""Fetchers for OHLCV, macro, sentiment, and on-chain data, cached under data/raw/."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import ccxt
import pandas as pd
import requests
import yfinance as yf

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0


def _request_with_backoff(
    url: str, params: dict[str, str | int] | None = None
) -> requests.Response:
    """GET a URL with exponential backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts") from last_exc


def _record_manifest(manifest_path: str | Path, entry: dict[str, object]) -> None:
    """Append a fetch record to the JSON manifest, creating it if absent."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else []
    entry["fetched_at"] = datetime.now(UTC).isoformat()
    manifest.append(entry)
    path.write_text(json.dumps(manifest, indent=2))


def _record_dataset(
    manifest_path: str | Path, source: str, dataset: str, df: pd.DataFrame, cache_path: Path
) -> None:
    """Record a fetched-and-cached dataset's shape and date range into the manifest."""
    _record_manifest(
        manifest_path,
        {
            "source": source,
            "dataset": dataset,
            "rows": len(df),
            "start": df.index.min().isoformat() if len(df) else None,
            "end": df.index.max().isoformat() if len(df) else None,
            "path": str(cache_path),
        },
    )


def _fetch_ohlcv_batch(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, since: int, limit: int
) -> list[list[float]]:
    """Fetch one paginated OHLCV batch, retrying transient failures with backoff."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:  # ccxt raises its own hierarchy of network/exchange errors
            last_exc = exc
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise RuntimeError(f"Failed to fetch OHLCV batch after {MAX_RETRIES} attempts") from last_exc


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: str,
    raw_data_dir: str | Path = "data/raw",
    manifest_path: str | Path = "data/manifest.json",
    exchange_id: str = "binance",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch paginated OHLCV candles, resuming from any existing cached parquet file.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
        timeframe: ccxt timeframe string, e.g. "1h" or "1d".
        start_date: ISO date to start from if no cache exists yet.
        raw_data_dir: Directory to read/write the cached parquet file.
        manifest_path: Path to the JSON fetch manifest.
        exchange_id: ccxt exchange id, e.g. "binance".
        limit: Candles per paginated request.

    Returns:
        The full cached OHLCV DataFrame, indexed by UTC timestamp.
    """
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "")
    cache_path = raw_dir / f"ohlcv_{safe_symbol}_{timeframe}.parquet"

    exchange = getattr(ccxt, exchange_id)()

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        since = int(existing.index[-1].timestamp() * 1000) + 1
    else:
        existing = pd.DataFrame()
        since = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)

    rows: list[list[float]] = []
    while True:
        batch = _fetch_ohlcv_batch(exchange, symbol, timeframe, since, limit)
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + 1
        if len(batch) < limit:
            break

    if rows:
        new_df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        new_df = new_df.set_index("timestamp")
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = existing

    combined.to_parquet(cache_path)
    _record_dataset(manifest_path, exchange_id, f"ohlcv_{symbol}_{timeframe}", combined, cache_path)
    return combined


def fetch_macro(
    tickers: dict[str, str],
    start_date: str,
    raw_data_dir: str | Path = "data/raw",
    manifest_path: str | Path = "data/manifest.json",
) -> dict[str, pd.DataFrame]:
    """Fetch daily macro series (DXY, SPX, gold, VIX, 10Y yield) via yfinance.

    Args:
        tickers: Mapping of short name -> yfinance ticker symbol.
        start_date: ISO date to start the history from.
        raw_data_dir: Directory to write cached parquet files.
        manifest_path: Path to the JSON fetch manifest.

    Returns:
        Mapping of short name -> fetched DataFrame.
    """
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        df.index = pd.to_datetime(df.index, utc=True)
        cache_path = raw_dir / f"macro_{name}.parquet"
        df.to_parquet(cache_path)
        _record_dataset(manifest_path, "yfinance", f"macro_{name}", df, cache_path)
        results[name] = df
    return results


def fetch_fear_greed(
    raw_data_dir: str | Path = "data/raw",
    manifest_path: str | Path = "data/manifest.json",
    url: str = "https://api.alternative.me/fng/",
) -> pd.DataFrame:
    """Fetch the full history of the Crypto Fear & Greed Index.

    Args:
        raw_data_dir: Directory to write the cached parquet file.
        manifest_path: Path to the JSON fetch manifest.
        url: Fear & Greed Index API endpoint.

    Returns:
        DataFrame indexed by UTC timestamp with "value" and "value_classification".
    """
    response = _request_with_backoff(url, params={"limit": 0, "format": "json"})
    payload = response.json()["data"]
    df = pd.DataFrame(payload)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    df = df.set_index("timestamp")[["value", "value_classification"]].sort_index()

    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "fear_greed.parquet"
    df.to_parquet(cache_path)
    _record_dataset(manifest_path, "alternative.me", "fear_greed", df, cache_path)
    return df


def fetch_onchain(
    charts: dict[str, str],
    raw_data_dir: str | Path = "data/raw",
    manifest_path: str | Path = "data/manifest.json",
    base_url: str = "https://api.blockchain.info/charts",
) -> dict[str, pd.DataFrame]:
    """Fetch on-chain metrics (hash rate, active addresses, etc.) from blockchain.com.

    Args:
        charts: Mapping of short name -> blockchain.com chart slug.
        raw_data_dir: Directory to write cached parquet files.
        manifest_path: Path to the JSON fetch manifest.
        base_url: blockchain.com charts API base URL.

    Returns:
        Mapping of short name -> fetched single-column DataFrame.
    """
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    for name, chart in charts.items():
        response = _request_with_backoff(
            f"{base_url}/{chart}", params={"timespan": "all", "format": "json"}
        )
        payload = response.json()["values"]
        df = pd.DataFrame(payload)
        df["x"] = pd.to_datetime(df["x"], unit="s", utc=True)
        df = df.rename(columns={"x": "timestamp", "y": name}).set_index("timestamp").sort_index()

        cache_path = raw_dir / f"onchain_{name}.parquet"
        df.to_parquet(cache_path)
        _record_dataset(manifest_path, "blockchain.com", f"onchain_{name}", df, cache_path)
        results[name] = df
    return results
