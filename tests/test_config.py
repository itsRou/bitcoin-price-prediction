"""Tests for the typed YAML config loaders."""

from __future__ import annotations

from btcpred.config import load_data_config


def test_load_data_config_matches_real_yaml() -> None:
    config = load_data_config("configs/data.yaml")

    assert config.exchange == "binance"
    assert config.symbol == "BTC/USDT"
    assert "1h" in config.timeframes and "1d" in config.timeframes
    assert "dxy" in config.macro_tickers
    assert "hash_rate" in config.onchain
