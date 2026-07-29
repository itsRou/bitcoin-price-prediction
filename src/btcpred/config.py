"""Typed loaders for the YAML configs under configs/."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Schema for configs/data.yaml."""

    exchange: str
    symbol: str
    timeframes: list[str]
    start_date: str
    macro_tickers: dict[str, str]
    sentiment: dict[str, str]
    onchain: dict[str, str]
    raw_data_dir: str
    manifest_path: str


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    """Load and validate the data ingestion config."""
    raw = yaml.safe_load(Path(path).read_text())
    return DataConfig(**raw)
