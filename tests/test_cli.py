"""Smoke tests for the Phase 1 CLI stubs."""

from __future__ import annotations

from btcpred.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_shows_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output


def test_fetch_stub_runs() -> None:
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 0


def test_features_stub_runs() -> None:
    result = runner.invoke(app, ["features"])
    assert result.exit_code == 0


def test_train_stub_requires_model() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0


def test_train_stub_runs_with_model() -> None:
    result = runner.invoke(app, ["train", "--model", "xgboost"])
    assert result.exit_code == 0


def test_report_stub_runs() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0
