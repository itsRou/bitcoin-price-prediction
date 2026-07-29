"""Headless smoke test for the Streamlit dashboard, via Streamlit's AppTest harness."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_app_runs_without_exceptions() -> None:
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120)

    at.run()

    assert not at.exception
    assert len(at.tabs) == 4
    assert at.title[0].value.startswith("Bitcoin Price Prediction")
