"""Command-line entry point for the btcpred pipeline."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="btcpred",
    help="Bitcoin log-return prediction benchmark: fetch, features, train, backtest, report.",
    no_args_is_help=True,
)


@app.command()
def fetch(
    config: str = typer.Option("configs/data.yaml", help="Path to data config YAML."),
) -> None:
    """Fetch raw OHLCV, macro, on-chain, and sentiment data into data/raw/."""
    typer.echo(f"[stub] fetch: would load config from {config}")
    raise typer.Exit(code=0)


@app.command()
def features(
    config: str = typer.Option("configs/features.yaml", help="Path to feature config YAML."),
) -> None:
    """Build the feature matrix and prediction targets into data/processed/."""
    typer.echo(f"[stub] features: would load config from {config}")
    raise typer.Exit(code=0)


@app.command()
def train(
    model: str = typer.Option(..., help="Registered model name, e.g. 'xgboost'."),
    config: str = typer.Option(None, help="Path to configs/models/<model>.yaml (optional)."),
) -> None:
    """Train (and tune) a single registered model via walk-forward validation."""
    typer.echo(f"[stub] train: model={model} config={config}")
    raise typer.Exit(code=0)


@app.command()
def backtest(
    model: str = typer.Option(..., help="Registered model name to backtest."),
) -> None:
    """Run the cost-aware backtest for a trained model's predictions."""
    typer.echo(f"[stub] backtest: model={model}")
    raise typer.Exit(code=0)


@app.command()
def report() -> None:
    """Generate the leaderboard and figures under reports/."""
    typer.echo("[stub] report: would write reports/results.md")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
