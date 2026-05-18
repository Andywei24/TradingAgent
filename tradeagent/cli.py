from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from tradeagent.config import get_settings
from tradeagent.data.db import init_db
from tradeagent.data.ingest import ingest_many, seed_nasdaq_universe

app = typer.Typer(add_completion=False, help="Trading agent CLI")
data_app = typer.Typer(help="Data management")
features_app = typer.Typer(help="Feature materialization")
rag_app = typer.Typer(help="RAG knowledge base")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(rag_app, name="rag")


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.callback()
def _main() -> None:
    _configure_logging()
    init_db()


@app.command("ingest")
def ingest(
    symbols: list[str] = typer.Argument(None, help="Symbols, e.g. AAPL MSFT NVDA"),
    interval: str = typer.Option("1d", help="1d | 60min | 15min | 5min"),
    full: bool = typer.Option(False, help="Force full history pull (outputsize=full)"),
    universe: str | None = typer.Option(None, help="nasdaq-top50 | nasdaq-all"),
    limit: int | None = typer.Option(None, help="Limit number of symbols when using --universe"),
) -> None:
    """Ingest OHLCV data from Alpha Vantage (or CSV fallback)."""
    if universe:
        if universe == "nasdaq-all":
            n = seed_nasdaq_universe(limit=limit)
            typer.echo(f"Seeded {n} NASDAQ instruments. Re-run with explicit symbols to fetch bars.")
            return
        if universe == "nasdaq-top50":
            symbols = NASDAQ_TOP50
        else:
            raise typer.BadParameter(f"unknown universe: {universe}")
    if not symbols:
        raise typer.BadParameter("Provide symbols or --universe.")
    result = ingest_many(symbols, interval=interval, full=full)
    typer.echo(json.dumps(result, indent=2))


@features_app.command("build")
def features_build(
    symbols: list[str] = typer.Argument(None),
    feature_set: str = typer.Option("core", "--set"),
    interval: str = typer.Option("1d"),
) -> None:
    """Materialize indicator features into the features table."""
    from tradeagent.data.features import materialize_features
    from tradeagent.data.queries import list_symbols

    syms = symbols or list_symbols()
    if not syms:
        typer.echo("No symbols in DB. Run `tradeagent ingest` first.")
        raise typer.Exit(1)
    total = 0
    for sym in syms:
        total += materialize_features(sym, interval=interval, feature_set=feature_set)
    typer.echo(f"Wrote {total} feature rows across {len(syms)} symbols.")


@rag_app.command("index")
def rag_index(
    path: Path = typer.Argument(..., help="Directory of documents to index"),
) -> None:
    """Embed and index documents under PATH into the FAISS store."""
    from tradeagent.rag.loader import load_directory
    from tradeagent.rag.vectorstore import build_index

    chunks = load_directory(path)
    n = build_index(chunks)
    typer.echo(f"Indexed {n} chunks into FAISS.")


@app.command("ask")
def ask(query: str = typer.Argument(...)) -> None:
    """Run the reasoning agent on a natural-language query."""
    from tradeagent.agent.chain import run_chain

    result = run_chain(query)
    typer.echo(result.answer)
    typer.echo(f"\n[run_id={result.run_id}] traces saved.")


@app.command("report")
def report(run_id: int = typer.Argument(...)) -> None:
    """Rebuild the markdown report for a prior agent run."""
    from tradeagent.viz.report import build_report

    path = build_report(run_id)
    typer.echo(f"Report written to {path}")


NASDAQ_TOP50 = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "PEP",
    "COST", "ADBE", "CSCO", "NFLX", "AMD", "TMUS", "INTC", "QCOM", "INTU", "AMAT",
    "TXN", "HON", "AMGN", "ISRG", "BKNG", "VRTX", "SBUX", "MDLZ", "ADP", "GILD",
    "ADI", "PYPL", "REGN", "LRCX", "MU", "PANW", "KLAC", "SNPS", "CDNS", "MELI",
    "ASML", "MRVL", "ABNB", "ORLY", "MAR", "FTNT", "CRWD", "CTAS", "MNST", "DXCM",
]


if __name__ == "__main__":
    app()
