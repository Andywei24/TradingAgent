from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine

from tradeagent.data.db import _migrate_sqlite, connect
from tradeagent.data.models import agent_runs
from tradeagent.viz.report import build_report


def test_report_omits_charts_section_when_none():
    with connect() as conn:
        res = conn.execute(
            agent_runs.insert().values(
                user_query="q",
                final_answer="a",
                tool_trace="[]",
                artifacts="[]",
            )
        )
        run_id = int(res.inserted_primary_key[0])

    md = build_report(run_id).read_text(encoding="utf-8")
    assert "## Charts" not in md
    assert "## Answer" in md


def test_report_uses_relative_chart_path(tmp_path: Path):
    # a chart living under <data_dir>/reports/charts should render as charts/<file>.png
    from tradeagent.config import get_settings

    charts_dir = Path(get_settings().data_dir) / "reports" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    png = charts_dir / "AAPL_forecast_test.png"
    png.write_bytes(b"\x89PNG\r\n")  # not a real image, only the path matters here

    import json

    with connect() as conn:
        res = conn.execute(
            agent_runs.insert().values(
                user_query="q",
                final_answer="a",
                tool_trace="[]",
                artifacts=json.dumps([str(png)]),
            )
        )
        run_id = int(res.inserted_primary_key[0])

    md = build_report(run_id).read_text(encoding="utf-8")
    assert "## Charts" in md
    assert "![AAPL_forecast_test](charts/AAPL_forecast_test.png)" in md


def test_migration_adds_artifacts_column(tmp_path: Path):
    db = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db}", future=True)
    # legacy agent_runs without the artifacts column
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE agent_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "started_at TIMESTAMP, user_query TEXT, final_answer TEXT, tool_trace TEXT)"
        )

    _migrate_sqlite(engine)

    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(agent_runs)")}
    assert "artifacts" in cols
