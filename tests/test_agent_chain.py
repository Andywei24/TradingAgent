from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tradeagent.agent.chain import run_chain
from tradeagent.data.queries import upsert_bars, upsert_instruments
from tradeagent.viz.report import build_report


def _seed(symbol: str = "AAPL", n: int = 60) -> None:
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    base = datetime(2024, 1, 1)
    upsert_bars(
        [
            {
                "symbol": symbol,
                "ts": base + timedelta(days=i),
                "interval": "1d",
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "adj_close": 100.5 + i,
                "volume": 1_000_000.0,
            }
            for i in range(n)
        ]
    )


def _mk_msg(content: str | None = None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _mk_resp(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _mk_call(call_id: str, name: str, args: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakeClient:
    """Scripted Deepseek replacement: emits planner JSON, one tool call, then a summary."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, response_format=None):
        self.calls += 1
        # 1) planner — JSON object
        if response_format and response_format.get("type") == "json_object":
            return _mk_resp(
                _mk_msg(
                    content=json.dumps(
                        {"subgoals": [{"id": 1, "goal": "look at AAPL recent prices", "tools": ["get_price_history"]}]}
                    )
                )
            )
        # 2) executor first turn — emit one tool call
        if tools and self.calls == 2:
            return _mk_resp(
                _mk_msg(
                    content=None,
                    tool_calls=[_mk_call("c1", "get_price_history", {"symbol": "AAPL", "last_n": 5})],
                )
            )
        # 3) executor second turn — text summary, no more tool calls
        if tools:
            return _mk_resp(_mk_msg(content="Saw 5 recent AAPL bars.", tool_calls=None))
        # 4) synthesizer
        return _mk_resp(_mk_msg(content="# Answer\nLooks fine.\n\nNot financial advice."))


def test_run_chain_end_to_end_mocked():
    _seed()
    result = run_chain("How is AAPL doing?", client=FakeClient())
    assert result.run_id >= 1
    assert "Not financial advice" in result.answer
    assert any(t.get("tool") == "get_price_history" for t in result.trace)


def _seed_long(symbol: str = "AAPL", n: int = 250) -> None:
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    rng = np.random.default_rng(3)
    prices = 100 + rng.standard_normal(n).cumsum() * 0.5
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        p = float(prices[i])
        rows.append(
            {
                "symbol": symbol,
                "ts": base + timedelta(days=i),
                "interval": "1d",
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "adj_close": p,
                "volume": 1_000_000.0,
            }
        )
    upsert_bars(rows)


class ForecastFakeClient:
    """Scripted client that drives a run_linear_forecast subgoal (which renders a chart)."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, response_format=None):
        self.calls += 1
        if response_format and response_format.get("type") == "json_object":
            return _mk_resp(
                _mk_msg(
                    content=json.dumps(
                        {"subgoals": [{"id": 1, "goal": "forecast AAPL", "tools": ["run_linear_forecast"]}]}
                    )
                )
            )
        if tools and self.calls == 2:
            return _mk_resp(
                _mk_msg(
                    content=None,
                    tool_calls=[_mk_call("c1", "run_linear_forecast", {"symbol": "AAPL", "horizon_days": 5})],
                )
            )
        if tools:
            return _mk_resp(_mk_msg(content="Forecast computed.", tool_calls=None))
        return _mk_resp(_mk_msg(content="# Answer\nSee the forecast chart below.\n\nNot financial advice."))


class MissingSymbolFakeClient:
    """Planner -> get_price_history on an un-ingested symbol -> summary -> synth."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, response_format=None):
        self.calls += 1
        if response_format and response_format.get("type") == "json_object":
            return _mk_resp(
                _mk_msg(content=json.dumps({"subgoals": [{"id": 1, "goal": "prices", "tools": ["get_price_history"]}]}))
            )
        if tools and self.calls == 2:
            return _mk_resp(
                _mk_msg(content=None, tool_calls=[_mk_call("c1", "get_price_history", {"symbol": "ZZZZ"})])
            )
        if tools:
            return _mk_resp(_mk_msg(content="No data available for ZZZZ.", tool_calls=None))
        return _mk_resp(_mk_msg(content="No data for ZZZZ. Run `tradeagent ingest ZZZZ`.\n\nNot financial advice."))


class WrongSymbolIndicatorFakeClient:
    """Simulates an LLM drifting from GOOG in the user query to AAPL in a tool call."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, response_format=None):
        self.calls += 1
        if response_format and response_format.get("type") == "json_object":
            return _mk_resp(
                _mk_msg(
                    content=json.dumps(
                        {"subgoals": [{"id": 1, "goal": "Compute indicators for GOOG", "tools": ["compute_indicator"]}]}
                    )
                )
            )
        if tools and self.calls == 2:
            return _mk_resp(
                _mk_msg(
                    content=None,
                    tool_calls=[
                        _mk_call(
                            "c1",
                            "compute_indicator",
                            {"symbol": "AAPL", "indicator": "rsi_14", "interval": "1d", "last_n": 5},
                        )
                    ],
                )
            )
        if tools:
            return _mk_resp(_mk_msg(content="RSI computed for GOOG.", tool_calls=None))
        return _mk_resp(_mk_msg(content="# Answer\nGOOG indicators were computed.\n\nNot financial advice."))


class FalseNoDataSynthFakeClient:
    """Planner omits indicators, then synthesizer wrongly emits an ingest warning."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.0, response_format=None):
        self.calls += 1
        if response_format and response_format.get("type") == "json_object":
            return _mk_resp(
                _mk_msg(
                    content=json.dumps(
                        {"subgoals": [{"id": 1, "goal": "Get price history for GOOG", "tools": ["get_price_history"]}]}
                    )
                )
            )
        if tools:
            goal = messages[0]["content"]
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if has_tool_result:
                return _mk_resp(_mk_msg(content="Tool returned data for GOOG.", tool_calls=None))
            if "RSI" in goal:
                return _mk_resp(
                    _mk_msg(
                        content=None,
                        tool_calls=[
                            _mk_call(
                                "c2",
                                "compute_indicator",
                                {"symbol": "GOOG", "indicator": "rsi_14", "interval": "1d", "last_n": 5},
                            )
                        ],
                    )
                )
            return _mk_resp(
                _mk_msg(content=None, tool_calls=[_mk_call("c1", "get_price_history", {"symbol": "GOOG", "last_n": 5})])
            )
        return _mk_resp(
            _mk_msg(
                content="# Answer\nGOOG data is available.\n\n**No data for GOOG — run `tradeagent ingest GOOG`.**\n\nNot financial advice."
            )
        )


def test_missing_symbol_error_reaches_trace():
    # ZZZZ is never seeded; auto_ingest defaults off.
    result = run_chain("How is ZZZZ doing?", client=MissingSymbolFakeClient())
    tool_entries = [t for t in result.trace if t.get("tool") == "get_price_history"]
    assert tool_entries and "error" in (tool_entries[0]["result_keys"] or [])
    assert not result.charts


def test_single_requested_symbol_corrects_wrong_tool_symbol():
    _seed("GOOG", n=80)
    result = run_chain("Is GOOG overbought?", client=WrongSymbolIndicatorFakeClient())

    tool_entries = [t for t in result.trace if t.get("tool") == "compute_indicator"]
    assert tool_entries
    entry = tool_entries[0]
    args = json.loads(entry["args"])
    assert args["symbol"] == "GOOG"
    assert entry["corrected_symbol_from"] == "AAPL"
    assert "values" in entry["result_keys"]


def test_overbought_run_injects_indicators_and_removes_false_no_data_warning():
    _seed("GOOG", n=80)
    result = run_chain("Is GOOG overbought?", client=FalseNoDataSynthFakeClient())

    assert any("compute_indicator" in (sg.get("tools") or []) for sg in result.plan["subgoals"])
    assert any(t.get("tool") == "compute_indicator" for t in result.trace)
    assert "No data for GOOG" not in result.answer
    assert "tradeagent ingest GOOG" not in result.answer


def test_run_chain_generates_and_persists_chart():
    _seed_long()
    result = run_chain("Forecast AAPL 5 days", client=ForecastFakeClient())

    # chart captured on the RunResult and the PNG actually exists
    assert result.charts, "expected at least one chart"
    assert Path(result.charts[0]).exists()
    # trace entry for the forecast tool carries the chart path
    assert any(t.get("chart_path") for t in result.trace)

    # persisted to agent_runs.artifacts
    from sqlalchemy import select

    from tradeagent.data.db import connect
    from tradeagent.data.models import agent_runs

    with connect() as conn:
        row = conn.execute(select(agent_runs).where(agent_runs.c.id == result.run_id)).mappings().one()
    assert json.loads(row["artifacts"])

    # report embeds it under a Charts section with a relative image link
    path = build_report(result.run_id)
    md = path.read_text(encoding="utf-8")
    assert "## Charts" in md
    assert "![" in md and "charts/" in md and ".png)" in md
