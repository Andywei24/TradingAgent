from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from tradeagent.agent.chain import run_chain
from tradeagent.data.queries import upsert_bars, upsert_instruments


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
