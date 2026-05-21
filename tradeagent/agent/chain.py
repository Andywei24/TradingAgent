from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tradeagent.agent.llm import LLMClient, make_llm_client
from tradeagent.agent.prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from tradeagent.agent.tools import REGISTRY, auto_ingest_var, tool_schemas
from tradeagent.data.db import connect
from tradeagent.data.models import agent_runs

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS_PER_GOAL = 5
SYMBOL_SCOPED_TOOLS = {
    "get_price_history",
    "compute_indicator",
    "run_linear_forecast",
    "decompose_signal",
    "plot_chart",
    "summarize_statistics",
}
SYMBOL_STOPWORDS = {
    "API",
    "ATR",
    "CSV",
    "DB",
    "EMA",
    "ETF",
    "FAISS",
    "FFT",
    "LLM",
    "MACD",
    "ML",
    "OHLCV",
    "RAG",
    "RSI",
    "SMA",
    "SQL",
    "USD",
}


@dataclass
class RunResult:
    run_id: int
    answer: str
    plan: dict
    trace: list[dict] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)


def _extract_chart_path(result: dict) -> str | None:
    cp = result.get("chart_path")
    if not cp:
        p = result.get("path")
        if isinstance(p, str) and p.lower().endswith(".png"):
            cp = p
    return cp


def _requested_symbol(query: str) -> str | None:
    candidates = []
    for match in re.findall(r"\b[A-Z][A-Z0-9.]{0,9}\b", query):
        symbol = match.rstrip(".").upper()
        if symbol and symbol not in SYMBOL_STOPWORDS:
            candidates.append(symbol)
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _apply_requested_symbol(raw_args: str, requested_symbol: str | None) -> tuple[str, str | None]:
    if not requested_symbol:
        return raw_args, None
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return raw_args, None
    if not isinstance(args, dict) or "symbol" not in args:
        return raw_args, None
    original = str(args["symbol"]).upper()
    if original == requested_symbol:
        return raw_args, None
    args["symbol"] = requested_symbol
    return json.dumps(args), original


def _dispatch_tool_calls(
    calls: list[Any],
    trace: list[dict],
    charts: list[str],
    requested_symbol: str | None,
) -> list[dict]:
    messages: list[dict] = []
    for call in calls:
        name = call.function.name
        raw_args = call.function.arguments or "{}"
        corrected_from: str | None = None
        if name in SYMBOL_SCOPED_TOOLS:
            raw_args, corrected_from = _apply_requested_symbol(raw_args, requested_symbol)
        try:
            if name not in REGISTRY:
                result: dict = {"error": f"unknown tool {name}"}
            else:
                result = REGISTRY[name].call(raw_args)
        except Exception as e:  # surface to model
            log.exception("tool %s failed", name)
            result = {"error": f"{type(e).__name__}: {e}"}
        entry: dict = {
            "tool": name,
            "args": raw_args,
            "result_keys": list(result.keys()) if isinstance(result, dict) else None,
        }
        if corrected_from:
            entry["corrected_symbol_from"] = corrected_from
        if isinstance(result, dict):
            chart = _extract_chart_path(result)
            if chart:
                charts.append(chart)
                entry["chart_path"] = chart
        trace.append(entry)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result, default=str),
            }
        )
    return messages


def _plan(client: LLMClient, query: str) -> dict:
    resp = client.chat(
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM + "\nAvailable tools: " + ", ".join(REGISTRY)},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log.warning("planner returned non-JSON, defaulting to single subgoal")
        return {"subgoals": [{"id": 1, "goal": query, "tools": list(REGISTRY)}]}


def _execute_subgoal(
    client: LLMClient,
    goal: str,
    tool_names: list[str],
    trace: list[dict],
    charts: list[str],
    requested_symbol: str | None,
) -> str:
    tools = tool_schemas([n for n in tool_names if n in REGISTRY] or None)
    messages: list[dict] = [
        {"role": "system", "content": EXECUTOR_SYSTEM.format(goal=goal)},
        {"role": "user", "content": goal},
    ]
    for _ in range(MAX_TOOL_ITERATIONS_PER_GOAL):
        resp = client.chat(messages=messages, tools=tools, tool_choice="auto", temperature=0.0)
        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )
        if not msg.tool_calls:
            return msg.content or ""
        messages.extend(_dispatch_tool_calls(msg.tool_calls, trace, charts, requested_symbol))
    return "[executor stopped after max iterations]"


def _synthesize(
    client: LLMClient,
    user_query: str,
    plan: dict,
    subgoal_summaries: list[str],
) -> str:
    context_blob = "\n\n".join(
        f"### Sub-goal {i+1}: {sg.get('goal')}\n{summary}"
        for i, (sg, summary) in enumerate(zip(plan.get("subgoals", []), subgoal_summaries))
    )
    resp = client.chat(
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {
                "role": "user",
                "content": f"User question:\n{user_query}\n\nEvidence:\n{context_blob}",
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def run_chain(
    query: str,
    client: LLMClient | None = None,
    provider: str | None = None,
    auto_ingest: bool = False,
) -> RunResult:
    client = client or make_llm_client(provider)
    trace: list[dict] = []
    charts: list[str] = []

    token = auto_ingest_var.set(auto_ingest)
    try:
        requested_symbol = _requested_symbol(query)
        plan = _plan(client, query)
        trace.append({"phase": "plan", "plan": plan})

        summaries: list[str] = []
        for sg in plan.get("subgoals", []):
            goal = sg.get("goal", "")
            tools = sg.get("tools", []) or list(REGISTRY)
            summary = _execute_subgoal(client, goal, tools, trace, charts, requested_symbol)
            summaries.append(summary)
            trace.append({"phase": "subgoal_done", "id": sg.get("id"), "summary": summary})

        answer = _synthesize(client, query, plan, summaries)
        trace.append({"phase": "synth"})
    finally:
        auto_ingest_var.reset(token)

    with connect() as conn:
        res = conn.execute(
            agent_runs.insert().values(
                started_at=datetime.utcnow(),
                user_query=query,
                final_answer=answer,
                tool_trace=json.dumps(trace, default=str),
                artifacts=json.dumps(charts, default=str),
            )
        )
        run_id = int(res.inserted_primary_key[0])

    return RunResult(run_id=run_id, answer=answer, plan=plan, trace=trace, charts=charts)
