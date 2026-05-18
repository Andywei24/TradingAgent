from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tradeagent.agent.deepseek_client import DeepseekClient
from tradeagent.agent.prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from tradeagent.agent.tools import REGISTRY, tool_schemas
from tradeagent.data.db import connect
from tradeagent.data.models import agent_runs

log = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS_PER_GOAL = 5


@dataclass
class RunResult:
    run_id: int
    answer: str
    plan: dict
    trace: list[dict] = field(default_factory=list)


def _dispatch_tool_calls(calls: list[Any], trace: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for call in calls:
        name = call.function.name
        raw_args = call.function.arguments or "{}"
        try:
            if name not in REGISTRY:
                result: dict = {"error": f"unknown tool {name}"}
            else:
                result = REGISTRY[name].call(raw_args)
        except Exception as e:  # surface to model
            log.exception("tool %s failed", name)
            result = {"error": f"{type(e).__name__}: {e}"}
        trace.append({"tool": name, "args": raw_args, "result_keys": list(result.keys()) if isinstance(result, dict) else None})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result, default=str),
            }
        )
    return messages


def _plan(client: DeepseekClient, query: str) -> dict:
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
    client: DeepseekClient,
    goal: str,
    tool_names: list[str],
    trace: list[dict],
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
        messages.extend(_dispatch_tool_calls(msg.tool_calls, trace))
    return "[executor stopped after max iterations]"


def _synthesize(
    client: DeepseekClient,
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


def run_chain(query: str, client: DeepseekClient | None = None) -> RunResult:
    client = client or DeepseekClient()
    trace: list[dict] = []

    plan = _plan(client, query)
    trace.append({"phase": "plan", "plan": plan})

    summaries: list[str] = []
    for sg in plan.get("subgoals", []):
        goal = sg.get("goal", "")
        tools = sg.get("tools", []) or list(REGISTRY)
        summary = _execute_subgoal(client, goal, tools, trace)
        summaries.append(summary)
        trace.append({"phase": "subgoal_done", "id": sg.get("id"), "summary": summary})

    answer = _synthesize(client, query, plan, summaries)
    trace.append({"phase": "synth"})

    with connect() as conn:
        res = conn.execute(
            agent_runs.insert().values(
                started_at=datetime.utcnow(),
                user_query=query,
                final_answer=answer,
                tool_trace=json.dumps(trace, default=str),
            )
        )
        run_id = int(res.inserted_primary_key[0])

    return RunResult(run_id=run_id, answer=answer, plan=plan, trace=trace)
