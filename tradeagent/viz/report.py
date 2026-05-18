from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select

from tradeagent.config import get_settings
from tradeagent.data.db import connect
from tradeagent.data.models import agent_runs


def build_report(run_id: int) -> Path:
    with connect() as conn:
        row = conn.execute(select(agent_runs).where(agent_runs.c.id == run_id)).mappings().one_or_none()
    if row is None:
        raise ValueError(f"no agent_run with id={run_id}")

    trace = json.loads(row["tool_trace"] or "[]")
    body = [
        f"# Agent run #{run_id}",
        f"_Started: {row['started_at']}_\n",
        "## Question",
        f"> {row['user_query']}\n",
        "## Answer",
        row["final_answer"] or "",
        "",
        "## Trace",
        "```json",
        json.dumps(trace, indent=2, default=str),
        "```",
    ]

    out_dir = Path(get_settings().data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{run_id}.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path
