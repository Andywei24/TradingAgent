from __future__ import annotations

import json
import os
from pathlib import Path

from sqlalchemy import select

from tradeagent.config import get_settings
from tradeagent.data.db import connect
from tradeagent.data.models import agent_runs


def _collect_chart_paths(row) -> list[str]:
    """Charts come from the artifacts column; fall back to scanning the trace."""
    artifacts = row.get("artifacts") if hasattr(row, "get") else row["artifacts"]
    if artifacts:
        try:
            paths = json.loads(artifacts)
            if isinstance(paths, list) and paths:
                return [str(p) for p in paths]
        except json.JSONDecodeError:
            pass
    trace = json.loads(row["tool_trace"] or "[]")
    return [e["chart_path"] for e in trace if isinstance(e, dict) and e.get("chart_path")]


def _rel_to_reports(chart_path: str, reports_dir: Path) -> str:
    """Markdown-friendly path relative to the report's directory (forward slashes)."""
    try:
        return Path(os.path.relpath(chart_path, reports_dir)).as_posix()
    except ValueError:  # e.g. different drive on Windows
        return Path(chart_path).as_posix()


def build_report(run_id: int) -> Path:
    with connect() as conn:
        row = conn.execute(select(agent_runs).where(agent_runs.c.id == run_id)).mappings().one_or_none()
    if row is None:
        raise ValueError(f"no agent_run with id={run_id}")

    out_dir = Path(get_settings().data_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = json.loads(row["tool_trace"] or "[]")
    body = [
        f"# Agent run #{run_id}",
        f"_Started: {row['started_at']}_\n",
        "## Question",
        f"> {row['user_query']}\n",
        "## Answer",
        row["final_answer"] or "",
        "",
    ]

    charts = _collect_chart_paths(row)
    if charts:
        body.append("## Charts")
        for cp in charts:
            rel = _rel_to_reports(cp, out_dir)
            body.append(f"![{Path(cp).stem}]({rel})")
        body.append("")

    body += [
        "## Trace",
        "```json",
        json.dumps(trace, indent=2, default=str),
        "```",
    ]

    path = out_dir / f"run_{run_id}.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path
