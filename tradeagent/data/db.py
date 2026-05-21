from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection

from tradeagent.config import get_settings
from tradeagent.data.models import metadata


def _apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cur = dbapi_connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA mmap_size=268435456")  # 256 MiB
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    engine = create_engine(settings.db_url, future=True)
    if engine.url.get_backend_name() == "sqlite":
        event.listen(engine, "connect", _apply_sqlite_pragmas)
    return engine


def _migrate_sqlite(engine: Engine) -> None:
    """Lightweight additive migrations. create_all() never alters existing tables, so
    add columns introduced after a DB was first created."""
    if engine.url.get_backend_name() != "sqlite":
        return
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(agent_runs)")}
        if cols and "artifacts" not in cols:
            conn.exec_driver_sql("ALTER TABLE agent_runs ADD COLUMN artifacts TEXT")


def init_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    metadata.create_all(engine)
    _migrate_sqlite(engine)


@contextmanager
def connect() -> Iterator[Connection]:
    engine = get_engine()
    with engine.begin() as conn:
        yield conn
