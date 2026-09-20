"""SQLite-backed caches for header scans and star-profile measurements.

Both caches are keyed on (path, mtime, size) so that an unchanged archive is
never re-read, while an edited or replaced file is picked up automatically.
SQLite is used rather than Parquet because the dominant access pattern is
incremental upsert of a few hundred new rows into a table of hundreds of
thousands.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS headers (
    path   TEXT PRIMARY KEY,
    mtime  REAL NOT NULL,
    size   INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    path    TEXT NOT NULL,
    backend TEXT NOT NULL,
    params  TEXT NOT NULL,
    mtime   REAL NOT NULL,
    size    INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (path, backend, params)
);
CREATE INDEX IF NOT EXISTS profiles_path ON profiles(path);
"""


def _stat(path: str) -> tuple[float, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return float(st.st_mtime), int(st.st_size)


class FrameCache:
    """Persistent cache of normalised headers and measured star profiles."""

    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = os.fspath(path) if path else ":memory:"
        if self.path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ headers
    def get_header(self, path: str) -> dict[str, Any] | None:
        st = _stat(path)
        if st is None:
            return None
        row = self._conn.execute(
            "SELECT mtime, size, payload FROM headers WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        if abs(row[0] - st[0]) > 1e-6 or row[1] != st[1]:
            return None
        payload = json.loads(row[2])
        payload["path"] = path
        return payload

    def put_headers(self, rows: Iterable[dict[str, Any]]) -> int:
        records = []
        for row in rows:
            path = row.get("path")
            if not path:
                continue
            st = _stat(path)
            if st is None:
                continue
            payload = {k: _jsonable(v) for k, v in row.items() if k != "path"}
            records.append((path, st[0], st[1], json.dumps(payload)))
        if records:
            self._conn.executemany(
                "INSERT OR REPLACE INTO headers(path, mtime, size, payload) "
                "VALUES(?, ?, ?, ?)",
                records,
            )
            self._conn.commit()
        return len(records)

    # ----------------------------------------------------------------- profiles
    def get_profile(self, path: str, backend: str, params_key: str) -> dict[str, Any] | None:
        st = _stat(path)
        if st is None:
            return None
        row = self._conn.execute(
            "SELECT mtime, size, payload FROM profiles "
            "WHERE path = ? AND backend = ? AND params = ?",
            (path, backend, params_key),
        ).fetchone()
        if row is None:
            return None
        if abs(row[0] - st[0]) > 1e-6 or row[1] != st[1]:
            return None
        payload = json.loads(row[2])
        payload["path"] = path
        payload["backend"] = backend
        return payload

    def put_profiles(
        self, rows: Sequence[dict[str, Any]], params_key: str
    ) -> int:
        records = []
        for row in rows:
            path = row.get("path")
            backend = row.get("backend")
            if not path or not backend:
                continue
            st = _stat(path)
            if st is None:
                continue
            payload = {
                k: _jsonable(v) for k, v in row.items() if k not in ("path", "backend")
            }
            records.append((path, backend, params_key, st[0], st[1], json.dumps(payload)))
        if records:
            self._conn.executemany(
                "INSERT OR REPLACE INTO profiles(path, backend, params, mtime, size, payload) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                records,
            )
            self._conn.commit()
        return len(records)

    def stats(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT COUNT(*) FROM headers")
        n_head = int(cur.fetchone()[0])
        cur = self._conn.execute("SELECT COUNT(*) FROM profiles")
        n_prof = int(cur.fetchone()[0])
        return {"headers": n_head, "profiles": n_prof}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

    def __enter__(self) -> FrameCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    import datetime as _dt

    import numpy as np

    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    try:
        import pandas as pd

        if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    return str(value)
