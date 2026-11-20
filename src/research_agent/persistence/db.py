"""SQLite persistence for application data.

LangGraph checkpoints (conversation state) live in a separate SQLite file managed by
`SqliteSaver`. This module stores everything else: threads, approvals, audit log,
per-node timings and LLM token usage.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    approval_type TEXT NOT NULL,
    status TEXT NOT NULL,            -- pending | approved | rejected
    thread_id TEXT,                  -- set for graph interrupts
    reason TEXT,
    payload TEXT NOT NULL,           -- JSON
    requested_by TEXT,
    reviewer TEXT,
    comment TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    thread_id TEXT,
    user_id TEXT,
    data TEXT
);
CREATE TABLE IF NOT EXISTS node_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    thread_id TEXT,
    turn_id TEXT,
    node TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_ms REAL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tool TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_node_runs_node ON node_runs(node);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        self._conn.close()

    # ---- threads ----
    def upsert_thread(self, thread_id: str, user_id: str, title: str | None = None) -> None:
        now = _now()
        self._exec(
            """INSERT INTO threads(thread_id, user_id, title, created_at, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(thread_id) DO UPDATE SET updated_at=excluded.updated_at""",
            (thread_id, user_id, (title or "")[:120], now, now),
        )

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM threads WHERE thread_id=?", (thread_id,))
        return rows[0] if rows else None

    def list_threads(self, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if user_id:
            return self._query("SELECT * FROM threads WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit))
        return self._query("SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?", (limit,))

    # ---- approvals ----
    def create_approval(
        self,
        approval_type: str,
        payload: dict[str, Any],
        reason: str,
        thread_id: str | None = None,
        requested_by: str = "anonymous",
        approval_id: str | None = None,
    ) -> str:
        approval_id = approval_id or f"apr_{uuid.uuid4().hex[:12]}"
        self._exec(
            """INSERT OR IGNORE INTO approvals(approval_id, approval_type, status, thread_id, reason, payload, requested_by, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (approval_id, approval_type, "pending", thread_id, reason, json.dumps(payload, default=str), requested_by, _now()),
        )
        return approval_id

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM approvals WHERE approval_id=?", (approval_id,))
        if not rows:
            return None
        row = rows[0]
        row["payload"] = json.loads(row["payload"])
        return row

    def pending_approval_for_thread(self, thread_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT approval_id FROM approvals WHERE thread_id=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        )
        return self.get_approval(rows[0]["approval_id"]) if rows else None

    def list_approvals(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._query("SELECT * FROM approvals WHERE status=? ORDER BY created_at DESC", (status,))
        else:
            rows = self._query("SELECT * FROM approvals ORDER BY created_at DESC")
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def decide_approval(self, approval_id: str, approved: bool, reviewer: str, comment: str = "") -> None:
        self._exec(
            "UPDATE approvals SET status=?, reviewer=?, comment=?, decided_at=? WHERE approval_id=? AND status='pending'",
            ("approved" if approved else "rejected", reviewer, comment, _now(), approval_id),
        )

    # ---- audit ----
    def audit(self, event: str, thread_id: str | None = None, user_id: str | None = None, **data: Any) -> None:
        self._exec(
            "INSERT INTO audit_log(ts, event, thread_id, user_id, data) VALUES (?,?,?,?,?)",
            (_now(), event, thread_id, user_id, json.dumps(data, default=str)),
        )

    def audit_events(self, thread_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if thread_id:
            rows = self._query("SELECT * FROM audit_log WHERE thread_id=? ORDER BY id DESC LIMIT ?", (thread_id, limit))
        else:
            rows = self._query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["data"] = json.loads(r["data"]) if r["data"] else {}
        return rows

    # ---- metrics ----
    def record_node(self, thread_id: str | None, turn_id: str | None, node: str, duration_ms: float, status: str, error: str | None = None) -> None:
        self._exec(
            "INSERT INTO node_runs(ts, thread_id, turn_id, node, duration_ms, status, error) VALUES (?,?,?,?,?,?,?)",
            (_now(), thread_id, turn_id, node, duration_ms, status, error),
        )

    def record_llm_call(self, purpose: str, model: str, input_tokens: int | None, output_tokens: int | None, duration_ms: float, status: str) -> None:
        self._exec(
            "INSERT INTO llm_calls(ts, purpose, model, input_tokens, output_tokens, duration_ms, status) VALUES (?,?,?,?,?,?,?)",
            (_now(), purpose, model, input_tokens, output_tokens, duration_ms, status),
        )

    def record_tool_call(self, tool: str, attempt: int, status: str, error: str | None = None) -> None:
        self._exec(
            "INSERT INTO tool_calls(ts, tool, attempt, status, error) VALUES (?,?,?,?,?)",
            (_now(), tool, attempt, status, error),
        )

    def metrics_summary(self) -> dict[str, Any]:
        nodes = self._query(
            """SELECT node, COUNT(*) AS runs, ROUND(AVG(duration_ms),1) AS avg_ms,
                      ROUND(MAX(duration_ms),1) AS max_ms,
                      SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
               FROM node_runs GROUP BY node ORDER BY runs DESC"""
        )
        llm = self._query(
            """SELECT purpose, model, COUNT(*) AS calls, SUM(COALESCE(input_tokens,0)) AS input_tokens,
                      SUM(COALESCE(output_tokens,0)) AS output_tokens, ROUND(AVG(duration_ms),1) AS avg_ms,
                      SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END) AS failures
               FROM llm_calls GROUP BY purpose, model ORDER BY calls DESC"""
        )
        tools = self._query(
            """SELECT tool, status, COUNT(*) AS n FROM tool_calls GROUP BY tool, status ORDER BY tool"""
        )
        approvals = self._query("SELECT approval_type, status, COUNT(*) AS n FROM approvals GROUP BY approval_type, status")
        turns = self._query("SELECT COUNT(DISTINCT turn_id) AS n FROM node_runs")[0]["n"]
        return {"turns": turns, "nodes": nodes, "llm": llm, "tools": tools, "approvals": approvals}
