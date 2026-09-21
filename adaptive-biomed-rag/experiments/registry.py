"""Run store: one SQLite row per run + one per-question parquet per run.

Runs are ids like R001. The per-question table (data/runs/R001.parquet) is what
paired comparisons are computed from, so every run is re-analysable later.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pandas as pd

from common import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  name         TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  status       TEXT NOT NULL,            -- queued | running | done | failed
  question_set TEXT NOT NULL,
  corpus       TEXT NOT NULL,
  n_questions  INTEGER NOT NULL,
  done         INTEGER NOT NULL DEFAULT 0,
  config       TEXT NOT NULL,            -- RetrievalConfig JSON
  metrics      TEXT,                     -- [{key,label,value,ci_low,ci_high,unit}]
  slices       TEXT,                     -- [{dimension,value,n,metrics}]
  duration_s   REAL,
  error        TEXT NOT NULL DEFAULT ''
)
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(paths.RUNS_DB)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def run_id(seq: int) -> str:
    return f"R{seq:03d}"


def _seq(rid: str) -> int:
    return int(rid.lstrip("R"))


def create(name: str, config: dict, question_set: str, corpus: str, n: int) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(_connect()) as con, con:
        cur = con.execute(
            "INSERT INTO runs (name, created_at, status, question_set, corpus, n_questions, config)"
            " VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (name, now, question_set, corpus, n, json.dumps(config)))
        seq = cur.lastrowid
        if not name:
            con.execute("UPDATE runs SET name = ? WHERE seq = ?", (run_id(seq), seq))
    return run_id(seq)


def start(rid: str) -> None:
    with closing(_connect()) as con, con:
        con.execute("UPDATE runs SET status = 'running' WHERE seq = ?", (_seq(rid),))


def progress(rid: str, done: int) -> None:
    with closing(_connect()) as con, con:
        con.execute("UPDATE runs SET done = ? WHERE seq = ?", (done, _seq(rid)))


def finish(rid: str, metrics: list[dict], slices: list[dict], per_query: pd.DataFrame,
           duration_s: float) -> None:
    per_query.to_parquet(paths.RUNS_DIR / f"{rid}.parquet", index=False)
    with closing(_connect()) as con, con:
        con.execute(
            "UPDATE runs SET status = 'done', done = n_questions, metrics = ?, slices = ?,"
            " duration_s = ? WHERE seq = ?",
            (json.dumps(metrics), json.dumps(slices), round(duration_s, 1), _seq(rid)))


def fail(rid: str, error: str) -> None:
    with closing(_connect()) as con, con:
        con.execute("UPDATE runs SET status = 'failed', error = ? WHERE seq = ?",
                    (error[:2000], _seq(rid)))


def fail_interrupted() -> None:
    """Runs still queued or running at startup died with the previous server process."""
    with closing(_connect()) as con, con:
        con.execute("UPDATE runs SET status = 'failed', error = 'interrupted: server restarted'"
                    " WHERE status IN ('queued', 'running')")


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["id"] = run_id(d.pop("seq"))
    d["config"] = json.loads(d["config"])
    d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else []
    d["slices"] = json.loads(d["slices"]) if d["slices"] else []
    return d


def get(rid: str) -> dict | None:
    with closing(_connect()) as con:
        r = con.execute("SELECT * FROM runs WHERE seq = ?", (_seq(rid),)).fetchone()
    return _row(r) if r else None


def all_runs() -> list[dict]:
    with closing(_connect()) as con:
        rows = con.execute("SELECT * FROM runs ORDER BY seq DESC").fetchall()
    return [_row(r) for r in rows]


def per_query(rid: str) -> pd.DataFrame:
    return pd.read_parquet(paths.RUNS_DIR / f"{rid}.parquet")
