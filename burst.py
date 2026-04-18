#!/usr/bin/env python3
"""
Raw burst storage — the immutable foundation of the keystroke lifecycle.
Bursts are captured as atomic typing events and stored without sessionization decisions.
Reasoning about sessions happens later, in reasoner.py.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Burst:
    """
    An atomic typing event: characters typed within a focused input window,
    captured at a moment in time. No sessionization — just the raw facts.

    A "burst" is what you get when someone finishes a unit of typing and
    moves to another window / pauses / gets interrupted. It's the atom
    of the capture layer.
    """
    timestamp: str          # ISO 8601: "2026-04-18T09:15:23"
    window_id: str          # Stable OS handle or hash identifying the window instance
    window_title: str        # Human-readable: "Codex - auth/middleware.py", "Gmail - Inbox"
    app_name: str           # Process name: "Codex", "Chrome", "Outlook"
    app_path: str            # Full path to executable
    chars: str               # The actual characters typed in this burst
    char_count: int         # len(chars) — stored for fast aggregation
    source: str              # "manual" | "voice" | "clipboard" — how the text arrived
    focus_active: bool      # Was this window focused when the burst was captured?

    def __post_init__(self):
        if self.char_count == 0 and self.chars:
            self.char_count = len(self.chars)


def burst_from_dict(d: dict) -> Burst:
    """Parse a dict (e.g. from JSON) into a Burst."""
    return Burst(
        timestamp=d["timestamp"],
        window_id=d.get("window_id", ""),
        window_title=d.get("window_title", ""),
        app_name=d.get("app_name", ""),
        app_path=d.get("app_path", ""),
        chars=d.get("chars", ""),
        char_count=d.get("char_count", len(d.get("chars", ""))),
        source=d.get("source", "manual"),
        focus_active=d.get("focus_active", True),
    )


# ---------------------------------------------------------------------------
# SQLite Storage
# ---------------------------------------------------------------------------

BURST_SCHEMA = """
CREATE TABLE IF NOT EXISTS bursts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,          -- ISO 8601
    window_id    TEXT    NOT NULL DEFAULT '', -- stable window identifier
    window_title TEXT    NOT NULL DEFAULT '',
    app_name     TEXT    NOT NULL DEFAULT '',
    app_path     TEXT    NOT NULL DEFAULT '',
    chars        TEXT    NOT NULL DEFAULT '',
    char_count   INTEGER NOT NULL DEFAULT 0,
    source       TEXT    NOT NULL DEFAULT 'manual',
    focus_active INTEGER NOT NULL DEFAULT 1,
    captured_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for time-range and window queries
CREATE INDEX IF NOT EXISTS idx_bursts_timestamp ON bursts(timestamp);
CREATE INDEX IF NOT EXISTS idx_bursts_window_id ON bursts(window_id);
CREATE INDEX IF NOT EXISTS idx_bursts_app_name ON bursts(app_name);
CREATE INDEX IF NOT EXISTS idx_bursts_date ON bursts(date(timestamp));
"""


def get_db_path(data_dir: Path = None) -> Path:
    """Path to the bursts SQLite database."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "bursts.db"


def init_db(db_path: Path = None) -> None:
    """Create the bursts table if it doesn't exist."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.executescript(BURST_SCHEMA)
    conn.commit()
    conn.close()


def insert_burst(burst: Burst, db_path: Path = None) -> int:
    """Insert a single burst. Returns the row id."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bursts (timestamp, window_id, window_title, app_name, app_path,
                            chars, char_count, source, focus_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (burst.timestamp, burst.window_id, burst.window_title, burst.app_name,
          burst.app_path, burst.chars, burst.char_count, burst.source,
          1 if burst.focus_active else 0))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def insert_bursts(bursts: List[Burst], db_path: Path = None) -> int:
    """Insert multiple bursts efficiently. Returns count inserted."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO bursts (timestamp, window_id, window_title, app_name, app_path,
                            chars, char_count, source, focus_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [(b.timestamp, b.window_id, b.window_title, b.app_name, b.app_path,
           b.chars, b.char_count, b.source, 1 if b.focus_active else 0)
          for b in bursts])
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def load_bursts_for_date(date_str: str, db_path: Path = None) -> List[Burst]:
    """
    Load all bursts for a given date (YYYY-MM-DD).
    Ordered by timestamp ascending.
    """
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM bursts
        WHERE date(timestamp) = ?
        ORDER BY timestamp ASC
    """, (date_str,))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_burst(r) for r in rows]


def load_bursts_in_range(start_date: str, end_date: str, db_path: Path = None) -> List[Burst]:
    """Load all bursts in a date range (inclusive)."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM bursts
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
    """, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_burst(r) for r in rows]


def load_bursts_by_window(window_id: str, db_path: Path = None) -> List[Burst]:
    """Load all bursts for a specific window (by window_id)."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM bursts
        WHERE window_id = ?
        ORDER BY timestamp ASC
    """, (window_id,))
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_burst(r) for r in rows]


def _row_to_burst(r: sqlite3.Row) -> Burst:
    return Burst(
        timestamp=r["timestamp"],
        window_id=r["window_id"],
        window_title=r["window_title"],
        app_name=r["app_name"],
        app_path=r["app_path"],
        chars=r["chars"],
        char_count=r["char_count"],
        source=r["source"],
        focus_active=bool(r["focus_active"]),
    )


# ---------------------------------------------------------------------------
# JSON import/export
# ---------------------------------------------------------------------------

def load_bursts_from_json(filepath: Path) -> List[Burst]:
    """Load bursts from a JSON file (our sample conversation format)."""
    with open(filepath) as f:
        data = json.load(f)
    bursts = []
    for d in data:
        b = burst_from_dict(d)
        # window_id: use window_title as a stand-in if not present
        if not b.window_id:
            b.window_id = b.window_title
        bursts.append(b)
    return bursts


def save_bursts_to_json(bursts: List[Burst], filepath: Path) -> None:
    """Export bursts to a JSON file."""
    with open(filepath, "w") as f:
        json.dump([asdict(b) for b in bursts], f, indent=2)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def print_bursts(bursts: List[Burst]) -> None:
    """Pretty-print a list of bursts."""
    for b in bursts:
        preview = b.chars[:60].replace("\n", " ") + ("..." if len(b.chars) > 60 else "")
        print(f"  {b.timestamp} | {b.app_name:<12} | {b.window_title[:40]:<40} | "
              f"{b.source:<8} | {b.char_count:>5} chars | {preview}")


if __name__ == "__main__":
    import sys
    # Demo: init DB and load sample data if provided
    db_path = Path(__file__).parent / "data" / "bursts.db"
    init_db(db_path)

    sample = Path(__file__).parent / "data" / "sample_conversation.json"
    if sample.exists():
        print(f"Loading sample data from {sample}")
        bursts = load_bursts_from_json(sample)
        count = insert_bursts(bursts, db_path)
        print(f"Inserted {count} bursts into {db_path}")
        print_bursts(bursts[:5])
    else:
        print("No sample file found, DB initialized at:", db_path)
