#!/usr/bin/env python3
"""
Sessionizer — groups raw bursts into candidate sessions.

IMPORTANT: This module does NOT make hard sessionization decisions.
It pre-groups bursts by window_id and time proximity to help the LLM reasoner,
but the LLM has the final say on session boundaries.

Architecture:
  Raw bursts (burst.py) → sessionizer.py pre-groups by window+time
                          → reasoner.py LLM decides final sessions
                          → sessions stored (db-query-layer compatible)
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from burst import Burst, get_db_path


@dataclass
class CandidateSession:
    """
    A pre-grouped session candidate — raw bursts grouped by window_id
    and time proximity. The LLM reasoner takes these and decides:
    - Should these be one session or split?
    - What work_type applies?
    - What is the topic?
    """
    date: str
    start_time: str
    end_time: str
    window_id: str
    window_title: str
    app_name: str
    app_path: str
    bursts: List[Burst]          # The raw bursts in this candidate
    total_chars: int             # Sum of all burst char_counts
    total_duration_seconds: int  # Wall-clock time from first to last burst
    burst_count: int             # Number of bursts
    sources: List[str]            # All source types used
    focus_changes: int           # How many times focus changed within this session

    @property
    def full_text(self) -> str:
        """Concatenated text of all bursts in this candidate."""
        return " ".join(b.chars for b in self.bursts)

    @property
    def burst_timestamps(self) -> List[str]:
        """Timestamps of each burst for timing analysis."""
        return [b.timestamp for b in self.bursts]


class Sessionizer:
    """
    Groups raw bursts into candidate sessions for LLM reasoning.

    The grouping strategy is intentionally lightweight and over-Merges:
    - Groups bursts that share a window_id AND are within a reasonable time window
    - errs on the side of grouping more, letting the LLM split if needed
    """

    # If bursts in the same window are more than this far apart, put them
    # in separate candidate sessions (but the LLM can still re-merge)
    MAX_BURST_GAP_MINUTES = 10.0

    def __init__(self, max_gap_minutes: float = 10.0):
        self.max_gap = timedelta(minutes=max_gap_minutes)

    def group_bursts(self, bursts: List[Burst]) -> List[CandidateSession]:
        """
        Group raw bursts into candidate sessions.
        Groups by window_id, splitting when gap exceeds max_gap_minutes.
        """
        if not bursts:
            return []

        # Sort by timestamp
        sorted_bursts = sorted(bursts, key=lambda b: b.timestamp)

        candidates = []
        current_group: List[Burst] = []
        current_window_id = None

        for burst in sorted_bursts:
            if current_window_id is None:
                # First burst — start first group
                current_group = [burst]
                current_window_id = burst.window_id
            elif burst.window_id != current_window_id:
                # Window changed — close current group, start new one
                if current_group:
                    candidates.append(self._build_candidate(current_group))
                current_group = [burst]
                current_window_id = burst.window_id
            else:
                # Same window — check gap
                last_ts = datetime.fromisoformat(current_group[-1].timestamp)
                curr_ts = datetime.fromisoformat(burst.timestamp)
                gap = curr_ts - last_ts

                if gap > self.max_gap:
                    # Gap too large — split candidates
                    if current_group:
                        candidates.append(self._build_candidate(current_group))
                    current_group = [burst]
                else:
                    # Within gap — keep grouping
                    current_group.append(burst)

        # Don't forget the last group
        if current_group:
            candidates.append(self._build_candidate(current_group))

        return candidates

    def _build_candidate(self, bursts: List[Burst]) -> CandidateSession:
        """Build a CandidateSession from a list of bursts in the same window."""
        first_ts = datetime.fromisoformat(bursts[0].timestamp)
        last_ts = datetime.fromisoformat(bursts[-1].timestamp)

        total_chars = sum(b.char_count for b in bursts)
        total_duration = int((last_ts - first_ts).total_seconds())
        sources = list({b.source for b in bursts})

        # Count focus changes
        focus_changes = sum(
            1 for i in range(1, len(bursts))
            if bursts[i].focus_active != bursts[i-1].focus_active
        )

        return CandidateSession(
            date=first_ts.strftime("%Y-%m-%d"),
            start_time=first_ts.strftime("%H:%M:%S"),
            end_time=last_ts.strftime("%H:%M:%S"),
            window_id=bursts[0].window_id,
            window_title=bursts[0].window_title,
            app_name=bursts[0].app_name,
            app_path=bursts[0].app_path,
            bursts=bursts,
            total_chars=total_chars,
            total_duration_seconds=total_duration,
            burst_count=len(bursts),
            sources=sources,
            focus_changes=focus_changes,
        )

    def group_bursts_from_db(self, date_str: str, db_path: Path = None) -> List[CandidateSession]:
        """Load bursts for a date and group them into candidates."""
        from burst import load_bursts_for_date
        bursts = load_bursts_for_date(date_str, db_path)
        return self.group_bursts(bursts)

    def group_bursts_in_range(
        self, start_date: str, end_date: str, db_path: Path = None
    ) -> List[CandidateSession]:
        """Load bursts in a range and group them."""
        from burst import load_bursts_in_range
        bursts = load_bursts_in_range(start_date, end_date, db_path)
        return self.group_bursts(bursts)


# ---------------------------------------------------------------------------
# Formatting for LLM consumption
# ---------------------------------------------------------------------------

def candidate_to_dict(c: CandidateSession) -> dict:
    """Serialize a CandidateSession for the LLM reasoner."""
    return {
        "date": c.date,
        "start_time": c.start_time,
        "end_time": c.end_time,
        "window_id": c.window_id,
        "window_title": c.window_title,
        "app_name": c.app_name,
        "app_path": c.app_path,
        "total_chars": c.total_chars,
        "total_duration_seconds": c.total_duration_seconds,
        "burst_count": c.burst_count,
        "sources": c.sources,
        "focus_changes": c.focus_changes,
        # Full concatenated text for LLM analysis
        "full_text": " ".join(b.chars for b in c.bursts),
        # Timestamps of each burst (for timing analysis)
        "burst_timestamps": [b.timestamp for b in c.bursts],
    }


def candidates_to_llm_prompt(candidates: List[CandidateSession]) -> str:
    """
    Format candidate sessions into a prompt for the LLM reasoner.
    The LLM will use this to decide final session boundaries,
    work types, and topics.
    """
    blocks = []
    for i, c in enumerate(candidates):
        block = f"""Candidate Session {i+1}:
  Time: {c.start_time} - {c.end_time} ({c.date})
  Window: {c.window_title}
  App: {c.app_name}
  Window ID: {c.window_id}
  Characters: {c.total_chars} (in {c.burst_count} bursts over {c.total_duration_seconds}s)
  Sources: {', '.join(c.sources)}
  Focus changes: {c.focus_changes}
  Text preview: {c.full_text[:300]}{"..." if len(c.full_text) > 300 else ""}
  Burst timestamps: {c.burst_timestamps}
"""
        blocks.append(block)

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Output: sessions in db-query-layer schema
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    Final reasoned session — matches db-query-layer schema exactly.
    Produced by reasoner.py after LLM reasoning over CandidateSessions.
    """
    date: str
    start_time: str
    end_time: str
    app_name: str
    window_title: str
    work_type: str
    topic: str
    chars: int
    duration_minutes: int


def save_sessions(sessions: List[Session], db_path: Path) -> None:
    """Save reasoned sessions to SQLite (db-query-layer schema)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            start_time TEXT,
            end_time TEXT,
            app_name TEXT,
            window_title TEXT,
            work_type TEXT,
            topic TEXT,
            chars INTEGER,
            duration_minutes INTEGER
        )
    """)

    for s in sessions:
        cursor.execute("""
            INSERT INTO sessions (date, start_time, end_time, app_name, window_title,
                                   work_type, topic, chars, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (s.date, s.start_time, s.end_time, s.app_name, s.window_title,
              s.work_type, s.topic, s.chars, s.duration_minutes))

    conn.commit()
    conn.close()


def load_sessions(db_path: Path) -> List[Session]:
    """Load reasoned sessions from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions ORDER BY date, start_time")
    rows = cursor.fetchall()
    conn.close()

    return [Session(
        date=r["date"],
        start_time=r["start_time"],
        end_time=r["end_time"],
        app_name=r["app_name"],
        window_title=r["window_title"],
        work_type=r["work_type"],
        topic=r["topic"],
        chars=r["chars"],
        duration_minutes=r["duration_minutes"],
    ) for r in rows]


def print_candidates(candidates: List[CandidateSession]) -> None:
    """Pretty-print candidate sessions."""
    current_date = None
    for c in candidates:
        if c.date != current_date:
            print(f"\n=== {c.date} ===")
            current_date = c.date
        print(f"  {c.start_time}-{c.end_time} | {c.app_name:<12} | "
              f"{c.window_title[:40]:<40} | {c.burst_count} bursts | "
              f"{c.total_chars} chars")


def print_sessions(sessions: List[Session]) -> None:
    """Pretty-print reasoned sessions."""
    current_date = None
    for s in sessions:
        if s.date != current_date:
            print(f"\n=== {s.date} ===")
            current_date = s.date
        print(f"  {s.start_time}-{s.end_time} | {s.app_name:<12} | "
              f"{s.work_type:<15} | {s.duration_minutes:>3}min | {s.topic[:60]}")


if __name__ == "__main__":
    from burst import get_db_path, init_db

    db_path = get_db_path()
    print(f"DB: {db_path}")

    # Group sample data if it exists
    sample = Path(__file__).parent / "data" / "sample_conversation.json"
    if sample.exists():
        from burst import load_bursts_from_json, insert_bursts
        bursts = load_bursts_from_json(sample)
        init_db(db_path)
        insert_bursts(bursts, db_path)

        sessionizer = Sessionizer()
        candidates = sessionizer.group_bursts(bursts)
        print(f"\n{len(candidates)} candidate sessions:")
        print_candidates(candidates)
