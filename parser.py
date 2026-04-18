#!/usr/bin/env python3
"""
Keystroke parser: converts raw keystroke bursts into time-boxed sessions.
Uses activity-gap detection for sessionization, with optional LLM for work_type + topic.
"""

import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Work type keywords for rule-based classification
WORK_TYPE_KEYWORDS = {
    "debugging": ["fix", "bug", "debug", "error", "issue", "problem", "crash", "fail", "race condition", "exhaust", "timeout", "connection"],
    "writing": ["implement", "write", "add", "create", "new", "build", "design", "architect", "schema", "migration", "model", "endpoint", "webhook", "component"],
    "reading": ["review", "read", "look", "check", "investigate", "analyze", "explore", "understand", "logs", "traces"],
    "communicating": ["slack", "email", "team", "reply", "update", "customer", "coordinat", "discuss", "sync", "announce", "ping", "ask"],
    "planning": ["plan", "outline", "scope", "sprint", "priorit", "capacity", "retrospective", "goals"],
}


@dataclass
class Session:
    """A time-boxed work session reconstructed from keystroke bursts."""
    date: str
    start_time: str
    end_time: str
    app_name: str
    window_title: str
    work_type: str
    topic: str
    chars: int
    duration_minutes: int


class KeystrokeParser:
    """Parses raw keystroke bursts into sessions."""

    def __init__(
        self,
        gap_threshold_minutes: float = 5.0,
        app_split_threshold_minutes: float = 2.0,
        use_llm: bool = False,
        llm_summarizer: Optional[Any] = None
    ):
        self.gap_threshold = timedelta(minutes=gap_threshold_minutes)
        self.app_split_threshold = timedelta(minutes=app_split_threshold_minutes)
        self.use_llm = use_llm
        self.llm = llm_summarizer

    def parse_file(self, filepath: Path) -> List[Session]:
        """Parse a JSON file of keystroke bursts into sessions."""
        with open(filepath) as f:
            bursts = json.load(f)
        return self.parse_bursts(bursts)

    def parse_bursts(self, bursts: List[Dict]) -> List[Session]:
        """Group bursts into sessions and classify each."""
        # Sort by timestamp
        sorted_bursts = sorted(bursts, key=lambda b: b["timestamp"])

        sessions = []
        current_session = None

        for burst in sorted_bursts:
            ts = datetime.fromisoformat(burst["timestamp"])
            app = burst.get("app_name", "Unknown")
            window = burst.get("window_title", "")
            chars = burst.get("chars", "")
            text = chars if isinstance(chars, str) else ""
            char_count = len(text)

            if current_session is None:
                # Start first session
                current_session = {
                    "start": ts,
                    "end": ts,
                    "app": app,
                    "window": window,
                    "texts": [text],
                    "char_counts": [char_count],
                }
            else:
                gap = ts - current_session["end"]

                if gap > self.gap_threshold:
                    # Close current session, start new one
                    sessions.append(self._build_session(current_session))
                    current_session = {
                        "start": ts,
                        "end": ts,
                        "app": app,
                        "window": window,
                        "texts": [text],
                        "char_counts": [char_count],
                    }
                elif gap > self.app_split_threshold and app != current_session["app"]:
                    # Same session but different app after a pause — split
                    sessions.append(self._build_session(current_session))
                    current_session = {
                        "start": ts,
                        "end": ts,
                        "app": app,
                        "window": window,
                        "texts": [text],
                        "char_counts": [char_count],
                    }
                else:
                    # Merge into current session
                    current_session["end"] = ts
                    current_session["texts"].append(text)
                    current_session["char_counts"].append(char_count)
                    if app != current_session["app"]:
                        current_session["app"] = app
                    if window:
                        current_session["window"] = window

        # Don't forget the last session
        if current_session is not None:
            sessions.append(self._build_session(current_session))

        return sessions

    def _build_session(self, raw: Dict) -> Session:
        """Convert raw session dict to Session object with classification."""
        total_chars = sum(raw["char_counts"])
        full_text = " ".join(raw["texts"])
        duration = int((raw["end"] - raw["start"]).total_seconds() / 60)

        # Determine work type
        if self.use_llm and self.llm:
            work_type = self.llm.classify_work_type(full_text)
        else:
            work_type = self._rule_based_work_type(full_text)

        # Extract topic
        if self.use_llm and self.llm:
            topic = self.llm.extract_topic(full_text)
        else:
            topic = self._rule_based_topic(full_text, raw["window"])

        return Session(
            date=raw["start"].strftime("%Y-%m-%d"),
            start_time=raw["start"].strftime("%H:%M:%S"),
            end_time=raw["end"].strftime("%H:%M:%S"),
            app_name=raw["app"],
            window_title=raw["window"],
            work_type=work_type,
            topic=topic,
            chars=total_chars,
            duration_minutes=max(duration, 1),
        )

    def _rule_based_work_type(self, text: str) -> str:
        """Classify work type from text using keyword matching."""
        text_lower = text.lower()
        scores = {}

        for work_type, keywords in WORK_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[work_type] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "writing"  # default

    def _rule_based_topic(self, text: str, window: str) -> str:
        """Extract a brief topic from text and window title."""
        # Try to pull a filename or meaningful identifier from window
        if window:
            # Extract last path component or anything after " - "
            parts = window.split(" - ")
            if len(parts) > 1:
                window_topic = parts[-1].strip()
                if window_topic and window_topic not in ("Terminal", "Discord"):
                    return window_topic

        # Fall back to first meaningful phrase in text
        words = text.split()[:8]
        if words:
            topic = " ".join(words)
            return topic[:80]  # cap at 80 chars
        return "general work"


def save_to_db(sessions: List[Session], db_path: Path):
    """Save parsed sessions to SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
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
            INSERT INTO sessions (date, start_time, end_time, app_name, window_title, work_type, topic, chars, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (s.date, s.start_time, s.end_time, s.app_name, s.window_title, s.work_type, s.topic, s.chars, s.duration_minutes))

    conn.commit()
    conn.close()


def load_from_db(db_path: Path) -> List[Session]:
    """Load sessions from SQLite."""
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


def print_sessions(sessions: List[Session]):
    """Pretty-print sessions."""
    current_date = None
    for s in sessions:
        if s.date != current_date:
            print(f"\n=== {s.date} ===")
            current_date = s.date
        print(f"  {s.start_time}-{s.end_time} | {s.app_name:<12} | {s.work_type:<15} | {s.duration_minutes:>3}min | {s.topic[:60]}")
