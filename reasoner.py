#!/usr/bin/env python3
"""
LLM Reasoner — the intelligence layer over raw bursts.

The reasoner receives ALL bursts for a time period (day/week) and produces:
1. Final session boundaries (reasoned, not just gap-based)
2. Work type per session
3. Topic per session
4. Daily narrative summary

This is where the meta-layer magic lives: the LLM sees the full picture
and can handle interleaved conversations, parallel workflows, and
context shifts that simple gap detection would miss.
"""

import json
import re
import os
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from sessionizer import (
    CandidateSession, Session, Sessionizer,
    candidate_to_dict, candidates_to_llm_prompt,
    save_sessions
)
from burst import Burst, get_db_path


class Reasoner:
    """
    LLM-powered reasoning over raw keystroke bursts.

    The reasoner:
    1. Groups bursts into candidate sessions (lightweight pre-grouping)
    2. Feeds all candidates to the LLM with full context
    3. LLM decides final sessionization, work_type, and topic
    4. Produces a daily narrative summary
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "MiniMax-M2.7"):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.io/v1/text"
        self.model = model

    # -------------------------------------------------------------------------
    # Main entry points
    # -------------------------------------------------------------------------

    def reason_day(
        self,
        bursts: List[Burst],
        return_sessions: bool = True,
        return_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Full reasoning pipeline for a list of bursts.

        Returns dict with:
          - sessions: List[Session] (db-query-layer compatible)
          - daily_summary: str
          - candidate_count: int (how many candidates were fed to LLM)
        """
        if not bursts:
            return {"sessions": [], "daily_summary": "No activity recorded.", "candidate_count": 0}

        # Pre-group into candidates (lightweight, LLM has final say)
        sessionizer = Sessionizer()
        candidates = sessionizer.group_bursts(bursts)

        # Let LLM reason over them
        sessions = self._reason_sessions(candidates) if return_sessions else []
        summary = self._reason_daily_summary(candidates) if return_summary else ""

        return {
            "sessions": sessions,
            "daily_summary": summary,
            "candidate_count": len(candidates),
        }

    def reason_day_from_db(
        self,
        date_str: str,
        db_path: Path = None,
        return_sessions: bool = True,
        return_summary: bool = True,
    ) -> Dict[str, Any]:
        """Full reasoning from SQLite burst store."""
        from burst import load_bursts_for_date
        if db_path is None:
            db_path = get_db_path()
        bursts = load_bursts_for_date(date_str, db_path)
        return self.reason_day(bursts, return_sessions, return_summary)

    # -------------------------------------------------------------------------
    # Session reasoning
    # -------------------------------------------------------------------------

    def _reason_sessions(self, candidates: List[CandidateSession]) -> List[Session]:
        """
        Ask the LLM to analyze candidate sessions and produce final sessions.
        The LLM can: merge candidates, split them, reassign work_type/topic.
        """
        if not self.api_key:
            # Fallback: just use candidates directly with rule-based classification
            return self._rule_based_sessions(candidates)

        prompt = self._build_session_prompt(candidates)

        response = self._call(prompt)
        sessions = self._parse_session_response(response, candidates)
        return sessions

    def _build_session_prompt(self, candidates: List[CandidateSession]) -> str:
        """Build the LLM prompt for session reasoning."""
        candidate_blocks = candidates_to_llm_prompt(candidates)

        return f"""You are analyzing a developer's keystroke activity for one day.

Below are pre-grouped "candidate sessions" — bursts of typing grouped by window and time proximity.
YOUR JOB is to decide the FINAL session boundaries and labels.

IMPORTANT RULES:
1. MERGE candidates that belong to the same logical work session (same task/conversation)
2. SPLIT candidates that cover different tasks even if they share a window
3. Interleaved conversations (typing in two Codex agents, two email threads) = SEPARATE sessions
4. When in doubt, lean toward MORE sessions (finer granularity is better than lumping unrelated work)
5. For each session you produce, you MUST assign ALL fields below

OUTPUT FORMAT — respond with a JSON array of sessions:
[
  {{
    "session_id": 1,
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "app_name": "Codex" | "VS Code" | "Chrome" | "Outlook" | "Terminal" | "Discord" | etc,
    "window_title": "brief window context",
    "work_type": "debugging" | "writing" | "reading" | "communicating" | "planning",
    "topic": "specific task in ≤10 words",
    "chars": total_characters_typed,
    "duration_minutes": wall_clock_minutes
  }},
  ...
]

Candidate Sessions:
{candidate_blocks}

Today's Date: {candidates[0].date if candidates else "unknown"}

Respond ONLY with the JSON array. No markdown, no explanation."""

    def _parse_session_response(
        self, response: str, candidates: List[CandidateSession]
    ) -> List[Session]:
        """Parse LLM JSON response into Session objects."""
        # Try to extract JSON from response
        json_str = self._extract_json(response)
        if not json_str:
            # Fallback to rule-based
            return self._rule_based_sessions(candidates)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"[reasoner] JSON parse failed, using rule-based. Response:\n{response[:200]}")
            return self._rule_based_sessions(candidates)

        sessions = []
        for item in data:
            try:
                sessions.append(Session(
                    date=candidates[0].date if candidates else "unknown",
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    app_name=item.get("app_name", "Unknown"),
                    window_title=item.get("window_title", ""),
                    work_type=item.get("work_type", "writing"),
                    topic=item.get("topic", "general work")[:80],
                    chars=int(item.get("chars", 0)),
                    duration_minutes=int(item.get("duration_minutes", 1)),
                ))
            except (KeyError, ValueError) as e:
                print(f"[reasoner] Skipping malformed session item: {item} ({e})")
                continue

        return sessions

    def _rule_based_sessions(self, candidates: List[CandidateSession]) -> List[Session]:
        """
        Fallback when no LLM is available: convert candidates directly to sessions
        using rule-based work_type and topic extraction.
        """
        from sessionizer import CandidateSession as CS
        WORK_TYPE_KEYWORDS = {
            "debugging": ["fix", "bug", "debug", "error", "issue", "problem", "crash", "fail", "race condition", "exhaust", "timeout", "connection"],
            "writing": ["implement", "write", "add", "create", "new", "build", "design", "architect", "schema", "migration", "model", "endpoint", "webhook", "component"],
            "reading": ["review", "read", "look", "check", "investigate", "analyze", "explore", "understand", "logs", "traces"],
            "communicating": ["slack", "email", "team", "reply", "update", "customer", "coordinat", "discuss", "sync", "announce", "ping", "ask"],
            "planning": ["plan", "outline", "scope", "sprint", "priorit", "capacity", "retrospective", "goals"],
        }

        sessions = []
        for c in candidates:
            full_text = " ".join(b.chars for b in c.bursts)
            text_lower = full_text.lower()

            # Rule-based work type
            scores = {}
            for wt, kws in WORK_TYPE_KEYWORDS.items():
                scores[wt] = sum(1 for kw in kws if kw in text_lower)
            work_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "writing"

            # Rule-based topic
            if c.window_title:
                parts = c.window_title.split(" - ")
                if len(parts) > 1:
                    window_topic = parts[-1].strip()
                    if window_topic and window_topic not in ("Terminal", "Discord"):
                        topic = window_topic
                    else:
                        words = full_text.split()[:8]
                        topic = " ".join(words)[:80]
                else:
                    words = full_text.split()[:8]
                    topic = " ".join(words)[:80]
            else:
                words = full_text.split()[:8]
                topic = " ".join(words)[:80] if words else "general work"

            sessions.append(Session(
                date=c.date,
                start_time=c.start_time,
                end_time=c.end_time,
                app_name=c.app_name,
                window_title=c.window_title,
                work_type=work_type,
                topic=topic,
                chars=c.total_chars,
                duration_minutes=max(1, int(c.total_duration_seconds / 60)),
            ))

        return sessions

    # -------------------------------------------------------------------------
    # Daily summary reasoning
    # -------------------------------------------------------------------------

    def _reason_daily_summary(self, candidates: List[CandidateSession]) -> str:
        """Generate a paragraph summary of the day's work."""
        if not self.api_key:
            return self._rule_based_summary(candidates)

        session_dicts = [candidate_to_dict(c) for c in candidates]

        prompt = f"""You are summarizing a developer's workday from keystroke activity logs.

Based on these {len(candidates)} work sessions, write 3 paragraphs summarizing the day:
- What were the major topics worked on?
- What was accomplished across different projects?
- Any notable patterns, switches between tasks, or parallel work?

Be specific. Use the session details to ground your summary.

Sessions:
{json.dumps(session_dicts, indent=2)[:3000]}

Write your summary in a natural, narrative style."""

        return self._call(prompt).strip()

    def _rule_based_summary(self, candidates: List[CandidateSession]) -> str:
        """Fallback summary using rule-based heuristics."""
        if not candidates:
            return "No activity recorded."

        total_chars = sum(c.total_chars for c in candidates)
        total_duration = sum(c.total_duration_seconds for c in candidates)
        apps = {c.app_name for c in candidates}
        work_types = self._rule_based_sessions(candidates)
        wt_counts = {}
        for s in work_types:
            wt_counts[s.work_type] = wt_counts.get(s.work_type, 0) + 1

        top_wt = max(wt_counts, key=wt_counts.get) if wt_counts else "writing"
        top_app = max(apps, key=lambda a: sum(c.total_chars for c in candidates if c.app_name == a)) if apps else "Unknown"

        return (
            f"Day summary: {len(candidates)} sessions, "
            f"{total_chars} characters typed over {int(total_duration/60)} minutes. "
            f"Primary app: {top_app}. Dominant work type: {top_wt}."
        )

    # -------------------------------------------------------------------------
    # API call
    # -------------------------------------------------------------------------

    def _call(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Make a MiniMax API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chatcompletion_v2",
                headers=headers,
                json=payload,
                timeout=30,
            )
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            elif "base_resp" in data:
                return f"API Error: {data['base_resp'].get('status_msg', 'unknown')}"
        except Exception as e:
            return f"Error: {str(e)}"
        return ""

    def _extract_json(self, text: str) -> Optional[str]:
        """Try to extract a JSON array from LLM response text."""
        # Look for array notation
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        # Try object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return "[" + text[start:end+1] + "]"
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    """Load MiniMax API key from Hermes config."""
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "MINIMAX_API_KEY" in line:
                    parts = line.strip().split("MINIMAX_API_KEY=", 1)
                    if len(parts) > 1:
                        key = parts[1].strip()
                        # Remove surrounding quotes if present
                        key = key.strip('"').strip("'")
                        return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", key)
    return ""


if __name__ == "__main__":
    import sys

    db_path = get_db_path()
    api_key = load_api_key()

    print("=" * 60)
    print("KEYSTROKE REASONER — Daily reasoning from raw bursts")
    print("=" * 60)
    print(f"DB: {db_path}")
    print(f"API key: {'loaded' if api_key else 'not found'}")

    # Reason over sample data if available
    sample = Path(__file__).parent / "data" / "sample_conversation.json"
    if sample.exists():
        from burst import load_bursts_from_json, insert_bursts, init_db
        bursts = load_bursts_from_json(sample)
        init_db(db_path)
        insert_bursts(bursts, db_path)

        reasoner = Reasoner(api_key=api_key)
        date_str = bursts[0].timestamp[:10] if bursts else "2026-04-18"

        print(f"\nReasoning over {len(bursts)} bursts...")
        result = reasoner.reason_day(bursts)

        print(f"\n--- {len(result['sessions'])} Final Sessions ---")
        from sessionizer import print_sessions
        print_sessions(result["sessions"])

        print(f"\n--- Daily Summary ---")
        print(result["daily_summary"])

        # Save to sessions DB
        sessions_db = Path(__file__).parent / "data" / "sessions.db"
        save_sessions(result["sessions"], sessions_db)
        print(f"\nSaved sessions to {sessions_db}")
    else:
        print("\nNo sample data found.")
