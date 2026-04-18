#!/usr/bin/env python3
"""
Test the keystroke parser using our actual Discord conversation as raw input.
"""

import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from parser import KeystrokeParser, save_to_db, print_sessions
from summarizer import LLMSummarizer, load_api_key


def main():
    data_file = Path(__file__).parent / "data" / "sample_conversation.json"
    db_file = Path(__file__).parent / "data" / "parsed.db"

    print("=" * 60)
    print("KEYSTROKE PARSER — Testing on Discord conversation")
    print("=" * 60)

    # Load API key from Hermes config
    api_key = load_api_key()
    print(f"\nAPI key: {'loaded' if api_key else 'not found'}")

    # --- Rule-based parsing (always works, no API needed) ---
    print("\n=== Rule-based parsing (no LLM) ===")
    parser = KeystrokeParser(gap_threshold_minutes=5.0, use_llm=False)
    sessions = parser.parse_file(data_file)
    print(f"Parsed {len(sessions)} sessions from 29 message bursts\n")

    # Print sessions
    print_sessions(sessions)

    # Work type breakdown
    from collections import Counter
    work_types = Counter(s.work_type for s in sessions)
    print("\n--- Work type breakdown ---")
    for wt, count in work_types.most_common():
        print(f"  {wt}: {count}")

    # Save to SQLite
    save_to_db(sessions, db_file)
    print(f"\nSaved to {db_file}")

    # --- LLM summary (one call for the whole day) ---
    if api_key:
        print("\n=== LLM daily summary ===")
        llm = LLMSummarizer(api_key=api_key)

        # Build session dicts for LLM (use rule-based work_type since it's fast)
        session_dicts = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "app_name": s.app_name,
                "work_type": s.work_type,
                "topic": s.topic,
                "chars": s.chars,
                "duration_minutes": s.duration_minutes,
            }
            for s in sessions
        ]

        summary = llm.summarize_day(session_dicts)
        print(summary)
    else:
        print("\n=== LLM summary skipped (no API key) ===")


if __name__ == "__main__":
    main()
