#!/usr/bin/env python3
"""
Full pipeline test: burst → sessionizer → reasoner.

Tests the complete keystroke lifecycle reasoning pipeline using
our actual Discord conversation as sample data.
"""

import sys
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent))

from burst import (
    Burst, init_db, insert_bursts, load_bursts_from_json,
    get_db_path, print_bursts
)
from sessionizer import (
    Sessionizer, CandidateSession,
    candidates_to_llm_prompt, print_candidates, save_sessions
)
from reasoner import Reasoner, load_api_key


def main():
    print("=" * 60)
    print("KEYSTROKE PIPELINE — Full end-to-end test")
    print("=" * 60)

    # Load API key
    api_key = load_api_key()
    print(f"\nAPI key: {'loaded' if api_key else 'not found'}")

    # Paths
    data_dir = Path(__file__).parent / "data"
    sample_file = data_dir / "sample_conversation.json"
    bursts_db = data_dir / "bursts.db"
    sessions_db = data_dir / "sessions.db"

    # -------------------------------------------------------------------------
    # Step 1: Load sample bursts
    # -------------------------------------------------------------------------
    print("\n=== Step 1: Load sample bursts ===")
    if not sample_file.exists():
        print(f"ERROR: Sample file not found: {sample_file}")
        return

    bursts = load_bursts_from_json(sample_file)
    print(f"Loaded {len(bursts)} bursts from {sample_file}")
    print_bursts(bursts[:3])

    # -------------------------------------------------------------------------
    # Step 2: Store in SQLite
    # -------------------------------------------------------------------------
    print("\n=== Step 2: Store bursts in SQLite ===")
    init_db(bursts_db)
    count = insert_bursts(bursts, bursts_db)
    print(f"Inserted {count} bursts into {bursts_db}")

    # -------------------------------------------------------------------------
    # Step 3: Pre-group into candidate sessions
    # -------------------------------------------------------------------------
    print("\n=== Step 3: Group into candidate sessions ===")
    sessionizer = Sessionizer(max_gap_minutes=10.0)
    candidates = sessionizer.group_bursts(bursts)
    print(f"Created {len(candidates)} candidate sessions")
    print_candidates(candidates)

    # -------------------------------------------------------------------------
    # Step 4: LLM reasoning
    # -------------------------------------------------------------------------
    print("\n=== Step 4: LLM reasoning ===")
    if not api_key:
        print("No API key — using rule-based fallback")
    else:
        print("API key loaded — using LLM reasoning")

    reasoner = Reasoner(api_key=api_key)
    result = reasoner.reason_day(bursts)

    print(f"\n--- {len(result['sessions'])} Final Sessions ---")
    from sessionizer import print_sessions
    print_sessions(result["sessions"])

    print(f"\n--- Daily Summary ---")
    print(result["daily_summary"])

    # -------------------------------------------------------------------------
    # Step 5: Save reasoned sessions
    # -------------------------------------------------------------------------
    print("\n=== Step 5: Save reasoned sessions ===")
    save_sessions(result["sessions"], sessions_db)
    print(f"Saved {len(result['sessions'])} sessions to {sessions_db}")

    # -------------------------------------------------------------------------
    # Step 6: Show candidate→session mapping for transparency
    # -------------------------------------------------------------------------
    print("\n=== Candidate → Final Session mapping ===")
    print("(For debugging: shows how LLM merged/split candidate sessions)")
    for i, c in enumerate(candidates):
        text_preview = c.full_text[:50].replace("\n", " ")
        print(f"  Candidate {i+1}: {c.start_time}-{c.end_time} | "
              f"{c.app_name} | {c.window_title[:30]} | "
              f"{c.burst_count} bursts | '{text_preview}...'")

    print("\n✓ Pipeline test complete")


if __name__ == "__main__":
    main()
