#!/usr/bin/env python3
"""
Time-boxed chunking for the reasoning pipeline.

Breaks a day's bursts into overlapping 1-hour windows so the LLM never
sees too many bursts at once. A final "fix" pass resolves inconsistencies
caused by splitting at chunk boundaries.

Chunk structure:
  - 1 hour primary window (e.g., 09:00-10:00)
  - 15 minutes overlap on each side (e.g., 08:45-09:00, 10:00-10:15)
  - LLM only produces sessions for the primary window portion
  - Overlap portions are used for context only

Final pass: receives all chunk sessions → deduplicates + resolves
conflicts caused by tasks split across chunk boundaries.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from burst import Burst
from sessionizer import CandidateSession, Sessionizer
from reasoner import Reasoner


@dataclass
class Chunk:
    """A time-boxed chunk with overlap for context."""
    chunk_id: int
    primary_start: datetime
    primary_end: datetime       # exclusive end of primary window
    overlap_start: datetime     # start of full chunk (including overlap)
    overlap_end: datetime       # end of full chunk (including overlap)

    def __repr__(self):
        return (f"Chunk {self.chunk_id}: "
                f"primary {self.primary_start.strftime('%H:%M')}-"
                f"{self.primary_end.strftime('%H:%M')} | "
                f"overlap {self.overlap_start.strftime('%H:%M')}-"
                f"{self.overlap_end.strftime('%H:%M')}")


class Chunker:
    """
    Partitions bursts into overlapping hourly chunks.

    Each chunk has:
    - primary window: the 1-hour window we're actually reasoning about
    - overlap_left / overlap_right: 15 minutes of context on each side

    The LLM only produces sessions for the primary window. Overlap is
    for context continuity only.
    """

    def __init__(
        self,
        primary_minutes: int = 60,
        overlap_minutes: int = 15,
    ):
        self.primary_minutes = primary_minutes
        self.overlap_minutes = overlap_minutes

    def chunk_for_date(self, bursts: List[Burst]) -> List[Tuple[Chunk, List[Burst]]]:
        """
        Partition bursts into overlapping chunks.

        Returns list of (Chunk, bursts) tuples where bursts includes
        both primary and overlap bursts, but the LLM should only produce
        sessions for the primary window portion.
        """
        if not bursts:
            return []

        # Determine the day's start and end
        sorted_bursts = sorted(bursts, key=lambda b: b.timestamp)
        first_ts = datetime.fromisoformat(sorted_bursts[0].timestamp)
        last_ts = datetime.fromisoformat(sorted_bursts[-1].timestamp)

        # Round first_ts down to nearest hour boundary
        day_start = first_ts.replace(minute=0, second=0, microsecond=0)
        # Round last_ts up to nearest hour boundary
        day_end = last_ts.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        chunks = []
        chunk_id = 0
        current = day_start

        while current < day_end:
            primary_start = current
            primary_end = current + timedelta(minutes=self.primary_minutes)

            overlap_start = primary_start - timedelta(minutes=self.overlap_minutes)
            overlap_end = primary_end + timedelta(minutes=self.overlap_minutes)

            chunk = Chunk(
                chunk_id=chunk_id,
                primary_start=primary_start,
                primary_end=primary_end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )

            # Collect bursts that fall within the full overlap window
            chunk_bursts = [
                b for b in sorted_bursts
                if overlap_start <= datetime.fromisoformat(b.timestamp) < overlap_end
            ]

            chunks.append((chunk, chunk_bursts))
            chunk_id += 1
            current += timedelta(minutes=self.primary_minutes)

        return chunks

    def partition_bursts(
        self, bursts: List[Burst]
    ) -> List[Tuple[Chunk, List[Burst], List[Burst]]]:
        """
        Partition bursts into chunks, splitting into three categories per chunk:

        Returns: List of (chunk, primary_bursts, overlap_bursts)
        - primary_bursts: bursts in the primary window (LLM reasons about these)
        - overlap_bursts: bursts in overlap regions (context only, not reasoned about)
        """
        chunk_data = self.chunk_for_date(bursts)
        result = []

        for chunk, all_bursts in chunk_data:
            primary_bursts = [
                b for b in all_bursts
                if chunk.primary_start <= datetime.fromisoformat(b.timestamp) < chunk.primary_end
            ]
            overlap_bursts = [b for b in all_bursts if b not in primary_bursts]
            result.append((chunk, primary_bursts, overlap_bursts))

        return result


class ChunkProcessor:
    """
    Processes each chunk through: sessionizer → reasoner → sessions.
    Then runs a final pass to fix cross-chunk inconsistencies.
    """

    def __init__(
        self,
        chunker: Chunker = None,
        reasoner: Reasoner = None,
        overlap_minutes: int = 15,
    ):
        self.chunker = chunker or Chunker()
        self.reasoner = reasoner or Reasoner()
        self.overlap_minutes = overlap_minutes

    def process_day(
        self, bursts: List[Burst], api_key: str = None
    ) -> Dict[str, Any]:
        """
        Full chunked reasoning pipeline:

        1. Partition bursts into overlapping hourly chunks
        2. Per chunk: sessionizer → LLM reasoner → sessions for primary window
        3. Final pass: fix inconsistencies from chunk boundaries
        4. Return: {sessions, daily_summary, chunk_count, candidates_count}
        """
        if not bursts:
            return {"sessions": [], "daily_summary": "No activity recorded.",
                    "chunk_count": 0, "candidate_count": 0}

        # Partition
        partitions = self.chunker.partition_bursts(bursts)
        chunk_count = len(partitions)

        all_chunk_sessions = []  # sessions from each chunk (before final fix)
        all_candidates = []      # all candidates for overall daily summary

        for chunk, primary_bursts, overlap_bursts in partitions:
            if not primary_bursts:
                continue

            # Sessionize primary bursts (include overlap for context in grouping)
            sessionizer = Sessionizer()
            all_bursts_for_sessionizer = primary_bursts + overlap_bursts
            candidates = sessionizer.group_bursts(all_bursts_for_sessionizer)

            # Filter candidates to only those with primary window bursts
            primary_candidates = [
                c for c in candidates
                if self._candidate_has_primary_bursts(c, chunk)
            ]

            all_candidates.extend(primary_candidates)

            # Reason only over primary candidates
            if primary_candidates:
                chunk_result = self.reasoner._reason_sessions(primary_candidates)
                # Tag each session with which chunk it came from
                for s in chunk_result:
                    s.chunk_id = chunk.chunk_id
                all_chunk_sessions.extend(chunk_result)

        # Final pass: fix cross-chunk inconsistencies
        fixed_sessions = self._fix_chunk_boundaries(all_chunk_sessions)

        # Generate daily summary from all candidates
        if all_candidates and api_key:
            daily_summary = self.reasoner._reason_daily_summary(all_candidates)
        else:
            daily_summary = self._rule_based_summary(fixed_sessions)

        return {
            "sessions": fixed_sessions,
            "daily_summary": daily_summary,
            "chunk_count": chunk_count,
            "candidate_count": len(all_candidates),
            "raw_chunk_sessions": all_chunk_sessions,  # for debugging
        }

    def _candidate_has_primary_bursts(
        self, candidate: CandidateSession, chunk: Chunk
    ) -> bool:
        """Check if candidate has any bursts in the primary window."""
        for b in candidate.bursts:
            ts = datetime.fromisoformat(b.timestamp)
            if chunk.primary_start <= ts < chunk.primary_end:
                return True
        return False

    def _fix_chunk_boundaries(self, chunk_sessions: List) -> List:
        """
        Final pass: resolve inconsistencies caused by chunk boundaries.

        Problems this fixes:
        1. Same task straddling two chunks → appears as two sessions
        2. Work type differs at boundary (chunk 1: writing, chunk 2: debugging)
        3. Topic differs for same window_id at boundary

        Strategy:
        - Sort sessions by start_time
        - Merge adjacent sessions with same app_name + window_id + work_type
        - For same window_id but different work_type: keep separate (genuine shift)
        - Cross-app sessions with similar topics: evaluate for merge
        """
        if not chunk_sessions:
            return []

        from sessionizer import Session

        # Sort by start_time then chunk_id (for stable ordering)
        sorted_sessions = sorted(
            chunk_sessions,
            key=lambda s: (s.start_time, getattr(s, 'chunk_id', 0))
        )

        fixed = []
        current = None

        for session in sorted_sessions:
            if current is None:
                fixed.append(session)
                current = session
                continue

            # Check if we should merge with current
            should_merge = self._should_merge(current, session)

            if should_merge == "merge":
                # Extend current session
                current.end_time = session.end_time
                current.chars += session.chars
                current.duration_minutes = self._compute_duration(
                    current.start_time, session.end_time
                )
            else:
                fixed.append(session)
                current = session

        return fixed

    def _should_merge(self, prev, curr) -> str:
        """
        Decide whether to merge two adjacent sessions.

        Returns: "merge" | "split"
        Merge if ALL of:
        - Same app_name AND same window_id
        - Adjacent in time (within overlap window gap)
        - Same work_type OR same topic (flexible on work_type at boundaries)
        """
        from datetime import time

        def ts_to_dt(ts_str):
            return datetime.strptime(ts_str, "%H:%M:%S")

        gap = ts_to_dt(curr.start_time) - ts_to_dt(prev.end_time)
        gap_minutes = gap.total_seconds() / 60

        # If gap > overlap + buffer, they're genuinely separate
        if gap_minutes > self.overlap_minutes * 2:
            return "split"

        # Same app + same window_id + same work_type → merge
        if (prev.app_name == curr.app_name and
            prev.window_title == curr.window_title and
            prev.work_type == curr.work_type):
            return "merge"

        # Same app + same window_id + similar topic → merge
        if (prev.app_name == curr.app_name and
            prev.window_title == curr.window_title and
            self._topics_similar(prev.topic, curr.topic)):
            return "merge"

        return "split"

    def _topics_similar(self, t1: str, t2: str) -> bool:
        """Rough similarity check for topic merging."""
        if not t1 or not t2:
            return False
        # Simple: check if first 30 chars match (same file/thread)
        return t1[:30].lower() == t2[:30].lower()

    def _compute_duration(self, start: str, end: str) -> int:
        """Compute duration in minutes from start/end time strings."""
        from datetime import datetime
        start_dt = datetime.strptime(start, "%H:%M:%S")
        end_dt = datetime.strptime(end, "%H:%M:%S")
        return max(1, int((end_dt - start_dt).total_seconds() / 60))

    def _rule_based_summary(self, sessions: List) -> str:
        """Fallback daily summary without LLM."""
        if not sessions:
            return "No activity recorded."
        from collections import Counter
        wt_counter = Counter(s.work_type for s in sessions)
        apps = Counter(s.app_name for s in sessions)
        total_chars = sum(s.chars for s in sessions)
        top_wt = wt_counter.most_common(1)[0] if wt_counter else ("writing", 0)
        top_app = apps.most_common(1)[0] if apps else ("Unknown", 0)
        return (f"{len(sessions)} sessions, {total_chars} chars. "
                f"Top work type: {top_wt[0]} ({top_wt[1]} sessions). "
                f"Top app: {top_app[0]} ({top_app[1]} sessions).")


if __name__ == "__main__":
    # Quick test
    from burst import load_bursts_from_json, insert_bursts, init_db, get_db_path
    from pathlib import Path

    sample = Path(__file__).parent / "data" / "sample_conversation.json"
    if sample.exists():
        print("Testing chunker on sample data...")
        bursts = load_bursts_from_json(sample)
        chunker = Chunker()
        partitions = chunker.chunk_for_date(bursts)

        print(f"\n{len(partitions)} chunks created:")
        for chunk, bursts_in_chunk in partitions:
            print(f"  {chunk}")
            print(f"    Primary: {len([b for b in bursts_in_chunk if chunk.primary_start <= datetime.fromisoformat(b.timestamp) < chunk.primary_end])} bursts")
            print(f"    Overlap: {len([b for b in bursts_in_chunk if not (chunk.primary_start <= datetime.fromisoformat(b.timestamp) < chunk.primary_end)])} bursts")

        # Test full pipeline
        processor = ChunkProcessor()
        result = processor.process_day(bursts)
        print(f"\nProcessed {result['chunk_count']} chunks, {result['candidate_count']} candidates")
        print(f"Final sessions: {len(result['sessions'])}")
