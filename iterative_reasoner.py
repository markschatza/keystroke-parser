#!/usr/bin/env python3
"""
Iterative LLM Reasoner — Option 2: Fully LLM-driven sessionization.

Unlike the heuristic sessionizer, this module gives the LLM raw bursts and asks it
to make ALL sessionization decisions with visible reasoning:

Pass 1 — GROUPING: The LLM groups raw bursts into logical sessions, explaining WHY
          it made each grouping decision. It sees the full picture: timing, app,
          window context, and text content.

Pass 2 — LABELING: Each session from Pass 1 gets work_type, topic, and a narrative
          description of what happened.

Pass 3 — SYNTHESIS: The full day's sessions are reviewed for consistency, cross-session
          patterns are identified, and a daily summary is generated.

This is more expensive than heuristic sessionization but handles:
- Interleaved parallel workflows (two Codex agents, two email threads)
- Context-aware splitting/merging that heuristics can't express
- The "why" behind each decision (useful for debugging and trust)
"""

import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from burst import Burst, get_db_path


# ----------------------------------------------------------------------------- 
# Data structures
# ----------------------------------------------------------------------------- 

@dataclass
class RawBurstRef:
    """A compact reference to a burst for the LLM prompt."""
    burst_id: int       # index in the raw list
    timestamp: str
    app_name: str
    window_title: str
    window_id: str
    source: str
    char_count: int
    text_preview: str    # first 80 chars of text


@dataclass 
class LLMReasonedSession:
    """
    A session produced by the LLM with full reasoning visible.
    """
    session_id: int
    burst_ids: List[int]          # which raw bursts belong here
    start_time: str
    end_time: str
    app_name: str
    window_title: str
    total_chars: int
    total_duration_seconds: int
    work_type: str                 # debugging | writing | reading | communicating | planning
    topic: str                     # specific task in ≤10 words
    narrative: str                 # what happened, in 1-2 sentences
    grouping_reasoning: str       # WHY these bursts were grouped together
    confidence: str                # high | medium | low — how confident the LLM is


# ----------------------------------------------------------------------------- 
# Iterative Reasoner
# ----------------------------------------------------------------------------- 

class IterativeReasoner:
    """
    Multi-pass LLM reasoner for raw keystroke bursts.
    
    Pass 1 (GROUPING):     Raw bursts → reasoned session groups
    Pass 2 (LABELING):    Each group → work_type, topic, narrative
    Pass 3 (SYNTHESIS):   All sessions → daily summary + cross-session analysis
    
    The LLM sees more context and makes better decisions than the heuristic
    sessionizer, at the cost of more tokens and API calls.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "MiniMax-M2"):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.io/v1/text"
        self.model = model

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def reason_day(self, bursts: List[Burst]) -> Dict[str, Any]:
        """
        Full iterative reasoning pipeline over raw bursts.
        
        Returns dict with:
          - sessions: List[LLMReasonedSession]
          - daily_summary: str
          - grouping_reasoning: str (full reasoning from pass 1)
          - pass1_tokens_used: int (approximate)
          - pass2_tokens_used: int (approximate)
          - total_api_calls: int
        """
        if not bursts:
            return {
                "sessions": [],
                "daily_summary": "No activity recorded.",
                "grouping_reasoning": "",
                "daily_narrative": "",
                "pass1_tokens_used": 0,
                "pass2_tokens_used": 0,
                "total_api_calls": 0,
            }

        if not self.api_key:
            return self._fallback(bursts)

        # Sort bursts by time
        sorted_bursts = sorted(bursts, key=lambda b: b.timestamp)

        # Build compact burst references
        burst_refs = self._build_burst_refs(sorted_bursts)

        # Pass 1: LLM groups bursts into sessions
        grouping_result = self._pass1_grouping(burst_refs, sorted_bursts)
        raw_sessions = grouping_result["sessions"]  # List of dicts from LLM

        # Pass 2: LLM labels each session
        labeled_sessions, pass2_usage = self._pass2_labeling(raw_sessions, sorted_bursts, burst_refs)
        
        # Pass 3: Synthesis
        daily_summary, daily_narrative, pass3_usage = self._pass3_synthesis(labeled_sessions, sorted_bursts)

        return {
            "sessions": labeled_sessions,
            "daily_summary": daily_summary,
            "daily_narrative": daily_narrative,
            "grouping_reasoning": grouping_result.get("full_reasoning", ""),
            "pass1_tokens_used": grouping_result.get("tokens_used", 0),
            "pass2_tokens_used": pass2_usage,
            "pass3_tokens_used": pass3_usage,
            "total_api_calls": grouping_result.get("num_chunks", 1) + 1 + 1,  # Pass1 chunks + Pass2 (1) + Pass3 (1)
        }

    def reason_day_from_db(self, date_str: str) -> Dict[str, Any]:
        """Load bursts from DB and reason over them."""
        from burst import load_bursts_for_date
        bursts = load_bursts_for_date(date_str)
        return self.reason_day(bursts)

    # -------------------------------------------------------------------------
    # Pass 1: Grouping
    # -------------------------------------------------------------------------

    def _pass1_grouping(
        self, burst_refs: List[RawBurstRef], sorted_bursts: List[Burst]
    ) -> Dict[str, Any]:
        """
        Pass 1: Chunk bursts and call LLM on each chunk in parallel.
        
        Each chunk gets a "carry context" message noting the last burst from the
        previous chunk, so the LLM can make correct cross-chunk decisions when
        a session spans a chunk boundary.
        
        Chunk size is 60 bursts (~3 hours of typical work), which keeps each
        prompt well under token limits even with the context preamble.
        """
        chunk_size = 60
        chunks = []
        for i in range(0, len(burst_refs), chunk_size):
            chunks.append(burst_refs[i:i + chunk_size])

        # Build carry context for each chunk (what was the last burst of the previous?)
        carry_contexts = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                prev_last = burst_refs[idx * chunk_size - 1]
                window = prev_last.window_title.split(' - ')[-1] if ' - ' in prev_last.window_title else prev_last.window_title
                carry_contexts.append(
                    f"[CARRIES FROM PREVIOUS CHUNK] Last burst before this chunk: "
                    f"[{idx * chunk_size - 1}] {prev_last.timestamp[11:16]} | "
                    f"{prev_last.app_name[:10]} | {window[:40]}"
                )
            else:
                carry_contexts.append(None)

        def process_chunk(chunk_refs, global_offset, carry):
            """Process one chunk and return session groups. Runs in thread pool."""
            # Annotate bursts with gap warnings for long gaps
            prev_ts = None
            annotated = []
            for i, ref in enumerate(chunk_refs):
                ts = ref.timestamp
                gap_note = ""
                if prev_ts:
                    try:
                        from datetime import datetime
                        t_prev = datetime.fromisoformat(prev_ts)
                        t_curr = datetime.fromisoformat(ts)
                        gap_min = int((t_curr - t_prev).total_seconds() / 60)
                        if gap_min >= 30:
                            gap_note = f" ← {gap_min}min gap"
                        elif gap_min >= 5:
                            gap_note = f" ({gap_min}min)"
                    except:
                        pass
                prev_ts = ts

                window = ref.window_title.split(' - ')[-1] if ' - ' in ref.window_title else ref.window_title
                text_snippet = ref.text_preview[:50].replace("\n", " ")
                annotated.append("[%d] %s | %-10s | %-25s |%s %s" % (
                    i,
                    ref.timestamp[11:16],
                    ref.app_name[:10],
                    window[:25],
                    gap_note,
                    text_snippet
                ))

            num_bursts = len(chunk_refs)
            max_burst_id = num_bursts - 1
            carry_note = f"\n\n{carry}" if carry else ""

            prompt = f"""You are sessionizing a developer's keystroke bursts.

RULES:
1. Gap > 30 min = NEW session (meeting/break)
2. Different SEMANTIC TASK = NEW session
   - Fixing a bug = different from implementing a new feature = different from code review
   - "Add handling fee for international" (bug fix in shippingService) = different from
     "user avatar upload" (new feature in avatarService) = different from
     "PR #234 review" (reading/coordination)
3. Same file/feature being iterated on = SAME session
   - Multiple Codex bursts on auth/middleware.py across 30 min = same auth session
4. Quick coordination (< 5 min, Slack/email) during active dev = part of that session{carry_note}

CRITICAL EXAMPLES of how to split:
- Bug fix (shippingService hotfix) at 12:05 ≠ New feature (avatarService) at 13:00 → SPLIT
- Implementing feature (avatarService.ts) at 13:10 ≠ PR review (reading) at 13:15 → SPLIT
- Fix session (postgres debug) at 14:00 ≠ EOD standup at 16:30 → SPLIT{carry_note}

When in doubt, SPLIT. Better 8 clean sessions than 5 messy ones.{carry_note}

INPUT — bursts {global_offset} to {global_offset + num_bursts - 1}:
{chr(10).join(annotated)}

OUTPUT — JSON object with sessions array:
{{"sessions": [
  {{"session_id": 1, "burst_ids": [0, 2, 4], "start_time": "08:00", "end_time": "08:10", "grouping_reasoning": "why"}}
]}}

Every burst 0 to {max_burst_id} must appear exactly once in this chunk's output. JSON only."""

            response = self._call(prompt, temperature=0.3, max_tokens=16000)
            json_str = self._extract_json_for_pass(response, "sessions")
            if not json_str:
                return []

            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "sessions" in data:
                    return data.get("sessions", [])
                elif isinstance(data, list):
                    return data
                else:
                    return []
            except json.JSONDecodeError as e:
                # If JSON is truncated/malformed, try to repair by truncating to last complete object
                repaired = self._repair_json(json_str)
                if repaired:
                    try:
                        data = json.loads(repaired)
                        if isinstance(data, dict) and "sessions" in data:
                            return data.get("sessions", [])
                        elif isinstance(data, list):
                            return data
                    except json.JSONDecodeError:
                        pass
                return []

        # Process all chunks in parallel
        all_chunk_sessions = []
        with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
            futures = {}
            for idx, chunk in enumerate(chunks):
                global_offset = idx * chunk_size
                future = executor.submit(process_chunk, chunk, global_offset, carry_contexts[idx])
                futures[future] = idx
            for future in as_completed(futures):
                chunk_sessions = future.result()
                all_chunk_sessions.append((futures[future], chunk_sessions))

        # Sort by chunk order
        all_chunk_sessions.sort(key=lambda x: x[0])
        
        # Re-index session IDs and validate all bursts are covered
        all_ids = set()
        final_sessions = []
        session_id = 1

        for chunk_idx, chunk_sessions in all_chunk_sessions:
            global_offset = chunk_idx * chunk_size
            for sess in chunk_sessions:
                burst_ids = [global_offset + bid for bid in sess.get("burst_ids", [])]
                final_sessions.append({
                    "session_id": session_id,
                    "burst_ids": burst_ids,
                    "start_time": sess.get("start_time", ""),
                    "end_time": sess.get("end_time", ""),
                    "grouping_reasoning": sess.get("grouping_reasoning", ""),
                })
                all_ids.update(burst_ids)
                session_id += 1

        # Mark unused bursts as orphan sessions
        if len(all_ids) < len(burst_refs):
            unused = [i for i in range(len(burst_refs)) if i not in all_ids]
            if unused:
                final_sessions.append({
                    "session_id": session_id,
                    "burst_ids": unused,
                    "grouping_reasoning": "Orphaned bursts not captured in main sessions",
                    "start_time": burst_refs[unused[0]].timestamp[11:16],
                    "end_time": burst_refs[unused[-1]].timestamp[11:16],
                })

        return {
            "sessions": final_sessions,
            "full_reasoning": f"Processed {len(chunks)} chunks in parallel",
            "tokens_used": 0,
            "num_chunks": len(chunks),
        }

    # -------------------------------------------------------------------------
    # Pass 2: Labeling
    # -------------------------------------------------------------------------

    def _pass2_labeling(
        self,
        raw_sessions: List[Dict],
        sorted_bursts: List[Burst],
        burst_refs: List[RawBurstRef],
    ) -> tuple:
        """
        Pass 2: Label each session. Defaults to a SINGLE batched call
        for efficiency (1 API call instead of N). Set parallel=True to
        fall back to per-session parallel calls.
        """
        return self._pass2_labeling_batched(raw_sessions, sorted_bursts, burst_refs)

    def _pass2_labeling_batched(
        self,
        raw_sessions: List[Dict],
        sorted_bursts: List[Burst],
        burst_refs: List[RawBurstRef],
    ) -> tuple:
        """
        Pass 2: Batch ALL sessions into a SINGLE API call.
        Much more efficient — one API call instead of one per session.
        """
        if not raw_sessions:
            return [], 0

        # Build session blocks for the prompt
        session_blocks = []
        for sess in raw_sessions:
            burst_ids = sess.get("burst_ids", [])
            if not burst_ids:
                continue

            burst_texts = []
            for bid in burst_ids:
                if 0 <= bid < len(sorted_bursts):
                    b = sorted_bursts[bid]
                    burst_texts.append(f"[{bid}] {b.chars[:200]}")

            full_text = "\n".join(burst_texts)
            if len(full_text) > 2500:
                full_text = full_text[:2500] + "\n... (truncated)"

            first_ts = sorted_bursts[burst_ids[0]].timestamp
            last_ts = sorted_bursts[burst_ids[-1]].timestamp

            try:
                t1 = datetime.fromisoformat(first_ts)
                t2 = datetime.fromisoformat(last_ts)
                duration_min = max(1, int((t2 - t1).total_seconds() / 60))
            except:
                duration_min = 1

            chars_in_sess = sum(
                sorted_bursts[i].char_count
                for i in burst_ids if 0 <= i < len(sorted_bursts)
            )

            session_blocks.append({
                "session_id": sess["session_id"],
                "grouping_reasoning": sess.get("grouping_reasoning", ""),
                "first_ts": first_ts[11:16],
                "last_ts": last_ts[11:16],
                "duration_min": duration_min,
                "num_bursts": len(burst_ids),
                "chars_in_sess": chars_in_sess,
                "text_preview": full_text,
            })

        prompt = f"""Analyze and label each keystroke session below.

work_type options:
- debugging: fixing bugs, investigating errors, tracing issues, troubleshooting
- writing: implementing new features, writing code/modules, creating files, building
- reading: code review, reading docs, investigating/understanding code, analyzing
- communicating: Slack, email, messages, discussions, code reviews with others
- planning: planning, scoping, sprint planning, architecture design, outlining

topic: specific task in ≤10 words. Be concrete.
narrative: 1-2 sentences describing what happened. Name files, features, bugs.
confidence: high (clear task context), medium (some context), low (minimal context)

=== SESSIONS ===
{json.dumps(session_blocks, indent=2)}
===

OUTPUT — JSON array of labels, one per session:
[
  {{"session_id": 1, "work_type": "...", "topic": "...", "narrative": "...", "confidence": "high"}},
  ...
]

Respond ONLY with the JSON array."""

        response = self._call(prompt, temperature=0.3, max_tokens=3000)
        json_str = self._extract_json(response)

        # Build lookup from response
        label_map = {}
        if json_str:
            try:
                labels = json.loads(json_str)
                if isinstance(labels, list):
                    for lbl in labels:
                        label_map[lbl["session_id"]] = lbl
            except json.JSONDecodeError:
                pass

        # Assemble final sessions, falling back to defaults for any missing
        labeled = []
        for sess in raw_sessions:
            burst_ids = sess.get("burst_ids", [])
            if not burst_ids:
                continue

            first_ts = sorted_bursts[burst_ids[0]].timestamp
            last_ts = sorted_bursts[burst_ids[-1]].timestamp

            try:
                dur_sec = int((datetime.fromisoformat(last_ts) - datetime.fromisoformat(first_ts)).total_seconds())
            except:
                dur_sec = 60

            chars_in_sess = sum(
                sorted_bursts[i].char_count
                for i in burst_ids if 0 <= i < len(sorted_bursts)
            )

            lbl = label_map.get(sess["session_id"], {})

            labeled.append(LLMReasonedSession(
                session_id=sess["session_id"],
                burst_ids=burst_ids,
                start_time=first_ts[11:16],
                end_time=last_ts[11:16],
                app_name=burst_refs[burst_ids[0]].app_name if burst_ids else "Unknown",
                window_title=burst_refs[burst_ids[0]].window_title if burst_ids else "",
                total_chars=lbl.get("chars_in_session", chars_in_sess),
                total_duration_seconds=dur_sec,
                work_type=lbl.get("work_type", "writing"),
                topic=lbl.get("topic", f"Session {sess['session_id']}")[:80],
                narrative=lbl.get("narrative", sess.get("grouping_reasoning", "")),
                grouping_reasoning=sess.get("grouping_reasoning", ""),
                confidence=lbl.get("confidence", "low"),
            ))

        labeled.sort(key=lambda x: x.session_id)
        tokens_used = len(prompt) // 4 + len(response) // 4
        return labeled, tokens_used

    def _pass2_labeling_parallel(
        self,
        raw_sessions: List[Dict],
        sorted_bursts: List[Burst],
        burst_refs: List[RawBurstRef],
    ) -> tuple:
        """
        Pass 2: Label each session in parallel using ThreadPoolExecutor.
        One API call per session — useful for debugging or when you need
        isolation between sessions (can retry individual failures).
        """

        def label_session(sess):
            """Label a single session. Runs in thread pool."""
            burst_ids = sess.get("burst_ids", [])
            if not burst_ids:
                return None

            # Collect burst texts
            burst_texts = []
            for bid in burst_ids:
                if 0 <= bid < len(sorted_bursts):
                    b = sorted_bursts[bid]
                    burst_texts.append(f"[{bid}] {b.chars[:200]}")

            full_text = "\n".join(burst_texts)
            if len(full_text) > 3000:
                full_text = full_text[:3000] + "\n... (truncated)"

            first_ts = sorted_bursts[burst_ids[0]].timestamp
            last_ts = sorted_bursts[burst_ids[-1]].timestamp

            try:
                t1 = datetime.fromisoformat(first_ts)
                t2 = datetime.fromisoformat(last_ts)
                duration_min = max(1, int((t2 - t1).total_seconds() / 60))
            except:
                duration_min = 1

            chars_in_sess = sum(
                sorted_bursts[i].char_count
                for i in burst_ids if 0 <= i < len(sorted_bursts)
            )

            prompt = f"""Analyze this keystroke session and label it.

Session {sess['session_id']}: {sess.get('grouping_reasoning', 'No reasoning provided')}
Time range: {first_ts[11:16]} - {last_ts[11:16]} (~{duration_min} min)
Number of bursts: {len(burst_ids)}

Text content (what was typed):
---
{full_text}
---

Your task: determine what work_type best describes this session,
what the specific topic/task was, and write a brief narrative.

work_type options:
- debugging: fixing bugs, investigating errors, tracing issues, troubleshooting
- writing: implementing new features, writing code/modules, creating files, building
- reading: code review, reading docs, investigating/understanding code, analyzing
- communicating: Slack, email, messages, discussions, code reviews with others
- planning: planning, scoping, sprint planning, architecture design, outlining

topic: specific task in ≤10 words. Be concrete. Examples:
- "Implement JWT token verification in auth middleware"
- "Fix race condition in WebSocket connection handler"
- "Write unit tests for payment processing module"
- "Debug memory leak in background job worker"
- "Review PR #342: add rate limiting to API"

narrative: 1-2 sentences describing what happened. Be specific — name files, features, bugs.

OUTPUT FORMAT — respond with a JSON object:
{{
  "session_id": {sess['session_id']},
  "work_type": "...",
  "topic": "...",
  "narrative": "...",
  "confidence": "high" | "medium" | "low",
  "chars_in_session": {chars_in_sess}
}}

Respond ONLY with the JSON object."""

            response = self._call(prompt, temperature=0.3, max_tokens=500)

            json_str = self._extract_json(response)
            if json_str:
                try:
                    label_data = json.loads(json_str)
                    try:
                        t1 = datetime.fromisoformat(first_ts)
                        t2 = datetime.fromisoformat(last_ts)
                        dur_sec = int((t2 - t1).total_seconds())
                    except:
                        dur_sec = 60

                    return LLMReasonedSession(
                        session_id=sess["session_id"],
                        burst_ids=burst_ids,
                        start_time=first_ts[11:16],
                        end_time=last_ts[11:16],
                        app_name=burst_refs[burst_ids[0]].app_name if burst_ids else "Unknown",
                        window_title=burst_refs[burst_ids[0]].window_title if burst_ids else "",
                        total_chars=label_data.get("chars_in_session", chars_in_sess),
                        total_duration_seconds=dur_sec,
                        work_type=label_data.get("work_type", "writing"),
                        topic=label_data.get("topic", "general work")[:80],
                        narrative=label_data.get("narrative", ""),
                        grouping_reasoning=sess.get("grouping_reasoning", ""),
                        confidence=label_data.get("confidence", "medium"),
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

            # Fallback
            try:
                t1 = datetime.fromisoformat(first_ts)
                t2 = datetime.fromisoformat(last_ts)
                dur_sec = int((t2 - t1).total_seconds())
            except:
                dur_sec = 60

            return LLMReasonedSession(
                session_id=sess["session_id"],
                burst_ids=burst_ids,
                start_time=first_ts[11:16],
                end_time=last_ts[11:16],
                app_name=burst_refs[burst_ids[0]].app_name if burst_ids else "Unknown",
                window_title=burst_refs[burst_ids[0]].window_title if burst_ids else "",
                total_chars=chars_in_sess,
                total_duration_seconds=dur_sec,
                work_type="writing",
                topic=f"Session {sess['session_id']}",
                narrative=sess.get("grouping_reasoning", "No description available"),
                grouping_reasoning=sess.get("grouping_reasoning", ""),
                confidence="low",
            )

        # Run all sessions in parallel
        labeled = []
        total_tokens = 0
        with ThreadPoolExecutor(max_workers=min(10, len(raw_sessions))) as executor:
            futures = {executor.submit(label_session, sess): sess for sess in raw_sessions}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    labeled.append(result)

        # Sort by session_id to preserve order
        labeled.sort(key=lambda x: x.session_id)
        return labeled, total_tokens

    # -------------------------------------------------------------------------
    # Pass 3: Synthesis
    # -------------------------------------------------------------------------

    def _pass3_synthesis(
        self,
        sessions: List[LLMReasonedSession],
        sorted_bursts: List[Burst],
    ) -> tuple:
        """
        Pass 3: Review all sessions for consistency and generate daily summary.
        
        The LLM checks:
        - Are there similar sessions that should be merged?
        - Are there patterns across the day?
        - Is the time breakdown realistic?
        """
        if not sessions:
            return "No activity recorded.", "", 0

        # Build session summaries for the LLM
        session_summaries = []
        for s in sessions:
            session_summaries.append({
                "id": s.session_id,
                "time": f"{s.start_time}-{s.end_time}",
                "app": s.app_name,
                "work_type": s.work_type,
                "topic": s.topic,
                "chars": s.total_chars,
                "duration_min": max(1, s.total_duration_seconds // 60),
                "confidence": s.confidence,
            })

        total_chars = sum(s.total_chars for s in sessions)
        total_min = sum(s.total_duration_seconds for s in sessions) // 60

        prompt = f"""Review this developer's work day and produce a summary.

Total: {total_chars} characters typed over approximately {total_min} minutes across {len(sessions)} sessions.

Sessions:
{json.dumps(session_summaries, indent=2)}

Your task:
1. Write a 2-paragraph daily summary — what were the major topics? What was accomplished?
2. Note any interesting patterns (parallel workstreams, task switching, notable accomplishments)
3. Rate the overall day: productive, mixed, scattered, etc.

OUTPUT FORMAT:
{{
  "daily_summary": "2-paragraph narrative summary of the day",
  "day_quality": "productive | mixed | scattered | focused | etc",
  "key_accomplishments": ["accomplishment 1", "accomplishment 2", ...],
  "patterns": "brief note on any interesting patterns observed"
}}

Respond ONLY with the JSON object."""

        response = self._call(prompt, temperature=0.3, max_tokens=800)
        tokens_used = len(prompt) // 4 + len(response) // 4

        json_str = self._extract_json_for_pass(response, "daily_summary")
        if json_str:
            try:
                data = json.loads(json_str)
                # Handle malformed responses that are arrays instead of objects
                if isinstance(data, list) and len(data) > 0:
                    # Try to extract summary from the first array item
                    data = data[0]
                if isinstance(data, dict):
                    return (
                        data.get("daily_summary", "Summary not available."),
                        data.get("patterns", ""),
                        tokens_used,
                    )
            except json.JSONDecodeError:
                pass

        return response[:500] if response else "Summary not available.", "", tokens_used

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _build_burst_refs(self, bursts: List[Burst]) -> List[RawBurstRef]:
        """Build compact burst references for LLM consumption."""
        refs = []
        for i, b in enumerate(bursts):
            preview = b.chars[:80].replace("\n", " ").replace('"', "'").strip()
            refs.append(RawBurstRef(
                burst_id=i,
                timestamp=b.timestamp,
                app_name=b.app_name,
                window_title=b.window_title,
                window_id=b.window_id,
                source=b.source,
                char_count=b.char_count,
                text_preview=preview,
            ))
        return refs

    def _call(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1500, retries: int = 3) -> str:
        """Make a MiniMax API call with automatic retry on rate limit."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "off"}},
        }

        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chatcompletion_v2",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                data = response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                elif "base_resp" in data:
                    status_msg = data["base_resp"].get("status_msg", "unknown")
                    if "rate" in status_msg.lower() or "limit" in status_msg.lower():
                        wait = 2 ** attempt
                        print(f"[DEBUG _call] Rate limited, retrying in {wait}s (attempt {attempt+1}/{retries})")
                        import time; time.sleep(wait)
                        continue
                    return f"API Error: {status_msg}"
                else:
                    print(f"[DEBUG _call] Unexpected response status={response.status_code}, data={str(data)[:200]}")
            except Exception as e:
                print(f"[DEBUG _call] Exception on attempt {attempt+1}: {e}")
                if attempt < retries - 1:
                    import time; time.sleep(2 ** attempt)
        return ""

    def _extract_json(self, text: str) -> Optional[str]:
        # Remove code fences
        for fence in ["```json", "```", "```yaml", "```python"]:
            if text.startswith(fence):
                text = text[len(fence):]
            if text.endswith(fence):
                text = text[:-len(fence)]
        text = text.strip()

        # Find the closing tag
        end_tag = text.find("</think>")
        if end_tag != -1:
            text = text[end_tag + len("</think>"):].strip()
        # Also try to find content wrapped in other tags
        for tag in ["<result>", "</result>", "<output>", "</output>", "<response>", "</response>"]:
            start = text.find(tag)
            if start != -1:
                text = text[start + len(tag):]
                end = text.find(tag.replace("<", "</"))
                if end != -1:
                    text = text[:end]

        # Prefer JSON array over object when both present (array spans more chars)
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        obj_start = text.find("{")
        obj_end = text.rfind("}")

        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                # Pick whichever spans more text
                if arr_end - arr_start > obj_end - obj_start:
                    return text[arr_start:arr_end+1]
            return text[arr_start:arr_end+1]

        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return text[obj_start:obj_end+1]

        return None

    def _repair_json(self, malformed_json: str) -> Optional[str]:
        """
        Attempt to repair truncated/malformed JSON by truncating to the last
        structurally complete prefix. Handles:
        - Unclosed strings with escape sequences
        - Trailing incomplete objects/arrays
        - Missing closing braces
        """
        if not malformed_json:
            return None

        # Strategy: find the last complete object or array item
        # Walk backwards from the end, find the last '}' or ']' that closes a top-level structure
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(len(malformed_json) - 1, -1, -1):
            c = malformed_json[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if c == '\\' and in_string:
                escape_next = True
                continue
            
            if c == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if c == '}' or c == ']':
                depth += 1
            elif c == '{' or c == '[':
                depth -= 1
            
            # Found a top-level close?
            if depth > 0 and c in '}]':
                continue
            if depth == 0 and c in '}]':
                # This might be a valid end point
                # Try parsing from here
                candidate = malformed_json[:i+1]
                try:
                    json.loads(candidate)
                    return candidate  # Valid! Return it
                except json.JSONDecodeError:
                    pass
        
        # Fallback: try truncating to last valid array of complete objects
        # Find last '}' that might close the outermost object
        last_brace = malformed_json.rfind('}')
        if last_brace > 0:
            candidate = malformed_json[:last_brace+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        
        return None

    def _extract_json_for_pass(self, text: str, expected_key: str) -> Optional[str]:
        """
        Extract JSON, preferring a top-level object with expected_key over a bare array.
        Used by passes that wrap responses in {"sessions": [...]} or similar.
        """
        # Try to find {"expected_key": ...} first
        if expected_key:
            key_pattern = f'"{expected_key}"'
            key_pos = text.find(key_pattern)
            if key_pos != -1:
                # Back up to the opening brace
                start = text.rfind("{", 0, key_pos)
                if start != -1:
                    # Count braces to find the matching close
                    depth = 0
                    for i in range(start, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                return text[start:i+1]
        # Fall back to generic extraction
        return self._extract_json(text)

    def _fallback(self, bursts: List[Burst]) -> Dict[str, Any]:
        """Fallback when no API key is available."""
        return {
            "sessions": [],
            "daily_summary": "No API key available for LLM reasoning.",
            "grouping_reasoning": "",
            "daily_narrative": "",
            "pass1_tokens_used": 0,
            "pass2_tokens_used": 0,
            "total_api_calls": 0,
        }


# ----------------------------------------------------------------------------- 
# CLI 
# ----------------------------------------------------------------------------- 

def load_api_key() -> str:
    """Load MiniMax API key from Hermes config."""
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        raw = env_path.read_bytes()
        decoded = raw.decode('utf-8', errors='replace')
        # Find FIRST uncommented MINIMAX_API_KEY= line (first one is the real key)
        for line in decoded.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'MINIMAX_API_KEY' in line and '=' in line:
                key = line.split('=', 1)[1].strip()
                # Strip any non-printable chars (like escape sequences)
                key = ''.join(c for c in key if c.isprintable()).strip()
                if key and not key.startswith('***'):
                    return key
    return ""


def print_iterative_sessions(sessions: List[LLMReasonedSession]) -> None:
    """Pretty-print LLM-reasoned sessions."""
    print("\n" + "=" * 80)
    print("ITERATIVE LLM-REASONED SESSIONS")
    print("=" * 80)
    for s in sorted(sessions, key=lambda x: x.start_time):
        print(f"\n  [{s.start_time}-{s.end_time}] {s.app_name} | {s.work_type}")
        print(f"  Topic: {s.topic}")
        print(f"  {s.total_chars} chars | ~{max(1,s.total_duration_seconds//60)} min | confidence: {s.confidence}")
        print(f"  Reasoning: {s.grouping_reasoning[:100]}...")
        if s.narrative:
            print(f"  Narrative: {s.narrative[:150]}")


if __name__ == "__main__":
    from burst import load_bursts_from_json, get_db_path
    from pathlib import Path

    db_path = get_db_path()
    api_key = load_api_key()

    print("=" * 60)
    print("ITERATIVE REASONER — Fully LLM-driven sessionization")
    print("=" * 60)
    print(f"DB: {db_path}")
    print(f"API key: {'loaded' if api_key else 'not found'}")

    # Test on difficult_day.json
    sample = Path(__file__).parent / "data" / "difficult_day.json"
    if sample.exists():
        bursts = load_bursts_from_json(sample)
        print(f"\nReasoning over {len(bursts)} bursts from difficult_day.json...")

        reasoner = IterativeReasoner(api_key=api_key)
        result = reasoner.reason_day(bursts)

        print(f"\nAPI calls made: {result['total_api_calls']}")
        print(f"Approx tokens — Pass 1: {result['pass1_tokens_used']}, "
              f"Pass 2: {result['pass2_tokens_used']}, "
              f"Pass 3: {result['pass3_tokens_used']}")

        print_iterative_sessions(result["sessions"])

        print(f"\n{'='*60}")
        print("DAILY SUMMARY")
        print("=" * 60)
        print(result["daily_summary"])

        if result.get("grouping_reasoning"):
            print(f"\n{'='*60}")
            print("GROUPING APPROACH")
            print("=" * 60)
            print(result["grouping_reasoning"][:500])
    else:
        print("\nNo sample data found.")
