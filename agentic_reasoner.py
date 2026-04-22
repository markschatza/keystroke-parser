#!/usr/bin/env python3
"""
Agentic Sessionization Reasoner — deliberate multi-step reasoning with tools.

This reasoner uses a scratchpad + deliberation model with explicit tool calls:

1. SURVEY PASS: Read all bursts, build an overview (app counts, time range, gap distribution)
2. HYPOTHESIS PASS: Form initial session hypotheses based on patterns
3. VERIFICATION PASS: Use tools to check ambiguous cases against raw burst content
4. COMMIT PASS: Output final sessions with reasoning

The agent has access to tools:
- read_bursts(offset, limit): read raw burst text content
- analyze_gaps(bursts): return gap analysis (where are the 30+ min gaps)
- get_session_context(session_candidate): look at raw text of a proposed session
- scratchpad_write(text): write to scratchpad for reasoning
- scratchpad_read(): read scratchpad
- suggest_split(burst_id, reason): propose a split point
- suggest_merge(burst_ids, reason): propose merging

Output format (compatible with IterativeReasoner):
{"sessions": [{"session_id": 1, "burst_ids": [...], "start_time": "HH:MM",
               "end_time": "HH:MM", "grouping_reasoning": "...", "confidence": "high/medium/low"}]}
"""

import json
import os
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from burst import Burst
from external_context import format_external_context_for_prompt


# ----------------------------------------------------------------------------- 
# API helpers
# ----------------------------------------------------------------------------- 

def load_api_key() -> str:
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        raw = env_path.read_bytes().decode('utf-8', errors='replace')
        for line in raw.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'MINIMAX_API_KEY' in line and '=' in line:
                key = line.split('=', 1)[1].strip()
                key = ''.join(c for c in key if c.isprintable()).strip()
                if key and not key.startswith('***'):
                    return key
    return ""


def make_api_call(prompt: str, api_key: str, model: str = "MiniMax-M2",
                  base_url: str = "https://api.minimax.io",
                  temperature: float = 0.3, max_tokens: int = 8000,
                  retries: int = 3) -> tuple:
    """Returns (response_text, tokens_used)"""
    if not api_key:
        return "", 0
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"thinking": {"type": "off"}},
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"]
                tokens = len(prompt) // 4 + len(content) // 4
                return content, tokens
            if "base_resp" in data:
                status_msg = data["base_resp"].get("status_msg", "unknown")
                if "rate" in status_msg.lower() or "limit" in status_msg.lower():
                    import time
                    time.sleep(2 ** attempt)
                    continue
                return f"API Error: {status_msg}", 0
        except Exception as e:
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            return f"Error: {str(e)}", 0
    return "", 0


def extract_json(text: str) -> Optional[str]:
    """Extract JSON array or object from LLM response."""
    for fence in ["```json", "```", "```yaml", "```python"]:
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith(fence):
            text = text[:-len(fence)]
    text = text.strip()
    
    end_tag = text.find("</think>")
    if end_tag != -1:
        text = text[end_tag + len("</think>"):].strip()
    
    for tag in ["<result>", "</result>", "<output>", "</output>", "<response>", "</response>"]:
        start = text.find(tag)
        if start != -1:
            text = text[start + len(tag):]
            end = text.find(tag.replace("<", "</"))
            if end != -1:
                text = text[:end]
    
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            if arr_end - arr_start > obj_end - obj_start:
                return text[arr_start:arr_end+1]
        return text[arr_start:arr_end+1]
    
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        return text[obj_start:obj_end+1]
    return None


def repair_json(malformed_json: str) -> Optional[str]:
    """Trim malformed JSON back to the last parseable top-level prefix."""
    if not malformed_json:
        return None

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
        if c in '}]':
            depth += 1
        elif c in '{[':
            depth -= 1
        if depth == 0 and c in '}]':
            candidate = malformed_json[:i+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    last_brace = malformed_json.rfind('}')
    if last_brace > 0:
        candidate = malformed_json[:last_brace+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def extract_json_for_key(text: str, expected_key: str) -> Optional[str]:
    """Prefer a top-level object containing expected_key, then fall back generically."""
    if expected_key:
        key_pattern = f'"{expected_key}"'
        key_pos = text.find(key_pattern)
        if key_pos != -1:
            start = text.rfind("{", 0, key_pos)
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            return text[start:i+1]
    return extract_json(text)


# ----------------------------------------------------------------------------- 
# Agentic Reasoner
# ----------------------------------------------------------------------------- 

class AgenticReasoner:
    """
    Multi-step deliberation agent for keystroke sessionization.
    
    The agent uses a scratchpad to track its reasoning across passes:
    - Survey: understand the data landscape
    - Hypothesis: form initial session groupings  
    - Verification: probe ambiguous cases with tools
    - Commit: produce final sessions
    """
    
    # Gap threshold for hard session split (minutes)
    GAP_THRESHOLD_MINUTES = 30
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.minimax.io", model: str = "MiniMax-M2"):
        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY", "") or load_api_key()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # Scratchpad for deliberation tracking
        self._scratchpad: List[str] = []
        self._tool_calls: int = 0
        self._survey_tokens: int = 0
        self._deliberation_tokens: int = 0
        
        # State built during reasoning
        self._bursts: List[Burst] = []
        self._sorted_bursts: List[Burst] = []
        self._gap_analysis: Dict[str, Any] = {}
        self._session_hypotheses: List[Dict] = []
        self._external_context: Dict[str, Any] = {}
    
    # -------------------------------------------------------------------------
    # Tools (implemented as methods the agent calls via LLM prompts)
    # -------------------------------------------------------------------------
    
    def read_bursts(self, offset: int = 0, limit: int = 20) -> str:
        """Read raw burst text content from offset, returning up to limit bursts."""
        self._tool_calls += 1
        if not self._sorted_bursts:
            return "No bursts loaded."
        
        end = min(offset + limit, len(self._sorted_bursts))
        if offset >= len(self._sorted_bursts):
            return f"Offset {offset} beyond {len(self._sorted_bursts)} bursts."
        
        lines = []
        for i in range(offset, end):
            b = self._sorted_bursts[i]
            ts = b.timestamp[11:16] if len(b.timestamp) > 16 else b.timestamp
            preview = b.chars[:60].replace("\n", " ").replace('"', "'")
            lines.append(f"[{i}] {ts} | {b.app_name:<12} | {b.window_title[:35]:<35} | {preview}")
        
        return "\n".join(lines)
    
    def analyze_gaps(self, burst_indices: List[int] = None) -> str:
        """Return gap analysis — where are the 30+ min gaps?"""
        self._tool_calls += 1
        if not self._sorted_bursts:
            return "No bursts loaded."
        
        indices = burst_indices if burst_indices is not None else range(len(self._sorted_bursts))
        gaps = []
        prev_ts = None
        
        for i in indices:
            if i >= len(self._sorted_bursts):
                continue
            ts = self._sorted_bursts[i].timestamp
            if prev_ts:
                try:
                    t_prev = datetime.fromisoformat(prev_ts)
                    t_curr = datetime.fromisoformat(ts)
                    gap_min = int((t_curr - t_prev).total_seconds() / 60)
                    if gap_min >= self.GAP_THRESHOLD_MINUTES:
                        gaps.append(f"  Gap {gap_min}min before burst[{i}] at {ts[11:16]}")
                except:
                    pass
            prev_ts = ts
        
        if gaps:
            return f"Large gaps (>= {self.GAP_THRESHOLD_MINUTES} min):\n" + "\n".join(gaps)
        return f"No gaps >= {self.GAP_THRESHOLD_MINUTES} min found."
    
    def get_session_context(self, session_candidate: Dict) -> str:
        """Look at raw text content of a proposed session candidate."""
        self._tool_calls += 1
        burst_ids = session_candidate.get("burst_ids", [])
        if not burst_ids:
            return "Empty session candidate."
        
        lines = []
        for bid in burst_ids:
            if 0 <= bid < len(self._sorted_bursts):
                b = self._sorted_bursts[bid]
                ts = b.timestamp[11:16] if len(b.timestamp) > 16 else b.timestamp
                lines.append(f"[{bid}] {ts} {b.app_name} | {b.chars[:100].replace(chr(10), ' ')}")
        
        return "\n".join(lines)
    
    def scratchpad_write(self, text: str) -> str:
        """Write to scratchpad for reasoning."""
        self._tool_calls += 1
        self._scratchpad.append(text)
        return f"Scratchpad entry added (total: {len(self._scratchpad)} entries)"
    
    def scratchpad_read(self) -> str:
        """Read current scratchpad contents."""
        self._tool_calls += 1
        if not self._scratchpad:
            return "(scratchpad empty)"
        return "\n---\n".join(f"[Entry {i}] {t}" for i, t in enumerate(self._scratchpad))
    
    def suggest_split(self, burst_id: int, reason: str) -> str:
        """Propose a split point at the given burst_id."""
        self._tool_calls += 1
        if 0 <= burst_id < len(self._sorted_bursts):
            b = self._sorted_bursts[burst_id]
            ts = b.timestamp[11:16] if len(b.timestamp) > 16 else b.timestamp
            return f"SPLIT at burst[{burst_id}] {ts}: {reason}"
        return f"SPLIT at burst[{burst_id}]: {reason}"
    
    def suggest_merge(self, burst_ids: List[int], reason: str) -> str:
        """Propose merging the given burst IDs into one session."""
        self._tool_calls += 1
        ts_info = []
        for bid in burst_ids[:5]:
            if 0 <= bid < len(self._sorted_bursts):
                ts = self._sorted_bursts[bid].timestamp[11:16]
                ts_info.append(f"{bid}:{ts}")
        suffix = "..." if len(burst_ids) > 5 else ""
        return f"MERGE [{', '.join(ts_info)}{suffix}]: {reason}"
    
    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------
    
    def reason_day(self, bursts: List[Burst], external_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Full agentic reasoning pipeline."""
        if not bursts:
            return self._empty_result()
        
        self._bursts = bursts
        for i, b in enumerate(bursts):
            b.burst_id = i
        self._sorted_bursts = sorted(bursts, key=lambda b: b.timestamp)
        self._scratchpad = []
        self._tool_calls = 0
        self._survey_tokens = 0
        self._deliberation_tokens = 0
        self._session_hypotheses = []
        self._external_context = external_context or {}
        
        # --- PASS 1: SURVEY ---
        survey_result = self._survey_pass()
        
        # --- PASS 2: HYPOTHESIS ---
        hypothesis_result = self._hypothesis_pass(survey_result)
        
        # --- PASS 3: VERIFICATION ---
        verified_sessions = self._verification_pass(hypothesis_result)
        
        # --- PASS 4: COMMIT ---
        final_result = self._commit_pass(verified_sessions)
        
        final_result["survey_tokens"] = self._survey_tokens
        final_result["deliberation_tokens"] = self._deliberation_tokens
        final_result["total_tokens"] = self._survey_tokens + self._deliberation_tokens
        final_result["num_tool_calls"] = self._tool_calls
        final_result["external_context"] = self._external_context
        
        return final_result
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "sessions": [],
            "daily_summary": "No activity recorded.",
            "grouping_reasoning": "",
            "daily_narrative": "",
            "pass1_tokens_used": 0,
            "pass2_tokens_used": 0,
            "pass3_tokens_used": 0,
            "total_api_calls": 0,
            "survey_tokens": 0,
            "deliberation_tokens": 0,
            "total_tokens": 0,
            "num_tool_calls": 0,
            "external_context": {},
        }
    
    # -------------------------------------------------------------------------
    # PASS 1: SURVEY
    # -------------------------------------------------------------------------
    
    def _survey_pass(self) -> Dict[str, Any]:
        """Build an overview of the data: app counts, time range, gap distribution."""
        
        # Build compact burst overview for the LLM
        app_counts: Dict[str, int] = {}
        time_start = self._sorted_bursts[0].timestamp if self._sorted_bursts else ""
        time_end = self._sorted_bursts[-1].timestamp if self._sorted_bursts else ""
        total_chars = 0
        sources: Dict[str, int] = {}
        
        for b in self._sorted_bursts:
            app_counts[b.app_name] = app_counts.get(b.app_name, 0) + 1
            total_chars += b.char_count
            sources[b.source] = sources.get(b.source, 0) + 1
        
        # Gap analysis
        gaps = []
        prev_ts = None
        for i, b in enumerate(self._sorted_bursts):
            if prev_ts:
                try:
                    t_prev = datetime.fromisoformat(prev_ts)
                    t_curr = datetime.fromisoformat(b.timestamp)
                    gap_min = int((t_curr - t_prev).total_seconds() / 60)
                    if gap_min >= 5:
                        gaps.append((i, gap_min, b.timestamp[11:16]))
                except:
                    pass
            prev_ts = b.timestamp
        
        self._gap_analysis = {
            "gaps": gaps,
            "large_gaps": [(i, g, t) for i, g, t in gaps if g >= self.GAP_THRESHOLD_MINUTES],
        }
        
        # Build annotated burst list for survey prompt
        annotated = []
        for i, b in enumerate(self._sorted_bursts):
            ts = b.timestamp[11:16]
            window = b.window_title.split(' - ')[-1] if ' - ' in b.window_title else b.window_title
            preview = b.chars[:80].replace("\n", " ").replace('"', "'")
            
            # Check for large gap before this burst
            gap_note = ""
            if i > 0:
                try:
                    t_prev = datetime.fromisoformat(self._sorted_bursts[i-1].timestamp)
                    t_curr = datetime.fromisoformat(b.timestamp)
                    gap_min = int((t_curr - t_prev).total_seconds() / 60)
                    if gap_min >= self.GAP_THRESHOLD_MINUTES:
                        gap_note = f" ← GAP {gap_min}min"
                    elif gap_min >= 5:
                        gap_note = f" ({gap_min}min)"
                except:
                    pass
            
            annotated.append(f"[{i}] {ts} | {b.app_name:<12} | {window[:30]:<30} |{gap_note} {preview}")
        
        overview = {
            "num_bursts": len(self._sorted_bursts),
            "time_range": f"{time_start[11:16] if time_start else '?'} - {time_end[11:16] if time_end else '?'}",
            "app_counts": app_counts,
            "total_chars": total_chars,
            "sources": sources,
            "large_gaps": [(i, g, t) for i, g, t in gaps if g >= self.GAP_THRESHOLD_MINUTES],
            "all_gaps_5min_plus": [(i, g, t) for i, g, t in gaps],
            "annotated_bursts": annotated,
            "external_context": self._external_context,
        }
        
        # Write survey to scratchpad
        survey_summary = (
            f"SURVEY: {len(self._sorted_bursts)} bursts, "
            f"{time_start[11:16] if time_start else '?'}-{time_end[11:16] if time_end else '?'}, "
            f"apps={app_counts}, large_gaps={len(self._gap_analysis['large_gaps'])}"
        )
        self.scratchpad_write(survey_summary)
        if self._external_context:
            self.scratchpad_write(
                "EXTERNAL: supplemental same-day context loaded from "
                f"GitHub/Hermes for {self._external_context.get('date', 'unknown date')}"
            )
        
        return overview
    
    # -------------------------------------------------------------------------
    # PASS 2: HYPOTHESIS
    # -------------------------------------------------------------------------
    
    def _hypothesis_pass(self, survey: Dict[str, Any]) -> List[Dict]:
        """Form initial session hypotheses based on survey patterns."""
        
        # If no API key, fall back to heuristic hypothesis formation
        if not self.api_key:
            return self._heuristic_hypothesis(survey)
        
        # Build the hypothesis prompt
        gap_lines = []
        for i, gap_min, ts in survey["all_gaps_5min_plus"][:15]:
            gap_lines.append(f"  burst[{i}] at {ts}: {gap_min}min gap")
        
        burst_lines = []
        for line in survey["annotated_bursts"][:40]:
            burst_lines.append("  " + line)
        
        prompt = f"""You are sessionizing a developer's keystroke bursts using deliberate multi-step reasoning.

CRITICAL RULES:
1. 30+ MINUTE GAP = HARD SPLIT. Nothing survives a 30+ min gap.
2. MEETINGS (Zoom, Meet, Teams, standup, sprint planning) = SPLIT even if brief
3. CODE REVIEW READING ≠ CODE BEING REVIEWED = SPLIT  
4. SAME TASK across app switches = MERGE (debugging that moves Chrome→VS Code→Chrome)
5. QUICK COORDINATION (< 5 min, e.g., Slack ping) during work = MERGE into surrounding session
6. DIFFERENT LOGICAL TASKS in same app = SPLIT (e.g., userService write → paymentService debug)
7. When uncertain = SPLIT. More clean sessions beat fewer messy ones.

SURVEY DATA:
- Total bursts: {survey['num_bursts']}
- Time range: {survey['time_range']}
- App distribution: {survey['app_counts']}
- Total chars: {survey['total_chars']}
- Sources: {survey['sources']}

GAPS (5+ minutes, potential session boundaries):
{chr(10).join(gap_lines[:20]) if gap_lines else "(no significant gaps)"}

BURSTS:
{chr(10).join(burst_lines)}

Your task: Form initial session HYPOTHESES. For each proposed session, provide:
- burst_ids: which burst indices belong together
- start_time, end_time
- grouping_reasoning: why these bursts form a session
- confidence: high/medium/low

Respond ONLY with JSON: {{"hypotheses": [...]}}
Every burst 0 to {survey['num_bursts']-1} must appear in exactly one hypothesis."""
        
        response, tokens = make_api_call(prompt, self.api_key, self.model, self.base_url, temperature=0.2, max_tokens=8000)
        self._survey_tokens += tokens
        self._deliberation_tokens += tokens
        
        json_str = extract_json(response)
        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "hypotheses" in data:
                    self._session_hypotheses = data["hypotheses"]
                    self.scratchpad_write(f"HYPOTHESIS: {len(self._session_hypotheses)} sessions proposed")
                    return self._session_hypotheses
            except json.JSONDecodeError:
                pass
        
        # Fallback to heuristic
        self.scratchpad_write("HYPOTHESIS: API failed, falling back to heuristic")
        return self._heuristic_hypothesis(survey)
    
    def _heuristic_hypothesis(self, survey: Dict[str, Any]) -> List[Dict]:
        """Fallback: use simple heuristics to form session hypotheses."""
        hypotheses = []
        current_group = []
        current_start = None
        current_app = None
        current_window = None
        
        large_gaps = set(i for i, g, t in self._gap_analysis.get("large_gaps", []))
        
        def end_current_group():
            """Close current group and add hypothesis."""
            nonlocal current_group, current_start, hypotheses
            if current_group:
                end_ts = self._sorted_bursts[current_group[-1]].timestamp[11:16]
                hypotheses.append(self._build_hypothesis(current_group, current_start, end_ts))
                current_group = []
                current_start = None
        
        for i, b in enumerate(self._sorted_bursts):
            ts = b.timestamp[11:16] if len(b.timestamp) > 16 else b.timestamp
            
            # Hard split on 30+ min gap
            if i in large_gaps:
                end_current_group()
                # Start new group
                current_group = [i]
                current_start = ts
                current_app = b.app_name
                current_window = b.window_title
                continue
            
            # New group if empty
            if not current_group:
                current_group = [i]
                current_start = ts
                current_app = b.app_name
                current_window = b.window_title
                continue
            
            # Check if this burst continues the current session
            # Rule 4: Same task = same session even with app switches if same logical work
            # Simplification: same window_title = same session
            same_window = (b.window_title == current_window)
            same_app = (b.app_name == current_app)
            
            # Brief coordination (Slack/Discord) during work → merge
            if b.app_name in ("Slack", "Discord") and current_app in ("Slack", "Discord", "VS Code", "Codex"):
                # Check gap - if brief, merge
                try:
                    t_prev = datetime.fromisoformat(self._sorted_bursts[i-1].timestamp)
                    t_curr = datetime.fromisoformat(b.timestamp)
                    gap = int((t_curr - t_prev).total_seconds() / 60)
                    if gap < 5:
                        current_group.append(i)
                        continue
                except:
                    pass
            
            if same_window:
                # Same window → definitely same session
                current_group.append(i)
            elif same_app and current_app in ("Chrome", "VS Code", "Codex"):
                # App switch but same primary app type (e.g., Chrome↔VS Code while debugging)
                # Check if gap is small enough to be continuous work
                try:
                    t_prev = datetime.fromisoformat(self._sorted_bursts[i-1].timestamp)
                    t_curr = datetime.fromisoformat(b.timestamp)
                    gap = int((t_curr - t_prev).total_seconds() / 60)
                    if gap < 10:  # 10 min threshold for same-app continuity
                        current_group.append(i)
                        current_window = b.window_title  # update window
                        continue
                except:
                    pass
                
                # Different window, different app context → SPLIT
                end_current_group()
                current_group = [i]
                current_start = ts
                current_app = b.app_name
                current_window = b.window_title
            else:
                # Different app entirely → SPLIT
                end_current_group()
                current_group = [i]
                current_start = ts
                current_app = b.app_name
                current_window = b.window_title
        
        # Don't forget last group
        end_current_group()
        
        # Re-index sessions
        for idx, h in enumerate(hypotheses):
            h["session_id"] = idx + 1
        
        self._session_hypotheses = hypotheses
        self.scratchpad_write(f"HEURISTIC HYPOTHESIS: {len(hypotheses)} sessions formed")
        return hypotheses
    
    def _build_hypothesis(self, burst_ids: List[int], start_time: str, end_time: str) -> Dict:
        """Build a hypothesis dict from a group of burst IDs."""
        if not burst_ids:
            return {}
        
        # Determine dominant app and window
        app_counts: Dict[str, int] = {}
        window_counts: Dict[str, int] = {}
        for bid in burst_ids:
            if 0 <= bid < len(self._sorted_bursts):
                b = self._sorted_bursts[bid]
                app_counts[b.app_name] = app_counts.get(b.app_name, 0) + 1
                window_counts[b.window_title] = window_counts.get(b.window_title, 0) + 1
        
        dominant_app = max(app_counts, key=app_counts.get) if app_counts else "Unknown"
        dominant_window = max(window_counts, key=window_counts.get) if window_counts else ""
        
        reasoning = f"{dominant_app} work"
        if len(burst_ids) > 1:
            reasoning += f" ({len(burst_ids)} bursts)"
        
        return {
            "session_id": 0,  # Will be re-indexed
            "burst_ids": burst_ids,
            "start_time": start_time,
            "end_time": end_time,
            "app_name": dominant_app,
            "window_title": dominant_window,
            "grouping_reasoning": reasoning,
            "confidence": "high" if len(burst_ids) >= 2 else "medium",
        }

    def _hypothesis_pass(self, survey: Dict[str, Any]) -> List[Dict]:
        """Form semantic session hypotheses using the full burst list."""
        if not self.api_key:
            raise RuntimeError("AgenticReasoner requires MINIMAX_API_KEY; heuristic mode is disabled.")

        gap_lines = [
            f"  burst[{i}] at {ts}: {gap_min}min gap"
            for i, gap_min, ts in survey["all_gaps_5min_plus"][:30]
        ]
        burst_lines = ["  " + line for line in survey["annotated_bursts"]]

        prompt = f"""You are sessionizing a developer's keystroke bursts into SEMANTIC WORK SESSIONS.

The goal is not app-by-app segmentation. The goal is to identify the underlying task, project, bug, deliverable, or conversation thread.

CRITICAL RULES:
1. 30+ MINUTE GAP = HARD SPLIT. Nothing survives a 30+ minute gap.
2. SAME UNDERLYING TASK across app switches = SAME SESSION.
3. A session MAY contain NON-CONSECUTIVE burst_ids if the user briefly detours and then returns to the same work.
4. Sessions are WORK-BASED, not TIME-BASED. Multiple sessions may overlap in clock time if the user is multitasking and bouncing between them.
5. Browser research + ChatGPT prompting + screenshots + Save As dialogs for the same artifact/design = ONE session.
6. Discord + terminal + browser + agent chat while investigating one issue = ONE session.
7. Unrelated browsing, watching YouTube, casual exploration, or reading updates should be a SEPARATE session even if it happens during the same time span as technical work.
8. Do not merge an incidental-but-different thread just because it is nearby in time. Keep "Hermes reset / infra debugging" separate from "casual browsing / YouTube watching" unless the browsing was directly supporting that debugging task.
9. Screenshotting, saving files, and quick file management usually belong to the surrounding task that produced those artifacts.
10. Meetings/calls are their own sessions. Code review reading is separate from implementation work.
11. Different logical tasks in the same app must split.
12. Prefer semantic workstream grouping over window boundaries.

Negative examples:
- Terminal gateway debugging + a separate "codex update" search = TWO sessions unless the update search was clearly part of the gateway fix.
- Infrastructure debugging + watching a YouTube video in a browser tab = TWO sessions unless the video was directly used as a reference for the debugging.
- Discord coordination about frontend automation + separate agent/infrastructure debugging = TWO sessions even if both happen in the same few minutes.
- "web", "codex update", general browsing, Gmail, or YouTube should not be folded into a bugfix session unless the burst text or window title clearly shows they were used to solve that bug.

SURVEY DATA:
- Total bursts: {survey['num_bursts']}
- Time range: {survey['time_range']}
- App distribution: {survey['app_counts']}
- Total chars: {survey['total_chars']}
- Sources: {survey['sources']}

EXTERNAL SAME-DAY CONTEXT:
{format_external_context_for_prompt(survey.get("external_context"))}

GAPS (5+ minutes):
{chr(10).join(gap_lines) if gap_lines else "(no significant gaps)"}

ALL BURSTS:
{chr(10).join(burst_lines)}

Return hypotheses that cover every burst id exactly once.

Important:
- start_time/end_time are just the outer clock bounds of that workstream and MAY overlap with other sessions.
- It is completely valid for two sessions to interleave in time.
- Optimize for "what distinct threads of work were active?" not "what contiguous block of minutes happened?"

Each hypothesis must include:
- session_id
- burst_ids
- start_time
- end_time
- topic: short semantic task name
- supporting_apps: list of apps materially involved
- grouping_reasoning: 1 sentence explaining the common underlying work
- confidence: high/medium/low

Example topics:
- "Inspire 5 patent poster design"
- "Hermes agent Docker gateway debugging"

Respond ONLY with JSON in this exact shape:
{{"hypotheses": [
  {{
    "session_id": 1,
    "burst_ids": [0, 1, 2],
    "start_time": "09:39",
    "end_time": "10:10",
    "topic": "Inspire 5 patent poster design",
    "supporting_apps": ["comet", "ChatGPT", "SnippingTool", "PickerHost"],
    "grouping_reasoning": "Research, prompt iteration, screenshots, and saved artifacts all supported the same patent poster design work.",
    "confidence": "high"
  }}
]}}"""

        response, tokens = make_api_call(
            prompt,
            self.api_key,
            self.model,
            self.base_url,
            temperature=0.2,
            max_tokens=8000,
        )
        self._survey_tokens += tokens
        self._deliberation_tokens += tokens

        json_str = extract_json_for_key(response, "hypotheses")
        if not json_str:
            repair_prompt = f"""{prompt}

Your previous response did not contain parseable JSON.
Return the same answer again, but this time respond with JSON only in the exact required shape and no extra text."""
            response, retry_tokens = make_api_call(
                repair_prompt,
                self.api_key,
                self.model,
                self.base_url,
                temperature=0.0,
                max_tokens=8000,
            )
            self._survey_tokens += retry_tokens
            self._deliberation_tokens += retry_tokens
            json_str = extract_json_for_key(response, "hypotheses")
            if not json_str:
                raise RuntimeError(f"Hypothesis pass returned no parseable JSON. Response preview: {response[:600]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = repair_json(json_str)
            if not repaired:
                raise RuntimeError(f"Hypothesis JSON was malformed and unrecoverable. Response preview: {response[:600]}")
            data = json.loads(repaired)

        hypotheses = data.get("hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise RuntimeError(f"Hypothesis pass returned invalid hypotheses payload: {str(data)[:600]}")

        self._session_hypotheses = hypotheses
        self.scratchpad_write(f"HYPOTHESIS: {len(self._session_hypotheses)} semantic sessions proposed")
        return self._session_hypotheses
    
    # -------------------------------------------------------------------------
    # PASS 3: VERIFICATION
    # -------------------------------------------------------------------------
    
    def _verification_pass(self, hypotheses: List[Dict]) -> List[Dict]:
        """Verify ambiguous hypotheses by checking raw burst content."""
        if not self.api_key:
            return hypotheses
        
        # Find potentially ambiguous hypotheses (single-burst, or large sessions)
        ambiguous = []
        for h in hypotheses:
            if len(h.get("burst_ids", [])) == 1:
                ambiguous.append(h)
            elif len(h.get("burst_ids", [])) > 6:
                ambiguous.append(h)
        
        if not ambiguous:
            self.scratchpad_write("VERIFICATION: No ambiguous cases, skipping deep verification")
            return hypotheses
        
        # Check content of ambiguous sessions
        verified_changes = 0
        for h in ambiguous[:5]:  # Check up to 5
            context = self.get_session_context(h)
            burst_id = h["burst_ids"][0]
            
            if 0 <= burst_id < len(self._sorted_bursts):
                b = self._sorted_bursts[burst_id]
                text_lower = b.chars.lower()
                
                # Check if it looks like a meeting/communication
                meeting_indicators = ["zoom", "meeting", "standup", "stand-up", "sprint", "sync", "call", "team"]
                if any(ind in text_lower for ind in meeting_indicators):
                    self.scratchpad_write(f"VERIFY: burst[{burst_id}] appears to be meeting/communication")
                    h["confidence"] = "medium"
                    h["grouping_reasoning"] += " [verified: appears to be communication]"
                    verified_changes += 1
                
                # Check for code review reading
                review_indicators = ["review", "pr #", "pull request", "code review", "lgtm", "looks good"]
                if any(ind in text_lower for ind in review_indicators):
                    self.scratchpad_write(f"VERIFY: burst[{burst_id}] appears to be code review reading")
                    h["confidence"] = "medium"
                    verified_changes += 1
        
        self.scratchpad_write(f"VERIFICATION: checked {min(5, len(ambiguous))} ambiguous cases, {verified_changes} updates")
        return hypotheses
    
    # -------------------------------------------------------------------------
    # PASS 4: COMMIT
    # -------------------------------------------------------------------------
    
    def _commit_pass(self, hypotheses: List[Dict]) -> Dict[str, Any]:
        """Commit final sessions, produce summary reasoning."""
        
        # Final consolidation: merge single-burst low-confidence sessions
        # with neighbors if gap is small
        final_sessions = self._consolidate_sessions(hypotheses)
        if self.api_key and final_sessions:
            try:
                final_sessions = self._refine_session_outliers(final_sessions)
            except Exception as e:
                self.scratchpad_write(f"REFINE WARNING: {str(e)}")
        
        # Build output — enrich with app_name, window_title from bursts
        sessions_output = []
        for h in final_sessions:
            burst_ids = h.get("burst_ids", [])
            # Determine dominant app from bursts
            app_counts: Dict[str, int] = {}
            window_counts: Dict[str, int] = {}
            for bid in burst_ids:
                if 0 <= bid < len(self._sorted_bursts):
                    b = self._sorted_bursts[bid]
                    app_counts[b.app_name] = app_counts.get(b.app_name, 0) + 1
                    window_counts[b.window_title] = window_counts.get(b.window_title, 0) + 1
            dominant_app = max(app_counts, key=app_counts.get) if app_counts else "Unknown"
            dominant_window = max(window_counts, key=window_counts.get) if window_counts else ""
            sessions_output.append({
                "session_id": h.get("session_id", 0),
                "burst_ids": burst_ids,
                "start_time": h.get("start_time", "00:00"),
                "end_time": h.get("end_time", "00:00"),
                "topic": h.get("topic", ""),
                "supporting_apps": h.get("supporting_apps", []),
                "app_name": h.get("app_name") or dominant_app,
                "window_title": h.get("window_title") or dominant_window,
                "grouping_reasoning": h.get("grouping_reasoning", ""),
                "confidence": h.get("confidence", "medium"),
            })

        if self.api_key and sessions_output:
            sessions_output = self._enrich_sessions(sessions_output)

        sessions_output = self._link_related_sessions(sessions_output)
        
        # Generate daily summary if we have API key
        daily_summary = ""
        daily_narrative = ""
        if self.api_key and sessions_output:
            daily_summary, daily_narrative = self._generate_summary(sessions_output)
        
        self.scratchpad_write(f"COMMIT: {len(final_sessions)} final sessions")
        
        return {
            "sessions": sessions_output,
            "daily_summary": daily_summary,
            "grouping_reasoning": "\n".join(self._scratchpad[-5:]),
            "daily_narrative": daily_narrative,
            "pass1_tokens_used": self._survey_tokens,
            "pass2_tokens_used": 0,
            "pass3_tokens_used": 0,
            "total_api_calls": 3,
        }
    
    def _consolidate_sessions(self, hypotheses: List[Dict]) -> List[Dict]:
        """Consolidate: merge tiny orphan sessions, fix gaps, finalize."""
        if not hypotheses:
            return []
        
        # Sort by burst_ids[0]
        sorted_h = sorted(hypotheses, key=lambda h: h.get("burst_ids", [9999])[0])
        
        # Merge adjacent sessions that are very short (single burst) with small time gap
        # and same app, into surrounding sessions
        consolidated = []
        skip_next = 0
        
        for i, h in enumerate(sorted_h):
            if skip_next > 0:
                skip_next -= 1
                continue
            
            burst_ids = h.get("burst_ids", [])
            
            # Single burst, low confidence: try to merge with prev or next
            if len(burst_ids) == 1 and h.get("confidence") == "low":
                merged = False
                
                # Try to merge with previous
                if consolidated:
                    prev = consolidated[-1]
                    prev_bursts = prev.get("burst_ids", [])
                    if prev_bursts:
                        last_prev_ts = self._sorted_bursts[prev_bursts[-1]].timestamp
                        curr_ts = self._sorted_bursts[burst_ids[0]].timestamp
                        try:
                            gap = (datetime.fromisoformat(curr_ts) - 
                                   datetime.fromisoformat(last_prev_ts)).total_seconds() / 60
                            if 0 < gap < 15:  # Small gap, merge
                                prev["burst_ids"] = prev_bursts + burst_ids
                                prev["end_time"] = h.get("end_time", prev["end_time"])
                                prev["confidence"] = "medium"
                                self.scratchpad_write(f"MERGE: single-burst session into previous")
                                merged = True
                        except:
                            pass
                
                # Try to merge with next
                if not merged and i + 1 < len(sorted_h):
                    next_h = sorted_h[i + 1]
                    next_bursts = next_h.get("burst_ids", [])
                    if next_bursts:
                        curr_ts = self._sorted_bursts[burst_ids[-1]].timestamp
                        first_next_ts = self._sorted_bursts[next_bursts[0]].timestamp
                        try:
                            gap = (datetime.fromisoformat(first_next_ts) - 
                                   datetime.fromisoformat(curr_ts)).total_seconds() / 60
                            if 0 < gap < 15:  # Small gap, merge
                                h["burst_ids"] = burst_ids + next_bursts
                                h["end_time"] = next_h.get("end_time", h["end_time"])
                                h["confidence"] = "medium"
                                self.scratchpad_write(f"MERGE: single-burst session into next")
                                skip_next = 1
                                merged = True
                        except:
                            pass
                
                if merged:
                    consolidated[-1] = h if skip_next == 0 else consolidated[-1]
                    if skip_next == 0:
                        continue
                    else:
                        consolidated[-1] = h
                        continue
            
            consolidated.append(h)
        
        # Re-number session IDs
        for idx, h in enumerate(consolidated):
            h["session_id"] = idx + 1
        
        return consolidated

    def _enrich_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Extract concrete session details from burst text and window context."""
        if not sessions:
            return sessions

        blocks = []
        for session in sessions:
            burst_ids = session.get("burst_ids", [])
            lines = []
            window_titles = []
            for bid in burst_ids[:40]:
                if 0 <= bid < len(self._sorted_bursts):
                    b = self._sorted_bursts[bid]
                    window_titles.append(b.window_title)
                    preview = b.chars[:180].replace("\n", " ")
                    lines.append(
                        f"[{bid}] {b.timestamp[11:16]} | {b.app_name} | {b.window_title[:80]} | {preview}"
                    )

            blocks.append({
                "session_id": session.get("session_id", 0),
                "topic": session.get("topic", ""),
                "time": f"{session.get('start_time', '?')}-{session.get('end_time', '?')}",
                "supporting_apps": session.get("supporting_apps", []),
                "window_titles": window_titles[:20],
                "bursts": lines,
            })

        prompt = f"""You are enriching already-grouped developer work sessions.

Your job is to write USEFUL session narratives for future retrieval and debugging, not generic summaries.

For each session, return:
- session_id
- narrative

The narrative should usually be one dense paragraph, and may be two paragraphs if the session has a lot of important detail.
It should be useful later for answering questions like:
- how was a bug fixed or investigated?
- what type of work happened?
- what important details or findings came up?
- what did we try?
- what did we discard?
- what references, searches, files, or tools mattered?

Rules:
- Be specific and evidence-based from the session data.
- Do not just restate "the user worked on X".
- Include concrete searches, prompts, source documents, files, saved outputs, failures, mismatches, revisions, and conclusions when present.
- Mention design/style direction only when it materially mattered to the work.
- Prefer concrete nouns and short factual clauses over vague abstractions.
- Write for future querying and recall, not for elegance.

SESSION DATA:
{json.dumps(blocks, indent=2)}

Respond ONLY with JSON:
{{"sessions": [
  {{
    "session_id": 1,
    "narrative": "..."
  }}
]}}"""

        response, tokens = make_api_call(
            prompt,
            self.api_key,
            self.model,
            self.base_url,
            temperature=0.2,
            max_tokens=5000,
        )
        self._deliberation_tokens += tokens

        json_str = extract_json_for_key(response, "sessions")
        if not json_str:
            raise RuntimeError(f"Session enrichment returned no parseable JSON. Response preview: {response[:600]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = repair_json(json_str)
            if not repaired:
                raise RuntimeError(f"Session enrichment JSON was malformed and unrecoverable. Response preview: {response[:600]}")
            data = json.loads(repaired)

        enriched = data.get("sessions")
        if not isinstance(enriched, list):
            raise RuntimeError(f"Session enrichment returned invalid payload: {str(data)[:600]}")

        by_id = {item.get("session_id"): item for item in enriched if isinstance(item, dict)}
        merged = []
        for session in sessions:
            details = by_id.get(session.get("session_id"), {})
            merged.append({
                **session,
                "narrative": details.get("narrative", ""),
            })

        self.scratchpad_write(f"ENRICH: extracted details for {len(merged)} sessions")
        return merged

    def _link_related_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Add explicit links between sessions that appear to continue the same work."""
        if not sessions:
            return sessions

        def normalize_topic(topic: str) -> set[str]:
            words = re.findall(r"[a-z0-9]+", (topic or "").lower())
            stop = {
                "and", "the", "for", "with", "from", "into", "that", "this", "work",
                "design", "creation", "research", "refinement", "debugging", "session",
                "casual", "activity", "discussion", "coordination", "discord", "codex",
                "comet", "chatgpt", "vision", "frontend",
            }
            return {w for w in words if w not in stop and len(w) > 2}

        def parse_minutes(ts: str) -> int:
            try:
                hh, mm = ts.split(":")
                return int(hh) * 60 + int(mm)
            except Exception:
                return -10**9

        linked = []
        prior = []
        for session in sessions:
            topic_terms = normalize_topic(session.get("topic", ""))
            app_set = set(session.get("supporting_apps", []))
            related = []
            for prev in prior:
                prev_terms = normalize_topic(prev.get("topic", ""))
                if not topic_terms or not prev_terms:
                    continue
                overlap = len(topic_terms & prev_terms)
                shared_apps = len(app_set & set(prev.get("supporting_apps", [])))
                gap_minutes = parse_minutes(session.get("start_time", "00:00")) - parse_minutes(prev.get("end_time", "00:00"))
                if overlap >= 2:
                    related.append({
                        "session_id": prev.get("session_id"),
                        "relation": "continues_same_topic",
                        "gap_minutes": max(0, gap_minutes),
                        "topic": prev.get("topic", ""),
                    })

            session_copy = dict(session)
            session_copy["related_sessions"] = related
            linked.append(session_copy)
            prior.append(session_copy)

        if any(s.get("related_sessions") for s in linked):
            self.scratchpad_write(
                f"LINK: added related-session links for {sum(1 for s in linked if s.get('related_sessions'))} sessions"
            )
        return linked

    def _refine_session_outliers(self, sessions: List[Dict]) -> List[Dict]:
        """Ask the model to make only surgical fixes for obvious outlier bursts."""
        if not sessions:
            return sessions

        session_blocks = []
        for session in sessions:
            burst_lines = []
            for bid in session.get("burst_ids", []):
                if 0 <= bid < len(self._sorted_bursts):
                    b = self._sorted_bursts[bid]
                    preview = b.chars[:140].replace("\n", " ")
                    burst_lines.append(
                        f"[{bid}] {b.timestamp[11:16]} | {b.app_name} | {b.window_title[:70]} | {preview}"
                    )
            session_blocks.append({
                "session_id": session.get("session_id", 0),
                "topic": session.get("topic", ""),
                "start_time": session.get("start_time", ""),
                "end_time": session.get("end_time", ""),
                "supporting_apps": session.get("supporting_apps", []),
                "grouping_reasoning": session.get("grouping_reasoning", ""),
                "bursts": burst_lines,
            })

        prompt = f"""You are doing a SURGICAL outlier-refinement pass on already-grouped work sessions.

Goal:
- Keep the existing sessions unless there is a clear mismatch.
- Fix only obvious outliers: bursts that clearly belong in a different session or should be split into a small separate session.

Important rules:
- Do NOT redo sessionization from scratch.
- Preserve the current topics whenever possible.
- A session is work-based, not time-based, and resumed work may stay in the same session.
- But bursts from unrelated Discord chatter, YouTube watching, casual browsing, or random searches should be removed from a technical/design session if they are clearly not supporting that session.
- Browser research that directly supports a design/debugging session should stay in that session.
- Lock screen unlocks, startup search box activity, accidental keystrokes, task switching noise, and unrelated Discord messages should NOT stay inside a focused design/debugging session.
- If a burst is weakly related or ambiguous, prefer putting it in a small incidental/noise/coordination session rather than polluting the main work session.
- Discord belongs in a design/debugging session only when the message content clearly supports that same work thread.
- SearchHost or explorer belongs in a work session only when it is clearly part of that work, not generic startup/task switching behavior.
- Return every burst exactly once across all refined sessions.
- Only make changes when the mismatch is clear from app, window title, and text.

Negative examples:
- LockApp + search box query "chat" should not be part of an Inspire patent-design session.
- Discord chatter in the middle of patent-image work should be split out unless it clearly discusses that same patent-image task.
- Random system/input noise should not remain inside a focused research or design session.

Current sessions:
{json.dumps(session_blocks, indent=2)}

Respond ONLY with JSON in this exact shape:
{{"sessions": [
  {{
    "session_id": 1,
    "topic": "existing or slightly improved topic",
    "burst_ids": [0, 1, 2],
    "start_time": "09:39",
    "end_time": "10:10",
    "supporting_apps": ["comet", "ChatGPT"],
    "grouping_reasoning": "short reason"
  }}
]}}"""

        all_original = sorted(bid for s in sessions for bid in s.get("burst_ids", []))

        def parse_refinement_response(raw_response: str) -> List[Dict]:
            json_str = extract_json_for_key(raw_response, "sessions")
            if not json_str:
                raise RuntimeError(f"Outlier refinement returned no parseable JSON. Response preview: {raw_response[:600]}")
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                repaired = repair_json(json_str)
                if not repaired:
                    raise RuntimeError(f"Outlier refinement JSON was malformed and unrecoverable. Response preview: {raw_response[:600]}")
                data = json.loads(repaired)
            refined_local = data.get("sessions")
            if not isinstance(refined_local, list) or not refined_local:
                raise RuntimeError(f"Outlier refinement returned invalid payload: {str(data)[:600]}")
            return refined_local

        def coverage_error(refined_local: List[Dict]) -> Optional[str]:
            refined_ids = [bid for s in refined_local for bid in s.get("burst_ids", [])]
            all_refined = sorted(refined_ids)
            if all_original == all_refined:
                return None
            original_set = set(all_original)
            refined_set = set(all_refined)
            missing = sorted(original_set - refined_set)
            extras = sorted(refined_set - original_set)
            duplicates = sorted(bid for bid in set(refined_ids) if refined_ids.count(bid) > 1)
            return (
                "Coverage mismatch. "
                f"Missing burst_ids: {missing}. "
                f"Duplicate burst_ids: {duplicates}. "
                f"Extra/invalid burst_ids: {extras}. "
                "Return corrected sessions with every original burst id exactly once."
            )

        response, tokens = make_api_call(
            prompt,
            self.api_key,
            self.model,
            self.base_url,
            temperature=0.1,
            max_tokens=6000,
        )
        self._deliberation_tokens += tokens
        refined = parse_refinement_response(response)

        error = coverage_error(refined)
        attempts = 0
        while error and attempts < 2:
            repair_prompt = f"""{prompt}

Your previous refinement response was structurally invalid.
{error}

Do not explain. Return corrected JSON only, preserving the same intent but fixing burst coverage exactly."""
            response, tokens = make_api_call(
                repair_prompt,
                self.api_key,
                self.model,
                self.base_url,
                temperature=0.0,
                max_tokens=6000,
            )
            self._deliberation_tokens += tokens
            refined = parse_refinement_response(response)
            error = coverage_error(refined)
            attempts += 1

        if error:
            raise RuntimeError(f"Outlier refinement could not repair coverage after retry loop. {error}")

        for idx, s in enumerate(refined):
            s["session_id"] = idx + 1

        self.scratchpad_write(f"REFINE: adjusted sessions with surgical outlier pass ({len(refined)} sessions)")
        return refined
    
    def _generate_summary(self, sessions: List[Dict]) -> tuple:
        """Generate daily summary using LLM."""
        if not sessions:
            return "", ""
        
        session_summaries = []
        for s in sessions:
            burst_ids = s.get("burst_ids", [])
            chars = sum(self._sorted_bursts[i].char_count 
                       for i in burst_ids if 0 <= i < len(self._sorted_bursts))
            session_summaries.append({
                "id": s.get("session_id", 0),
                "time": f"{s.get('start_time', '?')}-{s.get('end_time', '?')}",
                "topic": s.get("topic", ""),
                "app": s.get("app_name", "Unknown"),
                "bursts": len(burst_ids),
                "chars": chars,
                "narrative": s.get("narrative", ""),
                "reasoning": s.get("grouping_reasoning", ""),
                "confidence": s.get("confidence", "medium"),
            })
        
        prompt = f"""Write a daily recall note from these keystroke-derived work sessions.

This note is for future use by the same person doing the work, or by a future agent helping that same person.
Do NOT write like an outside observer describing "the developer."
Do NOT evaluate productivity or "day quality."

{len(sessions)} sessions detected:
{json.dumps(session_summaries, indent=2)}

EXTERNAL SAME-DAY CONTEXT:
{format_external_context_for_prompt(self._external_context)}

Write:
- `daily_summary`: one dense paragraph that helps future recall of what actually happened, what mattered, what was tried, what was learned, and what changed
- `patterns`: one dense paragraph covering cross-session context that will help later, such as interruptions, resumed threads, unresolved issues, discarded approaches, repeated bugs, decisions made, or things to revisit

Style rules:
- Prefer direct factual phrasing over narration about a person
- Avoid phrases like "the developer did..." or "the session reveals..."
- Include specific tools, bugs, files, searches, commands, GitHub activity, Hermes agent activity, and decisions when useful
- Optimize for future querying and handoff, not elegance

Respond with JSON: {{"daily_summary": "...", "patterns": "..."}}
Respond ONLY with JSON."""
        
        response, tokens = make_api_call(prompt, self.api_key, self.model, self.base_url, temperature=0.3, max_tokens=1000)
        self._deliberation_tokens += tokens
        
        json_str = extract_json_for_key(response, "daily_summary")
        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    return (
                        data.get("daily_summary", ""),
                        data.get("patterns", ""),
                    )
            except json.JSONDecodeError:
                repaired = repair_json(json_str)
                if repaired:
                    try:
                        data = json.loads(repaired)
                        if isinstance(data, list) and data:
                            data = data[0]
                        if isinstance(data, dict):
                            return (
                                data.get("daily_summary", ""),
                                data.get("patterns", ""),
                            )
                    except json.JSONDecodeError:
                        pass
        
        return f"{len(sessions)} sessions detected.", ""


# ----------------------------------------------------------------------------- 
# Convenience wrapper matching IterativeReasoner output shape
# ----------------------------------------------------------------------------- 

@dataclass 
class AgenticSession:
    """Matches LLMReasonedSession fields for compatibility."""
    session_id: int
    burst_ids: List[int]
    start_time: str
    end_time: str
    app_name: str = "Unknown"
    window_title: str = ""
    total_chars: int = 0
    total_duration_seconds: int = 0
    work_type: str = "writing"
    topic: str = ""
    narrative: str = ""
    grouping_reasoning: str = ""
    confidence: str = "medium"


def reason_day(
    bursts: List[Burst],
    api_key: str = None,
    external_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function matching IterativeReasoner.reason_day interface."""
    reasoner = AgenticReasoner(api_key=api_key)
    result = reasoner.reason_day(bursts, external_context=external_context)
    
    # Convert to AgenticSession objects for compatibility
    sessions = []
    for sess in result.get("sessions", []):
        burst_ids = sess.get("burst_ids", [])
        chars = sum(b.char_count for b in reasoner._sorted_bursts 
                   if b is not None and b in reasoner._sorted_bursts)
        
        # Calculate duration
        try:
            if burst_ids and reasoner._sorted_bursts:
                t_start = reasoner._sorted_bursts[burst_ids[0]].timestamp
                t_end = reasoner._sorted_bursts[burst_ids[-1]].timestamp
                dur = int((datetime.fromisoformat(t_end) - 
                          datetime.fromisoformat(t_start)).total_seconds())
            else:
                dur = 0
        except:
            dur = 0
        
        sessions.append(AgenticSession(
            session_id=sess.get("session_id", 0),
            burst_ids=burst_ids,
            start_time=sess.get("start_time", "00:00"),
            end_time=sess.get("end_time", "00:00"),
            app_name="",  # filled from burst data if available
            window_title="",
            total_chars=0,
            total_duration_seconds=dur,
            work_type="writing",
            topic="",
            narrative="",
            grouping_reasoning=sess.get("grouping_reasoning", ""),
            confidence=sess.get("confidence", "medium"),
        ))
    
    result["sessions"] = sessions
    return result


# ----------------------------------------------------------------------------- 
# CLI
# ----------------------------------------------------------------------------- 

if __name__ == "__main__":
    from pathlib import Path
    
    api_key = load_api_key()
    print(f"API key: {'loaded' if api_key else 'MISSING'}")
    
    # Test on challenge_long_haul
    sample = Path(__file__).parent / "data" / "challenge_long_haul.json"
    if sample.exists():
        import json as jsonmod
        with open(sample) as f:
            raw = jsonmod.load(f)
        
        bursts = []
        for r in raw:
            bursts.append(Burst(
                timestamp=r["timestamp"],
                window_id=r.get("window_id", r["window_title"]),
                window_title=r["window_title"],
                app_name=r["app_name"],
                app_path=r.get("app_path", r["app_name"]),
                chars=r["chars"],
                char_count=len(r["chars"]),
                source=r.get("source", "test"),
                focus_active=r.get("focus_active", True),
            ))
        
        print(f"\nReasoning over {len(bursts)} bursts...")
        reasoner = AgenticReasoner(api_key=api_key)
        result = reasoner.reason_day(bursts)
        
        print(f"\nTokens: survey={result['survey_tokens']}, deliberation={result['deliberation_tokens']}, total={result['total_tokens']}")
        print(f"Tool calls: {result['num_tool_calls']}")
        print(f"\n{len(result['sessions'])} sessions produced:")
        for s in result["sessions"]:
            print(f"  [{s['burst_ids'][0]}-{s['burst_ids'][-1]}] {s['start_time']}-{s['end_time']} | {s['grouping_reasoning'][:60]}")
    else:
        print("No challenge data found.")
