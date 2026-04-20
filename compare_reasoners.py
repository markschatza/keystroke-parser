#!/usr/bin/env python3
"""
Compare sessionization reasoners on challenge datasets.

Runs:
- Heuristic (Sessionizer + rule-based, no API)
- Agentic Reasoner (AgenticReasoner, with API, heuristic fallback if no API)

Prints a comparison table showing tokens, API calls, and session count accuracy.

Note: The LLM-based IterativeReasoner is NOT run in this comparison by default,
as it requires significant API calls and time. To include it, set INCLUDE_LLM_REASONER=True.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from burst import Burst

# Set to True to include IterativeReasoner (requires more time/API calls)
INCLUDE_LLM_REASONER = False

# Load API key
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


api_key = load_api_key()
print(f"API key: {'loaded' if api_key else 'MISSING (LLM methods use fallback)'}")

# Ground truth
GROUND_TRUTH = {
    "challenge_long_haul":       14,
    "challenge_meeting_day":      9,
    "challenge_parallel_tornado":  9,
    "challenge_rabbit_hole":      8,
}

CHALLENGES = sorted(GROUND_TRUTH.keys())


def load_bursts(name: str) -> List[Burst]:
    """Load challenge JSON and convert to Burst objects."""
    with open(Path(__file__).parent / "data" / f"{name}.json") as f:
        raw = json.load(f)
    
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
    return bursts


# --------------------------------------------------------------------------- 
# Heuristic reasoner (no API)
# --------------------------------------------------------------------------- 

def reason_heuristic(bursts: List[Burst]) -> tuple:
    """Heuristic: Sessionizer + rule-based labeling. No API calls."""
    from sessionizer import Sessionizer
    
    sessionizer = Sessionizer()
    candidates = sessionizer.group_bursts(bursts)
    
    WORK_TYPE_KEYWORDS = {
        "debugging": ["fix", "bug", "debug", "error", "issue", "problem", "crash", "fail", 
                      "race condition", "exhaust", "timeout", "connection"],
        "writing": ["implement", "write", "add", "create", "new", "build", "design", "architect", 
                    "schema", "migration", "model", "endpoint", "webhook", "component"],
        "reading": ["review", "read", "look", "check", "investigate", "analyze", "explore", 
                   "understand", "logs", "traces"],
        "communicating": ["slack", "email", "team", "reply", "update", "customer", "coordinat", 
                         "discuss", "sync", "announce", "ping", "ask"],
        "planning": ["plan", "outline", "scope", "sprint", "priorit", "capacity", 
                    "retrospective", "goals"],
    }
    
    sessions = []
    for c in candidates:
        full_text = " ".join(b.chars for b in c.bursts)
        text_lower = full_text.lower()
        
        scores = {}
        for wt, kws in WORK_TYPE_KEYWORDS.items():
            scores[wt] = sum(1 for kw in kws if kw in text_lower)
        work_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "writing"
        
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
        
        sessions.append({
            "session_id": len(sessions) + 1,
            "burst_ids": [b.timestamp for b in c.bursts],
            "start_time": c.start_time,
            "end_time": c.end_time,
            "work_type": work_type,
            "topic": topic,
            "chars": c.total_chars,
            "duration_minutes": max(1, int(c.total_duration_seconds / 60)),
        })
    
    # Map burst indices properly
    session_id = 1
    burst_idx = 0
    final_sessions = []
    for c in candidates:
        n = len(c.bursts)
        final_sessions.append({
            "session_id": session_id,
            "burst_ids": list(range(burst_idx, burst_idx + n)),
            "start_time": c.start_time,
            "end_time": c.end_time,
            "work_type": sessions[session_id-1]["work_type"] if session_id <= len(sessions) else "writing",
            "topic": sessions[session_id-1]["topic"] if session_id <= len(sessions) else "general",
            "chars": c.total_chars,
            "duration_minutes": max(1, int(c.total_duration_seconds / 60)),
        })
        session_id += 1
        burst_idx += n
    
    # Rough token estimate
    tokens = len(bursts) * 10
    return final_sessions, tokens, 0


# --------------------------------------------------------------------------- 
# Agentic reasoner (uses heuristic fallback if no API)
# --------------------------------------------------------------------------- 

def reason_agentic(bursts: List[Burst], api_key: str) -> tuple:
    """Use AgenticReasoner (uses heuristic fallback if no API)."""
    import agentic_reasoner
    
    reasoner = agentic_reasoner.AgenticReasoner(api_key=api_key)
    
    try:
        result = reasoner.reason_day(bursts)
        sessions = result.get("sessions", [])
        tokens = result.get("total_tokens", 0)
        tool_calls = result.get("num_tool_calls", 0)
        # Rough API call estimate based on tool calls
        api_equiv = max(1, tool_calls // 5) if api_key else 0
        return sessions, tokens, api_equiv
    except Exception as e:
        print(f"    Agentic error: {e}")
        return [], 0, 0


# --------------------------------------------------------------------------- 
# LLM reasoner (via IterativeReasoner) - optional
# --------------------------------------------------------------------------- 

def reason_llm_reasoner(bursts: List[Burst], api_key: str) -> tuple:
    """Use IterativeReasoner - optional, requires API and time."""
    if not INCLUDE_LLM_REASONER:
        return [], 0, 0
    
    if not api_key:
        return [], 0, 0
    
    from iterative_reasoner import IterativeReasoner
    
    try:
        reasoner = IterativeReasoner(api_key=api_key, model="MiniMax-M2")
        result = reasoner.reason_day(bursts)
        sessions = result.get("sessions", [])
        tokens = (
            result.get("pass1_tokens_used", 0) +
            result.get("pass2_tokens_used", 0) +
            result.get("pass3_tokens_used", 0)
        )
        api_calls = result.get("total_api_calls", 0)
        return sessions, tokens, api_calls
    except Exception as e:
        print(f"    LLM reasoner error: {e}")
        return [], 0, 0


# --------------------------------------------------------------------------- 
# Main comparison
# --------------------------------------------------------------------------- 

def main():
    results = {
        "heuristic": {"tokens": 0, "api_calls": 0, "challenges": {}},
        "agentic_reasoner": {"tokens": 0, "api_calls": 0, "challenges": {}},
    }
    
    if INCLUDE_LLM_REASONER and api_key:
        results["llm_reasoner"] = {"tokens": 0, "api_calls": 0, "challenges": {}}
    
    for challenge in CHALLENGES:
        print(f"\n>>> Processing {challenge}...", flush=True)
        bursts = load_bursts(challenge)
        print(f"    {len(bursts)} bursts, expected {GROUND_TRUTH[challenge]} sessions", flush=True)
        
        # Heuristic (fast, no API)
        try:
            h_sessions, h_tokens, h_calls = reason_heuristic(bursts)
            h_count = len(h_sessions)
            h_ok = "✓" if h_count == GROUND_TRUTH[challenge] else f"{h_count}/{GROUND_TRUTH[challenge]}"
            results["heuristic"]["challenges"][challenge] = {
                "produced": h_count, "expected": GROUND_TRUTH[challenge],
                "match": h_ok, "tokens": h_tokens, "api_calls": h_calls,
            }
            results["heuristic"]["tokens"] += h_tokens
            results["heuristic"]["api_calls"] += h_calls
            print(f"    heuristic: {h_count} sessions ({h_ok}), ~{h_tokens} tokens, 0 API calls", flush=True)
        except Exception as e:
            print(f"    heuristic ERROR: {e}", flush=True)
            results["heuristic"]["challenges"][challenge] = {"error": str(e)}
        
        time.sleep(0.3)
        
        # Agentic (uses heuristic fallback if no API)
        try:
            ag_sessions, ag_tokens, ag_calls = reason_agentic(bursts, api_key)
            ag_count = len(ag_sessions)
            ag_ok = "✓" if ag_count == GROUND_TRUTH[challenge] else f"{ag_count}/{GROUND_TRUTH[challenge]}"
            results["agentic_reasoner"]["challenges"][challenge] = {
                "produced": ag_count, "expected": GROUND_TRUTH[challenge],
                "match": ag_ok, "tokens": ag_tokens, "api_calls": ag_calls,
            }
            results["agentic_reasoner"]["tokens"] += ag_tokens
            results["agentic_reasoner"]["api_calls"] += ag_calls
            print(f"    agentic: {ag_count} sessions ({ag_ok}), ~{ag_tokens} tokens, ~{ag_calls} API equiv", flush=True)
        except Exception as e:
            print(f"    agentic ERROR: {e}", flush=True)
            results["agentic_reasoner"]["challenges"][challenge] = {"error": str(e)}
        
        time.sleep(0.3)
        
        # LLM Reasoner (optional)
        if INCLUDE_LLM_REASONER and api_key:
            try:
                llm_sessions, llm_tokens, llm_calls = reason_llm_reasoner(bursts, api_key)
                llm_count = len(llm_sessions)
                llm_ok = "✓" if llm_count == GROUND_TRUTH[challenge] else f"{llm_count}/{GROUND_TRUTH[challenge]}"
                results["llm_reasoner"]["challenges"][challenge] = {
                    "produced": llm_count, "expected": GROUND_TRUTH[challenge],
                    "match": llm_ok, "tokens": llm_tokens, "api_calls": llm_calls,
                }
                results["llm_reasoner"]["tokens"] += llm_tokens
                results["llm_reasoner"]["api_calls"] += llm_calls
                print(f"    llm_reasoner: {llm_count} sessions ({llm_ok}), ~{llm_tokens} tokens, {llm_calls} API calls", flush=True)
            except Exception as e:
                print(f"    llm_reasoner ERROR: {e}", flush=True)
                results["llm_reasoner"]["challenges"][challenge] = {"error": str(e)}
    
    # Print comparison table
    print("\n\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    
    # Header
    short_names = {c: c.replace("challenge_", "ch_") for c in CHALLENGES}
    header = f"{'Method':<20} | {'Tokens':>8} | {'API Calls':>10} |"
    for ch in CHALLENGES:
        header += f" {short_names[ch]:>16} |"
    print(header)
    print("-" * 110)
    
    # Rows
    for method in ["heuristic", "agentic_reasoner"]:
        if method not in results:
            continue
        r = results[method]
        total_tokens = r["tokens"]
        total_calls = r["api_calls"]
        
        row = f"{method:<20} | {total_tokens:>8} | {total_calls:>10} |"
        for ch in CHALLENGES:
            cr = r["challenges"].get(ch, {})
            if "error" in cr:
                val = f"ERR"
            else:
                val = cr.get("match", "?")
            row += f" {val:>16} |"
        print(row)
    
    if INCLUDE_LLM_REASONER and api_key and "llm_reasoner" in results:
        r = results["llm_reasoner"]
        total_tokens = r["tokens"]
        total_calls = r["api_calls"]
        row = f"{'llm_reasoner':<20} | {total_tokens:>8} | {total_calls:>10} |"
        for ch in CHALLENGES:
            cr = r["challenges"].get(ch, {})
            if "error" in cr:
                val = f"ERR"
            else:
                val = cr.get("match", "?")
            row += f" {val:>16} |"
        print(row)
    
    print("=" * 110)
    
    # Summary counts
    print("\nSUMMARY (exact matches / total challenges):")
    for method in ["heuristic", "agentic_reasoner"]:
        if method not in results:
            continue
        r = results[method]
        matches = 0
        for ch in CHALLENGES:
            cr = r["challenges"].get(ch, {})
            if "error" not in cr:
                match_str = str(cr.get("match", ""))
                if "✓" in match_str:
                    matches += 1
                elif "/" in match_str:
                    produced = int(match_str.split("/")[0])
                    if produced == cr.get("expected", 0):
                        matches += 1
        
        total_t = r["tokens"]
        total_c = r["api_calls"]
        print(f"  {method}: {matches}/{len(CHALLENGES)} exact matches, "
              f"~{total_t} tokens, ~{total_c} API calls")
    
    return results


if __name__ == "__main__":
    main()
