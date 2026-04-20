# Keystroke Parser

Parses raw keystroke bursts into reasoned, labeled sessions. Capture the keystrokes separately — this project handles the sessionization and summarization pipeline.

## Quick Start

```bash
pip install -r requirements.txt
python test_pipeline.py
```

## Input Format

Raw bursts as JSON — one burst per typing event:

```json
[
  {
    "timestamp": "2026-04-18T09:15:23",
    "window_id": "codex:auth/middleware.py",
    "window_title": "Codex - auth/middleware.py",
    "app_name": "Codex",
    "app_path": "C:\\Users\\...",
    "chars": "implementing the auth middleware",
    "char_count": 29,
    "source": "manual",
    "focus_active": true
  }
]
```

## Output

Sessionized and labeled sessions:

```json
{
  "sessions": [
    {
      "session_id": 1,
      "start_time": "09:15",
      "end_time": "09:45",
      "app_name": "Codex",
      "window_title": "Codex - auth/middleware.py",
      "work_type": "writing",
      "topic": "Implement JWT token verification in auth middleware",
      "narrative": "Worked on implementing JWT token verification...",
      "chars": 1250,
      "confidence": "high"
    }
  ],
  "daily_summary": "...",
  "daily_narrative": "...",
  "grouping_reasoning": "..."
}
```

## Components

- `burst.py` — Burst dataclass + SQLite storage for raw immutable bursts
- `sessionizer.py` — Heuristic pre-grouping by window_id + time gaps
- `reasoner.py` — Legacy LLM reasoner over heuristic candidates (2-pass)
- `iterative_reasoner.py` — **Recommended** fully iterative LLM reasoner (3-pass: grouping → labeling → synthesis)
- `chunker.py` — Time-boxed chunking for long days (60-burst chunks)
- `summarizer.py` — Lightweight single-session classifier
- `test_pipeline.py` — End-to-end test on sample data

## Reasoning Modes

### Mode A — Heuristic + LLM (reasoner.py)
Faster, deterministic pre-grouping, LLM refines. Good for real-time use.

### Mode B — Fully Iterative (iterative_reasoner.py) ⚡ Recommended
LLM sees ALL raw bursts and makes all grouping decisions. Handles interleaved workflows much better. Three passes:
1. **Grouping** — LLM groups bursts into logical sessions
2. **Labeling** — LLM labels each session (work_type, topic, narrative)
3. **Synthesis** — LLM produces daily summary and cross-session patterns

## Data

- `data/sample_conversation.json` — 30 bursts from a Discord conversation
- `data/difficult_day.json` — 51 bursts, hard interleaved workday (3 Codex agents, Slack, VS Code, Chrome, Terminal, Outlook)
