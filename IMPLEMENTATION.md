# Keystroke Parser — v0.2 Architecture

## Core Principle: Be Data-Rich at Capture Time, Reason Later

We do NOT make sessionization decisions at capture time. We capture everything,
store it raw, and let the LLM reason over the full picture later.

This enables:
- Re-reasoning with better prompts/strategies
- Parallel conversation detection
- Multiple sessionization strategies from the same raw data
- No loss of information from premature decisions

---

## Architecture

```
CAPTURE LAYER                    REASONING LAYER
─────────────────                ─────────────────
burst.py                         reasoner.py
  └─ Burst dataclass               └─ LLM reviews all candidates
  └─ SQLite raw storage             └─ Decides final sessions
  └─ window_id + window_title       └─ Assigns work_type + topic
  └─ source (manual/voice)          └─ Generates daily narrative
  └─ focus_active
        │
        ▼
sessionizer.py
  └─ Lightweight pre-grouping
  └─ Groups by window_id + time proximity
  └─ Produces CandidateSession[]
        │
        ▼                    sessions (db-query-layer schema)
                         ┌─────────────────────────────┐
                         │ date, start_time, end_time  │
                         │ app_name, window_title       │
                         │ work_type, topic, chars,     │
                         │ duration_minutes            │
                         └─────────────────────────────┘
```

---

## File Structure

```
keystroke-parser/
├── IMPLEMENTATION.md          # This file
├── README.md
├── burst.py                   # Raw burst storage (SQLite)
│   └── Burst dataclass
│   └── init_db, insert_bursts, load_bursts_*
├── sessionizer.py             # Lightweight pre-grouping
│   └── CandidateSession dataclass
│   └── Sessionizer.group_bursts()
│   └── Session (db-query-layer compatible output)
├── reasoner.py                # LLM reasoning over candidates
│   └── Reasoner.reason_day()
│   └── _reason_sessions() — final session boundaries
│   └── _reason_daily_summary()
├── summarizer.py              # Lightweight single-session classifier
├── test_pipeline.py           # End-to-end pipeline test
├── requirements.txt
└── data/
    ├── sample_conversation.json   # Our Discord conversation as raw input
    ├── bursts.db                   # SQLite: raw bursts (immutable)
    └── sessions.db                 # SQLite: reasoned sessions
```

---

## Data Model

### Burst (capture layer — immutable)

```json
{
  "timestamp": "2026-04-18T09:15:23",
  "window_id": "codex:auth/middleware.py",
  "window_title": "Codex - auth/middleware.py",
  "app_name": "Codex",
  "app_path": "C:\\Users\\...\\Codex.exe",
  "chars": "implementing the token refresh",
  "char_count": 34,
  "source": "manual",
  "focus_active": true
}
```

Key design decisions:
- `window_id`: stable identifier for the window instance (not just title).
  Enables tracking the same conversation across title changes.
- `source`: "manual" | "voice" | "clipboard" — how text arrived.
- `focus_active`: was this window focused when captured?
- `chars`: the actual typed/pasted text — full fidelity, no summarization.

### CandidateSession (pre-grouped bursts)

Pre-grouped by `window_id` + 10-minute max gap. Not a final session —
the LLM can merge or split these.

```json
{
  "date": "2026-04-18",
  "start_time": "09:00:00",
  "end_time": "09:45:00",
  "window_id": "codex:auth/middleware.py",
  "window_title": "Codex - auth/middleware.py",
  "app_name": "Codex",
  "total_chars": 1250,
  "total_duration_seconds": 2700,
  "burst_count": 4,
  "sources": ["manual", "voice"],
  "focus_changes": 1
}
```

### Session (output — db-query-layer compatible)

```json
{
  "date": "2026-04-18",
  "start_time": "09:15:23",
  "end_time": "10:30:00",
  "app_name": "VS Code",
  "window_title": "auth/middleware.py - VS Code",
  "work_type": "debugging",
  "topic": "token refresh issue in auth middleware",
  "chars": 1250,
  "duration_minutes": 75
}
```

---

## Sessionization Strategy

### v0.1 (old): Gap-based hard splits
If gap > 5 minutes → new session. Simple but wrong for interleaved conversations.

### v0.2 (new): LLM-guided session boundaries

1. **Capture**: Every keystroke event → Burst stored in SQLite
2. **Pre-group**: Sessionizer groups bursts by `window_id`, splitting at >10min gaps
   — This is intentionally over-merged. Better to group too much than too little.
3. **Reason**: LLM receives ALL candidates for the day and decides:
   - Should candidates be MERGED (same logical task)?
   - Should candidates be SPLIT (different tasks in same window)?
   - What work_type and topic per final session?
4. **Store**: Final sessions written to sessions.db

### Why this works for interleaved conversations

With multiple Codex agents running simultaneously:
- Each agent has its own window_id
- Sessionizer pre-groups by window_id → candidates are naturally separated
- LLM sees all candidates with full text → can reason about context switches
- "typing in Agent 1, then Agent 2, then back to Agent 1" → 3 candidates, LLM may merge or keep separate based on content

---

## Test Results

Tested on our actual Discord conversation (30 bursts, one day).

| Metric | Value |
|--------|-------|
| Raw bursts | 30 |
| Candidate sessions (pre-grouped) | 29 |
| Final sessions (LLM-reasoned) | 10 |
| LLM correctly merged interleaved Discord+Claude Code bursts | ✓ |
| LLM correctly classified work types | ✓ |
| Daily narrative | Coherent 3-paragraph summary |

---

## Next Steps

1. **Capture layer**: Build the actual keystroke capture daemon (Windows)
   - Global keyboard hook → GetAsyncKeyState
   - GetForegroundWindow + GetWindowText for window context
   - Text capture from focused edit control
2. **Voice capture**: VAD-based audio recording with Whisper transcription
3. **Real-time reasoning**: Stream bursts to reasoner for live session updates
4. **Meta layer**: Experiment with different sessionization prompts on the same raw data
