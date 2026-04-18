# Keystroke Parser — v0.3 Architecture

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

### Two Reasoning Approaches

The parser supports two reasoning modes:

#### Mode A — Heuristic Pre-grouping + LLM Refinement (reasoner.py)
```
Raw bursts → Sessionizer (heuristic: window_id + 10min gaps) → CandidateSessions
           → Reasoner (LLM: merge/split/label candidates) → Final Sessions
```
- Fast, deterministic pre-grouping
- LLM can only refine what heuristics pre-built
- Problem: heuristics fragment bursts when gaps exceed thresholds

#### Mode B — Fully Iterative LLM Reasoning (iterative_reasoner.py) ⚡ RECOMMENDED
```
Raw bursts → Pass 1: LLM groups bursts into sessions (visible reasoning)
           → Pass 2: LLM labels each session (work_type, topic, narrative)
           → Pass 3: LLM synthesizes daily summary + cross-session patterns
```
- LLM sees ALL raw bursts and makes ALL grouping decisions
- Grouping reasoning is visible and auditable
- Handles interleaved workflows, context switches, and ambiguous gaps natively
- **Significantly better grouping quality** — 50 fragmented sessions → 4 coherent sessions on difficult_day.json

### Architecture Diagram

```
CAPTURE LAYER                    REASONING LAYER
─────────────────                ──────────────────────────────────────
burst.py                         iterative_reasoner.py (RECOMMENDED)
  └─ Burst dataclass               └─ Pass 1: GROUPING (LLM sees raw bursts)
  └─ SQLite raw storage             └─ Pass 2: LABELING (work_type, topic, narrative)
  └─ window_id + window_title       └─ Pass 3: SYNTHESIS (daily summary)
  └─ source (manual/voice)        ──────────────────────────────────────
  └─ focus_active                reasoner.py (legacy heuristic approach)
                                   └─ Sessionizer (heuristic pre-grouping)
                                   └─ LLM refines pre-grouped candidates
                                         │
                                         ▼
                                   sessions (db-query-layer schema)
                              ┌─────────────────────────────────────────┐
                              │ date, start_time, end_time             │
                              │ app_name, window_title                  │
                              │ work_type, topic, chars,                │
                              │ duration_minutes                        │
                              └─────────────────────────────────────────┘
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
├── sessionizer.py             # Lightweight pre-grouping (used by reasoner.py)
│   └── CandidateSession dataclass
│   └── Sessionizer.group_bursts()
│   └── Session (db-query-layer compatible output)
├── reasoner.py                # Legacy LLM reasoning over heuristic candidates
│   └── Reasoner.reason_day() — with chunking for long days
│   └── _reason_sessions() — final session boundaries
│   └── _reason_daily_summary()
├── iterative_reasoner.py      # NEW: Fully LLM-driven reasoning (3-pass)
│   └── IterativeReasoner.reason_day()
│   └── Pass 1: _pass1_grouping() — LLM groups raw bursts
│   └── Pass 2: _pass2_labeling() — LLM labels sessions
│   └── Pass 3: _pass3_synthesis() — daily summary
├── summarizer.py              # Lightweight single-session classifier
├── chunker.py                 # Time-boxed chunking for long days
├── test_pipeline.py           # End-to-end pipeline test
├── requirements.txt
└── data/
    ├── sample_conversation.json   # Sample: Discord conversation as raw input
    ├── difficult_day.json         # Sample: hard interleaved workday (51 bursts)
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

### v0.2 (legacy): Heuristic pre-grouping + LLM refinement
1. **Capture**: Every keystroke event → Burst stored in SQLite
2. **Pre-group**: Sessionizer groups bursts by `window_id`, splitting at >10min gaps
   — Intentionally over-merged to avoid premature fragmentation
3. **Reason**: LLM receives ALL candidates for the day and decides:
   - Should candidates be MERGED (same logical task)?
   - Should candidates be SPLIT (different tasks in same window)?
   - What work_type and topic per final session?
4. **Store**: Final sessions written to sessions.db

### v0.3 (current): Fully Iterative LLM Reasoning — RECOMMENDED

3-pass approach where the LLM sees raw bursts and makes all decisions:

**Pass 1 — GROUPING**: The LLM sees ALL raw bursts for the day and groups them into
logical sessions with visible reasoning. It considers:
- Timing: gaps >30min = split; gaps <15min in same window = merge
- Content: are these bursts about the same task?
- Context: was this an interruption or parallel workflow?
- Every burst must appear in exactly one session

**Pass 2 — LABELING**: Each session from Pass 1 gets labeled:
- work_type: debugging | writing | reading | communicating | planning
- topic: specific task in ≤10 words
- narrative: 1-2 sentence description of what happened
- confidence: high | medium | low

**Pass 3 — SYNTHESIS**: All sessions reviewed together to:
- Generate a 2-paragraph daily summary
- Identify cross-session patterns
- Rate the overall day quality (productive/scattered/focused/etc)

### Why iterative (v0.3) beats heuristic (v0.2)

On difficult_day.json (51 bursts, interleaved morning work):
- **Heuristic (v0.2)**: 50 sessions — fragments every 3-5 min due to burst gaps
- **Iterative (v0.3)**: 4 sessions — properly groups auth work, Slack, afternoon coding, EOD wrap-up

The LLM can reason about CONTENT and INTENT, not just timing thresholds.

### Why this works for interleaved conversations

With multiple Codex agents running simultaneously:
- Each agent has its own window_id
- Sessionizer pre-groups by window_id → candidates are naturally separated
- LLM sees all candidates with full text → can reason about context switches
- "typing in Agent 1, then Agent 2, then back to Agent 1" → 3 candidates, LLM may merge or keep separate based on content

---

## Test Results

### difficult_day.json (hard interleaved workday)

A realistic developer day with frequent context-switching between Codex agents, VS Code, Slack, Chrome, Terminal, and Outlook — with 3-5 minute gaps between bursts that fool the heuristic sessionizer.

| Metric | Heuristic v0.2 | Iterative v0.3 |
|--------|--------------|----------------|
| Sessions produced | 50 (one per burst) | 4 (coherent) |
| Morning auth work | 30+ fragmented sessions | 1 merged session |
| Correct grouping | No | Yes |
| Grouping reasoning visible | No | Yes |
| LLM calls per session | 1 | 3 (one per pass) |

**v0.3 Iterative output** (4 sessions):
1. `[08:00-10:36] Codex | writing` — "Morning auth middleware + dashboard interleaved work"
2. `[12:00-12:08] Slack | writing` — "Brief afternoon Slack about deploy to staging"
3. `[13:00-14:08] VS Code | writing` — "PR review + avatar upload + database debugging"
4. `[16:30-16:40] Outlook | writing` — "EOD wrap-up: reviewing progress, Slack, git commit"

### Sample Conversation (our Discord thread)

| Metric | Value |
|--------|-------|
| Raw bursts | 30 |
| Candidate sessions (pre-grouped) | 29 |
| Final sessions (LLM-reasoned) | 10 |
| LLM correctly merged interleaved Discord+Claude Code bursts | ✓ |
| LLM correctly classified work types | ✓ |
| Daily narrative | Coherent 3-paragraph summary |

---

## Cost Analysis

The iterative reasoner (v0.3) costs more than the heuristic approach (v0.2):

| Pass | Tokens (approx) | API Calls |
|------|---------------|-----------|
| Pass 1 (Grouping) | ~6,000 | 1 |
| Pass 2 (Labeling, N sessions) | ~6,000 | 1 |
| Pass 3 (Synthesis) | ~1,000 | 1 |
| **Total per day** | **~13,000** | **3** |

vs. v0.2 heuristic: ~1 LLM call for the same data.

For a typical developer day (50-100 bursts → 3-8 sessions), v0.3 is still affordable at ~$0.01/day with MiniMax-M2.7.

---

## Next Steps

1. **Capture layer**: Build the actual keystroke capture daemon (Windows)
   - Global keyboard hook → GetAsyncKeyState
   - GetForegroundWindow + GetWindowText for window context
   - Text capture from focused edit control
2. **Voice capture**: VAD-based audio recording with Whisper transcription
3. **Real-time reasoning**: Stream bursts to reasoner for live session updates
4. **Meta layer**: Experiment with different sessionization prompts on the same raw data
