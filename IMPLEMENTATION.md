# Keystroke Parser — v0.1 Implementation

## Goal

Take raw keystroke data (timestamps + typed characters), parse it into time-boxed sessions, and generate summaries — mirroring the session structure in `db-query-layer`.

This is the **middle step** between raw keystroke capture and the queryable summary database:
```
Raw Keystrokes → Parser (sessionization) → Parsed Sessions → db-query-layer
```

## What We're Building

A standalone Python module that:
1. Takes raw keystroke input (timestamped character bursts)
2. Groups them into sessions based on activity patterns
3. Assigns work type labels (debugging, writing, communicating, etc.)
4. Extracts a topic from the content
5. Outputs structured session records

## Input Format (Raw Keystroke Log)

```json
{
  "timestamp": "2026-04-18T09:15:23",
  "window_title": "VS Code - auth/middleware.py",
  "app_name": "VS Code",
  "chars": "implementing the auth middleware"
}
```

Each entry represents a "burst" of typing — a sequence of characters typed in quick succession within a 30-second window.

## Output Format (Parsed Sessions)

```json
{
  "date": "2026-04-18",
  "start_time": "09:15:23",
  "end_time": "10:30:00",
  "app_name": "VS Code",
  "window_title": "VS Code - auth/middleware.py",
  "work_type": "debugging",
  "topic": "implementing token refresh in auth middleware",
  "chars": 1250,
  "duration_minutes": 75
}
```

## Sessionization Rules

1. **Time gap**: If >5 minutes of inactivity → new session starts
2. **App change**: If app changes mid-session → split if >2 minutes, else merge
3. **Window title change**: Track topic shifts within same app
4. **Work type inference**: Based on patterns in the text:
   - "fix", "debug", "error", "bug", "issue" → debugging
   - "implement", "write", "add", "create" → writing
   - "review", "read", "look at", "check" → reading
   - "@", "slack", "email", "reply", "team" → communicating
   - "plan", "outline", "scope", "sprint" → planning
5. **Topic extraction**: LLM reads the concatenated text and extracts a brief topic label

## File Structure

```
keystroke-parser/
├── IMPLEMENTATION.md
├── parser.py              # Core sessionization logic
├── summarizer.py          # LLM-based work type + topic extraction
├── requirements.txt
├── data/
│   └── sample_conversation.json   # Our Discord conversation as raw input
├── README.md
└── test_parser.py         # Quick validation script
```

## Key Design Decisions

- **No real-time inference** — batch processing only
- **LLM is optional** — can run with rule-based heuristics only
- **Keystroke bursts** are the atomic unit, not individual keys
- **Sessions** are the output unit, matching db-query-layer schema
- **LLM summarizer** generates work_type + topic from raw text

## Testing

Use our actual Discord conversation as the sample data:
- Timestamps from conversation metadata
- Content from both participants
- Real topic shifts: keystroke lifecycle tracker idea → architecture → storage → monetization → implementation planning

This validates the full pipeline: does this conversation reconstruct into sensible sessions?
