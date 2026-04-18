# Keystroke Parser

Parses raw keystroke bursts into time-boxed sessions. The middle step between raw capture and summary.

## Usage

```bash
pip install -r requirements.txt
python test_parser.py
```

## Input Format

```json
[
  {
    "timestamp": "2026-04-18T09:15:23",
    "window_title": "VS Code - auth/middleware.py",
    "app_name": "VS Code",
    "chars": "implementing the auth middleware"
  }
]
```

## Output

Sessionized and classified sessions matching the `db-query-layer` schema.

## Components

- `parser.py` — Core sessionization logic (gap detection, app changes, work type + topic classification)
- `summarizer.py` — Optional LLM-based work type and topic extraction via MiniMax
- `test_parser.py` — End-to-end test using our Discord conversation as sample data
- `data/sample_conversation.json` — 29 turns of our actual conversation as raw keystroke input
