#!/usr/bin/env python3
"""Focused tests for GitHub/Hermes supplemental context loaders."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from external_context import (
    fetch_github_activity,
    format_external_context_for_prompt,
    load_hermes_activity,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None):
        del url, headers, timeout
        page = int((params or {}).get("page", 1))
        self.calls += 1
        return _FakeResponse(self.pages.get(page, []))


class ExternalContextTests(unittest.TestCase):
    def test_fetch_github_activity_filters_same_day_events(self):
        fake_session = _FakeSession(
            {
                1: [
                    {
                        "type": "PushEvent",
                        "created_at": "2026-04-22T15:05:00Z",
                        "repo": {"name": "markschatza/keystroke-parser"},
                        "payload": {
                            "ref": "refs/heads/main",
                            "size": 2,
                            "commits": [
                                {"sha": "abcdef123456", "message": "Add external context loader"},
                                {"sha": "123456abcdef", "message": "Thread GitHub activity into summary"},
                            ],
                        },
                    },
                    {
                        "type": "PullRequestEvent",
                        "created_at": "2026-04-22T18:15:00Z",
                        "repo": {"name": "markschatza/keystroke-parser"},
                        "payload": {
                            "action": "closed",
                            "pull_request": {
                                "number": 12,
                                "title": "Merge daily context support",
                                "merged": True,
                                "merged_at": "2026-04-22T18:14:30Z",
                            },
                        },
                    },
                    {
                        "type": "PushEvent",
                        "created_at": "2026-04-21T23:55:00Z",
                        "repo": {"name": "markschatza/old"},
                        "payload": {"ref": "refs/heads/main", "size": 1, "commits": []},
                    },
                ]
            }
        )

        activity = fetch_github_activity(
            target_date=date(2026, 4, 22),
            username="markschatza",
            tz_name="America/Chicago",
            session=fake_session,
        )

        self.assertEqual(activity["counts"]["pushes"], 1)
        self.assertEqual(activity["counts"]["commits"], 2)
        self.assertEqual(activity["counts"]["merges"], 1)
        self.assertEqual(activity["pushes"][0]["ref"], "main")
        self.assertEqual(activity["commits"][0]["sha"], "abcdef1")

    def test_load_hermes_activity_reads_sessions_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sessions_dir = root / "sessions"
            logs_dir = root / "logs"
            sessions_dir.mkdir()
            logs_dir.mkdir()

            session_payload = {
                "session_id": "session_1",
                "session_start": "2026-04-22T10:47:30.875936",
                "last_updated": "2026-04-22T11:01:00.000000",
                "model": "MiniMax-M2.7",
                "message_count": 3,
                "messages": [
                    {"role": "user", "content": "Investigate the broken Discord gateway."},
                    {"role": "assistant", "content": "Checking the Hermes gateway logs now."},
                    {"role": "tool", "content": "ignored"},
                ],
            }
            (sessions_dir / "session_20260422_test.json").write_text(
                json.dumps(session_payload),
                encoding="utf-8",
            )
            (logs_dir / "agent.log").write_text(
                "\n".join(
                    [
                        "2026-04-21 23:59:00,000 INFO old line",
                        "2026-04-22 10:48:00,000 INFO run_agent: Loaded environment",
                    ]
                ),
                encoding="utf-8",
            )
            (logs_dir / "gateway.log").write_text(
                "\n".join(
                    [
                        "\u001b[33m2026-04-22 10:49:00,000 WARNING gateway.run: restart requested\u001b[0m",
                        "noise",
                    ]
                ),
                encoding="utf-8",
            )

            activity = load_hermes_activity(date(2026, 4, 22), root=root)

        self.assertEqual(activity["counts"]["sessions"], 1)
        self.assertEqual(activity["counts"]["agent_log_lines"], 1)
        self.assertEqual(activity["counts"]["gateway_log_lines"], 1)
        self.assertIn("Discord gateway", activity["sessions"][0]["user_prompts"][0])
        self.assertIn("restart requested", activity["gateway_log_lines"][0])

    def test_prompt_formatter_stays_compact(self):
        payload = {
            "date": "2026-04-22",
            "timezone": "America/Chicago",
            "github": {"username": "markschatza", "counts": {"pushes": 1}, "pushes": [{"repo": "a"}]},
            "hermes": {"counts": {"sessions": 1}, "sessions": [{"session_id": "x"}]},
        }

        rendered = format_external_context_for_prompt(payload)

        self.assertIn('"pushes": 1', rendered)
        self.assertIn('"session_id": "x"', rendered)


if __name__ == "__main__":
    unittest.main()
