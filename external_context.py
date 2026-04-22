#!/usr/bin/env python3
"""Load supplemental same-day context from GitHub and Hermes."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


DEFAULT_GITHUB_USERNAME = "markschatza"
DEFAULT_TIMEZONE = "America/Chicago"
DEFAULT_HERMES_ROOT = Path(r"\\wsl$\Ubuntu\home\marks\.hermes")
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
TIMEZONE_FALLBACKS = {
    "America/Chicago": timezone(timedelta(hours=-5)),
}


def _tzinfo(tz_name: str):
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return TIMEZONE_FALLBACKS.get(tz_name, timezone.utc)


def _local_day(value: str, tz_name: str) -> Optional[date]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if dt.tzinfo is None:
        return dt.date()
    return dt.astimezone(_tzinfo(tz_name)).date()


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _trim_text(value: Any, limit: int = 280) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _github_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "keystroke-parser",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_github_activity(
    target_date: date,
    username: str = DEFAULT_GITHUB_USERNAME,
    tz_name: str = DEFAULT_TIMEZONE,
    token: str = "",
    session: Optional[requests.Session] = None,
    per_page: int = 100,
    max_pages: int = 3,
) -> dict[str, Any]:
    """Fetch same-day GitHub events for a user from the Events API."""
    http = session or requests.Session()
    events: list[dict[str, Any]] = []
    source = "public"

    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/users/{username}/events/public"
        headers = _github_headers(token)
        params = {"per_page": per_page, "page": page}
        response = http.get(url, headers=headers, params=params, timeout=20)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        events.extend(batch)
        if all(_local_day(item.get("created_at", ""), tz_name) != target_date for item in batch):
            break

    pushes = []
    commits = []
    merges = []
    other = []

    for event in events:
        created_day = _local_day(event.get("created_at", ""), tz_name)
        if created_day != target_date:
            continue

        event_type = event.get("type", "")
        repo_name = event.get("repo", {}).get("name", "")
        created_at = event.get("created_at", "")
        payload = event.get("payload", {}) or {}

        if event_type == "PushEvent":
            ref = payload.get("ref", "").removeprefix("refs/heads/")
            push_item = {
                "repo": repo_name,
                "created_at": created_at,
                "ref": ref,
                "commit_count": payload.get("size", 0),
            }
            pushes.append(push_item)
            for commit in payload.get("commits", []) or []:
                commits.append(
                    {
                        "repo": repo_name,
                        "created_at": created_at,
                        "ref": ref,
                        "sha": (commit.get("sha") or "")[:7],
                        "message": _trim_text(commit.get("message", ""), 160),
                    }
                )
            continue

        if event_type == "PullRequestEvent":
            pr = payload.get("pull_request", {}) or {}
            action = payload.get("action", "")
            if action == "closed" and pr.get("merged"):
                merges.append(
                    {
                        "repo": repo_name,
                        "created_at": created_at,
                        "number": pr.get("number"),
                        "title": _trim_text(pr.get("title", ""), 160),
                        "merged_at": pr.get("merged_at", ""),
                    }
                )
            else:
                other.append(
                    {
                        "type": event_type,
                        "repo": repo_name,
                        "created_at": created_at,
                        "action": action,
                    }
                )
            continue

        other.append(
            {
                "type": event_type,
                "repo": repo_name,
                "created_at": created_at,
            }
        )

    return {
        "provider": "github",
        "username": username,
        "date": target_date.isoformat(),
        "source": source,
        "counts": {
            "pushes": len(pushes),
            "commits": len(commits),
            "merges": len(merges),
            "other_events": len(other),
        },
        "pushes": pushes,
        "commits": commits,
        "merges": merges,
        "other_events": other[:20],
    }


def _extract_hermes_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content or "")


def _summarize_hermes_session(path: Path, target_date: date) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    session_start = data.get("session_start", "")
    session_day = _local_day(session_start, DEFAULT_TIMEZONE)
    if session_day != target_date:
        return None

    messages = data.get("messages", []) or []
    user_messages = []
    assistant_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        text = _trim_text(_extract_hermes_message_text(msg), 220)
        if not text:
            continue
        if role == "user":
            user_messages.append(text)
        elif role == "assistant":
            assistant_messages.append(text)

    return {
        "session_id": data.get("session_id", path.stem),
        "session_start": session_start,
        "last_updated": data.get("last_updated", ""),
        "model": data.get("model", ""),
        "message_count": data.get("message_count", len(messages)),
        "user_prompts": user_messages[:6],
        "assistant_replies": assistant_messages[:4],
    }


def _extract_log_lines(path: Path, target_date: date, limit: int = 120) -> list[str]:
    if not path.exists():
        return []

    wanted = target_date.isoformat()
    matched = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = _strip_ansi(raw_line.rstrip("\n"))
            if line.startswith(wanted):
                matched.append(line)
    return matched[-limit:]


def load_hermes_activity(
    target_date: date,
    root: Path = DEFAULT_HERMES_ROOT,
) -> dict[str, Any]:
    """Load same-day Hermes session and log activity from WSL-backed storage."""
    sessions_dir = root / "sessions"
    logs_dir = root / "logs"

    sessions = []
    if sessions_dir.exists():
        for path in sorted(sessions_dir.glob("session_*.json")):
            session = _summarize_hermes_session(path, target_date)
            if session:
                sessions.append(session)

    agent_lines = _extract_log_lines(logs_dir / "agent.log", target_date)
    gateway_lines = _extract_log_lines(logs_dir / "gateway.log", target_date)

    return {
        "provider": "hermes",
        "date": target_date.isoformat(),
        "root": str(root),
        "counts": {
            "sessions": len(sessions),
            "agent_log_lines": len(agent_lines),
            "gateway_log_lines": len(gateway_lines),
        },
        "sessions": sessions,
        "agent_log_lines": agent_lines,
        "gateway_log_lines": gateway_lines,
    }


def build_external_day_context(
    target_date: date,
    tz_name: str = DEFAULT_TIMEZONE,
    github_username: str = DEFAULT_GITHUB_USERNAME,
    github_token: str = "",
    hermes_root: Path = DEFAULT_HERMES_ROOT,
) -> dict[str, Any]:
    """Load all configured same-day external sources."""
    github: dict[str, Any]
    hermes: dict[str, Any]

    try:
        github = fetch_github_activity(
            target_date=target_date,
            username=github_username,
            tz_name=tz_name,
            token=github_token,
        )
    except Exception as exc:
        github = {
            "provider": "github",
            "date": target_date.isoformat(),
            "username": github_username,
            "error": str(exc),
            "counts": {},
            "pushes": [],
            "commits": [],
            "merges": [],
            "other_events": [],
        }

    try:
        hermes = load_hermes_activity(target_date=target_date, root=hermes_root)
    except Exception as exc:
        hermes = {
            "provider": "hermes",
            "date": target_date.isoformat(),
            "root": str(hermes_root),
            "error": str(exc),
            "counts": {},
            "sessions": [],
            "agent_log_lines": [],
            "gateway_log_lines": [],
        }

    return {
        "date": target_date.isoformat(),
        "timezone": tz_name,
        "github": github,
        "hermes": hermes,
    }


def format_external_context_for_prompt(external_context: Optional[dict[str, Any]]) -> str:
    """Format supplemental context into a compact prompt block."""
    if not external_context:
        return "No external same-day context available."

    github = external_context.get("github", {}) or {}
    hermes = external_context.get("hermes", {}) or {}

    payload = {
        "date": external_context.get("date", ""),
        "timezone": external_context.get("timezone", DEFAULT_TIMEZONE),
        "github": {
            "username": github.get("username", ""),
            "counts": github.get("counts", {}),
            "pushes": github.get("pushes", [])[:12],
            "commits": github.get("commits", [])[:20],
            "merges": github.get("merges", [])[:12],
            "error": github.get("error", ""),
        },
        "hermes": {
            "counts": hermes.get("counts", {}),
            "sessions": hermes.get("sessions", [])[:8],
            "agent_log_lines": hermes.get("agent_log_lines", [])[-20:],
            "gateway_log_lines": hermes.get("gateway_log_lines", [])[-20:],
            "error": hermes.get("error", ""),
        },
    }
    return json.dumps(payload, indent=2)
