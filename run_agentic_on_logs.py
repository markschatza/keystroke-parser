#!/usr/bin/env python3
"""Parse keystroke logger .log files and run the agentic reasoner on them."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from agentic_reasoner import AgenticReasoner, load_api_key
from burst import Burst
from external_context import build_external_day_context


INTERESTING_EVENTS = {
    "TYPE",
    "PASTE",
    "PASTE INJECTED",
    "TYPE_REDACTED",
    "PASTE_REDACTED",
    "PASTE_REDACTED INJECTED",
}

SOURCE_PREFIXES = ("voice/", "clipboard/")
METADATA_PREFIXES = ("focus=", "reason=", "idle=")
CHAR_COUNT_RE = re.compile(r"^(?P<count>\d+)\s+chars$")
TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<event>.+)$")


def parse_log_line(line: str) -> Optional[Burst]:
    line = line.rstrip("\n")
    if not line:
        return None

    parts = line.split(" | ")
    if not parts:
        return None

    m = TIMESTAMP_RE.match(parts[0])
    if not m:
        return None

    event = m.group("event").strip()
    if event not in INTERESTING_EVENTS:
        return None

    timestamp = m.group("ts").replace(" ", "T")
    app_name = parts[1].strip() if len(parts) > 1 else "Unknown"

    char_idx = None
    for idx in range(len(parts) - 1, 1, -1):
        if CHAR_COUNT_RE.match(parts[idx].strip()):
            char_idx = idx
            break
    if char_idx is None:
        return None

    count_match = CHAR_COUNT_RE.match(parts[char_idx].strip())
    assert count_match is not None
    char_count = int(count_match.group("count"))
    chars = parts[char_idx + 1] if char_idx + 1 < len(parts) else ""

    middle = [p.strip() for p in parts[2:char_idx] if p.strip()]
    descriptive = [
        p for p in middle
        if not p.startswith(METADATA_PREFIXES)
        and not p.startswith(SOURCE_PREFIXES)
    ]
    source_fields = [p for p in middle if p.startswith(SOURCE_PREFIXES)]

    window_title = descriptive[-1] if descriptive else app_name
    window_id = " | ".join(descriptive[:-1] + [window_title]) if descriptive else window_title
    source = source_fields[-1] if source_fields else event.lower().replace(" ", "_")

    if event.endswith("REDACTED") or "REDACTED" in event:
        if not chars:
            chars = "[REDACTED_SECRET]"

    return Burst(
        timestamp=timestamp,
        window_id=window_id,
        window_title=window_title,
        app_name=app_name,
        app_path=app_name,
        chars=chars,
        char_count=char_count,
        source=source,
        focus_active=True,
    )


def load_bursts_from_log(path: Path) -> list[Burst]:
    bursts: list[Burst] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            burst = parse_log_line(line)
            if burst is not None:
                bursts.append(burst)
    bursts.sort(key=lambda b: b.timestamp)
    return bursts


def summarize_apps(bursts: Iterable[Burst]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for burst in bursts:
        counts[burst.app_name] = counts.get(burst.app_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def infer_log_date(path: Path, bursts: list[Burst]) -> date:
    if bursts:
        return date.fromisoformat(bursts[0].timestamp[:10])

    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
    if match:
        return date.fromisoformat(match.group(1))
    raise ValueError(f"Unable to infer log date from {path}")


def resolve_credentials() -> tuple[str, str]:
    api_key = os.environ.get("MINIMAX_API_KEY", "") or load_api_key()
    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io")

    auth_path = Path.home() / ".hermes" / "auth.json"
    if auth_path.exists():
        try:
            with auth_path.open("r", encoding="utf-8", errors="replace") as handle:
                auth = json.load(handle)
            creds = auth.get("credential_pool", {}).get("minimax", [])
            if creds:
                api_key = api_key or creds[0].get("access_token", "")
                base_url = creds[0].get("base_url", base_url) or base_url
        except Exception:
            pass

    return api_key, base_url


def run_one(path: Path, output_name: str, api_key: str, base_url: str) -> Optional[Path]:
    bursts = load_bursts_from_log(path)
    if not bursts:
        print(f"SKIP {path.name}: no parseable keystroke bursts")
        return None

    target_date = infer_log_date(path, bursts)
    external_context = build_external_day_context(
        target_date=target_date,
        github_token=os.environ.get("GITHUB_TOKEN", ""),
    )

    print(f"RUN  {path.name}: {len(bursts)} bursts")
    reasoner = AgenticReasoner(api_key=api_key, base_url=base_url)
    result = reasoner.reason_day(bursts, external_context=external_context)

    payload = {
        "source_log": str(path),
        "date": target_date.isoformat(),
        "burst_count": len(bursts),
        "apps": summarize_apps(bursts),
        "external_context": external_context,
        "result": result,
        "normalized_bursts": [asdict(b) for b in bursts],
    }

    if output_name.startswith("_"):
        output_filename = f"{path.stem}{output_name}.json"
    else:
        output_filename = f"{path.stem}.{output_name}.json"
    output_path = path.with_name(output_filename)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"WROTE {output_path.name}: {len(result.get('sessions', []))} sessions")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default=".", help="Log file or directory")
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern when target is a directory",
    )
    parser.add_argument(
        "--output-name",
        default="_parsed",
        help="Output suffix before .json",
    )
    args = parser.parse_args()

    target = Path(args.target)
    api_key, base_url = resolve_credentials()
    if not api_key:
        raise SystemExit("MINIMAX_API_KEY not found; agentic reasoner requires API access.")

    if target.is_dir():
        paths = sorted(target.glob(args.pattern))
    else:
        paths = [target]

    written = 0
    for path in paths:
        if path.suffix.lower() != ".log":
            continue
        if run_one(path, args.output_name, api_key, base_url):
            written += 1

    print(f"DONE wrote {written} output file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
