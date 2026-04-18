#!/usr/bin/env python3
"""
LLM-based summarizer for keystroke sessions.
Takes raw text from a session and generates work_type + topic.
"""

import os
import json
import re
import requests
from pathlib import Path
from typing import Optional


class LLMSummarizer:
    """Uses MiniMax to classify work type and extract topic."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.io/v1/text"
        self.model = "MiniMax-M2.7"

    def classify_work_type(self, text: str) -> str:
        """Classify a session's work type from its text content."""
        if not self.api_key or not text.strip():
            return "writing"  # fallback

        prompt = (
            "Classify this work session based on the typed content below.\n"
            "Return ONLY one word: debugging, writing, reading, communicating, or planning.\n\n"
            f"Content: {text[:500]}\n\n"
            "Work type:"
        )

        result = self._call(prompt)
        result = result.strip().lower()

        valid = {"debugging", "writing", "reading", "communicating", "planning"}
        if result in valid:
            return result
        return "writing"

    def extract_topic(self, text: str) -> str:
        """Extract a brief topic label from session text."""
        if not self.api_key or not text.strip():
            return text[:80]

        prompt = (
            "Read this work session content and produce a brief topic label (max 10 words).\n"
            "Focus on the specific task, not generic descriptions.\n\n"
            f"Content: {text[:800]}\n\n"
            "Topic:"
        )

        topic = self._call(prompt).strip()
        return topic[:80]

    def summarize_day(self, sessions: list) -> str:
        """Generate a paragraph summary of a day's sessions."""
        if not self.api_key or not sessions:
            return "No sessions to summarize."

        session_summaries = []
        for i, s in enumerate(sessions):
            session_summaries.append(
                f"Session {i+1}: {s['start_time']}-{s['end_time']}, "
                f"{s['app_name']}, {s['work_type']}, {s['topic']}"
            )

        prompt = (
            "You are summarizing a developer's workday from keystroke logs.\n"
            "Write 3 paragraphs summarizing the day based on these sessions:\n\n"
            + "\n".join(session_summaries)
            + "\n\nFocus on: what were the major topics, what was accomplished, any patterns or notable events."
        )

        return self._call(prompt).strip()

    def _call(self, prompt: str) -> str:
        """Make a MiniMax API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 200
        }

        try:
            response = requests.post(
                f"{self.base_url}/chatcompletion_v2",
                headers=headers,
                json=payload,
                timeout=15
            )
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            elif "base_resp" in data:
                return f"API Error: {data['base_resp'].get('status_msg', 'unknown')}"
        except Exception as e:
            return f"Error: {str(e)}"
        return ""


def load_api_key() -> str:
    """Load MiniMax API key from Hermes config."""
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "MINIMAX_API_KEY=" in line:
                    key = line.split("MINIMAX_API_KEY=", 1)[1].strip()
                    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", key)
    return ""
