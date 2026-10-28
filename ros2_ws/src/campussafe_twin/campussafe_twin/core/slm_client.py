"""Minimal Ollama client (standard library only) that asks for JSON output.

Install Ollama (https://ollama.com), then pull a small model, e.g.
    ollama pull llama3.2:3b     # or qwen3:4b, phi4-mini, gemma3:4b
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 60.0,
                 temperature: float = 0.0, max_tokens: int = 200):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.options = {"temperature": temperature, "num_predict": max_tokens}

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat_json(self, system: str, user: str) -> tuple[dict | None, str, float]:
        """Returns (parsed JSON or None, raw text, latency in ms)."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": self.options,
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = json.loads(r.read())["message"]["content"]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            return None, f"ERROR: {e}", (time.perf_counter() - t0) * 1000
        latency = (time.perf_counter() - t0) * 1000
        return _parse_json(raw), raw, latency


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                return None
    return None
