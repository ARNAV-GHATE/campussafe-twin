"""Decision policies. All take the state dict and return a Decision.

  rule  - deterministic rule-based baseline (also the validator's fallback)
  slm   - small language model through Ollama
  mock  - rule policy that makes deliberate mistakes; tests the validator
          without needing a GPU or Ollama
  none  - robot does nothing; measures what happens with no robot at all
"""
from __future__ import annotations

import json
import random
import time

from .actions import ACTION_HELP, Action, Decision
from .slm_client import OllamaClient


class RulePolicy:
    name = "rule"

    def decide(self, state: dict) -> Decision:
        t0 = time.perf_counter()
        a = self._decide(state)
        return Decision(a, (time.perf_counter() - t0) * 1000, "")

    @staticmethod
    def _decide(s: dict) -> Action:
        # 1. Report anything not yet reported.
        for h in s["known_hazards"]:
            if not h["reported"]:
                return Action("report_hazard", h["location"])
        safe_exits = [e for e in s["exits"] if e["status"] == "safe" and e["distance_m"] is not None]
        # 2. Pick the area with the most people (unknown counts as 1), then the nearest.
        areas = [a for a in s["areas_with_people"]
                 if a["distance_m"] is not None and a["nearest_safe_exit"]]
        if areas:
            target = max(areas, key=lambda a: (1 if a["people"] == "unknown" else a["people"],
                                               -a["distance_m"]))
            if s["guiding_to_exit"] != target["nearest_safe_exit"] or s["guidance_status"] == "unsafe":
                return Action("guide_to_exit", target["nearest_safe_exit"])
            if s["robot_location"] != target["id"]:
                return Action("navigate_to", target["id"])
            return Action("wait")
        # 3. Nobody left that we know of: make sure guidance is safe, then hold.
        if safe_exits and (s["guiding_to_exit"] is None or s["guidance_status"] == "unsafe"):
            return Action("guide_to_exit", min(safe_exits, key=lambda e: e["distance_m"])["id"])
        return Action("wait")


class NoRobotPolicy:
    name = "none"

    def decide(self, state: dict) -> Decision:
        return Decision(Action("wait"))


class MockSLMPolicy:
    """Rule policy plus deliberate errors and simulated latency."""
    name = "mock"

    def __init__(self, error_rate: float = 0.2, seed: int = 0):
        self.error_rate = error_rate
        self.rng = random.Random(seed)

    def decide(self, state: dict) -> Decision:
        latency = self.rng.uniform(300, 1200)
        if self.rng.random() < self.error_rate:
            bad = self.rng.choice([
                Action("guide_to_exit", next((e["id"] for e in state["exits"] if e["status"] != "safe"), "EXIT_NORTH")),
                Action("navigate_to", "ROOM_404"),
                Action("navigate_to", (state["known_hazards"] or [{"location": "C20"}])[0]["location"]),
                Action("evacuate_everyone"),
            ])
            return Decision(bad, latency, json.dumps(bad.to_dict()))
        return Decision(RulePolicy._decide(state), latency, "")


SYSTEM_PROMPT = """You are the decision module of CampusSafe, an autonomous humanoid emergency-response robot inside a university building.
Each turn you receive a JSON situation report. Reply with ONE JSON object and nothing else:
{"action": "<action>", "target": "<id or null>", "message": "<short text for people, or empty>"}

Actions:
""" + "\n".join(f"- {k}: {v}" for k, v in ACTION_HELP.items()) + """

Rules, in priority order:
1. Never guide people to, or move into, a location listed in known_hazards or an exit whose status is not "safe".
2. Report every hazard whose "reported" is false.
3. Make sure guiding_to_exit is the nearest_safe_exit of the area you are heading to; change it if guidance_status is "unsafe".
4. Go to the area with the most people (treat "unknown" as possibly occupied), preferring closer areas on ties.
5. When you are at that area, use "wait" so people can follow your guidance.
Use only ids that appear in the report."""


class SLMPolicy:
    name = "slm"

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434", timeout: float = 240.0):
        self.client = OllamaClient(model, host, timeout)
        self.name = f"slm:{model}"

    def decide(self, state: dict) -> Decision:
        report = {k: v for k, v in state.items() if k != "allowed_actions"}
        data, raw, latency = self.client.chat_json(SYSTEM_PROMPT, json.dumps(report))
        return Decision(Action.from_dict(data), latency, raw)


def make_policy(name: str, model: str = "llama3.2:3b", host: str = "http://localhost:11434", seed: int = 0):
    if name == "rule":
        return RulePolicy()
    if name == "none":
        return NoRobotPolicy()
    if name == "mock":
        return MockSLMPolicy(seed=seed)
    if name == "slm":
        return SLMPolicy(model, host)
    raise ValueError(f"unknown policy '{name}' (use rule | slm | mock | none)")
