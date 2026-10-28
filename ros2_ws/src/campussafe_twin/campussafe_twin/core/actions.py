"""The fixed action set the decision layer (SLM or rules) may choose from."""
from __future__ import annotations

from dataclasses import dataclass

ACTIONS = ("navigate_to", "guide_to_exit", "announce", "report_hazard", "wait")

ACTION_HELP = {
    "navigate_to": "move the robot to a room or corridor node (target = node id)",
    "guide_to_exit": "direct everyone near the robot, now and as it moves, to this exit (target = exit id)",
    "announce": "loudspeaker message that alerts unaware people within 12 m (message = text)",
    "report_hazard": "report a known hazard to campus police / EHS (target = hazard location)",
    "wait": "stay in place, keep guiding",
}


@dataclass
class Action:
    action: str
    target: str | None = None
    message: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "Action":
        if not isinstance(d, dict):
            return cls("invalid")
        target = d.get("target")
        return cls(str(d.get("action", "invalid")).strip(),
                   None if target in (None, "", "null") else str(target).strip(),
                   str(d.get("message") or ""))

    def to_dict(self) -> dict:
        return {"action": self.action, "target": self.target, "message": self.message}


@dataclass
class Decision:
    action: Action
    latency_ms: float = 0.0
    raw: str = ""
