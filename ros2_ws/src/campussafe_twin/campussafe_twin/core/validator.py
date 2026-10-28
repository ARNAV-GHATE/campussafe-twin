"""Safety validator: every proposed action is checked before the robot acts.

An SLM can hallucinate (a non-existent room, an exit that is on fire). The
validator rejects such actions; the caller then falls back to the rule-based
policy. The rejection rate is one of the thesis metrics.
"""
from __future__ import annotations

from .actions import ACTIONS, Action
from .building import Building

MAX_MESSAGE_CHARS = 240


def validate(action: Action, state: dict, building: Building) -> tuple[bool, str]:
    hazards = {h["location"] for h in state["known_hazards"]}
    robot = state["robot_location"]

    if action.action not in ACTIONS:
        return False, f"unknown action '{action.action}'"

    if action.action == "navigate_to":
        t = action.target
        if t not in building.nodes:
            return False, f"unknown location '{t}'"
        if t in hazards:
            return False, f"'{t}' is a known hazard"
        if building.nodes[t].kind == "exit" and t != _safe_exit_or_none(state, t):
            return False, f"exit '{t}' is unsafe"
        path, _ = building.shortest_path(robot, t, hazards)
        if path is None:
            return False, f"no safe route to '{t}'"

    elif action.action == "guide_to_exit":
        t = action.target
        if t not in building.exits:
            return False, f"'{t}' is not an exit"
        if _safe_exit_or_none(state, t) is None:
            return False, f"exit '{t}' is unsafe or unreachable"

    elif action.action == "report_hazard":
        if action.target not in hazards:
            return False, f"'{action.target}' is not a known hazard"

    elif action.action == "announce":
        if not action.message.strip():
            return False, "empty announcement"
        if len(action.message) > MAX_MESSAGE_CHARS:
            return False, "announcement too long"

    return True, "ok"


def _safe_exit_or_none(state: dict, exit_id: str) -> str | None:
    for e in state["exits"]:
        if e["id"] == exit_id and e["status"] == "safe" and e["distance_m"] is not None:
            return exit_id
    return None
