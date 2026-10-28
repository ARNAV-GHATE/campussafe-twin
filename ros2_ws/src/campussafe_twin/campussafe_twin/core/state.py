"""Builds the compact, structured situation report the decision layer reads.

The same dictionary goes to the SLM (as JSON in the prompt), the rule-based
baseline and the safety validator, so every controller sees identical inputs.
"""
from __future__ import annotations

import math

from .actions import ACTIONS
from .building import Building
from .knowledge import RobotKnowledge


def _r(d: float):
    return None if math.isinf(d) else round(d, 1)


def build_state(building: Building, know: RobotKnowledge, robot_node: str,
                guidance_exit: str | None, t: float) -> dict:
    avoid = set(know.hazards)
    exits = []
    for e in building.exits:
        safe = e not in avoid
        _, d = building.shortest_path(robot_node, e, avoid) if safe else (None, math.inf)
        exits.append({"id": e, "status": "safe" if safe else f"unsafe ({know.hazards[e]})",
                      "distance_m": _r(d)})
    areas = []
    for area, count in sorted(know.counts.items()):
        if count == 0 or area in avoid:
            continue
        _, d = building.shortest_path(robot_node, area, avoid)
        exit_id, _, _ = building.nearest_exit(area, avoid)
        areas.append({"id": area, "type": building.nodes[area].kind,
                      "people": "unknown" if count is None else count,
                      "distance_m": _r(d), "nearest_safe_exit": exit_id})
    guidance_status = None
    if guidance_exit:
        guidance_status = "unsafe" if guidance_exit in avoid else "safe"
    return {
        "time_s": round(t, 1),
        "robot_location": robot_node,
        "guiding_to_exit": guidance_exit,
        "guidance_status": guidance_status,
        "building_sensors": "online" if know.sensors_online else "OFFLINE",
        "known_hazards": [{"location": n, "type": k, "reported": n in know.reported}
                          for n, k in sorted(know.hazards.items())],
        "exits": exits,
        "areas_with_people": areas,
        "allowed_actions": list(ACTIONS),
    }
