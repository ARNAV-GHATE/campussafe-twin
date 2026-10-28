"""What the robot knows: hazards, people per area, and which hazards it reported.

Building sensors (smoke detectors, occupancy counters) feed everything while
online. After a sensor failure the robot only learns from what it perceives
itself, and room counts it has not re-checked go stale.
"""
from __future__ import annotations

import math

from .building import Building

HAZARD_SENSE_RADIUS = 6.0  # m, robot's own smoke/heat/camera range
PEOPLE_SEE_RADIUS = 8.0    # m, robot's own people-detection range


class RobotKnowledge:
    def __init__(self, building: Building):
        self.building = building
        self.hazards: dict[str, str] = {}
        self.reported: set[str] = set()
        self.counts: dict[str, int | None] = {r: None for r in building.rooms}
        self.sensors_online = True

    def update(self, snapshot: dict, robot_xy: tuple[float, float]) -> None:
        online = snapshot["sensors_online"]
        self.sensors_online = online
        for node, kind in snapshot["hazards"].items():
            if online or self._near(node, robot_xy, HAZARD_SENSE_RADIUS):
                self.hazards[node] = kind
        truth = snapshot["area_counts"]
        if online:
            self.counts = {a: c for a, c in truth.items() if c > 0 or a in self.building.rooms}
            return
        # offline: corridor knowledge is transient, rooms keep their last value
        self.counts = {a: c for a, c in self.counts.items() if a in self.building.rooms}
        for area in set(truth) | set(self.building.rooms):
            if self._near(area, robot_xy, PEOPLE_SEE_RADIUS):
                c = truth.get(area, 0)
                if c > 0 or area in self.building.rooms:
                    self.counts[area] = c

    def _near(self, node: str, xy, radius: float) -> bool:
        x, y = self.building.xy(node)
        return math.hypot(x - xy[0], y - xy[1]) <= radius
