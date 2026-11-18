"""What the robot knows: hazards, people per area, and which hazards it reported.

Two information sources:
  * the building-sensor feed (smoke detectors, occupancy counters), which can
    fail (sensor_loss) or be attacked; `feed` is None when no message arrived
  * the robot's own onboard sensors, trusted, but short range (`local`)

While the feed is online, it supplies hazards and people counts everywhere.
Offline (or declared untrusted by the defense layer), the robot only learns
from what it perceives itself, and room counts it has not re-checked go stale.

Defense additions:
  unverified  hazards the world-model monitor found implausible. They are
              still avoided (safe default) until the robot sees the spot
              itself: if nothing is there the claim is cleared as fake.
  cleared     fake hazards the robot disproved with its own sensors.
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
        self.unverified: set[str] = set()
        self.cleared: set[str] = set()
        self.counts: dict[str, int | None] = {r: None for r in building.rooms}
        self.sensors_online = True

    def update(self, feed: dict | None, robot_xy: tuple[float, float], local: dict | None = None) -> None:
        """feed: building-sensor reading that reached the robot (None = no message this step).
        local: ground truth used only within onboard sensor range (defaults to the feed)."""
        local = local if local is not None else feed
        if feed is not None:
            self.sensors_online = bool(feed["sensors_online"])
            if self.sensors_online:
                unver = set(feed.get("unverified", ()))
                for node, kind in feed["hazards"].items():
                    if node in self.cleared:
                        continue
                    if node not in self.hazards:
                        self.hazards[node] = kind
                        if node in unver:
                            self.unverified.add(node)
                    elif node not in unver:
                        self.hazards[node] = kind
                prev = self.counts
                self.counts = {}
                for a, c in feed["area_counts"].items():
                    if c is None:                                    # rejected by the defense layer
                        if prev.get(a) is not None:
                            self.counts[a] = prev[a]
                        elif a in self.building.rooms:
                            self.counts[a] = None
                    elif c > 0 or a in self.building.rooms:
                        self.counts[a] = c
                for r in self.building.rooms:
                    self.counts.setdefault(r, prev.get(r))
        if not self.sensors_online:
            # offline: corridor knowledge is transient, rooms keep their last value
            self.counts = {a: c for a, c in self.counts.items() if a in self.building.rooms}
        if local is None:
            return
        self._perceive(local, robot_xy, refresh_counts=not self.sensors_online)

    def _perceive(self, local: dict, robot_xy, refresh_counts: bool) -> None:
        for node in self.building.nodes:
            if not self._near(node, robot_xy, HAZARD_SENSE_RADIUS):
                continue
            kind = local["hazards"].get(node)
            if kind:
                self.hazards[node] = kind
                self.unverified.discard(node)
            elif node in self.unverified:                            # verified fake
                self.unverified.discard(node)
                self.hazards.pop(node, None)
                self.cleared.add(node)
        if refresh_counts:
            truth = local["area_counts"]
            for area in set(truth) | set(self.building.rooms):
                if self._near(area, robot_xy, PEOPLE_SEE_RADIUS):
                    c = truth.get(area, 0)
                    if c > 0 or area in self.building.rooms:
                        self.counts[area] = c

    def perceived_hazards(self, local: dict, robot_xy) -> dict[str, str]:
        """Hazards the onboard sensors can see right now (used for cross-checking the feed)."""
        return {n: k for n, k in local["hazards"].items() if self._near(n, robot_xy, HAZARD_SENSE_RADIUS)}

    def _near(self, node: str, xy, radius: float) -> bool:
        x, y = self.building.xy(node)
        return math.hypot(x - xy[0], y - xy[1]) <= radius
