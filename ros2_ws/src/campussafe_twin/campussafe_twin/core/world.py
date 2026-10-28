"""Occupant and hazard model for the digital twin.

The World advances simulated people and hazards. It does not move the robot:
the robot's position comes from outside (the offline GraphRobot, or Gazebo
odometry in ROS 2), so the same model runs in both modes.

Occupant behaviour (deliberately simple and documented, so it can be defended):
  * Each person reacts to the alarm after a random delay (15-90 s).
  * Unguided people head for the nearest exit and know nothing about hazards.
  * Within NOTICE_DISTANCE of a hazard they see it, turn back, pause
    (CONFUSION_DELAY) and re-plan. Fire/smoke counts as an exposure.
  * Within GUIDE_RADIUS of the robot, people react immediately, adopt the
    exit the robot is guiding to, and learn every hazard the robot knows.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .building import Building
from .scenarios import Scenario

WALK_SPEED = 1.2        # m/s
GUIDE_RADIUS = 8.0      # m, robot voice/gesture guidance range
ANNOUNCE_RADIUS = 12.0  # m, loudspeaker announcement range
NOTICE_DISTANCE = 3.0   # m, a person sees a hazard this close
CONFUSION_DELAY = 10.0  # s, pause after discovering a hazard
EXPOSURE_KINDS = {"fire", "smoke"}


@dataclass
class Occupant:
    id: str
    room: str
    x: float
    y: float
    node: str
    react_at: float
    status: str = "unaware"          # unaware | moving | waiting | trapped | evacuated
    path: list = field(default_factory=list)
    target_exit: str | None = None
    guided: bool = False
    known_hazards: set = field(default_factory=set)
    wait_until: float = 0.0
    exposures: int = 0
    exposed_at: set = field(default_factory=set)
    evac_time: float | None = None


class World:
    def __init__(self, building: Building, scenario: Scenario, seed: int = 0):
        self.building = building
        self.scenario = scenario
        self.t = 0.0
        self.hazards: dict[str, str] = scenario.hazards_at(0.0)
        self.new_hazards: list[str] = list(self.hazards)
        rng = random.Random(seed)
        self.occupants: list[Occupant] = []
        for room, count in scenario.occupants.items():
            rx, ry = building.xy(room)
            for i in range(count):
                self.occupants.append(Occupant(
                    id=f"{room}-{i}", room=room,
                    x=rx + rng.uniform(-1.5, 1.5), y=ry + rng.uniform(-1.5, 1.5),
                    node=room, react_at=rng.uniform(15, 90)))

    # ---- queries ----------------------------------------------------------
    @property
    def sensors_online(self) -> bool:
        s = self.scenario.sensor_loss_at
        return s is None or self.t < s

    @property
    def done(self) -> bool:
        return self.t >= self.scenario.duration or all(o.status == "evacuated" for o in self.occupants)

    def area_counts(self) -> dict[str, int]:
        """People present per area: every room (zeros included) and occupied corridor nodes."""
        counts = {r: 0 for r in self.building.rooms}
        for o in self.occupants:
            if o.status != "evacuated" and self.building.nodes[o.node].kind != "exit":
                counts[o.node] = counts.get(o.node, 0) + 1
        return counts

    def snapshot(self) -> dict:
        return {
            "t": round(self.t, 2),
            "scenario": self.scenario.name,
            "duration": self.scenario.duration,
            "done": self.done,
            "sensors_online": self.sensors_online,
            "hazards": dict(self.hazards),
            "area_counts": self.area_counts(),
            "occupants": [
                {"id": o.id, "x": round(o.x, 2), "y": round(o.y, 2), "node": o.node,
                 "status": o.status, "guided": o.guided, "target_exit": o.target_exit,
                 "exposures": o.exposures, "evac_time": o.evac_time}
                for o in self.occupants
            ],
        }

    # ---- simulation -------------------------------------------------------
    def step(self, dt: float, robot_xy=None, guidance_exit: str | None = None,
             robot_hazards=(), announce: bool = False) -> None:
        self.t += dt
        current = self.scenario.hazards_at(self.t)
        self.new_hazards = [n for n in current if self.hazards.get(n) != current[n]]
        self.hazards = current
        robot_hazards = set(robot_hazards)
        for o in self.occupants:
            if o.status == "evacuated":
                continue
            self._exposure_check(o)
            self._robot_influence(o, robot_xy, guidance_exit, robot_hazards, announce)
            if o.status == "unaware":
                if self.t >= o.react_at:
                    o.status = "moving"
                    self._plan(o)
            elif o.status in ("waiting", "trapped"):
                if self.t >= o.wait_until:
                    o.status = "moving"
                    self._plan(o)
            elif o.status == "moving":
                self._move(o, dt)

    def _exposure_check(self, o: Occupant) -> None:
        kind = self.hazards.get(o.node)
        if kind in EXPOSURE_KINDS and self._dist_to(o, o.node) < NOTICE_DISTANCE:
            if o.node not in o.exposed_at:
                o.exposures += 1
                o.exposed_at.add(o.node)
            o.known_hazards.add(o.node)
            if o.status == "unaware":         # you notice a fire in your own room
                o.status = "moving"
                self._plan(o)

    def _robot_influence(self, o, robot_xy, guidance_exit, robot_hazards, announce) -> None:
        if robot_xy is None:
            return
        d = math.hypot(o.x - robot_xy[0], o.y - robot_xy[1])
        if guidance_exit and d <= GUIDE_RADIUS:
            learned = robot_hazards - o.known_hazards
            if not o.guided or o.target_exit != guidance_exit or learned:
                o.known_hazards |= robot_hazards
                o.guided = True
                o.target_exit = guidance_exit
                if o.status == "unaware":
                    o.status = "moving"
                if o.status == "moving":
                    self._plan(o)
        elif announce and d <= ANNOUNCE_RADIUS and o.status == "unaware":
            o.status = "moving"
            self._plan(o)

    def _plan(self, o: Occupant) -> None:
        b = self.building
        avoid = set(o.known_hazards)
        path = None
        if o.guided and o.target_exit and o.target_exit not in avoid:
            path, _ = b.shortest_path(o.node, o.target_exit, avoid)
        if path is None:
            exit_id, path, _ = b.nearest_exit(o.node, avoid)
            if exit_id is None:
                o.status, o.wait_until, o.path = "trapped", self.t + CONFUSION_DELAY, []
                return
            o.target_exit = exit_id
        # walk back to the last node first if standing between nodes
        o.path = path if self._dist_to(o, o.node) > 0.1 else path[1:]

    def _move(self, o: Occupant, dt: float) -> None:
        if not o.path:
            self._plan(o)
            return
        nxt = o.path[0]
        kind = self.hazards.get(nxt)
        if kind and nxt not in o.known_hazards and self._dist_to(o, nxt) <= NOTICE_DISTANCE:
            o.known_hazards.add(nxt)
            if kind in EXPOSURE_KINDS and nxt not in o.exposed_at:
                o.exposures += 1
                o.exposed_at.add(nxt)
            o.status, o.wait_until, o.path = "waiting", self.t + CONFUSION_DELAY, []
            return
        nx, ny = self.building.xy(nxt)
        d = math.hypot(nx - o.x, ny - o.y)
        step = WALK_SPEED * dt
        if d <= step:
            o.x, o.y, o.node = nx, ny, nxt
            o.path.pop(0)
            if self.building.nodes[nxt].kind == "exit" and nxt not in self.hazards:
                o.status, o.evac_time, o.path = "evacuated", round(self.t, 2), []
        else:
            o.x += (nx - o.x) / d * step
            o.y += (ny - o.y) / d * step

    def _dist_to(self, o: Occupant, node: str) -> float:
        x, y = self.building.xy(node)
        return math.hypot(o.x - x, o.y - y)
