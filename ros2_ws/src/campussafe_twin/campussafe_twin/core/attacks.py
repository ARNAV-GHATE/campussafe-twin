"""Cyberattack models for resilience experiments.

Threat model (documented in docs/NCM_WORLD_MODEL.md):

  Building sensors (smoke detectors, occupancy counters) report to the robot
  over a network feed. The robot's own onboard sensors (camera, smoke/heat,
  6-8 m range) are trusted. Attacks hit three surfaces:

  compromised sensor  The attacker controls the building management system
                      (BMS) or a sensor, so false data is *validly signed*.
                      Cryptography cannot catch this; only a plausibility
                      check (world model) and onboard cross-checks can.
  network             The attacker is on the network but has no key: it can
                      drop, delay or replay messages and forge unsigned ones.
  command channel     The attacker injects forged robot actions
                      (guide_to_exit / navigate_to) after the safety validator.

Every attack starts at `start` seconds after the alarm (default 10 s), the
most damaging moment: people are just starting to move.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .building import Building
from .security import forge


@dataclass(frozen=True)
class Attack:
    name: str
    description: str
    start: float = 10.0
    sensor: tuple[str, ...] = ()     # compromised-sensor tampering: false_hazard | suppress | count_spoof
    network: str | None = None       # replay | dos
    command: bool = False            # forged action injection
    command_period: float = 15.0
    dos_drop_rate: float = 0.95
    params: dict = field(default_factory=dict)


ATTACKS: dict[str, Attack] = {
    "none": Attack("none", "Normal operation, no attack."),
    "false_hazard": Attack(
        "false_hazard", "Compromised smoke detector reports a fake fire at the busiest safe exit "
        "(denial-of-exit). Validly signed.", sensor=("false_hazard",)),
    "hazard_suppression": Attack(
        "hazard_suppression", "Compromised BMS hides every hazard that appears after the attack starts "
        "(stealthy: earlier hazards are still reported). Validly signed.", sensor=("suppress",)),
    "occupancy_spoof": Attack(
        "occupancy_spoof", "Compromised occupancy counters report a decoy crowd in the room farthest "
        "from the people and empty rooms everywhere else, luring the robot away.", sensor=("count_spoof",)),
    "replay": Attack(
        "replay", "Network attacker records the sensor feed at the attack start and replays it, "
        "freezing the robot's view of the building.", network="replay"),
    "command_injection": Attack(
        "command_injection", "Network attacker injects forged robot commands every 15 s: guide people "
        "to the worst exit, and send the robot to a far room.", command=True),
    "dos": Attack(
        "dos", "Network attacker drops 95% of sensor messages (jamming / flooding).", network="dos"),
    "coordinated": Attack(
        "coordinated", "Fake exit fire + hazard suppression + command injection at once.",
        sensor=("false_hazard", "suppress"), command=True),
}


def bms_reading(snapshot: dict) -> dict:
    """What the building management system can measure (no individual tracking)."""
    return {"t": snapshot["t"], "sensors_online": snapshot["sensors_online"],
            "hazards": dict(snapshot["hazards"]), "area_counts": dict(snapshot["area_counts"])}


class Attacker:
    """Stateful attacker for one episode. Same object is used offline and by the ROS attack_injector node."""

    def __init__(self, attack: Attack, building: Building, seed: int = 0):
        self.a = attack
        self.b = building
        self.rng = random.Random(10_000 + seed)
        self.fake_exit: str | None = None
        self.frozen_hazards: dict | None = None
        self.decoy_room: str | None = None
        self.recorded_env: dict | None = None
        self.next_cmd = attack.start
        self.cmd_toggle = False
        self.forged_sent = 0

    def active(self, t: float) -> bool:
        return self.a.name != "none" and t >= self.a.start

    # ---- compromised sensor: modifies the reading before it is signed ------------
    def tamper_reading(self, reading: dict, truth: dict) -> dict:
        t = reading["t"]
        if "suppress" in self.a.sensor and not self.active(t):
            self.frozen_hazards = dict(reading["hazards"])     # what was public before the attack
        if not self.active(t) or not self.a.sensor or not reading["sensors_online"]:
            return reading
        r = copy.deepcopy(reading)
        if "suppress" in self.a.sensor:
            if self.frozen_hazards is None:
                self.frozen_hazards = dict(reading["hazards"])
            r["hazards"] = dict(self.frozen_hazards)
        if "false_hazard" in self.a.sensor:
            if self.fake_exit is None:
                self.fake_exit = self._busiest_safe_exit(truth)
            if self.fake_exit:
                r["hazards"][self.fake_exit] = "fire"
        if "count_spoof" in self.a.sensor:
            if self.decoy_room is None:
                self.decoy_room = self._decoy(truth)
            r["area_counts"] = {a: 0 for a in r["area_counts"]}
            r["area_counts"][self.decoy_room] = self.a.params.get("decoy_count", 25)
        return r

    # ---- network: acts on signed envelopes --------------------------------------
    def tamper_envelope(self, env: dict | None, t: float) -> dict | None:
        if env is None or not self.active(t) or self.a.network is None:
            if self.a.network == "replay" and env is not None:
                self.recorded_env = copy.deepcopy(env)
            return env
        if self.a.network == "dos":
            return None if self.rng.random() < self.a.dos_drop_rate else env
        if self.a.network == "replay":
            return copy.deepcopy(self.recorded_env) if self.recorded_env else env
        return env

    # ---- command channel ---------------------------------------------------------
    def forged_command(self, t: float, truth: dict, robot_node: str) -> dict | None:
        if not self.a.command or not self.active(t) or t < self.next_cmd:
            return None
        self.next_cmd += self.a.command_period
        self.cmd_toggle = not self.cmd_toggle
        if self.cmd_toggle:
            payload = {"action": "guide_to_exit", "target": self._worst_exit(truth), "message": ""}
        else:
            far = max(self.b.rooms, key=lambda r: self.b.shortest_path(robot_node, r)[1]
                      if self.b.shortest_path(robot_node, r)[0] else -1)
            payload = {"action": "navigate_to", "target": far, "message": ""}
        self.forged_sent += 1
        return forge(payload, t)

    # ---- attacker's target selection (it can read the true sensor data) -----------
    def _busiest_safe_exit(self, truth: dict) -> str | None:
        hz = set(truth["hazards"])
        load = {e: 0 for e in self.b.exits if e not in hz}
        for area, c in truth["area_counts"].items():
            if c and area not in hz:
                e, _, _ = self.b.nearest_exit(area, hz)
                if e in load:
                    load[e] += c
        return max(load, key=load.get) if load else None

    def _worst_exit(self, truth: dict) -> str:
        hz = set(truth["hazards"])
        if any(e in hz for e in self.b.exits):
            return next(e for e in self.b.exits if e in hz)
        occupied = [a for a, c in truth["area_counts"].items() if c] or self.b.rooms

        def cost(e):
            return sum(self.b.shortest_path(a, e)[1] for a in occupied)
        return max(self.b.exits, key=cost)

    def _decoy(self, truth: dict) -> str:
        """Room farthest from the biggest crowd: luring the robot there costs the most."""
        hz = set(truth["hazards"])
        counts = truth["area_counts"]
        crowd = max(counts, key=lambda a: counts[a] or 0)
        rooms = [r for r in self.b.rooms if r not in hz and r != crowd]
        return max(rooms, key=lambda r: self.b.distance(r, crowd))
