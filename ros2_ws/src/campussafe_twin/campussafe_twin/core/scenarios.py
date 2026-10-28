"""Emergency scenarios: timed hazard events, occupants and sensor failures.

Hazard kinds
    fire    - impassable; people who get close are exposed
    smoke   - impassable; people who get close are exposed
    blocked - impassable (e.g. a jammed exit door); no exposure
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HazardEvent:
    t: float     # seconds after the alarm
    node: str
    kind: str    # "fire" | "smoke" | "blocked"


DEFAULT_OCCUPANTS = {"N1": 4, "N2": 6, "N3": 5, "N4": 3, "S1": 4, "LOBBY": 6, "S3": 4}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    events: tuple[HazardEvent, ...]
    occupants: dict = field(default_factory=lambda: dict(DEFAULT_OCCUPANTS))
    robot_start: str = "C20"
    sensor_loss_at: float | None = None   # building sensors go offline at this time
    duration: float = 600.0

    def hazards_at(self, t: float) -> dict[str, str]:
        active: dict[str, str] = {}
        for ev in sorted(self.events, key=lambda e: e.t):
            if ev.t <= t:
                active[ev.node] = ev.kind
        return active

    @property
    def hazard_nodes(self) -> list[str]:
        return sorted({e.node for e in self.events})


SCENARIOS: dict[str, Scenario] = {
    "fire_north": Scenario(
        "fire_north",
        "Fire starts in room N2; smoke spreads west along the corridor.",
        (HazardEvent(0, "N2", "fire"), HazardEvent(45, "C15", "smoke"), HazardEvent(90, "C10", "smoke")),
    ),
    "blocked_exit": Scenario(
        "blocked_exit",
        "West exit door is jammed; later smoke cuts off the east exit route.",
        (HazardEvent(0, "EXIT_WEST", "blocked"), HazardEvent(60, "C38", "smoke")),
    ),
    "sensor_loss": Scenario(
        "sensor_loss",
        "Corridor fire at C30; sensors fail at 20 s; smoke reaches N3's door at 50 s.",
        (HazardEvent(0, "C30", "fire"), HazardEvent(50, "C25", "smoke")),
        sensor_loss_at=20.0,
    ),
    "multi_hazard": Scenario(
        "multi_hazard",
        "Fire in N3, east exit blocked, west corridor smoke, sensor loss at 60 s.",
        (HazardEvent(0, "N3", "fire"), HazardEvent(30, "EXIT_EAST", "blocked"), HazardEvent(90, "C10", "smoke")),
        sensor_loss_at=60.0,
    ),
}
