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


def random_scenario(building, seed: int, name: str | None = None) -> Scenario:
    """Randomized emergency for training the world model and NCM planner.

    Physics (kept simple and documented):
      * 1 fire at t=0 in a random room or corridor node (2 fires with p=0.2)
      * every 15 s, smoke spreads from each fire/smoke node to each adjacent
        room/corridor node with probability p_spread ~ U(0.08, 0.25)
      * p=0.15: a secondary, independent fire starts at t ~ U(20, 120)
      * p=0.35: one exit is blocked at t ~ U(0, 60)
      * p=0.25: building sensors fail at t ~ U(10, 80)
      * 0-7 people per room, robot starts on a random corridor node
    The four named scenarios are not used for training; they are the test set.
    """
    import random as _random
    rng = _random.Random(seed)
    inner = [n for n, v in building.nodes.items() if v.kind != "exit"]
    corridors = [n for n, v in building.nodes.items() if v.kind == "corridor"]
    events: list[HazardEvent] = []
    burning: dict[str, float] = {}
    for _ in range(2 if rng.random() < 0.2 else 1):
        n = rng.choice(inner)
        burning.setdefault(n, 0.0)
        events.append(HazardEvent(0.0, n, "fire"))
    secondary = (rng.uniform(20, 120), rng.choice(inner)) if rng.random() < 0.15 else None
    p_spread = rng.uniform(0.08, 0.25)
    t = 15.0
    while t <= 300:
        if secondary and secondary[0] <= t and secondary[1] not in burning:
            burning[secondary[1]] = t
            events.append(HazardEvent(round(secondary[0], 1), secondary[1], "fire"))
        for src in [n for n, t0 in burning.items() if t0 < t]:
            for nb, _ in building.adj[src]:
                if nb in burning or building.nodes[nb].kind == "exit":
                    continue
                if rng.random() < p_spread:
                    burning[nb] = t
                    events.append(HazardEvent(t, nb, "smoke"))
        t += 15.0
    if rng.random() < 0.35:
        events.append(HazardEvent(round(rng.uniform(0, 60), 1), rng.choice(building.exits), "blocked"))
    occupants = {r: rng.randint(0, 7) for r in building.rooms}
    if sum(occupants.values()) == 0:
        occupants[rng.choice(building.rooms)] = 3
    start = rng.choice([c for c in corridors if c not in burning] or corridors)
    return Scenario(name or f"random_{seed}", "randomized training scenario", tuple(events), occupants,
                    robot_start=start, sensor_loss_at=round(rng.uniform(10, 80), 1) if rng.random() < 0.25 else None,
                    duration=300.0)
