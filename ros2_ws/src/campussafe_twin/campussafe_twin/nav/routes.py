"""Task 2 routes: where the robot has to drive, and what happens on the way.

A route is a start node, a goal ("EXIT" = nearest exit that is safe right now, re-evaluated
whenever a new hazard appears, exactly like the evacuation task) and timed hazard events.
The eight TEST_ROUTES are held out: the learned models are trained on random_route() only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.building import Building, lamar_cs_floor
from .sim import HazardEvent, Pose

E = HazardEvent


@dataclass(frozen=True)
class Route:
    name: str
    start: str
    goal: str                          # node id, or "EXIT" for the nearest safe exit
    events: tuple = ()
    heading: float = 0.0               # initial yaw (rad)
    t_max: float = 120.0

    def start_pose(self, b: Building) -> Pose:
        x, y = b.xy(self.start)
        return Pose(x, y, self.heading)


TEST_ROUTES: dict[str, Route] = {r.name: r for r in (
    Route("corridor_west", "C20", "EXIT_WEST", (E(0, "N2", "fire"),), heading=math.pi),
    Route("room_to_main", "N1", "EXIT_MAIN", (E(0, "N4", "fire"),), heading=-math.pi / 2),
    Route("lobby_to_n4", "LOBBY", "N4", (E(0, "S1", "smoke"),), heading=math.pi / 2),
    Route("long_haul", "S3", "EXIT_WEST", (E(0, "LOBBY", "fire"),), heading=math.pi / 2),
    Route("blocked_east", "N4", "EXIT", (E(0, "EXIT_EAST", "blocked"),), heading=-math.pi / 2),
    Route("smoke_reroute", "C30", "EXIT", (E(2.5, "C35", "smoke"),), heading=0.0),
    Route("fire_behind", "N3", "EXIT", (E(0, "N2", "fire"), E(5, "C20", "smoke")), heading=-math.pi / 2),
    Route("west_room", "S1", "EXIT", (E(0, "EXIT_WEST", "blocked"), E(8, "LOBBY", "smoke")), heading=math.pi / 2),
)}


def resolve_goal(route: Route, b: Building, node: str, hazards: dict) -> tuple[str | None, list[str] | None]:
    """Goal node and node path from `node`, avoiding every active hazard."""
    avoid = set(hazards)
    if route.goal == "EXIT":
        goal, path, _ = b.nearest_exit(node, avoid)
        return goal, path
    path, _ = b.shortest_path(node, route.goal, avoid)
    return (route.goal, path) if path else (None, None)


def random_route(rng: np.random.Generator, b: Building | None = None, max_tries: int = 50) -> Route:
    """Random training route: random start / goal, 0-3 hazards (some appearing mid-route),
    guaranteed reachable at t=0. Never identical to a test route."""
    b = b or lamar_cs_floor()
    places = [n for n in b.nodes if b.nodes[n].kind != "exit"]
    test_keys = {(r.start, r.goal, tuple((e.node, e.kind) for e in r.events)) for r in TEST_ROUTES.values()}
    for _ in range(max_tries):
        start = str(rng.choice(places))
        goal = "EXIT" if rng.random() < 0.5 else str(rng.choice([n for n in b.nodes if n != start]))
        events = []
        for _ in range(int(rng.integers(0, 4))):
            kind = str(rng.choice(["fire", "smoke", "smoke", "blocked"]))
            pool = b.exits if kind == "blocked" else [n for n in places if n != start]
            node = str(rng.choice(pool))
            t = 0.0 if rng.random() < 0.5 else float(rng.uniform(2, 25))
            if node != goal:
                events.append(HazardEvent(round(t, 1), node, kind))
        r = Route("train", start, goal, tuple(events), heading=float(rng.uniform(-math.pi, math.pi)))
        if (r.start, r.goal, tuple((e.node, e.kind) for e in r.events)) in test_keys:
            continue
        g, path = resolve_goal(r, b, start, {e.node: e.kind for e in events if e.t == 0})
        if path and len(path) >= 2:
            return r
    return Route("train", "C20", "EXIT_MAIN", (), 0.0)
