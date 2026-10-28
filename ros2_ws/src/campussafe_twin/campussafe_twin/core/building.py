"""Building model: a graph of rooms, corridor waypoints and exits, plus wall geometry.

The layout is a placeholder single-corridor floor (40 m x 20 m). Replace the
coordinates in `lamar_cs_floor()` and `wall_segments()` with measurements from
real Lamar University floor plans; everything else adapts automatically.

Layout (y up):
    y=10  +---------+---------+---------+---------+
          |   N1    |   N2    |   N3    |   N4    |
    y=2   +--d------+--d------+--d------+--d------+
 EXIT_W   C02  C05  C10  C15  C20  C25  C30  C35  C38   EXIT_E   (corridor, y=0)
    y=-2  +--d-----------+---DOOR---+-----------d--+
          |      S1      |  LOBBY   |      S3      |
    y=-10 +--------------+---EXIT---+--------------+
                           EXIT_MAIN
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Node:
    id: str
    x: float
    y: float
    kind: str  # "corridor" | "room" | "exit"


class Building:
    def __init__(self, name: str, nodes: Iterable[Node], edges: Iterable[tuple[str, str]]):
        self.name = name
        self.nodes: dict[str, Node] = {n.id: n for n in nodes}
        self.adj: dict[str, list[tuple[str, float]]] = {n: [] for n in self.nodes}
        for a, b in edges:
            d = self.distance(a, b)
            self.adj[a].append((b, d))
            self.adj[b].append((a, d))

    # ---- geometry -------------------------------------------------------
    def distance(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    def xy(self, node_id: str) -> tuple[float, float]:
        n = self.nodes[node_id]
        return n.x, n.y

    @property
    def rooms(self) -> list[str]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "room")

    @property
    def exits(self) -> list[str]:
        return sorted(n.id for n in self.nodes.values() if n.kind == "exit")

    def nearest_node(self, x: float, y: float, kinds: tuple[str, ...] | None = None) -> str:
        cands = [n for n in self.nodes.values() if kinds is None or n.kind in kinds]
        return min(cands, key=lambda n: math.hypot(n.x - x, n.y - y)).id

    # ---- planning -------------------------------------------------------
    def shortest_path(self, start: str, goal: str, avoid: Iterable[str] = ()) -> tuple[list[str] | None, float]:
        """Dijkstra over the graph. Nodes in `avoid` are impassable, except the
        start node (you can always walk out of where you are)."""
        avoid = set(avoid) - {start}
        if goal in avoid or start not in self.nodes or goal not in self.nodes:
            return None, math.inf
        dist = {start: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w in self.adj[u]:
                if v in avoid:
                    continue
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            return None, math.inf
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return path[::-1], dist[goal]

    def nearest_exit(self, start: str, avoid: Iterable[str] = ()) -> tuple[str | None, list[str] | None, float]:
        avoid = set(avoid)
        best = (None, None, math.inf)
        for e in self.exits:
            if e in avoid:
                continue
            path, d = self.shortest_path(start, e, avoid)
            if path is not None and d < best[2]:
                best = (e, path, d)
        return best


def lamar_cs_floor() -> Building:
    corridor_x = [2, 5, 10, 15, 20, 25, 30, 35, 38]
    corridor = [Node(f"C{x:02d}", float(x), 0.0, "corridor") for x in corridor_x]
    rooms = [
        Node("N1", 5, 6, "room"), Node("N2", 15, 6, "room"),
        Node("N3", 25, 6, "room"), Node("N4", 35, 6, "room"),
        Node("S1", 5, -6, "room"), Node("LOBBY", 20, -6, "room"), Node("S3", 35, -6, "room"),
    ]
    exits = [
        Node("EXIT_WEST", -1, 0, "exit"), Node("EXIT_EAST", 41, 0, "exit"),
        Node("EXIT_MAIN", 20, -11, "exit"),
    ]
    edges = [(corridor[i].id, corridor[i + 1].id) for i in range(len(corridor) - 1)]
    edges += [
        ("N1", "C05"), ("N2", "C15"), ("N3", "C25"), ("N4", "C35"),
        ("S1", "C05"), ("LOBBY", "C20"), ("S3", "C35"),
        ("EXIT_WEST", "C02"), ("EXIT_EAST", "C38"), ("EXIT_MAIN", "LOBBY"),
    ]
    return Building("Lamar CS Building (placeholder floor)", corridor + rooms + exits, edges)


def _with_gaps(a: float, b: float, gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Split the interval [a, b] around door gaps."""
    pieces, cur = [], a
    for g0, g1 in sorted(gaps):
        if g0 > cur:
            pieces.append((cur, g0))
        cur = max(cur, g1)
    if cur < b:
        pieces.append((cur, b))
    return pieces


def wall_segments() -> list[tuple[float, float, float, float]]:
    """Wall centre-lines (x1, y1, x2, y2) matching `lamar_cs_floor()`."""
    walls: list[tuple[float, float, float, float]] = []

    def horiz(y, x0, x1, gaps=()):
        walls.extend((p0, y, p1, y) for p0, p1 in _with_gaps(x0, x1, list(gaps)))

    def vert(x, y0, y1, gaps=()):
        walls.extend((x, p0, x, p1) for p0, p1 in _with_gaps(y0, y1, list(gaps)))

    door = 0.8  # half door width
    horiz(10, 0, 40)                                   # outer north
    horiz(-10, 0, 40, [(19, 21)])                      # outer south, main exit
    vert(0, -10, 10, [(-1, 1)])                        # outer west, west exit
    vert(40, -10, 10, [(-1, 1)])                       # outer east, east exit
    horiz(2, 0, 40, [(x - door, x + door) for x in (5, 15, 25, 35)])
    horiz(-2, 0, 40, [(5 - door, 5 + door), (18, 22), (35 - door, 35 + door)])
    for x in (10, 20, 30):
        vert(x, 2, 10)                                 # north room dividers
    for x in (15, 25):
        vert(x, -10, -2)                               # south room dividers
    return walls
