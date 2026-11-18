"""Task 2 simulator: a differential-drive robot with a 2-D LiDAR in the campus building.

This is the offline twin of the Gazebo robot (`campus_robot` in worlds/campus_building.sdf):
same walls (core/building.py: wall_segments), same LiDAR (32 beams, 360 deg, 8 m),
same speed limits (1.0 m/s, 1.5 rad/s), same /odom + /scan + /cmd_vel interface.
It only needs numpy, so thousands of episodes run in minutes for training and
statistics; the ROS 2 nodes in nodes/nav_*.py run the same controllers against Gazebo.

Physics (kept simple and documented):
  * unicycle kinematics, first-order velocity response with acceleration limits
  * the robot is a disc of radius ROBOT_R; walls have thickness WALL_T; a move that would
    bring the disc into a wall is cancelled (the robot stops) and counts as a collision
  * fire / smoke are discs of radius HAZARD_R around a building node; they do not block
    the LiDAR or the robot, but time spent inside one is "hazard exposure"
  * a "blocked" exit is a physical barrier across the doorway (the LiDAR sees it)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.building import Building, lamar_cs_floor, wall_segments

DT = 0.1                 # s, simulation / sensor period (10 Hz, like the Gazebo bridge)
ROBOT_R = 0.30           # m, footprint radius
WALL_T = 0.20            # m, wall thickness (matches tools/generate_world.py)
V_MAX, W_MAX = 1.0, 1.5  # m/s, rad/s
A_V, A_W = 1.5, 4.0      # m/s^2, rad/s^2 acceleration limits
N_BEAMS = 32
LIDAR_MAX = 8.0          # m
LIDAR_NOISE = 0.02       # m (1 sigma)
ODOM_NOISE = 0.004       # m per step, pose noise of the localization source
PROPRIO_NOISE = (0.02, 0.03)  # wheel-encoder velocity noise (m/s, rad/s): dead reckoning drifts
HAZARD_R = 2.0           # m, fire / smoke radius
HAZARD_SENSE = 6.0       # m, onboard thermal / smoke sensing range (same as core/knowledge.py)
EXPOSURE_KINDS = ("fire", "smoke")

BEAM_ANGLES = np.linspace(-math.pi, math.pi, N_BEAMS, endpoint=False)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Pose:
    x: float
    y: float
    th: float

    def as_array(self):
        return np.array([self.x, self.y, self.th])


class WallMap:
    """Static wall geometry plus dynamic barriers (blocked exits). Raycasting and collision."""

    def __init__(self, building: Building | None = None):
        self.b = building or lamar_cs_floor()
        self.static = np.array(wall_segments(), dtype=float)       # [S, 4] x1 y1 x2 y2
        self.barriers: dict[str, np.ndarray] = {}
        self.segs = self.static

    def set_blocked(self, nodes) -> None:
        """Put a barrier across each blocked exit doorway (exits sit just outside a door gap)."""
        bars = {}
        for n in nodes:
            if self.b.nodes[n].kind != "exit":
                continue
            x, y = self.b.xy(n)
            if abs(y) < 2:   # west / east doors in vertical walls at x = 0 / 40
                xw = 0.0 if x < 20 else 40.0
                bars[n] = np.array([[xw, -1.2, xw, 1.2]])
            else:            # main exit door in the south wall
                bars[n] = np.array([[18.8, -10.0, 21.2, -10.0]])
        self.barriers = bars
        self.segs = np.vstack([self.static] + list(bars.values())) if bars else self.static

    # ------------------------------------------------------------------ geometry
    def raycast(self, x: float, y: float, th: float, angles=BEAM_ANGLES, max_r=LIDAR_MAX) -> np.ndarray:
        ang = th + angles
        d = np.stack([np.cos(ang), np.sin(ang)], 1)                     # [B, 2]
        a, b = self.segs[:, :2], self.segs[:, 2:]
        e = b - a                                                       # [S, 2]
        ap = a - np.array([x, y])                                       # [S, 2]
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]          # [B, S]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ap[None, :, 0] * e[None, :, 1] - ap[None, :, 1] * e[None, :, 0]) / denom
            u = (ap[None, :, 0] * d[:, None, 1] - ap[None, :, 1] * d[:, None, 0]) / denom
        ok = (np.abs(denom) > 1e-9) & (t > 0) & (u >= -0.02) & (u <= 1.02)
        r = np.where(ok, t, np.inf).min(1) - WALL_T / 2
        return np.clip(r, 0.0, max_r)

    def clearance(self, x: float, y: float) -> float:
        """Distance from a point to the nearest wall surface."""
        a, b = self.segs[:, :2], self.segs[:, 2:]
        e = b - a
        p = np.array([x, y])
        t = np.clip(((p - a) * e).sum(1) / np.maximum((e * e).sum(1), 1e-12), 0, 1)
        q = a + t[:, None] * e
        return float(np.sqrt(((q - p) ** 2).sum(1)).min() - WALL_T / 2)


@dataclass
class HazardEvent:
    t: float
    node: str
    kind: str  # fire | smoke | blocked


@dataclass
class NavState:
    pose: Pose
    v: float = 0.0
    w: float = 0.0
    t: float = 0.0
    collisions: int = 0
    in_contact: bool = False
    exposure_s: float = 0.0
    path_len: float = 0.0
    min_clearance: float = math.inf
    hazards: dict = field(default_factory=dict)


class NavWorld:
    """Ground-truth robot + building. Sensors return clean values; attacks happen later."""

    def __init__(self, start: Pose, events: tuple[HazardEvent, ...] = (), seed: int = 0,
                 building: Building | None = None):
        self.map = WallMap(building)
        self.b = self.map.b
        self.events = tuple(sorted(events, key=lambda e: e.t))
        self.rng = np.random.default_rng(seed)
        self.s = NavState(Pose(start.x, start.y, start.th))
        self.new_hazards: list[str] = []
        self._update_hazards()

    # ------------------------------------------------------------------ hazards
    def _update_hazards(self) -> None:
        active = {}
        for ev in self.events:
            if ev.t <= self.s.t:
                active[ev.node] = ev.kind
        self.new_hazards = [n for n in active if self.s.hazards.get(n) != active[n]]
        if self.new_hazards or len(active) != len(self.s.hazards):
            self.map.set_blocked([n for n, k in active.items() if k == "blocked"])
        self.s.hazards = active

    def exposure_hazards(self):
        return [(self.b.xy(n), k) for n, k in self.s.hazards.items() if k in EXPOSURE_KINDS]

    # ------------------------------------------------------------------ sensors (truth)
    def lidar(self) -> np.ndarray:
        p = self.s.pose
        r = self.map.raycast(p.x, p.y, p.th)
        return np.clip(r + self.rng.normal(0, LIDAR_NOISE, r.shape), 0.0, LIDAR_MAX)

    def odom(self) -> np.ndarray:
        p = self.s.pose
        return np.array([p.x, p.y, p.th]) + self.rng.normal(0, ODOM_NOISE, 3) * np.array([1, 1, 0.5])

    def proprio(self) -> np.ndarray:
        """Wheel-encoder velocities: onboard, not networked, trusted (but noisy)."""
        return np.array([self.s.v, self.s.w]) + self.rng.normal(0, 1, 2) * np.array(PROPRIO_NOISE)

    def hazard_sense(self, pose: np.ndarray | None = None) -> np.ndarray:
        """Onboard thermal/smoke sensor: (dist/6, cos, sin) of the nearest fire/smoke centre
        within HAZARD_SENSE in the robot frame, or (1, 0, 0) if none."""
        p = self.s.pose
        best = None
        for (hx, hy), _ in self.exposure_hazards():
            d = math.hypot(hx - p.x, hy - p.y)
            if d <= HAZARD_SENSE and (best is None or d < best[0]):
                best = (d, math.atan2(hy - p.y, hx - p.x) - p.th)
        if best is None:
            return np.array([1.0, 0.0, 0.0])
        return np.array([best[0] / HAZARD_SENSE, math.cos(best[1]), math.sin(best[1])])

    # ------------------------------------------------------------------ dynamics
    def step(self, v_cmd: float, w_cmd: float, dt: float = DT) -> None:
        s = self.s
        v_cmd = float(np.clip(v_cmd, -0.3, V_MAX))
        w_cmd = float(np.clip(w_cmd, -W_MAX, W_MAX))
        s.v += float(np.clip(v_cmd - s.v, -A_V * dt, A_V * dt))
        s.w += float(np.clip(w_cmd - s.w, -A_W * dt, A_W * dt))
        p = s.pose
        th = p.th + s.w * dt
        nx = p.x + s.v * math.cos(p.th + 0.5 * s.w * dt) * dt
        ny = p.y + s.v * math.sin(p.th + 0.5 * s.w * dt) * dt
        if self.map.clearance(nx, ny) < ROBOT_R:
            if not s.in_contact:
                s.collisions += 1
            s.in_contact = True
            s.v = 0.0
            nx, ny = p.x, p.y
        else:
            # hysteresis: a contact ends only once the robot is 15 cm clear of the wall again (one step
            # at full speed is 10 cm, so a blocked move can stop that far out),
            # so pushing against a wall counts as one collision, not one per step
            s.in_contact = s.in_contact and self.map.clearance(nx, ny) < ROBOT_R + 0.15
            s.path_len += math.hypot(nx - p.x, ny - p.y)
        s.pose = Pose(nx, ny, float(wrap(th)))
        s.min_clearance = min(s.min_clearance, self.map.clearance(nx, ny))
        for (hx, hy), _ in self.exposure_hazards():
            if math.hypot(hx - nx, hy - ny) < HAZARD_R:
                s.exposure_s += dt
                break
        s.t += dt
        self._update_hazards()
