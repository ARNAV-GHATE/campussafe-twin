"""Task 2 controllers and the observation they all share.

  WaypointPath     global plan on the building graph (re-planned when a hazard appears)
  ClassicalController  pure-pursuit path tracker + LiDAR / hazard reflexes: the baseline,
                   and (fed the TRUE pose) the privileged teacher the NCM planner imitates
  make_obs         42-D egocentric observation used by the world model and the NCM planner
  tokens           (v, w) commands <-> 35 discrete motion tokens (5 speeds x 7 turn rates)
"""
from __future__ import annotations

import math

import numpy as np

from ..core.building import Building
from .routes import Route, resolve_goal
from .sim import BEAM_ANGLES, LIDAR_MAX, V_MAX, W_MAX, wrap

# ------------------------------------------------------------------ motion tokens
V_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
W_LEVELS = np.array([-1.5, -0.8, -0.35, 0.0, 0.35, 0.8, 1.5])
N_TOKENS = len(V_LEVELS) * len(W_LEVELS)
STOP_TOKEN = 3                                  # v = 0, w = 0


def to_token(v: float, w: float) -> int:
    return int(np.abs(V_LEVELS - v).argmin()) * len(W_LEVELS) + int(np.abs(W_LEVELS - w).argmin())


def from_token(tok: int) -> tuple[float, float]:
    return float(V_LEVELS[tok // len(W_LEVELS)]), float(W_LEVELS[tok % len(W_LEVELS)])


# ------------------------------------------------------------------ global path
class WaypointPath:
    """Node path to the goal, tracked from the robot's *estimated* pose."""

    REACHED = 0.5   # m

    def __init__(self, route: Route, b: Building):
        self.route, self.b = route, b
        self.goal: str | None = None
        self.nodes: list[str] = []
        self.pts = np.zeros((0, 2))
        self.i = 1                                   # index of the waypoint being driven to
        self.done = False
        self.trapped = False

    def plan(self, x: float, y: float, hazards: dict) -> None:
        start = self.b.nearest_node(x, y, ("corridor", "room"))
        goal, path = resolve_goal(self.route, self.b, start, hazards)
        if path is None:
            self.trapped, self.nodes, self.pts = True, [], np.zeros((0, 2))
            return
        self.trapped = False
        self.goal = goal
        self.nodes = path
        self.pts = np.array([[x, y]] + [self.b.xy(n) for n in path[1:]], dtype=float)
        if len(self.pts) == 1:
            self.pts = np.vstack([self.pts, self.pts])
        self.i = 1

    def update(self, x: float, y: float) -> None:
        if self.trapped or len(self.pts) == 0:
            return
        while self.i < len(self.pts) - 1:
            a, b = self.pts[self.i - 1], self.pts[self.i]
            seg = b - a
            L2 = float(seg @ seg)
            prog = float((np.array([x, y]) - a) @ seg / L2) if L2 > 1e-9 else 1.0
            if math.hypot(*(b - [x, y])) < self.REACHED or prog > 1.0:
                self.i += 1
            else:
                break
        if math.hypot(*(self.pts[-1] - [x, y])) < self.REACHED:
            self.done = True

    def lookahead(self, x: float, y: float, L: float = 1.0) -> np.ndarray:
        """Point L metres ahead along the path from the projection of (x, y)."""
        a, b = self.pts[self.i - 1], self.pts[self.i]
        seg = b - a
        L2 = float(seg @ seg)
        prog = float(np.clip((np.array([x, y]) - a) @ seg / L2, 0, 1)) if L2 > 1e-9 else 1.0
        p = a + prog * seg
        rem = L
        j = self.i
        cur = p
        while j < len(self.pts):
            nxt = self.pts[j]
            d = math.hypot(*(nxt - cur))
            if d >= rem:
                return cur + (nxt - cur) * (rem / d)
            rem -= d
            cur = nxt
            j += 1
        return self.pts[-1].copy()

    def next_points(self, k: int = 2) -> np.ndarray:
        idx = [min(self.i + j, len(self.pts) - 1) for j in range(k)]
        return self.pts[idx]

    def remaining(self, x: float, y: float) -> float:
        if len(self.pts) == 0:
            return 0.0
        d = math.hypot(*(self.pts[self.i] - [x, y]))
        return d + float(np.sum(np.linalg.norm(np.diff(self.pts[self.i:], axis=0), axis=1)))


# ------------------------------------------------------------------ observation
OBS_DIM = 32 + 4 + 2 + 2 + 3 + 1


def to_robot_frame(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    d = pts - pose[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def make_obs(scan: np.ndarray, pose: np.ndarray, path: WaypointPath, proprio: np.ndarray,
             hazard: np.ndarray) -> np.ndarray:
    """[32 LiDAR / 8 m | next 2 waypoints in robot frame / 5 m | 1 m look-ahead point in robot frame |
        v, w | hazard (d, cos, sin) | remaining path / 40 m]   (44-D)"""
    if len(path.pts) and not path.trapped:
        wp = to_robot_frame(path.next_points(2), pose)
        n = np.linalg.norm(wp, axis=1, keepdims=True)
        wp = wp * np.minimum(1.0, 5.0 / np.maximum(n, 1e-6)) / 5.0
        la = to_robot_frame(path.lookahead(pose[0], pose[1], 1.0)[None], pose)[0]
        rem = min(path.remaining(pose[0], pose[1]), 40.0) / 40.0
    else:
        wp, la, rem = np.zeros((2, 2)), np.zeros(2), 0.0
    return np.concatenate([scan / LIDAR_MAX, wp.ravel(), la, proprio / np.array([V_MAX, W_MAX]), hazard,
                           [rem]]).astype(np.float32)


# ------------------------------------------------------------------ classical controller
FRONT = np.abs(BEAM_ANGLES) <= math.radians(30)
LEFT = (BEAM_ANGLES > math.radians(30)) & (BEAM_ANGLES < math.radians(120))
RIGHT = (BEAM_ANGLES < -math.radians(30)) & (BEAM_ANGLES > -math.radians(120))


class ClassicalController:
    """Pure pursuit on the waypoint path, slowed by large heading error, with three reflexes:
    slow down for close obstacles ahead, keep off side walls, steer away from fire/smoke."""

    name = "classical"

    def __init__(self, lookahead: float = 1.0):
        self.L = lookahead

    def act(self, pose: np.ndarray, path: WaypointPath, scan: np.ndarray, hazard: np.ndarray) -> tuple[float, float]:
        if path.done or path.trapped or len(path.pts) == 0:
            return 0.0, 0.0
        x, y, th = pose
        tx, ty = path.lookahead(x, y, self.L)
        err = float(wrap(math.atan2(ty - y, tx - x) - th))
        w = 2.5 * err
        v = V_MAX * max(0.0, math.cos(err)) ** 3 if abs(err) < 1.2 else 0.0
        dist_goal = math.hypot(*(path.pts[-1] - [x, y]))
        v = min(v, 0.4 + 0.6 * min(1.0, dist_goal / 2.0))
        front = float(scan[FRONT].min())
        if front < 0.8:
            v = min(v, max(0.0, (front - 0.35) * 1.2))
        left, right = float(scan[LEFT].min()), float(scan[RIGHT].min())
        if min(left, right) < 0.55:
            w += 1.2 * (0.55 - right) if right < left else -1.2 * (0.55 - left)
        if hazard[0] < 0.45:                         # fire / smoke within 2.7 m
            bearing = math.atan2(hazard[2], hazard[1])
            if abs(bearing) < math.pi / 2:
                w -= 1.0 * math.copysign(1.0, bearing) * (0.45 - hazard[0]) / 0.45
        return float(np.clip(v, 0.0, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))
