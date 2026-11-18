"""Fixed-size encodings shared by the world model and the NCM planner.

Node order is the building graph's insertion order (corridor, rooms, exits),
so the encodings stay valid as long as lamar_cs_floor() keeps its node list.
If you change the floor plan, retrain (python -m campussafe_twin.learning.train).
"""
from __future__ import annotations


import numpy as np

from ..core.actions import Action
from ..core.building import Building
from ..core.knowledge import HAZARD_SENSE_RADIUS, PEOPLE_SEE_RADIUS

HAZ_KINDS = ("none", "fire", "smoke", "blocked")
COUNT_SCALE = 10.0
T_SCALE = 300.0
ANNOUNCE_TEXT = "Emergency! Follow the robot or the signs to the nearest safe exit."


class Layout:
    """Index tables for one building."""

    def __init__(self, b: Building):
        self.b = b
        self.nodes = list(b.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.robot_nodes = [n for n in self.nodes if b.nodes[n].kind != "exit"]
        self.ridx = {n: i for i, n in enumerate(self.robot_nodes)}
        self.exits = list(b.exits)
        self.N = len(self.nodes)
        self.R = len(self.robot_nodes)
        self.E = len(self.exits)
        xy = np.array([b.xy(n) for n in self.nodes])
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        self.near_haz = d <= HAZARD_SENSE_RADIUS      # [N, N] node j visible from robot at node i
        self.near_ppl = d <= PEOPLE_SEE_RADIUS
        # action vocabulary (one token = one decision)
        self.vocab = (["wait", "announce"] + [f"guide_to_exit:{e}" for e in self.exits]
                      + [f"navigate_to:{n}" for n in self.robot_nodes]
                      + [f"report_hazard:{n}" for n in self.nodes])
        self.tok = {v: i for i, v in enumerate(self.vocab)}
        self.V = len(self.vocab)

    # ---- dimensions -------------------------------------------------------
    @property
    def obs_dim(self) -> int:          # hazard one-hot + counts
        return self.N * len(HAZ_KINDS) + self.N

    @property
    def act_dim(self) -> int:          # robot node + guidance exit (or none) + time
        return self.R + self.E + 1 + 1

    # ---- actions <-> tokens -------------------------------------------------
    def action_to_token(self, a: Action) -> int:
        key = a.action if a.action in ("wait", "announce") else f"{a.action}:{a.target}"
        return self.tok.get(key, self.tok["wait"])

    def token_to_action(self, t: int) -> Action:
        v = self.vocab[int(t)]
        if v == "wait":
            return Action("wait")
        if v == "announce":
            return Action("announce", None, ANNOUNCE_TEXT)
        act, target = v.split(":", 1)
        return Action(act, target)

    # ---- encodings ------------------------------------------------------------
    def hazards_to_idx(self, hazards: dict) -> np.ndarray:
        h = np.zeros(self.N, np.int64)
        for n, k in hazards.items():
            if n in self.idx:
                h[self.idx[n]] = HAZ_KINDS.index(k)
        return h

    def counts_to_vec(self, counts: dict) -> np.ndarray:
        c = np.zeros(self.N, np.float32)
        for n, v in counts.items():
            if n in self.idx and v is not None:
                c[self.idx[n]] = float(v) / COUNT_SCALE
        return c

    def action_vec(self, robot_node: str, guidance: str | None, t: float) -> np.ndarray:
        a = np.zeros(self.act_dim, np.float32)
        a[self.ridx[robot_node]] = 1
        a[self.R + (self.exits.index(guidance) if guidance in self.exits else self.E)] = 1
        a[-1] = t / T_SCALE
        return a

    def local_mask(self, robot_node: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.idx[robot_node]
        return self.near_haz[i].astype(np.float32), self.near_ppl[i].astype(np.float32)

    def state_obs(self, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Robot-knowledge view from a decision state dict: (haz_idx, haz_mask, counts, count_mask).

        Online sensors: every node is observed. Offline: only what the robot can
        see from where it is, plus hazards it already knows."""
        online = state["building_sensors"] == "online"
        haz = self.hazards_to_idx({h["location"]: h["type"] for h in state["known_hazards"]})
        hm, cm = (np.ones(self.N, np.float32), np.ones(self.N, np.float32)) if online else self.local_mask(state["robot_location"])
        hm = np.maximum(hm, (haz > 0).astype(np.float32))
        cnt = np.zeros(self.N, np.float32)
        for a in state["areas_with_people"]:
            j = self.idx[a["id"]]
            if a["people"] == "unknown":
                cm[j] = 0
            else:
                cnt[j] = a["people"] / COUNT_SCALE
                cm[j] = 1
        cm = cm * (haz == 0)          # areas on hazards are not listed -> unknown
        return haz, hm, cnt, cm


def onehot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(idx.shape + (k,), np.float32)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


def advance_robot(b: Building, node: str, target: str | None, avoid, meters: float) -> str:
    """Where a robot walking `meters` along the safe route to `target` ends up (nearest graph node)."""
    if not target or target == node:
        return node
    path, _ = b.shortest_path(node, target, avoid)
    if not path:
        return node
    left = meters
    for a, c in zip(path, path[1:]):
        if b.nodes[c].kind == "exit":         # the robot stays inside to keep guiding
            return a
        d = b.distance(a, c)
        if d > left:
            return a if left < d / 2 else c
        left -= d
        node = c
    return node
