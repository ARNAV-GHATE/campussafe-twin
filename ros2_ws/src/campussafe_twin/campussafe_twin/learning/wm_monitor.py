"""World-model plausibility monitor: detects false sensor data that is validly signed.

Idea: the world model knows how emergencies evolve in this building (smoke
spreads along corridors, hazards don't vanish, people only leave). Data from
a compromised sensor breaks that physics. Every incoming reading is compared
with the model's prediction:

  new hazard      P(this hazard KIND at this node) under the prediction < tau
                  (a fire can appear in a room; it cannot appear at an exit door)
                  -> quarantined as "unverified" (still avoided, robot verifies)
  vanished hazard hazards persist in this world -> strong evidence
  people counts   |observed - predicted| / predicted std > z_th (per node)
                  -> those counts are dropped for this reading; evidence
                  min(z / z_th, 3) per anomalous count

Evidence is accumulated with a CUSUM detector,
    S_t = max(0, S_{t-1} + evidence_t - drift),  alarm when S_t > H,
which marks the whole feed compromised. tau, z_th, drift and H are
calibrated on attack-free runs (train.py -> monitor_calibration.json) to keep
false alarms rare; the thesis reports the resulting false-alarm rate.
"""
from __future__ import annotations

import json

import numpy as np

from ..core.defense import Verdict
from .features import HAZ_KINDS

DEFAULT_CALIB = {"tau": 0.02, "z_th": 4.0, "count_floor": 0.05, "drift": 1.0, "H": 4.0,
                 "w_new_hazard": 1.0, "w_removed": 3.0, "period": 5.0}


def load_calibration(path: str) -> dict:
    with open(path) as f:
        return DEFAULT_CALIB | json.load(f)


class WMPlausibilityMonitor:
    def __init__(self, runtime, calib: dict | None = None, record: bool = False):
        self.wm = runtime
        self.L = runtime.L
        self.c = DEFAULT_CALIB | (calib or {})
        self.record = record
        self.trace: dict[str, list] = {"p_new": [], "z": [], "cusum": []}
        self.belief = None
        self.accepted: dict[str, str] = {}
        self.flagged: set[str] = set()
        self.removed_flagged: set[str] = set()
        self.bad_counts: set[str] = set()       # flagged at the last tick; kept filtered until the next one
        self.last_tick = 0.0
        self.cusum = 0.0
        self.pending = 0.0
        self.p_now = self.p_next = self.mu = self.std = None

    # --------------------------------------------------------------------------------
    def check(self, reading: dict, t: float, robot_node: str, guidance: str | None) -> Verdict:
        L = self.L
        haz = L.hazards_to_idx(reading["hazards"])
        cnt = L.counts_to_vec(reading["area_counts"])
        ones = np.ones(L.N, np.float32)
        v = Verdict()
        if self.belief is None:                              # alarm-time snapshot: trusted baseline
            self.belief = self.wm.init_belief((haz, ones, cnt, ones))
            self.accepted = dict(reading["hazards"])
            self.last_tick = t
            self._predict(robot_node, guidance, t)
            return v

        for n in reading["hazards"]:
            if n in self.accepted or n in self.flagged:
                continue
            j = L.idx[n]
            k = HAZ_KINDS.index(reading["hazards"][n])
            p = float(max(self.p_now[j, k], self.p_next[j, k]))
            if self.record:
                self.trace["p_new"].append(p)
            if p < self.c["tau"]:
                self.flagged.add(n)
                v.new_suspicious.add(n)
                self.pending += self.c["w_new_hazard"]
                v.notes.append(f"implausible new hazard '{reading['hazards'][n]}' at {n} "
                               f"(model gives P={p:.4f}) -> quarantined, robot will verify")
            else:
                self.accepted[n] = reading["hazards"][n]
        v.suspicious_hazards = {n for n in self.flagged if n in reading["hazards"]}

        removed = [n for n in self.accepted if n not in reading["hazards"] and n not in self.removed_flagged]
        if removed:
            self.removed_flagged.update(removed)
            self.pending += self.c["w_removed"] * len(removed)
            v.notes.append(f"hazard(s) {removed} vanished from the feed (hazards do not disappear)")

        if t - self.last_tick >= self.c["period"] - 1e-6:
            z = np.abs(cnt - self.mu) / np.maximum(self.std, self.c["count_floor"])
            if self.record:
                self.trace["z"].extend(z.tolist())
            bad = np.where(z > self.c["z_th"])[0]
            self.bad_counts = {L.nodes[j] for j in bad}
            if len(bad):
                v.filtered.append(f"implausible people counts at {sorted(v.bad_counts)} "
                                  f"(max z={z.max():.1f}) -> ignored")
            severity = float(np.minimum(z[bad] / self.c["z_th"], 3.0).sum())   # 1 per barely-anomalous count, up to 3
            self.cusum = max(0.0, self.cusum + severity + self.pending - self.c["drift"])
            self.pending = 0.0
            if self.record:
                self.trace["cusum"].append(self.cusum)
            if self.cusum > self.c["H"]:
                v.compromised = True
                v.notes.append(f"CUSUM evidence {self.cusum:.1f} > {self.c['H']} -> sensor feed compromised")
            hm = ones.copy()
            cm = ones.copy()
            for n in self.flagged:
                hm[L.idx[n]] = 0
            hz = haz.copy()
            for n in self.flagged:
                hz[L.idx[n]] = 0
            for j in bad:
                cm[j] = 0
            act = L.action_vec(robot_node, guidance, self.last_tick)
            self.belief = self.wm.observe(self.belief, act, (hz, hm, cnt, cm))
            self.last_tick = t
            self._predict(robot_node, guidance, t)
        v.bad_counts = set(self.bad_counts)
        return v

    def _predict(self, robot_node, guidance, t):
        self.p_now, _, _ = self.wm.decode(self.belief)
        self.p_next, self.mu, self.std = self.wm.predict(self.belief, self.L.action_vec(robot_node, guidance, t))
