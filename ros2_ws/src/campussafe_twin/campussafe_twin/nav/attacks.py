"""Task 2 cyberattacks on the robot's own channels: /odom, /scan and /cmd_vel.

Threat model (same split as Task 1, core/attacks.py):
  * compromised sensor / driver: holds the device key, so its lies are validly signed
        odom_spoof      localization source drifts the reported pose sideways (GPS-spoofing style,
                        0.25 m/s, slow enough to look like normal noise at first)
        lidar_phantom   LiDAR driver injects a phantom obstacle 0.5 m ahead (stalls the robot)
  * network attacker without keys
        cmd_injection   forged /cmd_vel messages (full speed, hard turn) at 10 Hz
        sensor_replay   replays /odom + /scan captured 3 s earlier, in a loop
        dos             drops 90 % of /odom and /scan messages
  * coordinated: odom_spoof + LiDAR erasure (walls vanish from /scan) + cmd_injection

Onboard proprioception (wheel encoders) and the onboard thermal sensor are not networked
and are trusted, exactly like the robot's own 6-8 m sensors in Task 1.
All attacks start at ATTACK_START seconds.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.security import forge, resign
from .sim import BEAM_ANGLES, LIDAR_MAX

ATTACK_START = 5.0


@dataclass(frozen=True)
class NavAttack:
    name: str
    description: str
    odom_spoof: bool = False
    lidar_phantom: bool = False
    lidar_erase: bool = False
    cmd_injection: bool = False
    replay: bool = False
    dos: bool = False

    @property
    def sensor(self) -> bool:      # needs the device key (compromised device)
        return self.odom_spoof or self.lidar_phantom or self.lidar_erase


NAV_ATTACKS: dict[str, NavAttack] = {a.name: a for a in (
    NavAttack("none", "no attack"),
    NavAttack("odom_spoof", "compromised localization: pose drifts sideways at 0.25 m/s (signed)", odom_spoof=True),
    NavAttack("lidar_phantom", "compromised LiDAR driver: phantom obstacle 0.5 m ahead (signed)", lidar_phantom=True),
    NavAttack("cmd_injection", "network: forged /cmd_vel at 10 Hz (no key)", cmd_injection=True),
    NavAttack("sensor_replay", "network: /odom and /scan from 3 s ago replayed in a loop (no key)", replay=True),
    NavAttack("dos", "network: 90 % of /odom and /scan dropped", dos=True),
    NavAttack("coordinated", "odom drift + LiDAR erasure + forged /cmd_vel",
              odom_spoof=True, lidar_erase=True, cmd_injection=True),
)}

FRONT_SECTOR = np.abs(BEAM_ANGLES) <= math.radians(25)


class NavAttacker:
    def __init__(self, attack: NavAttack, seed: int = 0, start: float = ATTACK_START):
        self.a, self.start = attack, start
        self.rng = np.random.default_rng(10_000 + seed)
        self.history: deque = deque()
        self.replay_buf: list = []
        self.replay_i = 0
        self.forged_sent = 0

    def active(self, t: float) -> bool:
        return self.a.name != "none" and t >= self.start

    # ------------------------------------------------------------------ sensor messages
    def tamper_sensors(self, t: float, odom_env: dict, scan_env: dict):
        """Returns the (odom, scan) envelopes the robot receives; None = message lost."""
        a = self.a
        self.history.append((t, odom_env, scan_env))
        while self.history and self.history[0][0] < t - 3.5:
            self.history.popleft()
        if not self.active(t):
            return odom_env, scan_env
        if a.odom_spoof:
            pose = list(odom_env["payload"]["pose"])
            # the reported position slides to the left of the heading the robot had when the
            # attack began, 0.25 m/s, capped at 4 m; the controller "corrects" into the wall
            d = min(0.25 * (t - self.start), 4.0)
            hd = self._drift_heading(pose[2])
            pose[0] -= d * math.sin(hd)
            pose[1] += d * math.cos(hd)
            odom_env = resign(odom_env, {**odom_env["payload"], "pose": pose})
        if a.lidar_phantom or a.lidar_erase:
            r = np.array(scan_env["payload"]["ranges"])
            if a.lidar_phantom:
                r[FRONT_SECTOR] = np.minimum(r[FRONT_SECTOR], 0.5 + self.rng.normal(0, 0.02, FRONT_SECTOR.sum()))
            if a.lidar_erase:
                r[:] = LIDAR_MAX
            scan_env = resign(scan_env, {**scan_env["payload"], "ranges": r.round(3).tolist()})
        if a.replay:
            if not self.replay_buf:
                self.replay_buf = [(o, s) for (tt, o, s) in self.history if tt <= t - 3.0][-10:] or \
                                  [(odom_env, scan_env)]
            odom_env, scan_env = self.replay_buf[self.replay_i % len(self.replay_buf)]
            self.replay_i += 1
        if a.dos and self.rng.random() < 0.9:
            return None, None
        return odom_env, scan_env

    _drift_dir = None

    def _drift_heading(self, th: float) -> float:
        """Fix the drift direction when the attack starts (perpendicular to the initial heading)."""
        if self._drift_dir is None:
            self._drift_dir = th
        return self._drift_dir

    # ------------------------------------------------------------------ command channel
    def forged_command(self, t: float) -> dict | None:
        if not (self.active(t) and self.a.cmd_injection):
            return None
        self.forged_sent += 1
        w = 1.5 if (int(t / 2.0) % 2 == 0) else -1.5
        return forge({"v": 1.0, "w": w}, t, seq=10 ** 6 + self.forged_sent)
