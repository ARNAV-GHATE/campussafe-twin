"""Task 2 defense gateway between the robot's networked sensors / actuators and its controller.

    /odom, /scan --[crypto]--[world-model monitor]--> pose estimate + scan the controller uses
    controller --signed /cmd_vel--> [network] --[crypto at the motor driver]--> wheels

Modes (same names as Task 1, core/defense.py):
  none    naive robot: uses whatever arrives last; forged commands reach the wheels
  crypto  HMAC + sequence + freshness on /odom, /scan and /cmd_vel. Rejects forged and replayed
          messages. If /odom or /scan go quiet (DoS, or all replays rejected) the robot falls
          back to dead reckoning and slows down
  wm      world-model monitor (wm_monitor.py): per-channel surprise + CUSUM, then recovery:
            odom channel flagged   -> dead reckoning from the wheel encoders, rolled back to the
                                      moment the evidence started to build up (CUSUM onset)
            scan channel flagged   -> scan rendered from the digital-twin map at the estimated pose
            command channel flagged-> hardware motor interlock (safe stop): a forged command can
                                      be detected without keys, but only crypto can reject it
  full    crypto + wm

All flags latch for the rest of the episode (fail-safe), as in Task 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.security import DEFAULT_KEY, Verifier
from .sim import DT, WallMap, wrap

NAV_DEFENSES = ("none", "crypto", "wm", "full")
STALE_S = 0.5          # s without an accepted /odom or /scan -> dead reckoning / last good scan
STALE_ALARM_S = 1.0    # s -> security alarm (starved channel)
DEGRADED_V = 0.5       # m/s speed cap while running on dead reckoning


class NavGuard:
    def __init__(self, defense: str, wall_map: WallMap, monitor=None, key: bytes = DEFAULT_KEY):
        if defense not in NAV_DEFENSES:
            raise ValueError(f"unknown defense '{defense}' (use {' | '.join(NAV_DEFENSES)})")
        self.defense = defense
        self.crypto = defense in ("crypto", "full")
        self.semantic = defense in ("wm", "full")
        if self.semantic and monitor is None:
            raise ValueError("defense 'wm'/'full' needs a world-model monitor (trained models)")
        self.monitor = monitor
        self.map = wall_map
        self.v_odom = Verifier(key, max_age_s=STALE_S) if self.crypto else None
        self.v_scan = Verifier(key, max_age_s=STALE_S) if self.crypto else None
        self.pose = None                 # estimate handed to the controller
        self.dr = None                   # dead-reckoned pose (always integrated from encoders)
        self.scan = None
        self.t_odom = self.t_scan = 0.0
        self.odom_bad = self.scan_bad = self.cmd_bad = False
        self.alarms: list[tuple[float, str]] = []
        self.pose_hist: list[tuple[float, np.ndarray]] = []
        self.dr_hist: list[np.ndarray] = []

    # ------------------------------------------------------------------ helpers
    def _alarm(self, t: float, what: str) -> None:
        if not any(w == what for _, w in self.alarms):
            self.alarms.append((t, what))

    @property
    def degraded(self) -> bool:
        return self.dr is not None and self.pose is self.dr

    def _accept(self, verifier, env, t):
        if env is None:
            return None
        if verifier is None:
            return env["payload"]
        ok, why = verifier.verify(env, t)
        if not ok:
            self._alarm(t, "replay" if "replay" in why or "stale" in why else "forged")
            return None
        return env["payload"]

    # ------------------------------------------------------------------ per step
    def receive(self, t: float, odom_env, scan_env, proprio: np.ndarray, cmd: tuple[float, float],
                hazard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pose estimate, scan) for the controller."""
        odom = self._accept(self.v_odom, odom_env, t)
        scan = self._accept(self.v_scan, scan_env, t)
        odom_pose = np.array(odom["pose"]) if odom else None
        scan_r = np.array(scan["ranges"]) if scan else None

        # dead reckoning from the trusted wheel encoders
        if self.dr is None:
            self.dr = odom_pose.copy() if odom_pose is not None else np.zeros(3)
        else:
            v, w = proprio
            th = self.dr[2]
            self.dr = np.array([self.dr[0] + v * math.cos(th + 0.5 * w * DT) * DT,
                                self.dr[1] + v * math.sin(th + 0.5 * w * DT) * DT,
                                float(wrap(th + w * DT))])

        # world-model monitor sees what arrived (before we decide what to trust)
        if self.semantic:
            verdict = self.monitor.update(t, odom_pose, scan_r, proprio, cmd, hazard)
            for ch in verdict.new_alarms:
                self._alarm(t, f"wm:{ch}")
                if ch == "odom" and not self.odom_bad:
                    self.odom_bad = True
                    self._rollback(verdict.onset.get("odom", t))
                elif ch == "scan":
                    self.scan_bad = True
                elif ch == "cmd":
                    self.cmd_bad = True

        if self.defense == "none":
            # naive: last received values, no fallback
            if odom_pose is not None:
                self.pose = odom_pose
            if scan_r is not None:
                self.scan = scan_r
            if self.pose is None:
                self.pose = self.dr
            if self.scan is None:
                self.scan = self.map.raycast(*self.pose)
            return self.pose, self.scan

        # ---- pose
        if odom_pose is not None and not self.odom_bad:
            self.t_odom = t
            self.dr = odom_pose.copy()               # re-anchor dead reckoning
            self.pose = odom_pose
        else:
            self.pose = self.dr
            if self.crypto and t - self.t_odom > STALE_ALARM_S and not self.odom_bad:
                self._alarm(t, "stale:odom")
        self.pose_hist.append((t, self.pose.copy()))
        self.dr_hist.append(np.array(proprio, dtype=float))

        # ---- scan
        if self.scan_bad:
            self.scan = self.map.raycast(*self.pose)            # digital-twin rendering
        elif scan_r is not None:
            self.t_scan = t
            self.scan = scan_r
        elif self.scan is None or t - self.t_scan > STALE_S:
            self.scan = self.map.raycast(*self.pose)
            if self.crypto and t - self.t_scan > STALE_ALARM_S:
                self._alarm(t, "stale:scan")
        return self.pose, self.scan

    def _rollback(self, onset: float) -> None:
        """Re-integrate the encoders from the last pose recorded before the CUSUM onset."""
        idx = next((i for i in range(len(self.pose_hist) - 1, -1, -1) if self.pose_hist[i][0] < onset - 1e-6), None)
        if idx is None:
            return
        p = self.pose_hist[idx][1].copy()
        for v, w in self.dr_hist[idx + 1:]:
            th = p[2]
            p = np.array([p[0] + v * math.cos(th + 0.5 * w * DT) * DT, p[1] + v * math.sin(th + 0.5 * w * DT) * DT,
                          float(wrap(th + w * DT))])
        self.dr = p

    def speed_cap(self) -> float:
        if self.cmd_bad:
            return 0.0                                          # motor interlock engaged
        return DEGRADED_V if (self.defense != "none" and self.degraded) else 10.0


class MotorDriver:
    """Receives /cmd_vel. With crypto it verifies every command; otherwise the newest wins."""

    def __init__(self, crypto: bool, key: bytes = DEFAULT_KEY):
        self.verifier = Verifier(key, max_age_s=STALE_S) if crypto else None
        self.cmd = (0.0, 0.0)
        self.forged_executed = 0
        self.forged_blocked = 0
        self.rejected_at: float | None = None       # first time a command failed verification

    def receive(self, env: dict, t: float, forged: bool = False) -> None:
        if self.verifier is not None:
            ok, _ = self.verifier.verify(env, t)
            if not ok:
                self.forged_blocked += int(forged)
                if self.rejected_at is None:
                    self.rejected_at = t
                return
        if forged:
            self.forged_executed += 1
        self.cmd = (float(env["payload"]["v"]), float(env["payload"]["w"]))
