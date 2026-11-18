"""Defense layer between the building-sensor feed and the robot's knowledge.

    envelope --[crypto]--> reading --[onboard cross-check]--[world-model monitor]--> sanitized reading

Defense modes (compare them in experiments):
  none    accept whatever arrives (naive robot)
  crypto  HMAC + sequence + freshness (security.py). Stops forging, replay and
          detects a starved feed (DoS) -> falls back to onboard sensing.
  wm      semantic checks: onboard cross-check + world-model plausibility.
          Catches *validly signed* false data from a compromised sensor.
  full    crypto + wm

Responses are deliberately conservative (fail-safe):
  * an implausible NEW hazard is quarantined as "unverified": it is still
    avoided, and the robot is sent to look at it; if its own sensors see
    nothing there, the claim is cleared (knowledge.py)
  * implausible people counts are dropped (last trusted value kept)
  * accumulated evidence (CUSUM over world-model surprise) or an onboard
    contradiction marks the whole feed compromised: from then on the robot
    uses onboard sensing + world-model belief only, like a sensor failure
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .building import Building
from .knowledge import HAZARD_SENSE_RADIUS
from .security import DEFAULT_KEY, Verifier

DEFENSES = ("none", "crypto", "wm", "full")
OFFLINE = {"sensors_online": False, "hazards": {}, "area_counts": {}}


@dataclass
class Verdict:
    """What a plausibility monitor says about one sensor reading."""
    suspicious_hazards: set = field(default_factory=set)   # hazards currently quarantined
    new_suspicious: set = field(default_factory=set)       # quarantined by this reading
    bad_counts: set = field(default_factory=set)            # areas with implausible people counts
    surprise: float = 0.0                                   # per-reading anomaly score
    compromised: bool = False                               # CUSUM alarm: whole channel untrusted
    notes: list = field(default_factory=list)               # alarms (quarantine, vanished hazard, CUSUM)
    filtered: list = field(default_factory=list)            # routine data filtering, not an alarm


class SensorGateway:
    def __init__(self, building: Building, defense: str = "none", monitor=None,
                 key: bytes = DEFAULT_KEY, stale_after_s: float = 3.0):
        if defense not in DEFENSES:
            raise ValueError(f"unknown defense '{defense}' (use {' | '.join(DEFENSES)})")
        self.b = building
        self.defense = defense
        self.crypto = defense in ("crypto", "full")
        self.semantic = defense in ("wm", "full")
        if self.semantic and monitor is None:
            raise ValueError("defense 'wm'/'full' needs a plausibility monitor (trained world model)")
        self.monitor = monitor
        self.verifier = Verifier(key) if self.crypto else None
        self.stale_after_s = stale_after_s
        self.last_rx: float | None = None
        self.stale = False
        self.channel_trusted = True
        self.events: list[dict] = []

    def _event(self, t: float, layer: str, kind: str, detail: str, **extra) -> None:
        self.events.append({"t": round(t, 1), "layer": layer, "type": kind, "detail": detail} | extra)

    def receive(self, env: dict | None, now: float, onboard_hazards: dict,
                robot_node: str, guidance: str | None) -> dict | None:
        """Returns the reading to give the robot's knowledge, OFFLINE, or None (nothing new)."""
        reading = None
        if env is not None:
            if self.crypto:
                ok, reason = self.verifier.verify(env, now)
                if ok:
                    reading = env["payload"]
                else:
                    self._event(now, "crypto", "rejected", reason)
            else:
                reading = env["payload"]
        if reading is not None:
            self.last_rx = now

        if self.crypto and (self.last_rx is None or now - self.last_rx > self.stale_after_s):
            if not self.stale:
                self._event(now, "crypto", "stale_feed", "no authentic sensor data -> onboard sensing only")
            self.stale = True
            return dict(OFFLINE)
        self.stale = False
        if reading is None:
            return None
        if not self.semantic:
            return reading
        if not self.channel_trusted:
            return dict(OFFLINE)
        if not reading["sensors_online"]:
            return reading

        hidden = sorted(n for n in onboard_hazards if n not in reading["hazards"])
        if hidden:
            self._event(now, "onboard", "contradiction", f"feed hides hazard(s) {hidden} the robot can see")
            self._compromise(now)
            return dict(OFFLINE)

        v: Verdict = self.monitor.check(reading, now, robot_node, guidance)
        for n in v.notes:
            node = next((h for h in v.new_suspicious if f" at {h} " in n), None)
            self._event(now, "world_model", "quarantine" if node else "anomaly", n, **({"node": node} if node else {}))
        for n in v.filtered:
            self._event(now, "world_model", "filtered", n)
        if v.compromised:
            self._compromise(now)
            return dict(OFFLINE)
        out = dict(reading)
        out["unverified"] = set(v.suspicious_hazards)
        if v.bad_counts:
            out["area_counts"] = {a: (None if a in v.bad_counts else c) for a, c in reading["area_counts"].items()}
        return out

    def _compromise(self, now: float) -> None:
        if self.channel_trusted:
            self.channel_trusted = False
            self._event(now, "defense", "channel_compromised", "building feed distrusted for the rest of the run")

    @property
    def alarms(self) -> list[dict]:
        """Events that raise an alarm (routine count filtering does not; it feeds the CUSUM)."""
        return [e for e in self.events if e["type"] != "filtered"]

    @property
    def true_alarms(self) -> list[dict]:
        """Alarms minus quarantines of hazards that were actually real (labelled in simulation)."""
        return [e for e in self.alarms if not e.get("false_positive")]

    @property
    def first_alarm(self) -> float | None:
        a = self.true_alarms
        return a[0]["t"] if a else None


class CommandGuard:
    """Robot-side check on the action channel (validator -> controller)."""

    def __init__(self, crypto: bool, key: bytes = DEFAULT_KEY):
        self.verifier = Verifier(key, max_age_s=5.0) if crypto else None
        self.blocked = 0
        self.executed_forged = 0

    def accept(self, env: dict, now: float, forged: bool) -> dict | None:
        if self.verifier is not None:
            ok, _ = self.verifier.verify(env, now)
            if not ok:
                self.blocked += int(forged)
                return None
        if forged:
            self.executed_forged += 1
        return env["payload"]


def verification_target(know, building: Building, robot_node: str, max_route_m: float = 35.0) -> tuple[str, str] | None:
    """Closest safe spot from which the robot can check an unverified hazard with its own sensors."""
    best = None
    avoid = set(know.hazards)
    for h in know.unverified:
        hx, hy = building.xy(h)
        for n, node in building.nodes.items():
            if node.kind == "exit" or n in avoid:
                continue
            if ((node.x - hx) ** 2 + (node.y - hy) ** 2) ** 0.5 > HAZARD_SENSE_RADIUS - 0.5:
                continue
            path, d = building.shortest_path(robot_node, n, avoid)
            if path is not None and d <= max_route_m and (best is None or d < best[2]):
                best = (n, h, d)
    return (best[0], best[1]) if best else None
