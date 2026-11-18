"""Message authentication for robot data channels (HMAC-SHA256 + sequence + timestamp).

Two channels use this:
  * the building-sensor feed (building management system -> robot)
  * the action channel (safety validator -> robot controller)

Each message is wrapped in an envelope {seq, ts, payload, mac}. The receiver
rejects a message whose MAC does not verify (forged / tampered in transit),
whose sequence number is not newer than the last one accepted (replay), or
whose timestamp is too old (delayed / replayed). This stops a *network*
attacker who does not hold the key. It cannot stop a *compromised sensor*
that holds a valid key and signs false data; that is the world-model
monitor's job (see defense.py).

In a real ROS 2 deployment use SROS2 (DDS-Security) for transport security;
this module makes the same guarantees visible and measurable in simulation.
"""
from __future__ import annotations

import hashlib
import hmac
import json

DEFAULT_KEY = b"campussafe-thesis-demo-key-change-me"


def _mac(key: bytes, seq: int, ts: float, payload) -> str:
    body = json.dumps({"seq": seq, "ts": round(ts, 3), "payload": payload}, sort_keys=True, separators=(",", ":"))
    return hmac.new(key, body.encode(), hashlib.sha256).hexdigest()


class Signer:
    """Sender side: signs each payload with a fresh sequence number."""

    def __init__(self, key: bytes = DEFAULT_KEY):
        self.key = key
        self.seq = 0

    def sign(self, payload, ts: float) -> dict:
        self.seq += 1
        return {"seq": self.seq, "ts": round(ts, 3), "payload": payload, "mac": _mac(self.key, self.seq, ts, payload)}


class Verifier:
    """Receiver side: MAC check, replay check (monotonic seq) and freshness check."""

    def __init__(self, key: bytes = DEFAULT_KEY, max_age_s: float = 2.0):
        self.key = key
        self.max_age_s = max_age_s
        self.last_seq = 0

    def verify(self, env: dict, now: float) -> tuple[bool, str]:
        try:
            seq, ts, payload, mac = env["seq"], env["ts"], env["payload"], env["mac"]
        except (KeyError, TypeError):
            return False, "malformed envelope"
        if not hmac.compare_digest(str(mac), _mac(self.key, seq, ts, payload)):
            return False, "bad MAC (forged or tampered)"
        if seq <= self.last_seq:
            return False, f"replayed message (seq {seq} <= {self.last_seq})"
        if now - ts > self.max_age_s:
            return False, f"stale message ({now - ts:.1f} s old)"
        self.last_seq = seq
        return True, "ok"


def forge(payload, ts: float, seq: int = 10**6) -> dict:
    """What an attacker without the key can produce: a plausible envelope with a wrong MAC."""
    return {"seq": seq, "ts": round(ts, 3), "payload": payload, "mac": "0" * 64}


def resign(env: dict, payload, key: bytes = DEFAULT_KEY) -> dict:
    """A compromised sensor/BMS holds the key: it can sign altered data with the original seq/ts."""
    return {"seq": env["seq"], "ts": env["ts"], "payload": payload, "mac": _mac(key, env["seq"], env["ts"], payload)}
