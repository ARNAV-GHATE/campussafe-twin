"""Records every decision and writes the run's metrics when the scenario ends.

Subscribes: /campussafe/world, /campussafe/safe_action, /campussafe/guidance, /campussafe/security
Writes:     <output_dir>/<scenario>_<policy>_<attack>_<defense>_<timestamp>_summary.csv, _decisions.csv
            and _security.csv
"""
from __future__ import annotations

import csv
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.attacks import ATTACKS
from campussafe_twin.core.metrics import summarize

from .common import T_GUIDANCE, T_SAFE, T_SECURITY, T_WORLD, declare_security, from_msg


class MetricsLogger(Node):
    def __init__(self):
        super().__init__("metrics_logger")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("policy", "slm")
        self.out_dir = self.get_parameter("output_dir").value or os.path.expanduser("~/campussafe_results")
        self.policy = self.get_parameter("policy").value
        self.attack, self.defense = declare_security(self)
        self.events: list[dict] = []
        self.forged_executed = 0
        self.forged_blocked = 0
        self.decisions: list[dict] = []
        self.replan_times: list[float] = []
        self.guidance = None
        self.unsafe_since = None
        self.snapshot = None
        self.written = False
        self.create_subscription(String, T_WORLD, self.on_world, 10)
        self.create_subscription(String, T_SAFE, self.on_action, 10)
        self.create_subscription(String, T_GUIDANCE, self.on_guidance, 10)
        self.create_subscription(String, T_SECURITY, self.on_security, 10)

    def on_security(self, msg):
        e = from_msg(msg)
        if e["type"] == "forged_command_executed":
            self.forged_executed += 1
            return
        if e["type"] == "forged_command_blocked":
            self.forged_blocked += 1
        self.events.append(e)
        if e["type"] == "filtered":            # routine count filtering, not an alarm
            self.events.pop()

    def on_guidance(self, msg):
        new = msg.data or None
        if new != self.guidance and self.unsafe_since is not None and self.snapshot:
            self.replan_times.append(self.snapshot["t"] - self.unsafe_since)
            self.unsafe_since = None
        self.guidance = new

    def on_action(self, msg):
        env = from_msg(msg)
        a = env.get("payload", env)
        if env.get("_forged") or "proposed" not in a:
            return
        self.decisions.append({"t": a.get("state_time_s"), "proposed_action": a["proposed"]["action"],
                               "proposed_target": a["proposed"]["target"], "valid": a["valid"],
                               "reason": a["reason"], "executed_action": a["action"],
                               "executed_target": a.get("target"), "latency_ms": a.get("latency_ms", 0.0)})

    def on_world(self, msg):
        self.snapshot = from_msg(msg)
        if self.guidance and self.guidance in self.snapshot["hazards"] and self.unsafe_since is None:
            self.unsafe_since = self.snapshot["t"]
        if self.snapshot["done"] and not self.written:
            self.write()

    def write(self):
        self.written = True
        result = summarize(self.snapshot, self.decisions, self.replan_times) | {"policy": self.policy}
        start = ATTACKS[self.attack].start if self.attack in ATTACKS else 0.0
        first = min((e["t"] for e in self.events if not e.get("false_positive")), default=None)
        attacked = self.attack != "none"
        detected = attacked and first is not None and first >= start
        fp = sum(1 for e in self.events if e.get("false_positive"))
        result |= {"attack": self.attack, "defense": self.defense,
                   "detected": int(detected) if attacked else None,
                   "detection_delay_s": round(first - start, 1) if detected else None,
                   "false_alarm": int(fp > 0 or (first is not None and (not attacked or first < start))),
                   "false_quarantines": fp,
                   "security_events": len(self.events),
                   "forged_cmds_executed": self.forged_executed, "forged_cmds_blocked": self.forged_blocked}
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.join(self.out_dir, f"{result['scenario']}_{self.policy.replace(':', '-')}_{self.attack}_"
                                          f"{self.defense}_{time.strftime('%Y%m%d-%H%M%S')}")
        with open(stem + "_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result))
            w.writeheader()
            w.writerow(result)
        if self.decisions:
            with open(stem + "_decisions.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.decisions[0]))
                w.writeheader()
                w.writerows(self.decisions)
        if self.events:
            with open(stem + "_security.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["t", "layer", "type", "detail", "node", "false_positive"])
                w.writeheader()
                w.writerows(self.events)
        self.get_logger().info(
            f"RESULTS: attack {self.attack} / defense {self.defense}, detected={result['detected']}, "
            f"forged executed={self.forged_executed}; "
            f" {result['pct_evacuated']}% evacuated, mean {result['mean_evacuation_time_s']} s, "
            f"{result['people_exposed']} exposed, {result['invalid_rate_pct']}% invalid actions, "
            f"latency {result['mean_latency_ms']} ms -> {stem}_summary.csv")


def main():
    rclpy.init()
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
