"""Records every decision and writes the run's metrics when the scenario ends.

Subscribes: /campussafe/world, /campussafe/safe_action, /campussafe/guidance
Writes:     <output_dir>/<scenario>_<policy>_<timestamp>_summary.csv and _decisions.csv
"""
from __future__ import annotations

import csv
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.metrics import summarize

from .common import T_GUIDANCE, T_SAFE, T_WORLD, from_msg


class MetricsLogger(Node):
    def __init__(self):
        super().__init__("metrics_logger")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("policy", "slm")
        self.out_dir = self.get_parameter("output_dir").value or os.path.expanduser("~/campussafe_results")
        self.policy = self.get_parameter("policy").value
        self.decisions: list[dict] = []
        self.replan_times: list[float] = []
        self.guidance = None
        self.unsafe_since = None
        self.snapshot = None
        self.written = False
        self.create_subscription(String, T_WORLD, self.on_world, 10)
        self.create_subscription(String, T_SAFE, self.on_action, 10)
        self.create_subscription(String, T_GUIDANCE, self.on_guidance, 10)

    def on_guidance(self, msg):
        new = msg.data or None
        if new != self.guidance and self.unsafe_since is not None and self.snapshot:
            self.replan_times.append(self.snapshot["t"] - self.unsafe_since)
            self.unsafe_since = None
        self.guidance = new

    def on_action(self, msg):
        a = from_msg(msg)
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
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.join(self.out_dir, f"{result['scenario']}_{self.policy.replace(':', '-')}_{time.strftime('%Y%m%d-%H%M%S')}")
        with open(stem + "_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result))
            w.writeheader()
            w.writerow(result)
        if self.decisions:
            with open(stem + "_decisions.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.decisions[0]))
                w.writeheader()
                w.writerows(self.decisions)
        self.get_logger().info(
            f"RESULTS: {result['pct_evacuated']}% evacuated, mean {result['mean_evacuation_time_s']} s, "
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
