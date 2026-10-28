"""Safety validator: vetoes unsafe or invalid actions and substitutes the rule fallback.

Subscribes: /campussafe/state, /campussafe/proposed_action
Publishes:  /campussafe/safe_action  (includes valid / reason / fallback fields)
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.actions import Action
from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.policies import RulePolicy
from campussafe_twin.core.validator import validate

from .common import T_PROPOSED, T_SAFE, T_STATE, from_msg, to_msg


class SafetyValidator(Node):
    def __init__(self):
        super().__init__("safety_validator")
        self.building = lamar_cs_floor()
        self.fallback = RulePolicy()
        self.state = None
        self.create_subscription(String, T_STATE, lambda m: setattr(self, "state", from_msg(m)), 10)
        self.create_subscription(String, T_PROPOSED, self.on_proposed, 10)
        self.pub = self.create_publisher(String, T_SAFE, 10)

    def on_proposed(self, msg):
        if self.state is None:
            return
        proposed = from_msg(msg)
        action = Action.from_dict(proposed)
        ok, reason = validate(action, self.state, self.building)
        executed = action if ok else self.fallback.decide(self.state).action
        if not ok:
            self.get_logger().warn(f"REJECTED {action.action} {action.target}: {reason} "
                                   f"-> fallback {executed.action} {executed.target or ''}")
        self.pub.publish(to_msg(executed.to_dict() | {
            "valid": ok, "reason": reason, "fallback": not ok,
            "latency_ms": proposed.get("latency_ms", 0.0), "proposed": action.to_dict(),
            "state_time_s": self.state["time_s"]}))


def main():
    rclpy.init()
    node = SafetyValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
