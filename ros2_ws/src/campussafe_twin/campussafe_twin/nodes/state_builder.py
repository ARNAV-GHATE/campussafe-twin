"""Turns raw world data + robot pose into the structured decision state.

Subscribes: /campussafe/world, /odom, /campussafe/guidance, /campussafe/safe_action
Publishes:  /campussafe/state  (every decision_period seconds)
"""
from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.knowledge import RobotKnowledge
from campussafe_twin.core.state import build_state

from .common import T_GUIDANCE, T_SAFE, T_STATE, T_WORLD, declare_spawn, from_msg, odom_to_world, to_msg


class StateBuilder(Node):
    def __init__(self):
        super().__init__("state_builder")
        self.declare_parameter("decision_period", 5.0)
        self.spawn = declare_spawn(self)
        self.building = lamar_cs_floor()
        self.know = RobotKnowledge(self.building)
        self.robot_xy = None
        self.guidance = None
        self.snapshot = None

        self.create_subscription(String, T_WORLD, self.on_world, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, T_GUIDANCE, lambda m: setattr(self, "guidance", m.data or None), 10)
        self.create_subscription(String, T_SAFE, self.on_safe_action, 10)
        self.pub = self.create_publisher(String, T_STATE, 10)
        self.create_timer(self.get_parameter("decision_period").value, self.publish_state)

    def on_odom(self, msg):
        x, y, _ = odom_to_world(msg, *self.spawn)
        self.robot_xy = (x, y)

    def on_world(self, msg):
        self.snapshot = from_msg(msg)
        if self.robot_xy is not None:
            self.know.update(self.snapshot, self.robot_xy)

    def on_safe_action(self, msg):
        a = from_msg(msg)
        if a["action"] == "report_hazard" and a.get("target"):
            self.know.reported.add(a["target"])

    def publish_state(self):
        if self.snapshot is None or self.robot_xy is None or self.snapshot["done"]:
            return
        loc = self.building.nearest_node(*self.robot_xy, kinds=("corridor", "room"))
        state = build_state(self.building, self.know, loc, self.guidance, self.snapshot["t"])
        self.pub.publish(to_msg(state))


def main():
    rclpy.init()
    node = StateBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
