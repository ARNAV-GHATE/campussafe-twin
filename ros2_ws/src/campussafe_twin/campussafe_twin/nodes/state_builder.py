"""Turns sensor data + robot pose into the structured decision state, through the defense layer.

Subscribes: /campussafe/bms_feed_rx (building sensors, as received over the network)
            /campussafe/world       (ground truth, used ONLY within onboard sensor range)
            /odom, /campussafe/guidance, /campussafe/safe_action
Publishes:  /campussafe/state     (every decision_period seconds)
            /campussafe/security  (defense events: rejected messages, anomalies, alarms)
Parameters: decision_period, defense (none | crypto | wm | full)

The processing per world tick is identical to core/episode.py:
    gateway.receive(feed) -> knowledge.update(feed, onboard truth) -> build_state
"""
from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.defense import SensorGateway, verification_target
from campussafe_twin.core.knowledge import RobotKnowledge
from campussafe_twin.core.state import build_state

from .common import (T_BMS_RX, T_GUIDANCE, T_SAFE, T_SECURITY, T_STATE, T_WORLD, declare_security, declare_spawn,
                     from_msg, odom_to_world, to_msg)


class StateBuilder(Node):
    def __init__(self):
        super().__init__("state_builder")
        self.declare_parameter("decision_period", 5.0)
        self.spawn = declare_spawn(self)
        _, defense = declare_security(self)
        self.building = lamar_cs_floor()
        self.know = RobotKnowledge(self.building)
        factory = None
        if defense in ("wm", "full"):
            from campussafe_twin.learning import monitor_factory   # needs PyTorch + trained models
            factory = monitor_factory()
        self.gateway = SensorGateway(self.building, defense, factory() if factory else None)
        self.semantic = defense in ("wm", "full")
        self.n_events = 0
        self.robot_xy = None
        self.guidance = None
        self.truth = None
        self.pending_env = None

        self.create_subscription(String, T_WORLD, self.on_world, 10)
        self.create_subscription(String, T_BMS_RX, lambda m: setattr(self, "pending_env", from_msg(m)), 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, T_GUIDANCE, lambda m: setattr(self, "guidance", m.data or None), 10)
        self.create_subscription(String, T_SAFE, self.on_safe_action, 10)
        self.pub = self.create_publisher(String, T_STATE, 10)
        self.sec_pub = self.create_publisher(String, T_SECURITY, 10)
        self.create_timer(self.get_parameter("decision_period").value, self.publish_state)
        self.get_logger().info(f"Defense: {defense}")

    def on_odom(self, msg):
        x, y, _ = odom_to_world(msg, *self.spawn)
        self.robot_xy = (x, y)

    def _robot_node(self):
        return self.building.nearest_node(*self.robot_xy, kinds=("corridor", "room"))

    def on_world(self, msg):
        self.truth = from_msg(msg)
        if self.robot_xy is None:
            return
        env, self.pending_env = self.pending_env, None
        feed = self.gateway.receive(env, self.truth["t"], self.know.perceived_hazards(self.truth, self.robot_xy),
                                    self._robot_node(), self.guidance)
        self.know.update(feed, self.robot_xy, local=self.truth)
        for e in self.gateway.events[self.n_events:]:
            if "node" in e:                       # evaluation label: was the quarantined hazard real?
                e["false_positive"] = e["node"] in self.truth["hazards"]
            self.get_logger().warn(f"SECURITY t={e['t']:.0f}s [{e['layer']}] {e['type']}: {e['detail']}")
            self.sec_pub.publish(to_msg(e))
        self.n_events = len(self.gateway.events)

    def on_safe_action(self, msg):
        env = from_msg(msg)
        a = env.get("payload", env)
        if a.get("action") == "report_hazard" and a.get("target") and "proposed" in a:
            self.know.reported.add(a["target"])

    def publish_state(self):
        if self.truth is None or self.robot_xy is None or self.truth["done"]:
            return
        loc = self._robot_node()
        state = build_state(self.building, self.know, loc, self.guidance, self.truth["t"])
        if self.semantic and self.know.unverified:
            vt = verification_target(self.know, self.building, loc)
            if vt:
                state["verify_target"] = {"go_to": vt[0], "hazard": vt[1]}
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
