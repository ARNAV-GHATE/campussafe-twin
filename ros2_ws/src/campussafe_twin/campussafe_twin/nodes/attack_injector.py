"""Cyberattack injector: sits on the network between the building sensors and the robot.

Subscribes: /campussafe/bms_feed (signed sensor feed), /campussafe/world (truth, for targeting),
            /campussafe/state (robot location, for forged commands)
Publishes:  /campussafe/bms_feed_rx (feed as the robot receives it)
            /campussafe/safe_action (forged commands, command_injection / coordinated)
Parameters: attack (see core/attacks.py), seed

With attack:=none it is a transparent relay, so it always runs.
Compromised-sensor attacks re-sign the altered data (the attacker holds the
device key); network attacks (replay, dos, forged commands) do not have the key.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.attacks import ATTACKS, Attacker
from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.security import resign

from .common import T_BMS, T_BMS_RX, T_SAFE, T_STATE, T_WORLD, declare_security, from_msg, to_msg


class AttackInjector(Node):
    def __init__(self):
        super().__init__("attack_injector")
        attack, _ = declare_security(self)
        self.declare_parameter("seed", 0)
        if attack not in ATTACKS:
            raise ValueError(f"unknown attack '{attack}', choose from {list(ATTACKS)}")
        self.attacker = Attacker(ATTACKS[attack], lamar_cs_floor(), self.get_parameter("seed").value)
        self.truth = None
        self.robot_node = "C20"
        self.create_subscription(String, T_WORLD, lambda m: setattr(self, "truth", from_msg(m)), 10)
        self.create_subscription(String, T_STATE, self.on_state, 10)
        self.create_subscription(String, T_BMS, self.on_bms, 10)
        self.rx_pub = self.create_publisher(String, T_BMS_RX, 10)
        self.cmd_pub = self.create_publisher(String, T_SAFE, 10)
        self.get_logger().info(f"Attack '{attack}': {ATTACKS[attack].description}")

    def on_state(self, msg):
        self.robot_node = from_msg(msg)["robot_location"]

    def on_bms(self, msg):
        env = from_msg(msg)
        t = env["ts"]
        if self.truth is not None and self.attacker.a.sensor:
            tampered = self.attacker.tamper_reading(env["payload"], self.truth)
            if tampered is not env["payload"]:
                env = resign(env, tampered)
        env = self.attacker.tamper_envelope(env, t)
        if env is not None:
            self.rx_pub.publish(to_msg(env))
        if self.truth is not None:
            forged = self.attacker.forged_command(t, self.truth, self.robot_node)
            if forged is not None:
                forged["_forged"] = True      # ground-truth label for metrics only; not covered by the MAC
                self.get_logger().warn(f"t={t:.0f}s injecting forged command {forged['payload']}")
                self.cmd_pub.publish(to_msg(forged))


def main():
    rclpy.init()
    node = AttackInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
