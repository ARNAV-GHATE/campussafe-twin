"""Runs the emergency scenario: hazards, simulated occupants and sensor failures.

Subscribes: /odom, /campussafe/guidance, /campussafe/state, /campussafe/announce
Publishes:  /campussafe/world  (JSON snapshot, every tick)
Gazebo:     raises the red hazard marker models when a hazard activates
"""
from __future__ import annotations

import shutil
import subprocess

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.scenarios import SCENARIOS
from campussafe_twin.core.world import World

from .common import T_ANNOUNCE, T_GUIDANCE, T_STATE, T_WORLD, declare_spawn, from_msg, odom_to_world, to_msg


class ScenarioManager(Node):
    def __init__(self):
        super().__init__("scenario_manager")
        self.declare_parameter("scenario", "fire_north")
        self.declare_parameter("seed", 0)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("world_name", "campus")
        self.declare_parameter("show_hazards_in_gazebo", True)
        self.spawn = declare_spawn(self)

        name = self.get_parameter("scenario").value
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario '{name}', choose from {list(SCENARIOS)}")
        self.building = lamar_cs_floor()
        self.world = World(self.building, SCENARIOS[name], self.get_parameter("seed").value)
        self.world_name = self.get_parameter("world_name").value
        self.show_hazards = self.get_parameter("show_hazards_in_gazebo").value and shutil.which("gz")

        self.robot_xy = None
        self.guidance = None
        self.robot_hazards: list[str] = []
        self.announce = False
        self.last_time = None
        self.finished = False

        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, T_GUIDANCE, lambda m: setattr(self, "guidance", m.data or None), 10)
        self.create_subscription(String, T_STATE, self.on_state, 10)
        self.create_subscription(String, T_ANNOUNCE, lambda m: setattr(self, "announce", True), 10)
        self.pub = self.create_publisher(String, T_WORLD, 10)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(f"Scenario '{name}': {SCENARIOS[name].description}")
        self._show_markers(list(self.world.hazards))

    def on_odom(self, msg: Odometry):
        x, y, _ = odom_to_world(msg, *self.spawn)
        self.robot_xy = (x, y)

    def on_state(self, msg: String):
        self.robot_hazards = [h["location"] for h in from_msg(msg)["known_hazards"]]

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is None or self.robot_xy is None:   # wait for Gazebo clock + odometry
            self.last_time = now
            return
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0 or self.finished:
            return
        self.world.step(min(dt, 0.5), self.robot_xy, self.guidance, self.robot_hazards, self.announce)
        self.announce = False
        if self.world.new_hazards:
            for n in self.world.new_hazards:
                self.get_logger().warn(f"t={self.world.t:.0f}s  HAZARD {self.world.hazards[n]} at {n}")
            self._show_markers(self.world.new_hazards)
        snap = self.world.snapshot()
        self.pub.publish(to_msg(snap))
        if snap["done"]:
            self.finished = True
            self.get_logger().info("Scenario finished.")

    def _show_markers(self, nodes):
        """Move the pre-placed hazard_<node> marker from underground to the hazard location."""
        if not self.show_hazards:
            return
        for n in nodes:
            x, y = self.building.xy(n)
            req = f'name: "hazard_{n}", position: {{x: {x}, y: {y}, z: 1.0}}'
            subprocess.Popen(["gz", "service", "-s", f"/world/{self.world_name}/set_pose",
                              "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                              "--timeout", "1000", "--req", req],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    rclpy.init()
    node = ScenarioManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
