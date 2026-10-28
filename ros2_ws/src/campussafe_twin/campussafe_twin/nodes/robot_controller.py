"""Executes validated actions in Gazebo.

navigate_to  -> plans a route on the building graph (avoiding known hazards)
                and drives the waypoints with a simple heading controller
guide_to_exit -> sets the guidance exit (people near the robot follow it)
announce     -> publishes the spoken message

Subscribes: /campussafe/safe_action, /campussafe/state, /odom
Publishes:  /cmd_vel, /campussafe/guidance, /campussafe/announce

To move to Nav2 later, replace `_drive()` with a NavigateThroughPoses action
client that sends the same waypoints.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor

from .common import T_ANNOUNCE, T_GUIDANCE, T_SAFE, T_STATE, declare_spawn, from_msg, odom_to_world

MAX_LINEAR = 1.0      # m/s
MAX_ANGULAR = 1.5     # rad/s
WAYPOINT_TOL = 0.4    # m


class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")
        self.spawn = declare_spawn(self)
        self.building = lamar_cs_floor()
        self.pose = None
        self.goal = None
        self.waypoints: list[tuple[float, float]] = []
        self.node_path: list[str] = []
        self.guidance = ""
        self.hazards: set[str] = set()

        self.create_subscription(String, T_SAFE, self.on_action, 10)
        self.create_subscription(String, T_STATE, self.on_state, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.guid_pub = self.create_publisher(String, T_GUIDANCE, 10)
        self.ann_pub = self.create_publisher(String, T_ANNOUNCE, 10)
        self.create_timer(0.1, self._drive)
        self.create_timer(1.0, lambda: self.guid_pub.publish(String(data=self.guidance)))

    def on_odom(self, msg):
        self.pose = odom_to_world(msg, *self.spawn)

    def on_state(self, msg):
        self.hazards = {h["location"] for h in from_msg(msg)["known_hazards"]}
        if self.goal and any(n in self.hazards for n in self.node_path):
            self.get_logger().warn("Route crosses a new hazard, re-planning")
            self._plan(self.goal)

    def on_action(self, msg):
        a = from_msg(msg)
        act, target = a["action"], a.get("target")
        if act == "navigate_to":
            self._plan(target)
        elif act == "guide_to_exit":
            self.guidance = target
            self.guid_pub.publish(String(data=target))
            self.get_logger().info(f"Guiding people to {target}")
        elif act == "announce":
            self.ann_pub.publish(String(data=a.get("message", "")))
            self.get_logger().info(f'ANNOUNCE: "{a.get("message", "")}"')
        elif act == "report_hazard":
            self.get_logger().warn(f"Reported hazard at {target} to campus police / EHS")

    def _plan(self, target: str):
        if self.pose is None:
            return
        start = self.building.nearest_node(self.pose[0], self.pose[1], ("corridor", "room"))
        path, _ = self.building.shortest_path(start, target, self.hazards)
        if path is None:
            self.get_logger().warn(f"No safe route to {target}")
            self.goal, self.waypoints, self.node_path = None, [], []
            return
        self.goal, self.node_path = target, path
        self.waypoints = [self.building.xy(n) for n in path]

    def _drive(self):
        twist = Twist()
        if self.pose is not None and self.waypoints:
            x, y, yaw = self.pose
            wx, wy = self.waypoints[0]
            dist = math.hypot(wx - x, wy - y)
            if dist < WAYPOINT_TOL:
                self.waypoints.pop(0)
                self.node_path.pop(0)
            else:
                err = math.atan2(wy - y, wx - x) - yaw
                err = math.atan2(math.sin(err), math.cos(err))
                twist.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, 2.0 * err))
                twist.linear.x = min(MAX_LINEAR, 0.8 * dist) if abs(err) < 0.4 else 0.05
        self.cmd.publish(twist)


def main():
    rclpy.init()
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
