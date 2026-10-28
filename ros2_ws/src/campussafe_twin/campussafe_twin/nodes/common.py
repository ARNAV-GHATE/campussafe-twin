"""Shared helpers for the ROS 2 nodes."""
from __future__ import annotations

import json
import math

from std_msgs.msg import String

# Topic names in one place
T_WORLD = "/campussafe/world"            # occupants + hazards snapshot (scenario_manager)
T_STATE = "/campussafe/state"            # decision state (state_builder)
T_PROPOSED = "/campussafe/proposed_action"  # raw policy output (decision_node)
T_SAFE = "/campussafe/safe_action"       # validated action (safety_validator)
T_GUIDANCE = "/campussafe/guidance"      # exit the robot is guiding to (robot_controller)
T_ANNOUNCE = "/campussafe/announce"      # spoken announcements (robot_controller)


def to_msg(data: dict) -> String:
    return String(data=json.dumps(data))


def from_msg(msg: String) -> dict:
    return json.loads(msg.data)


def odom_to_world(odom, spawn_x: float, spawn_y: float, spawn_yaw: float = 0.0):
    """Gazebo DiffDrive odometry starts at (0, 0) at the spawn pose; convert to world frame."""
    p, q = odom.pose.pose.position, odom.pose.pose.orientation
    c, s = math.cos(spawn_yaw), math.sin(spawn_yaw)
    x = spawn_x + c * p.x - s * p.y
    y = spawn_y + s * p.x + c * p.y
    yaw = spawn_yaw + math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return x, y, yaw


def declare_spawn(node) -> tuple[float, float, float]:
    node.declare_parameter("spawn_x", 20.0)
    node.declare_parameter("spawn_y", 0.0)
    node.declare_parameter("spawn_yaw", 0.0)
    return (node.get_parameter("spawn_x").value, node.get_parameter("spawn_y").value,
            node.get_parameter("spawn_yaw").value)
