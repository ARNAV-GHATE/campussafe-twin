"""Task 2, robot side: defense gateway + navigation policy (classical | ncm | ncm_wm).

Subscribes: /campussafe/nav/odom_rx, /campussafe/nav/scan_rx (signed, possibly attacked),
            /campussafe/nav/onboard (trusted encoders + thermal), /campussafe/nav/hazards
Publishes:  /campussafe/nav/cmd (signed velocity commands), /campussafe/nav/status
Parameters: route, nav_policy, nav_defense, spawn_*

Runs exactly the offline loop (nav/episode.py) at 10 Hz: NavGuard -> WaypointPath -> policy.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.security import Signer
from campussafe_twin.nav.controllers import WaypointPath
from campussafe_twin.nav.defense import NavGuard, WallMap
from campussafe_twin.nav.episode import make_nav_policy
from campussafe_twin.nav.routes import TEST_ROUTES

from .common import T_NAV_CMD, T_NAV_HAZARDS, T_NAV_ODOM, T_NAV_ONBOARD, T_NAV_SCAN, T_NAV_STATUS, from_msg, to_msg


class NavController(Node):
    def __init__(self):
        super().__init__("nav_controller")
        self.declare_parameter("route", "corridor_west")
        self.declare_parameter("nav_policy", "ncm_wm")
        self.declare_parameter("nav_defense", "full")
        route = TEST_ROUTES[self.get_parameter("route").value]
        pol = self.get_parameter("nav_policy").value
        defense = self.get_parameter("nav_defense").value
        self.b = lamar_cs_floor()
        models = None
        if pol != "classical" or defense in ("wm", "full"):
            from campussafe_twin.nav.learned import NavModels       # PyTorch
            models = NavModels()
        self.policy = make_nav_policy(pol, models)
        self.policy.reset(route)
        self.map = WallMap(self.b)
        self.guard = NavGuard(defense, self.map, models.monitor() if defense in ("wm", "full") else None)
        self.path = WaypointPath(route, self.b)
        self.signer = Signer()
        self.odom_env = self.scan_env = None
        self.onboard = None
        self.hazards = None
        self.known = None
        self.cmd = (0.0, 0.0)
        self.held = (0.0, 0.0)
        self.k = 0
        self.finished = False

        self.create_subscription(String, T_NAV_ODOM, lambda m: setattr(self, "odom_env", from_msg(m)), 10)
        self.create_subscription(String, T_NAV_SCAN, lambda m: setattr(self, "scan_env", from_msg(m)), 10)
        self.create_subscription(String, T_NAV_ONBOARD, self.on_onboard, 10)
        self.create_subscription(String, T_NAV_HAZARDS, lambda m: setattr(self, "hazards", from_msg(m)), 10)
        self.p_cmd = self.create_publisher(String, T_NAV_CMD, 10)
        self.p_status = self.create_publisher(String, T_NAV_STATUS, 10)
        self.get_logger().info(f"nav policy '{self.policy.name}', defense '{defense}', route '{route.name}'")

    def on_onboard(self, msg):
        """The onboard message closes each 10 Hz sensor cycle: run one control step."""
        self.onboard = from_msg(msg)
        if self.finished or self.hazards is None:
            return
        t = self.onboard["t"]
        proprio = np.array(self.onboard["proprio"])
        hazard = np.array(self.onboard["hazard"])
        odom_env, scan_env = self.odom_env, self.scan_env
        self.odom_env = self.scan_env = None                  # each message is used once
        pose, scan = self.guard.receive(t, odom_env, scan_env, proprio, self.cmd, hazard)

        hz = self.hazards["hazards"]
        if self.known != hz:
            self.known = dict(hz)
            self.path.plan(pose[0], pose[1], self.known)
        self.path.update(pose[0], pose[1])
        if self.path.done or self.path.trapped:
            self.finished = True
            self.cmd = (0.0, 0.0)
            self.get_logger().info("goal reached (by the robot's estimate)" if self.path.done else "no safe path")
        else:
            if self.k % self.policy.period == 0:
                self.held = self.policy.act({"t": t, "pose": pose, "path": self.path, "scan": scan,
                                             "hazard": hazard, "proprio": proprio, "guard": self.guard})
            self.k += 1
            v, w = self.held
            cap = self.guard.speed_cap()
            if cap <= 0:
                v, w = 0.0, 0.0
            self.cmd = (min(v, cap), w)
        self.p_cmd.publish(to_msg(self.signer.sign({"v": self.cmd[0], "w": self.cmd[1]}, t)))
        self.p_status.publish(to_msg({
            "t": t, "goal": self.path.goal, "done": self.path.done, "trapped": self.path.trapped,
            "finished": self.finished, "pose_est": [round(float(v), 3) for v in pose],
            "alarms": [[round(ta, 2), a] for ta, a in self.guard.alarms], "interlock": self.guard.cmd_bad,
            "policy": self.policy.name, "defense": self.guard.defense}))
        new = self.guard.alarms[-1] if self.guard.alarms else None
        if new and abs(new[0] - t) < 1e-6:
            self.get_logger().warn(f"t={t:.1f}s SECURITY ALARM: {new[1]}")


def main():
    rclpy.init()
    node = NavController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
