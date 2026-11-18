"""Task 2, sensor side + attacker: the robot's sensor drivers and the network between them.

Subscribes: /odom_gz, /scan_gz (Gazebo ground truth via ros_gz_bridge), /campussafe/nav/cmd
Publishes:  /campussafe/nav/odom_rx, /campussafe/nav/scan_rx   signed envelopes, possibly tampered
            /campussafe/nav/onboard                           trusted onboard sensors (not attackable)
            /campussafe/nav/hazards                           active hazards of the route (building feed)
            /campussafe/nav/cmd_rx                            commands after the network (+ forged ones)
Parameters: route (nav/routes.py TEST_ROUTES), nav_attack (nav/attacks.py), seed, spawn_*

The drivers sign each reading (HMAC, core/security.py). The attacker logic is the same code as
the offline experiments (nav/attacks.py), so offline and Gazebo runs are directly comparable.
With nav_attack:=none the node is a transparent relay.
"""
from __future__ import annotations

import math
import shutil
import subprocess

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.security import Signer
from campussafe_twin.nav.attacks import NAV_ATTACKS, NavAttacker
from campussafe_twin.nav.routes import TEST_ROUTES
from campussafe_twin.nav.sim import EXPOSURE_KINDS, HAZARD_SENSE, LIDAR_MAX

from .common import (T_NAV_CMD, T_NAV_CMD_RX, T_NAV_HAZARDS, T_NAV_ODOM, T_NAV_ONBOARD, T_NAV_SCAN, declare_spawn,
                     odom_to_world, to_msg, from_msg)


class NavAttackInjector(Node):
    def __init__(self):
        super().__init__("nav_attack_injector")
        self.declare_parameter("route", "corridor_west")
        self.declare_parameter("nav_attack", "none")
        self.declare_parameter("seed", 0)
        self.declare_parameter("world_name", "campus")
        self.spawn = declare_spawn(self)
        route = self.get_parameter("route").value
        attack = self.get_parameter("nav_attack").value
        if route not in TEST_ROUTES:
            raise ValueError(f"unknown route '{route}', choose from {list(TEST_ROUTES)}")
        if attack not in NAV_ATTACKS:
            raise ValueError(f"unknown nav_attack '{attack}', choose from {list(NAV_ATTACKS)}")
        self.route = TEST_ROUTES[route]
        self.b = lamar_cs_floor()
        self.attacker = NavAttacker(NAV_ATTACKS[attack], self.get_parameter("seed").value)
        self.s_odom, self.s_scan = Signer(), Signer()
        self.world_name = self.get_parameter("world_name").value
        self.t0 = None
        self.truth = None          # (x, y, th, v, w)
        self.scan = None
        self.hazards: dict = {}

        self.create_subscription(Odometry, "/odom_gz", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan_gz", self.on_scan, 10)
        self.create_subscription(String, T_NAV_CMD, self.on_cmd, 10)
        self.p_odom = self.create_publisher(String, T_NAV_ODOM, 10)
        self.p_scan = self.create_publisher(String, T_NAV_SCAN, 10)
        self.p_onboard = self.create_publisher(String, T_NAV_ONBOARD, 10)
        self.p_haz = self.create_publisher(String, T_NAV_HAZARDS, 10)
        self.p_cmd = self.create_publisher(String, T_NAV_CMD_RX, 10)
        self.create_timer(0.1, self.tick)                      # 10 Hz, like the offline simulator
        self._teleport_to_start()
        self.get_logger().info(f"Route '{route}' -> {self.route.goal}; attack '{attack}': "
                               f"{NAV_ATTACKS[attack].description}")

    def now(self) -> float:
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    # ------------------------------------------------------------------ truth from Gazebo
    def on_odom(self, msg: Odometry):
        x, y, th = odom_to_world(msg, *self.spawn)
        self.truth = (x, y, th, msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def on_scan(self, msg: LaserScan):
        r = np.nan_to_num(np.array(msg.ranges, dtype=float), nan=LIDAR_MAX, posinf=LIDAR_MAX, neginf=0.0)
        self.scan = np.clip(r, 0.0, LIDAR_MAX)

    # ------------------------------------------------------------------ 10 Hz sensor cycle
    def tick(self):
        if self.truth is None or self.scan is None:
            return
        t = self.now()
        active = {e.node: e.kind for e in self.route.events if e.t <= t}
        new = [n for n in active if self.hazards.get(n) != active[n]]
        self.hazards = active
        if new:
            self._show_markers(new)
        self.p_haz.publish(to_msg({"t": round(t, 2), "hazards": active}))

        x, y, th, v, w = self.truth
        odom_env = self.s_odom.sign({"pose": [round(x, 4), round(y, 4), round(th, 4)]}, t)
        scan_env = self.s_scan.sign({"ranges": [round(float(r), 3) for r in self.scan]}, t)
        odom_env, scan_env = self.attacker.tamper_sensors(t, odom_env, scan_env)
        if odom_env is not None:
            self.p_odom.publish(to_msg(odom_env))
        if scan_env is not None:
            self.p_scan.publish(to_msg(scan_env))
        self.p_onboard.publish(to_msg({"t": round(t, 3), "proprio": [v, w], "hazard": self._thermal(x, y, th)}))

        forged = self.attacker.forged_command(t)
        if forged is not None:
            forged["_forged"] = True          # ground-truth label for metrics only
            self.p_cmd.publish(to_msg(forged))

    def on_cmd(self, msg: String):
        self.p_cmd.publish(msg)               # legitimate commands cross the network unchanged

    def _teleport_to_start(self):
        """The world spawns the robot at C20 facing east; routes start elsewhere. Gazebo's DiffDrive
        odometry integrates wheel motion from zero, so moving the robot before it drives keeps
        'world pose = route start + odometry' valid for every node."""
        sx, sy, syaw = self.spawn
        if (abs(sx - 20.0) < 1e-6 and abs(sy) < 1e-6 and abs(syaw) < 1e-6) or not shutil.which("gz"):
            return
        req = (f'name: "campus_robot", position: {{x: {sx}, y: {sy}, z: 0.0}}, '
               f'orientation: {{x: 0, y: 0, z: {math.sin(syaw / 2)}, w: {math.cos(syaw / 2)}}}')
        subprocess.run(["gz", "service", "-s", f"/world/{self.world_name}/set_pose", "--reqtype", "gz.msgs.Pose",
                        "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        self.get_logger().info(f"robot moved to route start ({sx:.1f}, {sy:.1f}, yaw {syaw:.2f})")

    def _thermal(self, x, y, th):
        best = None
        for n, k in self.hazards.items():
            if k not in EXPOSURE_KINDS:
                continue
            hx, hy = self.b.xy(n)
            d = math.hypot(hx - x, hy - y)
            if d <= HAZARD_SENSE and (best is None or d < best[0]):
                best = (d, math.atan2(hy - y, hx - x) - th)
        if best is None:
            return [1.0, 0.0, 0.0]
        return [best[0] / HAZARD_SENSE, math.cos(best[1]), math.sin(best[1])]

    def _show_markers(self, nodes):
        if not shutil.which("gz"):
            return
        for n in nodes:
            hx, hy = self.b.xy(n)
            req = f'name: "hazard_{n}", position: {{x: {hx}, y: {hy}, z: 1.0}}'
            subprocess.Popen(["gz", "service", "-s", f"/world/{self.world_name}/set_pose",
                              "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                              "--timeout", "1000", "--req", req],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    rclpy.init()
    node = NavAttackInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
