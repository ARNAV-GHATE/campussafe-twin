"""Task 2 evaluation probe: scores the run from Gazebo ground truth and writes a CSV row.

Subscribes: /odom_gz (truth), /campussafe/nav/hazards, /campussafe/nav/status, /campussafe/security
Writes:     <output_dir>/nav_<route>_<policy>_<attack>_<defense>_<timestamp>.csv
Same columns as the offline experiments (nav/episode.py), so Gazebo runs can be appended to
the offline results. Collisions are counted when the true footprint touches a wall in the
building model; exposure is time spent inside a fire / smoke radius.
"""
from __future__ import annotations

import csv
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.nav.attacks import ATTACK_START
from campussafe_twin.nav.episode import SUCCESS_R
from campussafe_twin.nav.routes import TEST_ROUTES
from campussafe_twin.nav.sim import EXPOSURE_KINDS, HAZARD_R, ROBOT_R, WallMap

from .common import T_NAV_HAZARDS, T_NAV_STATUS, T_SECURITY, declare_spawn, from_msg, odom_to_world


class NavMetrics(Node):
    def __init__(self):
        super().__init__("nav_metrics")
        for k, d in (("route", "corridor_west"), ("nav_policy", "ncm_wm"), ("nav_attack", "none"),
                     ("nav_defense", "full"), ("output_dir", "")):
            self.declare_parameter(k, d)
        self.spawn = declare_spawn(self)
        self.p = {k: self.get_parameter(k).value for k in ("route", "nav_policy", "nav_attack", "nav_defense")}
        self.route = TEST_ROUTES[self.p["route"]]
        self.out_dir = self.get_parameter("output_dir").value or os.path.expanduser("~/campussafe_results")
        self.b = lamar_cs_floor()
        self.map = WallMap(self.b)
        self.hazards: dict = {}
        self.prev = None
        self.t = 0.0
        self.collisions = 0
        self.contact = False
        self.exposure = 0.0
        self.path_len = 0.0
        self.min_clear = math.inf
        self.status = None
        self.written = False
        self.create_subscription(Odometry, "/odom_gz", self.on_odom, 10)
        self.create_subscription(String, T_NAV_HAZARDS, self.on_hazards, 10)
        self.create_subscription(String, T_NAV_STATUS, self.on_status, 10)
        self.extra_alarms: list = []
        self.create_subscription(String, T_SECURITY, lambda m: self.extra_alarms.append(
            (from_msg(m)["t"], from_msg(m)["alarm"])), 10)

    def on_hazards(self, msg):
        h = from_msg(msg)
        self.t = h["t"]
        if h["hazards"] != self.hazards:
            self.hazards = h["hazards"]
            self.map.set_blocked([n for n, k in self.hazards.items() if k == "blocked"])
        if not self.written and self.t >= self.route.t_max - 0.15:
            self.write()

    def on_odom(self, msg):
        x, y, _ = odom_to_world(msg, *self.spawn)
        if self.prev is not None:
            dt = 0.05
            self.path_len += math.hypot(x - self.prev[0], y - self.prev[1])
            for n, k in self.hazards.items():
                if k in EXPOSURE_KINDS and math.hypot(x - self.b.xy(n)[0], y - self.b.xy(n)[1]) < HAZARD_R:
                    self.exposure += dt
                    break
        c = self.map.clearance(x, y)
        self.min_clear = min(self.min_clear, c)
        touching = c < ROBOT_R + 0.02
        if touching and not self.contact:
            self.collisions += 1
        self.contact = touching
        self.prev = (x, y)

    def on_status(self, msg):
        self.status = from_msg(msg)
        if self.status.get("finished") and not self.written:
            self.write()

    def write(self):
        self.written = True
        st = self.status or {}
        goal = st.get("goal")
        gx, gy = self.b.xy(goal) if goal else (math.nan, math.nan)
        d = math.hypot(self.prev[0] - gx, self.prev[1] - gy) if self.prev else math.nan
        alarms = [(ta, a) for ta, a in st.get("alarms", [])] + self.extra_alarms
        after = [ta for ta, _ in alarms if ta >= ATTACK_START]
        attacked = self.p["nav_attack"] != "none"
        row = {"route": self.p["route"], "policy": self.p["nav_policy"], "attack": self.p["nav_attack"],
               "defense": self.p["nav_defense"], "sim": "gazebo", "success": int(d <= SUCCESS_R),
               "time_s": round(self.t, 1) if d <= SUCCESS_R else None, "final_dist_m": round(d, 2),
               "collisions": self.collisions, "exposure_s": round(self.exposure, 1),
               "min_clearance_m": round(self.min_clear, 2), "path_len_m": round(self.path_len, 1),
               "detected": int(attacked and bool(after)),
               "detection_delay_s": round(min(after) - ATTACK_START, 1) if attacked and after else None,
               "false_alarm": int(not attacked and bool(alarms)), "alarms": ";".join(sorted({a for _, a in alarms}))}
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"nav_{self.p['route']}_{self.p['nav_policy']}_{self.p['nav_attack']}_"
                                          f"{self.p['nav_defense']}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        self.get_logger().info(f"RESULT: success={row['success']} time={row['time_s']} collisions={row['collisions']} "
                               f"exposure={row['exposure_s']}s detected={row['detected']} -> {path}")


def main():
    rclpy.init()
    node = NavMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
