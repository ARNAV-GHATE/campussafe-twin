"""Minimal stand-in for rclpy + message packages, so the ROS 2 node code can be tested without ROS.

    from ros_stub import install, Bus, GazeboStub
    bus = install()                         # must run BEFORE importing campussafe_twin.nodes.*

The real node classes are instantiated unchanged. Publications are delivered synchronously to
subscribers on the same in-process Bus; timers are fired by Bus.spin_for(). GazeboStub plays the
part of Gazebo + ros_gz_bridge using the offline simulator (nav/sim.py): it publishes /odom_gz and
/scan_gz and applies /cmd_vel. This checks message formats, topic wiring, parameters and the
closed loop of the Task 2 nodes; it does not replace a real Gazebo run.
"""
from __future__ import annotations

import math
import sys
import types
from collections import defaultdict


class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _vec():
    return _Msg(x=0.0, y=0.0, z=0.0)


class String(_Msg):
    def __init__(self, data=""):
        super().__init__(data=data)


class Twist(_Msg):
    def __init__(self):
        super().__init__(linear=_vec(), angular=_vec())


class Odometry(_Msg):
    def __init__(self):
        super().__init__(pose=_Msg(pose=_Msg(position=_vec(), orientation=_Msg(x=0.0, y=0.0, z=0.0, w=1.0))),
                         twist=_Msg(twist=_Msg(linear=_vec(), angular=_vec())))


class LaserScan(_Msg):
    def __init__(self):
        super().__init__(ranges=[])


class Bus:
    def __init__(self):
        self.subs = defaultdict(list)
        self.timers = []                     # [period, next_due, callback]
        self.t = 0.0
        self.params: dict = {}
        self.log: list = []

    def publish(self, topic, msg):
        for cb in list(self.subs[topic]):
            cb(msg)

    def spin_for(self, seconds, dt=0.05, hook=None):
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + dt, 6)
            for tm in self.timers:
                if self.t + 1e-9 >= tm[1]:
                    tm[1] += tm[0]
                    tm[2]()
            if hook:
                hook()


BUS: Bus | None = None


class _Param:
    def __init__(self, v):
        self.value = v


class _Logger:
    def __init__(self, name):
        self.name = name

    def _log(self, lvl, m):
        BUS.log.append((lvl, self.name, m))

    def info(self, m):
        self._log("info", m)

    def warn(self, m):
        self._log("warn", m)

    warning = warn

    def error(self, m):
        self._log("error", m)


class _Clock:
    def now(self):
        return _Msg(nanoseconds=int(BUS.t * 1e9))


class Node:
    def __init__(self, name):
        self._name = name
        self._params = {}

    def declare_parameter(self, name, default=None):
        self._params[name] = BUS.params.get(name, default)

    def get_parameter(self, name):
        return _Param(self._params[name])

    def create_subscription(self, _type, topic, cb, _qos):
        BUS.subs[topic].append(cb)

    def create_publisher(self, _type, topic, _qos):
        return _Msg(publish=lambda m, t=topic: BUS.publish(t, m))

    def create_timer(self, period, cb):
        BUS.timers.append([period, BUS.t + period, cb])

    def get_logger(self):
        return _Logger(self._name)

    def get_clock(self):
        return _Clock()

    def destroy_node(self):
        pass


def install() -> Bus:
    global BUS
    BUS = Bus()
    mods = {
        "rclpy": types.ModuleType("rclpy"),
        "rclpy.node": types.ModuleType("rclpy.node"),
        "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": types.ModuleType("std_msgs.msg"),
        "geometry_msgs": types.ModuleType("geometry_msgs"), "geometry_msgs.msg": types.ModuleType("geometry_msgs.msg"),
        "nav_msgs": types.ModuleType("nav_msgs"), "nav_msgs.msg": types.ModuleType("nav_msgs.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
    }
    mods["rclpy"].init = lambda *a, **k: None
    mods["rclpy"].spin = lambda *a, **k: None
    mods["rclpy"].try_shutdown = lambda *a, **k: None
    mods["rclpy.node"].Node = Node
    mods["std_msgs.msg"].String = String
    mods["geometry_msgs.msg"].Twist = Twist
    mods["nav_msgs.msg"].Odometry = Odometry
    mods["sensor_msgs.msg"].LaserScan = LaserScan
    sys.modules.update(mods)
    return BUS


class GazeboStub:
    """Plays Gazebo + bridge with the offline simulator: /odom_gz and /scan_gz out, /cmd_vel in.
    Odometry is reported in the odom frame (zero at the spawn pose), like DiffDrive."""

    def __init__(self, bus: Bus, world, spawn):
        self.bus, self.w, self.spawn = bus, world, spawn
        self.cmd = (0.0, 0.0)
        bus.subs["/cmd_vel"].append(lambda m: setattr(self, "cmd", (m.linear.x, m.angular.z)))
        bus.timers.append([0.1, bus.t + 0.1, self.step])

    def step(self):
        w = self.w
        w.step(*self.cmd)
        p = w.s.pose
        sx, sy, syaw = self.spawn
        c, s = math.cos(-syaw), math.sin(-syaw)
        dx, dy = p.x - sx, p.y - sy
        od = Odometry()
        od.pose.pose.position.x = c * dx - s * dy
        od.pose.pose.position.y = s * dx + c * dy
        yaw = p.th - syaw
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        od.twist.twist.linear.x, od.twist.twist.angular.z = w.s.v, w.s.w
        self.bus.publish("/odom_gz", od)
        sc = LaserScan()
        sc.ranges = [float(r) for r in w.lidar()]
        self.bus.publish("/scan_gz", sc)
