"""Task 2, actuator side: the motor driver that turns received commands into /cmd_vel.

Subscribes: /campussafe/nav/cmd_rx (legitimate + possibly forged commands), /campussafe/nav/status
Publishes:  /cmd_vel (geometry_msgs/Twist, to Gazebo via ros_gz_bridge)
            /campussafe/security (first rejected command -> "forged:cmd" alarm)
Parameters: nav_defense - with crypto | full every command's HMAC / sequence / age is verified
            and forged ones are dropped; otherwise the newest command wins.

The motor interlock (status.interlock) models a hard-wired e-stop line from the robot
computer: when the world-model monitor flags the command channel, the motors stop and no
network message can restart them.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.nav.defense import MotorDriver

from .common import T_NAV_CMD_RX, T_NAV_STATUS, T_SECURITY, from_msg, to_msg


class NavMotorDriver(Node):
    def __init__(self):
        super().__init__("nav_motor_driver")
        self.declare_parameter("nav_defense", "full")
        defense = self.get_parameter("nav_defense").value
        self.driver = MotorDriver(crypto=defense in ("crypto", "full"))
        self.interlock = False
        self.finished = False
        self.create_subscription(String, T_NAV_CMD_RX, self.on_cmd, 10)
        self.create_subscription(String, T_NAV_STATUS, self.on_status, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sec = self.create_publisher(String, T_SECURITY, 10)
        self.reported = False
        self.create_timer(0.05, self.publish)

    def on_status(self, msg):
        st = from_msg(msg)
        self.interlock = bool(st.get("interlock"))
        self.finished = bool(st.get("finished"))
        self.t = st.get("t", 0.0)

    def on_cmd(self, msg):
        env = from_msg(msg)
        now = getattr(self, "t", env.get("ts", 0.0))           # robot time from the latest status
        self.driver.receive(env, now, forged=bool(env.get("_forged")))
        if self.driver.rejected_at is not None and not self.reported:
            self.reported = True
            self.sec.publish(to_msg({"t": self.driver.rejected_at, "alarm": "forged:cmd", "source": "nav_motor_driver"}))
            self.get_logger().warn(f"t={now:.1f}s rejected a command with a bad MAC (forged /cmd_vel)")

    def publish(self):
        tw = Twist()
        if not self.interlock:
            tw.linear.x, tw.angular.z = self.driver.cmd
        self.pub.publish(tw)


def main():
    rclpy.init()
    node = NavMotorDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
