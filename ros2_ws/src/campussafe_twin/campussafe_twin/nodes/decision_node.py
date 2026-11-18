"""Decision layer: runs the SLM (or a baseline policy) on each state.

Subscribes: /campussafe/state
Publishes:  /campussafe/proposed_action
Parameters: policy (slm | rule | mock | none | ncm | wm_ncm), model, ollama_host

If the state carries a verify_target (the world-model monitor quarantined a
hazard claim), the robot goes to check it with its own sensors first; the
policy still runs so learned policies keep their world-model belief current.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from campussafe_twin.core.actions import Action, Decision
from campussafe_twin.core.policies import make_policy
from campussafe_twin.core.slm_client import OllamaClient

from .common import T_PROPOSED, T_STATE, from_msg, to_msg


class DecisionNode(Node):
    def __init__(self):
        super().__init__("decision_node")
        self.declare_parameter("policy", "slm")
        self.declare_parameter("model", "llama3.2:3b")
        self.declare_parameter("ollama_host", "http://localhost:11434")
        self.declare_parameter("seed", 0)
        pol = self.get_parameter("policy").value
        model = self.get_parameter("model").value
        host = self.get_parameter("ollama_host").value
        if pol == "slm" and not OllamaClient(model, host).available():
            self.get_logger().error(f"Ollama not reachable at {host}; every decision will fall back to rules.")
        self.policy = make_policy(pol, model, host, self.get_parameter("seed").value)
        self.create_subscription(String, T_STATE, self.on_state, 10)
        self.pub = self.create_publisher(String, T_PROPOSED, 10)
        self.get_logger().info(f"Decision policy: {self.policy.name}")

    def on_state(self, msg):
        state = from_msg(msg)
        # The SLM call blocks this callback; states arrive every few seconds, so that is fine.
        dec = self.policy.decide(state)
        if state.get("verify_target"):
            vt = state["verify_target"]
            dec = Decision(Action("navigate_to", vt["go_to"], f"verify {vt['hazard']}"), dec.latency_ms, dec.raw)
        out = dec.action.to_dict() | {"latency_ms": round(dec.latency_ms, 1), "raw": dec.raw[:500],
                                      "policy": self.policy.name, "state_time_s": state["time_s"]}
        self.pub.publish(to_msg(out))
        self.get_logger().info(f"t={state['time_s']:.0f}s proposed {dec.action.action} "
                               f"{dec.action.target or ''} ({dec.latency_ms:.0f} ms)")


def main():
    rclpy.init()
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
