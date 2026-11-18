"""Offline episode runner: the full decision loop without ROS or Gazebo.

    world snapshot -> building sensors -> [attacks] -> defense gateway -> robot knowledge
                   -> state -> policy (rule | slm | ncm) -> validator -> signed command -> act

The ROS 2 nodes run exactly this loop, with Gazebo moving the robot instead of
GraphRobot. Use this for fast batch experiments (hundreds of runs) and to
debug prompts before going to Gazebo.
"""
from __future__ import annotations

import math

from .actions import Action, Decision
from .attacks import ATTACKS, Attack, Attacker, bms_reading
from .building import Building
from .defense import CommandGuard, SensorGateway, verification_target
from .knowledge import RobotKnowledge
from .metrics import security_metrics, summarize
from .policies import RulePolicy
from .scenarios import Scenario
from .security import Signer
from .state import build_state
from .validator import validate
from .world import World

ROBOT_SPEED = 1.0  # m/s, a realistic walking speed for a humanoid


class GraphRobot:
    """Kinematic robot that walks along the building graph."""

    def __init__(self, building: Building, start: str):
        self.b = building
        self.node = start
        self.x, self.y = building.xy(start)
        self.goal: str | None = None
        self.path: list[str] = []

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def location(self) -> str:
        return self.b.nearest_node(self.x, self.y, ("corridor", "room"))

    def go_to(self, target: str, avoid) -> bool:
        path, _ = self.b.shortest_path(self.node, target, avoid)
        if path is None:
            self.goal, self.path = None, []
            return False
        self.goal = target
        at_node = math.hypot(self.x - self.b.xy(self.node)[0], self.y - self.b.xy(self.node)[1]) < 0.1
        self.path = path[1:] if at_node else path
        return True

    def step(self, dt: float, avoid) -> None:
        if not self.path:
            return
        if self.path[0] in avoid and self.goal:
            self.go_to(self.goal, avoid)
            if not self.path:
                return
        nx, ny = self.b.xy(self.path[0])
        d = math.hypot(nx - self.x, ny - self.y)
        step = ROBOT_SPEED * dt
        if d <= step:
            self.x, self.y, self.node = nx, ny, self.path.pop(0)
        else:
            self.x += (nx - self.x) / d * step
            self.y += (ny - self.y) / d * step


def run_episode(building: Building, scenario: Scenario, policy, seed: int = 0,
                dt: float = 0.5, decision_period: float = 5.0, log=None,
                attack: Attack | str = "none", defense: str = "none", monitor_factory=None) -> dict:
    """One emergency episode.

    attack   name in ATTACKS (or an Attack) - cyberattack applied to the robot's data channels
    defense  none | crypto | wm | full - see defense.py
    monitor_factory  zero-arg callable returning a plausibility monitor (needed for wm / full)
    """
    attack = ATTACKS[attack] if isinstance(attack, str) else attack
    world = World(building, scenario, seed)
    robot = GraphRobot(building, scenario.robot_start)
    know = RobotKnowledge(building)
    fallback = RulePolicy()
    robot_active = getattr(policy, "name", "") != "none"
    if hasattr(policy, "reset"):
        policy.reset()

    attacker = Attacker(attack, building, seed)
    sensor_signer, cmd_signer = Signer(), Signer()
    semantic = defense in ("wm", "full")
    monitor = monitor_factory() if semantic and monitor_factory else None
    gateway = SensorGateway(building, defense, monitor)
    guard = CommandGuard(crypto=defense in ("crypto", "full"))

    guidance: str | None = None
    announce = False
    decisions: list[dict] = []
    replan_times: list[float] = []
    unsafe_since: float | None = None
    next_decision = 0.0
    n_verify = 0

    def execute(action: Action) -> None:
        nonlocal guidance, announce, unsafe_since
        if action.action == "navigate_to":
            robot.go_to(action.target, know.hazards)
        elif action.action == "guide_to_exit":
            if unsafe_since is not None and action.target != guidance:
                replan_times.append(world.t - unsafe_since)
                unsafe_since = None
            guidance = action.target
        elif action.action == "report_hazard":
            know.reported.add(action.target)
        elif action.action == "announce":
            announce = True

    while not world.done:
        snap = world.snapshot()

        # building sensors -> (attacker) -> network -> (attacker) -> defense gateway -> knowledge
        reading = attacker.tamper_reading(bms_reading(snap), snap)
        env = attacker.tamper_envelope(sensor_signer.sign(reading, world.t), world.t)
        n_ev = len(gateway.events)
        feed = gateway.receive(env, world.t, know.perceived_hazards(snap, robot.xy),
                               robot.location, guidance)
        for e in gateway.events[n_ev:]:          # evaluation only: was a quarantined hazard actually real?
            if "node" in e:
                e["false_positive"] = e["node"] in snap["hazards"]
        know.update(feed, robot.xy, local=snap)

        # resilience metric: time from "guided exit known unsafe" to switching exits
        if guidance and guidance in know.hazards and unsafe_since is None:
            unsafe_since = world.t

        if robot_active and world.t >= next_decision:
            state = build_state(building, know, robot.location, guidance, world.t)
            dec = policy.decide(state)
            source = "policy"
            vt = verification_target(know, building, robot.location) if semantic and know.unverified else None
            if vt:
                dec = Decision(Action("navigate_to", vt[0], f"verify {vt[1]}"), dec.latency_ms, dec.raw)
                source = "verify"
                n_verify += 1
            action = dec.action
            ok, reason = validate(action, state, building)
            if not ok:
                action = fallback.decide(state).action
            decisions.append({"t": round(world.t, 1), "proposed": dec.action.to_dict(), "valid": ok,
                              "reason": reason, "executed": action.to_dict(), "latency_ms": dec.latency_ms,
                              "source": source})
            if log:
                flag = "" if ok else f"  [REJECTED: {reason} -> fallback]"
                tag = "  [verify unverified hazard]" if source == "verify" else ""
                log(f"t={world.t:6.1f}s  {dec.action.action} {dec.action.target or ''}{flag}{tag}")
            payload = guard.accept(cmd_signer.sign(action.to_dict(), world.t), world.t, forged=False)
            if payload:
                execute(Action.from_dict(payload))
            next_decision += decision_period

        forged = attacker.forged_command(world.t, snap, robot.location) if robot_active else None
        if forged is not None:
            payload = guard.accept(forged, world.t, forged=True)
            if payload is None:
                gateway._event(world.t, "crypto", "forged_command_blocked",
                               f"{forged['payload']['action']} {forged['payload']['target']} (bad MAC)")
            if log:
                p = forged["payload"]
                log(f"t={world.t:6.1f}s  !! forged command {p['action']} {p['target']} -> "
                    f"{'EXECUTED' if payload else 'BLOCKED (bad MAC)'}")
            if payload:
                execute(Action.from_dict(payload))

        robot.step(dt, know.hazards)
        world.step(dt, robot.xy if robot_active else None, guidance,
                   know.hazards.keys() - know.unverified, announce, know.cleared)
        announce = False

    if log:
        for e in gateway.events:
            log(f"   security t={e['t']:6.1f}s  [{e['layer']}] {e['type']}: {e['detail']}")

    result = summarize(world.snapshot(), decisions, replan_times)
    result["policy"] = getattr(policy, "name", "?")
    result["seed"] = seed
    result["hazards_reported"] = len(know.reported)
    result.update(security_metrics(attack, defense, gateway, guard, attacker, know, n_verify))
    return result
