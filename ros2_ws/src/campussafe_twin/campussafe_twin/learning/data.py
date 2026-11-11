"""Collects training data from the offline simulator (randomized scenarios only).

At every decision tick (5 s) we record
  * ground truth: hazard kind and people count per node   -> world-model targets
  * the robot's knowledge view (what it would really see)   -> model inputs
  * robot node, guidance exit, time                         -> world-model actions
  * the EXPERT action                                       -> NCM plan targets
  * the action actually executed (expert / rules / random)  -> state diversity

The expert is a privileged teacher: the rule policy, but given the true
current hazards, the hazards that will exist 30 s from now, and true people
counts. The NCM planner learns to imitate it from the robot's *own* knowledge
plus the world model's forecast, so a well-trained planner becomes proactive.
"""
from __future__ import annotations

import copy
import random
from multiprocessing import Pool

import numpy as np

from ..core.attacks import bms_reading
from ..core.building import Building, lamar_cs_floor
from ..core.episode import GraphRobot
from ..core.knowledge import RobotKnowledge
from ..core.policies import RulePolicy
from ..core.scenarios import random_scenario
from ..core.state import build_state
from ..core.validator import validate
from ..core.world import World
from .features import Layout

FORECAST_S = 30.0


def oracle_action(b: Building, know: RobotKnowledge, world: World, robot_node: str, guidance):
    k = copy.copy(know)
    k.hazards = dict(know.hazards)
    k.reported = set(know.reported)
    k.unverified = set()
    future = world.scenario.hazards_at(world.t + FORECAST_S)
    for n, kind in list(world.hazards.items()) + list(future.items()):
        if n not in k.hazards:
            k.hazards[n] = kind
            k.reported.add(n)                     # privileged info: route around it, don't "report" it
    k.counts = {a: c for a, c in world.area_counts().items()}
    k.sensors_online = True
    return RulePolicy._decide(build_state(b, k, robot_node, guidance, world.t))


def run_collect_episode(seed: int, dt: float = 0.5, period: float = 5.0) -> dict:
    b = lamar_cs_floor()
    L = Layout(b)
    rng = random.Random(seed)
    sc = random_scenario(b, seed)
    world = World(b, sc, seed)
    robot = GraphRobot(b, sc.robot_start)
    know = RobotKnowledge(b)
    active = rng.random() > 0.1
    p_expert, p_rule = rng.choice([(0.9, 0.05), (0.7, 0.2), (0.5, 0.3)])
    guidance = None
    rec = {k: [] for k in ("haz", "cnt", "khaz", "khm", "kcnt", "kcm", "rep", "online",
                           "loc", "guid", "t", "expert", "executed")}
    next_dec = 0.0
    while True:
        snap = world.snapshot()
        know.update(bms_reading(snap), robot.xy, local=snap)
        if world.t >= next_dec or world.done:
            loc = robot.location
            state = build_state(b, know, loc, guidance, world.t)
            expert = oracle_action(b, know, world, loc, guidance)
            if not active:
                act = expert
            else:
                u = rng.random()
                if u < p_expert:
                    act = expert
                elif u < p_expert + p_rule:
                    act = RulePolicy._decide(state)
                else:
                    act = L.token_to_action(rng.randrange(L.V))
                if not validate(act, state, b)[0]:
                    act = RulePolicy._decide(state)
            kh, khm, kc, kcm = L.state_obs(state)
            rec["haz"].append(L.hazards_to_idx(snap["hazards"]))
            rec["cnt"].append(L.counts_to_vec(snap["area_counts"]))
            rec["khaz"].append(kh)
            rec["khm"].append(khm)
            rec["kcnt"].append(kc)
            rec["kcm"].append(kcm)
            rec["rep"].append(np.array([n in know.reported for n in L.nodes], np.float32))   # BEFORE acting
            rec["online"].append(float(know.sensors_online))
            rec["loc"].append(L.ridx[loc])
            g_after = act.target if (active and act.action == "guide_to_exit") else guidance
            rec["guid"].append(L.exits.index(g_after) if g_after in L.exits else L.E)   # guidance AFTER acting
            rec["t"].append(world.t)
            rec["expert"].append(L.action_to_token(expert) if active else -1)
            rec["executed"].append(L.action_to_token(act))
            if active:                                   # act only after recording the pre-action state
                if act.action == "navigate_to":
                    robot.go_to(act.target, know.hazards)
                elif act.action == "guide_to_exit":
                    guidance = act.target
                elif act.action == "report_hazard":
                    know.reported.add(act.target)
            next_dec += period
            if world.done:
                break
        robot.step(dt, know.hazards)
        world.step(dt, robot.xy if active else None, guidance if active else None,
                   know.hazards.keys(), False)
    return {k: np.array(v) for k, v in rec.items()}


def collect(n_episodes: int, seed0: int = 0, workers: int = 2) -> list[dict]:
    seeds = list(range(seed0, seed0 + n_episodes))
    if workers <= 1:
        return [run_collect_episode(s) for s in seeds]
    with Pool(workers) as p:
        return p.map(run_collect_episode, seeds, chunksize=20)


def save(episodes: list[dict], path: str) -> None:
    flat = {}
    for i, ep in enumerate(episodes):
        for k, v in ep.items():
            flat[f"{i}/{k}"] = v
    np.savez_compressed(path, n=len(episodes), **flat)


def load(path: str) -> list[dict]:
    z = np.load(path)
    n = int(z["n"])
    keys = {k.split("/", 1)[1] for k in z.files if "/" in k}
    return [{k: z[f"{i}/{k}"] for k in keys} for i in range(n)]
