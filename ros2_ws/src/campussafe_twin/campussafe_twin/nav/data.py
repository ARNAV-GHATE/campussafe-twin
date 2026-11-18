"""Task 2 training data from random routes (the eight TEST_ROUTES are never used).

Each episode is driven at 5 Hz by either the privileged teacher (classical controller fed the
TRUE pose) or, in the DAgger round, by the current NCM student; with probability EXPLORE per
control step a short burst of random motion tokens is executed instead, so the world model
also sees near-collisions and recoveries.

Recorded
  world-model stream (10 Hz): o_t, mask_t, a_t                      -> nav world model
  planner samples (every 2nd control step): 42-D observation, sim-step index, and the plan label:
      the 16 tokens the teacher produces when it drives on from a COPY of the current world
      (a 3.2 s privileged roll-out)                                   -> NCM tokenizer / generator
"""
from __future__ import annotations

import copy
import os
from multiprocessing import Pool

import numpy as np

from ..core.building import lamar_cs_floor
from .controllers import OBS_DIM, ClassicalController, WaypointPath, from_token, make_obs, to_token
from .learned import CONTROL_PERIOD, PLAN_L, act_vec, odom_increment, wm_obs
from .routes import random_route
from .sim import NavWorld

EXPLORE = 0.08


def _teacher_token(world: NavWorld, path: WaypointPath, ctl: ClassicalController) -> int:
    p = world.s.pose.as_array()
    path.update(p[0], p[1])
    v, w = ctl.act(p, path, world.lidar(), world.hazard_sense())
    return to_token(v, w)


def _plan_label(world: NavWorld, path: WaypointPath, ctl: ClassicalController, known: dict) -> list[int]:
    w2, p2 = copy.deepcopy(world), copy.deepcopy(path)
    toks = []
    for _ in range(PLAN_L):
        if w2.s.hazards != known:
            known = dict(w2.s.hazards)
            p = w2.s.pose
            p2.plan(p.x, p.y, known)
        if p2.trapped:
            tok = 3                                       # no safe path: stop
        else:
            tok = _teacher_token(w2, p2, ctl) if not p2.done else 3
            if p2.done:
                # Goal reached inside the 3.2 s look-ahead: keep the last MOVING token. Padding with
                # STOP made the plan's coarse "concept" read "arrive and stop", and because NCM decodes
                # coarse-to-fine the student braked ~1 m before every goal. Stopping at the goal is the
                # job of WaypointPath (the run ends when the robot arrives), not of the plan.
                moving = [t for t in toks if t // 7 > 0]
                tok = moving[-1] if moving else 17        # 17 = 0.5 m/s straight
        toks.append(tok)
        for _ in range(CONTROL_PERIOD):
            w2.step(*from_token(tok))
    return toks


def collect_episode(args) -> dict:
    seed, student = args
    rng = np.random.default_rng(seed)
    b = lamar_cs_floor()
    route = random_route(rng, b)
    world = NavWorld(route.start_pose(b), route.events, seed=seed, building=b)
    path = WaypointPath(route, b)
    tpath = WaypointPath(route, b)                    # teacher's path from the true pose
    ctl = ClassicalController()
    if student is not None:
        student.reset(route)
    O, M, A, cond, step_idx, plans = [], [], [], [], [], []
    known = None
    prev_odom = None
    cmd = (0.0, 0.0)
    burst = 0
    burst_tok = 0
    k = 0
    best_rem, best_t = float("inf"), 0.0
    while world.s.t < route.t_max:
        tpath.update(world.s.pose.x, world.s.pose.y)  # teacher's path always tracks the TRUE pose
        if student is not None:                       # DAgger: stop once the student makes no progress
            rem = tpath.remaining(world.s.pose.x, world.s.pose.y) if len(tpath.pts) else 0.0
            if rem < best_rem - 0.5:
                best_rem, best_t = rem, world.s.t
            elif world.s.t - best_t > 15.0:
                break
        odom = world.odom()
        scan = world.lidar()
        proprio = world.proprio()
        hazard = world.hazard_sense()
        if known != world.s.hazards:
            known = dict(world.s.hazards)
            path.plan(odom[0], odom[1], known)
            p = world.s.pose
            tpath.plan(p.x, p.y, known)
        path.update(odom[0], odom[1])
        o, m = wm_obs(scan, odom_increment(prev_odom, odom), proprio, hazard)
        prev_odom = odom
        if k % CONTROL_PERIOD == 0:
            if tpath.done or tpath.trapped:
                break
            obs = make_obs(scan, odom, path, proprio, hazard)
            if (k // CONTROL_PERIOD) % 2 == 0:
                cond.append(obs)
                step_idx.append(len(O))
                plans.append(_plan_label(world, tpath, ctl, known))
            if burst > 0:
                burst -= 1
                tok = burst_tok
            elif rng.random() < EXPLORE:
                burst = int(rng.integers(2, 7))
                burst_tok = int(rng.integers(0, 35)) if rng.random() < 0.6 else int(rng.integers(14, 35))
                tok = burst_tok
            elif student is not None:
                v, w = student.act({"t": world.s.t, "pose": odom, "path": path, "scan": scan, "hazard": hazard,
                                    "proprio": proprio})
                tok = to_token(v, w)
            else:
                tok = _teacher_token(world, tpath, ctl)
            cmd = from_token(tok)
        elif student is not None and student.with_wm:
            student.act({"t": world.s.t, "pose": odom, "path": path, "scan": scan, "hazard": hazard,
                         "proprio": proprio})
        O.append(o)
        M.append(m)
        A.append(act_vec(cmd))
        world.step(*cmd)
        k += 1
    return {"O": np.array(O, np.float32), "M": np.array(M, np.float32), "A": np.array(A, np.float32),
            "cond": np.array(cond, np.float32).reshape(-1, OBS_DIM), "step_idx": np.array(step_idx, np.int64),
            "plans": np.array(plans, np.int64).reshape(-1, PLAN_L), "collisions": world.s.collisions,
            "seed": seed}


_STUDENT = None


def _init_worker(student):
    global _STUDENT
    import torch
    torch.set_num_threads(1)
    _STUDENT = student


def _collect_seed(seed):
    return collect_episode((seed, _STUDENT))


def collect(n: int, seed0: int, student=None, workers: int | None = None, log=print) -> list[dict]:
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    seeds = list(range(seed0, seed0 + n))
    if workers == 1:
        out = [collect_episode((s, student)) for s in seeds]
    else:
        with Pool(workers, initializer=_init_worker, initargs=(student,)) as pool:
            out = pool.map(_collect_seed, seeds, chunksize=4)
    log(f"  collected {n} episodes: {sum(len(e['O']) for e in out):,} world-model steps, "
        f"{sum(len(e['plans']) for e in out):,} plan samples, "
        f"{sum(e['collisions'] for e in out)} collisions (exploration)")
    return out
