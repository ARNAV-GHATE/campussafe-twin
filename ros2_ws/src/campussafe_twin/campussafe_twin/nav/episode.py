"""Task 2 episode: drive a route under (optional) attack with (optional) defenses.

    truth --sign--> /odom, /scan --[attacker]--> NavGuard --> controller --sign--> /cmd_vel
          --[attacker may inject]--> MotorDriver --> wheels

Success uses the TRUE pose: the robot must physically end within 0.75 m of its goal. A robot
that stops because a spoofed /odom says it has arrived has failed.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..core.building import lamar_cs_floor
from ..core.security import Signer
from .attacks import ATTACK_START, NAV_ATTACKS, NavAttacker
from .controllers import ClassicalController, WaypointPath
from .defense import MotorDriver, NavGuard
from .routes import Route
from .sim import DT, NavWorld

SUCCESS_R = 0.75


class ClassicalPolicy:
    """Adapter so the classical controller has the same interface as the learned ones."""
    name = "classical"
    period = 1           # control every sim step (10 Hz)

    def __init__(self):
        self.c = ClassicalController()

    def reset(self, route: Route) -> None:
        pass

    def act(self, ctx: dict) -> tuple[float, float]:
        return self.c.act(ctx["pose"], ctx["path"], ctx["scan"], ctx["hazard"])


def make_nav_policy(name: str, models=None):
    if name == "classical":
        return ClassicalPolicy()
    from .learned import make_learned_policy            # needs PyTorch
    return make_learned_policy(name, models)


def run_nav_episode(route: Route, policy, attack: str = "none", defense: str = "none", seed: int = 0,
                    monitor=None, log=None) -> dict:
    b = lamar_cs_floor()
    world = NavWorld(route.start_pose(b), route.events, seed=seed, building=b)
    attacker = NavAttacker(NAV_ATTACKS[attack], seed)
    s_odom, s_scan, s_cmd = Signer(), Signer(), Signer()
    guard = NavGuard(defense, world.map, monitor)
    driver = MotorDriver(crypto=defense in ("crypto", "full"))
    path = WaypointPath(route, b)
    policy.reset(route)
    if monitor is not None:
        monitor.reset()
    known: dict | None = None
    cmd = (0.0, 0.0)
    held = (0.0, 0.0)
    lat = []
    steps = int(route.t_max / DT)
    believed_arrived = False
    for k in range(steps):
        t = world.s.t
        odom_env = s_odom.sign({"pose": [round(float(v), 4) for v in world.odom()]}, t)
        scan_env = s_scan.sign({"ranges": [round(float(v), 3) for v in world.lidar()]}, t)
        odom_env, scan_env = attacker.tamper_sensors(t, odom_env, scan_env)
        proprio = world.proprio()
        hazard = world.hazard_sense()
        pose, scan = guard.receive(t, odom_env, scan_env, proprio, cmd, hazard)

        if known != world.s.hazards:                     # building feed (trusted here; attacked in Task 1)
            known = dict(world.s.hazards)
            path.plan(pose[0], pose[1], known)
        path.update(pose[0], pose[1])
        if path.done:
            believed_arrived = True
            break
        if path.trapped:
            break

        if k % policy.period == 0:
            t0 = time.perf_counter()
            held = policy.act({"t": t, "pose": pose, "path": path, "scan": scan, "hazard": hazard,
                               "proprio": proprio, "guard": guard})
            lat.append((time.perf_counter() - t0) * 1000)
        v, w = held
        cap = guard.speed_cap()
        if cap <= 0:
            v, w = 0.0, 0.0
        v = min(v, cap)
        cmd = (v, w)

        driver.receive(s_cmd.sign({"v": v, "w": w}, t), t)
        forged = attacker.forged_command(t)
        if forged is not None:
            driver.receive(forged, t, forged=True)
            if driver.rejected_at is not None:
                guard._alarm(driver.rejected_at, "forged:cmd")
        if guard.cmd_bad:
            world.step(0.0, 0.0)                         # hardware interlock: motors off
        else:
            world.step(*driver.cmd)
        if log and k % 50 == 0:
            p = world.s.pose
            log(f"t={t:5.1f}s true=({p.x:5.1f},{p.y:5.1f}) est=({pose[0]:5.1f},{pose[1]:5.1f}) "
                f"cmd=({driver.cmd[0]:.2f},{driver.cmd[1]:+.2f}) alarms={[a for _, a in guard.alarms]}")

    s = world.s
    goal_xy = b.xy(path.goal) if path.goal else (math.nan, math.nan)
    final_d = math.hypot(s.pose.x - goal_xy[0], s.pose.y - goal_xy[1])
    success = final_d <= SUCCESS_R
    attacked = attack != "none"
    alarms_after = [(ta, a) for ta, a in guard.alarms if ta >= ATTACK_START]
    first = min((ta for ta, _ in alarms_after), default=None)
    return {
        "route": route.name, "policy": policy.name, "attack": attack, "defense": defense, "seed": seed,
        "success": int(success), "time_s": round(s.t, 1) if success else None,
        "final_dist_m": round(final_d, 2), "believed_arrived": int(believed_arrived),
        "collisions": s.collisions, "exposure_s": round(s.exposure_s, 1),
        "min_clearance_m": round(s.min_clearance, 2), "path_len_m": round(s.path_len, 1),
        "detected": int(attacked and bool(alarms_after)),
        "detection_delay_s": round(first - ATTACK_START, 1) if (attacked and first is not None) else None,
        "false_alarm": int(not attacked and bool(guard.alarms)),
        "alarms": ";".join(sorted({a for _, a in guard.alarms})),
        "forged_executed": driver.forged_executed, "forged_blocked": driver.forged_blocked,
        "mean_latency_ms": round(float(np.mean(lat)), 3) if lat else 0.0,
    }
