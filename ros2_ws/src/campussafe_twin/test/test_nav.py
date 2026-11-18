"""Task 2 tests: simulator, routes, attacks, defenses, learned models, and the ROS 2 nodes (stubbed).

    python3 -m pytest ros2_ws/src/campussafe_twin/test/test_nav.py
"""
import importlib
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.nav.attacks import NAV_ATTACKS  # noqa: E402
from campussafe_twin.nav.controllers import from_token, to_token  # noqa: E402
from campussafe_twin.nav.episode import ClassicalPolicy, run_nav_episode  # noqa: E402
from campussafe_twin.nav.routes import TEST_ROUTES, random_route, resolve_goal  # noqa: E402
from campussafe_twin.nav.sim import BEAM_ANGLES, LIDAR_MAX, NavWorld, Pose, WallMap  # noqa: E402

B = lamar_cs_floor()
MODELS_DIR = os.path.join(os.path.dirname(HERE), "campussafe_twin", "nav", "models")
HAVE_MODELS = os.path.exists(os.path.join(MODELS_DIR, "nav_ncm.pt"))
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
needs_models = pytest.mark.skipif(not (HAVE_MODELS and HAVE_TORCH), reason="needs PyTorch + nav/models")


# ------------------------------------------------------------------ simulator
def test_lidar_geometry_matches_building():
    m = WallMap(B)
    fwd = int(np.argmin(np.abs(BEAM_ANGLES)))
    assert m.raycast(20, 0, 0)[fwd] == pytest.approx(LIDAR_MAX)          # long corridor ahead
    assert m.raycast(20, 0, math.pi / 2)[fwd] == pytest.approx(1.9, abs=0.01)   # wall at y = 2
    m.set_blocked(["EXIT_WEST"])
    assert m.raycast(2, 0, math.pi)[fwd] == pytest.approx(1.9, abs=0.01)      # barrier in the doorway


def test_walls_stop_the_robot():
    w = NavWorld(Pose(20, 0, math.pi / 2))
    for _ in range(50):
        w.step(1.0, 0.0)
    assert w.s.collisions == 1 and w.s.pose.y < 2 - 0.29


def test_tokens_round_trip():
    for tok in range(35):
        assert to_token(*from_token(tok)) == tok


# ------------------------------------------------------------------ routes
def test_test_routes_are_reachable_and_held_out():
    for r in TEST_ROUTES.values():
        g, path = resolve_goal(r, B, r.start, {e.node: e.kind for e in r.events if e.t == 0})
        assert path is not None, r.name
    rng = np.random.default_rng(0)
    keys = {(r.start, r.goal, tuple((e.node, e.kind) for e in r.events)) for r in TEST_ROUTES.values()}
    for _ in range(200):
        r = random_route(rng, B)
        assert (r.start, r.goal, tuple((e.node, e.kind) for e in r.events)) not in keys


# ------------------------------------------------------------------ classical baseline, attacks, crypto
def test_classical_completes_every_route():
    for r in TEST_ROUTES.values():
        res = run_nav_episode(r, ClassicalPolicy(), seed=1)
        assert res["success"] and res["collisions"] == 0, r.name


def test_command_injection_needs_crypto():
    r = TEST_ROUTES["room_to_main"]
    naive = run_nav_episode(r, ClassicalPolicy(), "cmd_injection", "none")
    signed = run_nav_episode(r, ClassicalPolicy(), "cmd_injection", "crypto")
    assert naive["forged_executed"] > 0 and not naive["success"]
    assert signed["forged_executed"] == 0 and signed["success"] and signed["detected"]


def test_crypto_detects_replay_and_dos():
    r = TEST_ROUTES["corridor_west"]
    assert run_nav_episode(r, ClassicalPolicy(), "sensor_replay", "crypto")["detected"]
    assert run_nav_episode(r, ClassicalPolicy(), "dos", "crypto")["detected"]


def test_crypto_alone_cannot_stop_a_compromised_sensor():
    r = TEST_ROUTES["corridor_west"]
    res = run_nav_episode(r, ClassicalPolicy(), "odom_spoof", "crypto")
    assert not res["success"] and not res["detected"]


def test_all_attacks_run():
    for a in NAV_ATTACKS:
        run_nav_episode(TEST_ROUTES["smoke_reroute"], ClassicalPolicy(), a, "crypto")


# ------------------------------------------------------------------ learned models
@needs_models
def test_learned_policies_drive_a_route():
    from campussafe_twin.nav.learned import NavModels, make_learned_policy
    m = NavModels(MODELS_DIR)
    for name in ("ncm", "ncm_wm"):
        res = run_nav_episode(TEST_ROUTES["corridor_west"], make_learned_policy(name, m))
        assert res["success"] and res["collisions"] == 0, name


@needs_models
def test_world_model_monitor_catches_signed_lies_without_false_alarm():
    from campussafe_twin.nav.learned import NavModels
    m = NavModels(MODELS_DIR)
    r = TEST_ROUTES["corridor_west"]
    clean = run_nav_episode(r, ClassicalPolicy(), "none", "wm", monitor=m.monitor())
    assert not clean["false_alarm"]
    for attack in ("odom_spoof", "lidar_phantom"):
        res = run_nav_episode(r, ClassicalPolicy(), attack, "full", monitor=m.monitor())
        assert res["detected"], attack
    spoof = run_nav_episode(r, ClassicalPolicy(), "odom_spoof", "full", monitor=m.monitor())
    assert spoof["success"], "dead reckoning after detection should still reach the goal"


# ------------------------------------------------------------------ ROS 2 nodes on a stub
def _ros_run(route, policy, attack, defense):
    import ros_stub
    bus = ros_stub.install()
    r = TEST_ROUTES[route]
    start = r.start_pose(B)
    bus.params = dict(route=route, nav_policy=policy, nav_attack=attack, nav_defense=defense, seed=0,
                      spawn_x=start.x, spawn_y=start.y, spawn_yaw=start.th, output_dir="/tmp/campussafe_test_nav")
    from campussafe_twin.nodes import nav_attack_injector, nav_controller, nav_metrics, nav_motor_driver
    for mod in (nav_attack_injector, nav_controller, nav_motor_driver, nav_metrics):
        importlib.reload(mod)
    world = NavWorld(start, r.events, seed=0)
    ros_stub.GazeboStub(bus, world, (start.x, start.y, start.th))
    nodes = [nav_attack_injector.NavAttackInjector(), nav_controller.NavController(),  # noqa: F841
             nav_motor_driver.NavMotorDriver(), nav_metrics.NavMetrics()]
    bus.spin_for(r.t_max)
    results = [m for m in bus.log if m[2].startswith("RESULT")]
    return world, results[-1][2] if results else ""


def test_ros_nodes_closed_loop_on_stub():
    world, res = _ros_run("corridor_west", "classical", "none", "none")
    assert "success=1" in res and "collisions=0" in res
    world, res = _ros_run("room_to_main", "classical", "cmd_injection", "crypto")
    assert "success=1" in res and "detected=1" in res


@needs_models
def test_ros_nodes_with_learned_stack_on_stub():
    world, res = _ros_run("corridor_west", "ncm_wm", "odom_spoof", "full")
    assert "detected=1" in res
