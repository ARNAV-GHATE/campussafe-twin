"""Tests for attacks, message authentication, the defense gateway, and (if PyTorch and trained
models are present) the world model + NCM components. Run: python3 -m pytest ros2_ws/src/campussafe_twin/test"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campussafe_twin.core.attacks import ATTACKS, Attacker, bms_reading  # noqa: E402
from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.core.episode import run_episode  # noqa: E402
from campussafe_twin.core.policies import RulePolicy  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS, random_scenario  # noqa: E402
from campussafe_twin.core.security import Signer, Verifier, forge, resign  # noqa: E402
from campussafe_twin.core.world import World  # noqa: E402

B = lamar_cs_floor()


def test_hmac_rejects_forgery_replay_and_stale():
    s, v = Signer(), Verifier(max_age_s=2.0)
    e1 = s.sign({"x": 1}, 0.0)
    assert v.verify(e1, 0.5)[0]
    assert not v.verify(e1, 0.6)[0]                        # replay
    assert not v.verify(forge({"x": 2}, 1.0), 1.0)[0]      # no key
    e2 = s.sign({"x": 3}, 1.0)
    tampered = dict(e2, payload={"x": 4})
    assert not v.verify(tampered, 1.0)[0]                  # modified in transit
    assert v.verify(resign(e2, {"x": 4}), 1.0)[0]          # compromised sensor holds the key
    assert not v.verify(s.sign({"x": 5}, 1.0), 9.0)[0]     # stale


def test_attacks_tamper_as_described():
    w = World(B, SCENARIOS["fire_north"], 0)
    w.t = 20.0
    truth = w.snapshot()
    a = Attacker(ATTACKS["false_hazard"], B)
    r = a.tamper_reading(bms_reading(truth), truth)
    assert r["hazards"][a.fake_exit] == "fire" and a.fake_exit in B.exits
    spoof = Attacker(ATTACKS["occupancy_spoof"], B).tamper_reading(bms_reading(truth), truth)
    assert max(spoof["area_counts"].values()) == 25


def test_attacks_hurt_and_crypto_blocks_network_attacks():
    sc = SCENARIOS["fire_north"]
    clean = run_episode(B, sc, RulePolicy(), 0)
    inj = run_episode(B, sc, RulePolicy(), 0, attack="command_injection")
    assert inj["forged_cmds_executed"] > 0
    assert inj["mean_evacuation_time_s"] > clean["mean_evacuation_time_s"]
    blocked = run_episode(B, sc, RulePolicy(), 0, attack="command_injection", defense="crypto")
    assert blocked["forged_cmds_executed"] == 0 and blocked["detected"] == 1
    assert blocked["mean_evacuation_time_s"] == clean["mean_evacuation_time_s"]
    rep = run_episode(B, sc, RulePolicy(), 0, attack="replay", defense="crypto")
    assert rep["detected"] == 1
    for name in ("none", "crypto"):
        assert run_episode(B, sc, RulePolicy(), 0, defense=name)["false_alarm"] == 0


def test_random_scenarios_are_valid():
    for s in range(20):
        sc = random_scenario(B, s)
        assert all(e.node in B.nodes for e in sc.events)
        assert sum(sc.occupants.values()) > 0


# ---- learned components (skipped without PyTorch / trained models) ----------------------
torch = pytest.importorskip("torch")
from campussafe_twin.learning import MODEL_DIR, monitor_factory  # noqa: E402

needs_models = pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "ncm.pt")),
                                  reason="trained models not found; run python -m campussafe_twin.learning.train")


@needs_models
def test_ncm_policy_proposes_valid_actions():
    from campussafe_twin.core.policies import make_policy
    pol = make_policy("ncm")
    r = run_episode(B, SCENARIOS["blocked_exit"], pol, 0)
    assert r["pct_evacuated"] > 90 and r["invalid_rate_pct"] < 50
    assert len(pol.last_plan) == 8


@needs_models
def test_world_model_monitor_catches_compromised_sensors():
    f = monitor_factory()
    fh = run_episode(B, SCENARIOS["fire_north"], RulePolicy(), 0, attack="false_hazard", defense="wm", monitor_factory=f)
    assert fh["detected"] == 1 and fh["fake_hazards_cleared"] >= 1
    cs = run_episode(B, SCENARIOS["fire_north"], RulePolicy(), 0, attack="occupancy_spoof", defense="wm",
                     monitor_factory=f)
    assert cs["detected"] == 1
    clean = run_episode(B, SCENARIOS["fire_north"], RulePolicy(), 0, defense="full", monitor_factory=f)
    assert clean["false_alarm"] == 0
