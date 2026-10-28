"""Unit tests for the ROS-independent core. Run: python3 -m pytest ros2_ws/src/campussafe_twin/test"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campussafe_twin.core.actions import Action  # noqa: E402
from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.core.episode import run_episode  # noqa: E402
from campussafe_twin.core.knowledge import RobotKnowledge  # noqa: E402
from campussafe_twin.core.policies import MockSLMPolicy, NoRobotPolicy, RulePolicy  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS  # noqa: E402
from campussafe_twin.core.state import build_state  # noqa: E402
from campussafe_twin.core.validator import validate  # noqa: E402

B = lamar_cs_floor()


def test_paths_avoid_hazards():
    path, _ = B.shortest_path("N1", "EXIT_EAST")
    assert "C20" in path
    assert B.shortest_path("N1", "EXIT_EAST", avoid={"C20"})[0] is None
    exit_id, _, _ = B.nearest_exit("N1", avoid={"EXIT_WEST"})
    assert exit_id == "EXIT_MAIN"


def _state(hazards):
    k = RobotKnowledge(B)
    k.hazards = dict(hazards)
    k.counts = {"N1": 3}
    return build_state(B, k, "C20", None, 0.0)


def test_validator_rejects_unsafe_and_hallucinated_actions():
    s = _state({"EXIT_WEST": "blocked", "C15": "smoke"})
    assert not validate(Action("guide_to_exit", "EXIT_WEST"), s, B)[0]
    assert not validate(Action("navigate_to", "C15"), s, B)[0]
    assert not validate(Action("navigate_to", "ROOM_404"), s, B)[0]
    assert not validate(Action("fly"), s, B)[0]
    assert validate(Action("guide_to_exit", "EXIT_MAIN"), s, B)[0]


def test_rule_policy_reports_hazard_first():
    s = _state({"C15": "smoke"})
    assert RulePolicy().decide(s).action.action == "report_hazard"


def test_robot_improves_evacuation():
    for name in ("blocked_exit", "multi_hazard"):
        with_robot = run_episode(B, SCENARIOS[name], RulePolicy(), seed=0)
        without = run_episode(B, SCENARIOS[name], NoRobotPolicy(), seed=0)
        assert with_robot["mean_evacuation_time_s"] < without["mean_evacuation_time_s"]


def test_validator_catches_mock_errors():
    r = run_episode(B, SCENARIOS["fire_north"], MockSLMPolicy(error_rate=0.5, seed=1), seed=0)
    assert r["invalid_actions"] > 0
