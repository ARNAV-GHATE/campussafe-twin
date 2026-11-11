"""Learning components (need PyTorch): Dreamer-style world model and Next-Concept Modeling planner.

    features.py     fixed-size encodings of the building, observations and actions
    data.py         training data from randomized offline scenarios
    world_model.py  RSSM world model: belief filtering, forecasting, imagination
    wm_monitor.py   world-model plausibility monitor (cyberattack detection)
    ncm.py          multiscale residual-VQ plan tokenizer + next-scale generator
    ncm_policy.py   the ncm decision policy (and the wm_ncm ablation)
    train.py        trains everything: python -m campussafe_twin.learning.train
"""
import os

MODEL_DIR = os.environ.get(
    "CAMPUSSAFE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))


def monitor_factory(model_dir: str = MODEL_DIR):
    """Zero-arg factory for run_episode(..., defense="wm"|"full"): one fresh monitor per episode."""
    from .wm_monitor import WMPlausibilityMonitor, load_calibration
    from .world_model import WorldModelRuntime
    runtime = WorldModelRuntime(os.path.join(model_dir, "world_model.pt"))
    calib = load_calibration(os.path.join(model_dir, "monitor_calibration.json"))
    return lambda: WMPlausibilityMonitor(runtime, calib)
