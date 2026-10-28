#!/usr/bin/env python3
<<<<<<< HEAD
"""Run CampusSafe Twin experiments without ROS or Gazebo."""
from __future__ import annotations
=======
"""Run CampusSafe Twin experiments without ROS or Gazebo.

Examples
    python run_offline.py                                  # rule baseline, all scenarios
    python run_offline.py --policy rule mock none --trials 5
    python run_offline.py --policy slm --model llama3.2:3b --scenario fire_north -v
    python run_offline.py --policy slm rule none --trials 10 --out results/run1.csv
"""
from __future__ import annotations

>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)
import argparse
import csv
import os
import statistics
import sys
<<<<<<< HEAD
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ros2_ws", "src", "campussafe_twin"))
from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.episode import run_episode
from campussafe_twin.core.policies import make_policy
from campussafe_twin.core.scenarios import SCENARIOS
from campussafe_twin.core.slm_client import OllamaClient
=======

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ros2_ws", "src", "campussafe_twin"))

from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.core.episode import run_episode  # noqa: E402
from campussafe_twin.core.policies import make_policy  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS  # noqa: E402
from campussafe_twin.core.slm_client import OllamaClient  # noqa: E402
>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)

SUMMARY_COLS = ["pct_evacuated", "total_evacuation_time_s", "mean_evacuation_time_s", "people_exposed",
                "invalid_rate_pct", "mean_latency_ms", "guidance_replan_time_s"]

<<<<<<< HEAD
=======

>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", nargs="+", default=["all"], help=f"all | {' | '.join(SCENARIOS)}")
    p.add_argument("--policy", nargs="+", default=["rule"], help="rule | slm | mock | none")
    p.add_argument("--model", default="llama3.2:3b", help="Ollama model tag for --policy slm")
    p.add_argument("--host", default="http://localhost:11434")
<<<<<<< HEAD
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", default="results/offline_results.csv")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    scenarios = list(SCENARIOS) if "all" in args.scenario else args.scenario
    if "slm" in args.policy and not OllamaClient(args.model, args.host).available():
        sys.exit(f"Ollama is not reachable at {args.host}. Start Ollama and pull {args.model}.")
=======
    p.add_argument("--trials", type=int, default=3, help="runs per scenario/policy (different seeds)")
    p.add_argument("--out", default="results/offline_results.csv")
    p.add_argument("-v", "--verbose", action="store_true", help="print every decision")
    args = p.parse_args()

    scenarios = list(SCENARIOS) if "all" in args.scenario else args.scenario
    if "slm" in args.policy and not OllamaClient(args.model, args.host).available():
        sys.exit(f"Ollama is not reachable at {args.host}. Start it with `ollama serve` "
                 f"and `ollama pull {args.model}`, or use --policy mock to test without it.")

>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)
    building = lamar_cs_floor()
    rows = []
    for sc in scenarios:
        for pol in args.policy:
            for seed in range(args.trials):
                policy = make_policy(pol, args.model, args.host, seed)
<<<<<<< HEAD
                print(f"running {sc:13s} {policy.name:20s} seed {seed}", flush=True)
                r = run_episode(building, SCENARIOS[sc], policy, seed, log=print if args.verbose else None)
                rows.append(r)
=======
                print(f"\n=== {sc} | {policy.name} | seed {seed} ===" if args.verbose else
                      f"running {sc:13s} {policy.name:20s} seed {seed}", flush=True)
                r = run_episode(building, SCENARIOS[sc], policy, seed, log=print if args.verbose else None)
                rows.append(r)

>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
<<<<<<< HEAD
    print(f"Saved {len(rows)} runs to {args.out}")
    for sc in scenarios:
        for pol in sorted({r["policy"] for r in rows}):
            group = [r for r in rows if r["scenario"] == sc and r["policy"] == pol]
            if group:
                print(sc, pol, {c: round(statistics.mean([r[c] for r in group if r[c] is not None]), 2)
                               for c in SUMMARY_COLS if any(r[c] is not None for r in group)})
=======

    print(f"\nMean over {args.trials} trial(s)   (saved all runs to {args.out})")
    head = f"{'scenario':13s} {'policy':20s} " + " ".join(f"{c[:14]:>14s}" for c in SUMMARY_COLS)
    print(head + "\n" + "-" * len(head))
    for sc in scenarios:
        for pol in sorted({r["policy"] for r in rows}):
            group = [r for r in rows if r["scenario"] == sc and r["policy"] == pol]
            if not group:
                continue
            cells = []
            for c in SUMMARY_COLS:
                vals = [r[c] for r in group if r[c] is not None]
                cells.append(f"{statistics.mean(vals):14.1f}" if vals else f"{'-':>14s}")
            print(f"{sc:13s} {pol:20s} " + " ".join(cells))


>>>>>>> a6fb636 (Initial commit: CampusSafe Twin robotics project)
if __name__ == "__main__":
    main()
