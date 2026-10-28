#!/usr/bin/env python3
"""Run CampusSafe Twin experiments without ROS or Gazebo."""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ros2_ws", "src", "campussafe_twin"))
from campussafe_twin.core.building import lamar_cs_floor
from campussafe_twin.core.episode import run_episode
from campussafe_twin.core.policies import make_policy
from campussafe_twin.core.scenarios import SCENARIOS
from campussafe_twin.core.slm_client import OllamaClient

SUMMARY_COLS = ["pct_evacuated", "total_evacuation_time_s", "mean_evacuation_time_s", "people_exposed",
                "invalid_rate_pct", "mean_latency_ms", "guidance_replan_time_s"]

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", nargs="+", default=["all"], help=f"all | {' | '.join(SCENARIOS)}")
    p.add_argument("--policy", nargs="+", default=["rule"], help="rule | slm | mock | none")
    p.add_argument("--model", default="llama3.2:3b", help="Ollama model tag for --policy slm")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", default="results/offline_results.csv")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    scenarios = list(SCENARIOS) if "all" in args.scenario else args.scenario
    if "slm" in args.policy and not OllamaClient(args.model, args.host).available():
        sys.exit(f"Ollama is not reachable at {args.host}. Start Ollama and pull {args.model}.")
    building = lamar_cs_floor()
    rows = []
    for sc in scenarios:
        for pol in args.policy:
            for seed in range(args.trials):
                policy = make_policy(pol, args.model, args.host, seed)
                print(f"running {sc:13s} {policy.name:20s} seed {seed}", flush=True)
                r = run_episode(building, SCENARIOS[sc], policy, seed, log=print if args.verbose else None)
                rows.append(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} runs to {args.out}")
    for sc in scenarios:
        for pol in sorted({r["policy"] for r in rows}):
            group = [r for r in rows if r["scenario"] == sc and r["policy"] == pol]
            if group:
                print(sc, pol, {c: round(statistics.mean([r[c] for r in group if r[c] is not None]), 2)
                               for c in SUMMARY_COLS if any(r[c] is not None for r in group)})
if __name__ == "__main__":
    main()
