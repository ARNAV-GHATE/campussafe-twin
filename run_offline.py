#!/usr/bin/env python3
"""Run CampusSafe Twin experiments without ROS or Gazebo.

Examples
    python run_offline.py                                  # rule baseline, all scenarios
    python run_offline.py --policy rule mock none --trials 5
    python run_offline.py --policy slm --model llama3.2:3b --scenario fire_north -v
    python run_offline.py --policy slm rule none --trials 10 --out results/run1.csv

World model + Next-Concept Modeling, and cyberattacks (need PyTorch, see README):
    python run_offline.py --policy rule ncm --attack none false_hazard --defense none full
    python run_offline.py --policy ncm --scenario blocked_exit --attack coordinated --defense full -v
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ros2_ws", "src", "campussafe_twin"))

from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.core.episode import run_episode  # noqa: E402
from campussafe_twin.core.policies import make_policy  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS  # noqa: E402
from campussafe_twin.core.slm_client import OllamaClient  # noqa: E402
from campussafe_twin.core.attacks import ATTACKS  # noqa: E402
from campussafe_twin.core.defense import DEFENSES  # noqa: E402

SUMMARY_COLS = ["pct_evacuated", "total_evacuation_time_s", "mean_evacuation_time_s", "people_exposed",
                "invalid_rate_pct", "mean_latency_ms", "guidance_replan_time_s", "detected",
                "detection_delay_s", "forged_cmds_executed"]
LEARNED = ("ncm", "wm_ncm")


def need_torch(what: str) -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit(f"{what} needs PyTorch: pip install -r requirements.txt  (CPU is enough)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", nargs="+", default=["all"], help=f"all | {' | '.join(SCENARIOS)}")
    p.add_argument("--policy", nargs="+", default=["rule"], help="rule | slm | mock | none | ncm | wm_ncm")
    p.add_argument("--attack", nargs="+", default=["none"], help=f"{' | '.join(ATTACKS)} | all")
    p.add_argument("--defense", nargs="+", default=["none"], help=f"{' | '.join(DEFENSES)}")
    p.add_argument("--model", default="llama3.2:3b", help="Ollama model tag for --policy slm")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--trials", type=int, default=3, help="runs per scenario/policy (different seeds)")
    p.add_argument("--out", default="results/offline_results.csv")
    p.add_argument("-v", "--verbose", action="store_true", help="print every decision")
    args = p.parse_args()

    scenarios = list(SCENARIOS) if "all" in args.scenario else args.scenario
    if "slm" in args.policy and not OllamaClient(args.model, args.host).available():
        sys.exit(f"Ollama is not reachable at {args.host}. Start it with `ollama serve` "
                 f"and `ollama pull {args.model}`, or use --policy mock to test without it.")

    attacks = list(ATTACKS) if "all" in args.attack else args.attack
    if any(p_ in LEARNED for p_ in args.policy):
        need_torch("--policy ncm / wm_ncm")
    factory = None
    if any(d in ("wm", "full") for d in args.defense):
        need_torch("--defense wm / full")
        from campussafe_twin.learning import monitor_factory
        factory = monitor_factory()

    building = lamar_cs_floor()
    rows = []
    policies = {}
    for sc in scenarios:
        for pol in args.policy:
            for atk in attacks:
                for dfn in args.defense:
                    for seed in range(args.trials):
                        if pol in LEARNED:                 # load the networks once, reset per episode
                            policy = policies.setdefault(pol, make_policy(pol, args.model, args.host, seed))
                            policy.seed = seed
                        else:
                            policy = make_policy(pol, args.model, args.host, seed)
                        tag = f"{sc} | {policy.name} | attack {atk} | defense {dfn} | seed {seed}"
                        print(f"\n=== {tag} ===" if args.verbose else f"running {tag}", flush=True)
                        r = run_episode(building, SCENARIOS[sc], policy, seed, log=print if args.verbose else None,
                                        attack=atk, defense=dfn, monitor_factory=factory)
                        rows.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nMean over {args.trials} trial(s)   (saved all runs to {args.out})")
    head = (f"{'scenario':13s} {'policy':14s} {'attack':19s} {'defense':7s} "
            + " ".join(f"{c[:14]:>14s}" for c in SUMMARY_COLS))
    print(head + "\n" + "-" * len(head))
    keys = sorted({(r["scenario"], r["policy"], r["attack"], r["defense"]) for r in rows},
                  key=lambda k: (scenarios.index(k[0]), k[1], attacks.index(k[2]), k[3]))
    for sc, pol, atk, dfn in keys:
        group = [r for r in rows if (r["scenario"], r["policy"], r["attack"], r["defense"]) == (sc, pol, atk, dfn)]
        cells = []
        for c in SUMMARY_COLS:
            vals = [r[c] for r in group if r[c] is not None]
            cells.append(f"{statistics.mean(vals):14.1f}" if vals else f"{'-':>14s}")
        print(f"{sc:13s} {pol:14s} {atk:19s} {dfn:7s} " + " ".join(cells))


if __name__ == "__main__":
    main()
