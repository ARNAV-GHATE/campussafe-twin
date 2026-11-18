"""Train every Task 2 model (CPU, ~1 h on 2 cores). Pre-trained weights ship in nav/models/.

    cd ros2_ws/src/campussafe_twin
    python -m campussafe_twin.nav.train            # full
    python -m campussafe_twin.nav.train --quick    # small smoke run (~3 min, worse models)

Steps
  1. collect teacher episodes on random routes (multiprocessing)
  2. train the navigation world model (NavRSSM)
  3. train the NCM plan tokenizer, then the ncm and ncm_wm generators
  4. DAgger (2 rounds): let the ncm student drive, label its states with the teacher, retrain
  5. calibrate the attack monitor's CUSUM (k, H) on attack-free runs
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..learning.ncm import tokenizer_accuracy, train_generator, train_tokenizer
from .controllers import N_TOKENS
from .data import collect
from .episode import ClassicalPolicy, run_nav_episode
from .learned import (CHANNELS, MODEL_DIR, PLAN_L, SCALES, NavMonitor, NavWMRuntime, NCMNavPolicy, train_nav_wm,
                      wm_obs)
from .routes import random_route


def wm_features(rt: NavWMRuntime, episodes) -> list[np.ndarray]:
    """Belief features [h, z] after observing o_t, at every planner sample of every episode."""
    out = []
    for e in episodes:
        b = rt.init()
        a_prev = np.zeros(2, np.float32)
        feats = np.zeros((len(e["O"]), rt.m.deter + rt.m.stoch), np.float32)
        for t in range(len(e["O"])):
            b = rt.observe(b, a_prev, e["O"][t], e["M"][t])
            feats[t] = rt.features(b)
            a_prev = e["A"][t]
        out.append(feats[e["step_idx"]])
    return out


def build_planner_set(rt, episodes):
    feats = wm_features(rt, episodes)
    cond = np.concatenate([e["cond"] for e in episodes])
    plans = np.concatenate([e["plans"] for e in episodes])
    f = np.concatenate(feats)
    return cond, np.concatenate([cond, f], 1), plans


def calibrate(models_dir, rt, n, seed0, log=print):
    """Run attack-free random routes with the monitor in 'observe only' mode; set k and H."""
    big = {c: {"k": 1e9, "H": 1e9} for c in CHANNELS}
    es = {c: [] for c in CHANNELS}
    per_ep = []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        mon = NavMonitor(rt, big)
        run_nav_episode(random_route(rng), ClassicalPolicy(), "none", "wm", seed=seed0 + i, monitor=mon)
        per_ep.append({c: np.array(mon.log_e[c]) for c in CHANNELS})
        for c in CHANNELS:
            es[c].extend(mon.log_e[c])
    calib = {}
    for c in CHANNELS:
        # k: 95th percentile of clean per-step surprise. Higher quantiles are dominated by rare
        # legitimate spikes (U-turns at re-plans, wall contacts) and made the odometry slack so large
        # that a 0.25 m/s pose drift never accumulated evidence. H absorbs those spikes instead.
        k = float(np.quantile(es[c], 0.95))
        maxS = []
        for ep in per_ep:
            S, m = 0.0, 0.0
            for e in ep[c]:
                S = max(0.0, S + e - k)
                m = max(m, S)
            maxS.append(m)
        H = float(max(np.quantile(maxS, 0.99) * 1.5, 2 * k))
        calib[c] = {"k": k, "H": H, "clean_runs_over_H": float(np.mean(np.array(maxS) > H))}
        log(f"  monitor {c:4s}: k = {k:7.2f}   H = {H:7.2f}   "
            f"(clean runs that would alarm: {100 * calib[c]['clean_runs_over_H']:.1f} %)")
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dagger", type=int, default=250, help="episodes per DAgger round")
    ap.add_argument("--rounds", type=int, default=2, help="DAgger rounds")
    ap.add_argument("--wm-steps", type=int, default=3000)
    ap.add_argument("--tok-steps", type=int, default=1500)
    ap.add_argument("--gen-steps", type=int, default=2500)
    ap.add_argument("--calib", type=int, default=150)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=MODEL_DIR)
    ap.add_argument("--reuse-wm", action="store_true", help="load nav_world_model.pt from --out instead of training")
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.dagger, args.rounds, args.wm_steps, args.tok_steps, args.gen_steps, args.calib = \
            200, 40, 1, 600, 500, 600, 30
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    report = {"torch": torch.__version__, "episodes": args.episodes, "dagger_episodes": args.dagger}

    print("1. collecting teacher episodes on random routes", flush=True)
    eps = collect(args.episodes, seed0=0)

    print("2. training the navigation world model", flush=True)
    wm_path = os.path.join(args.out, "nav_world_model.pt")
    if args.reuse_wm and os.path.exists(wm_path):
        print("   (reusing the trained world model in --out)")
        rt = NavWMRuntime(wm_path)
    else:
        wm = train_nav_wm([e["O"] for e in eps], [e["M"] for e in eps], [e["A"] for e in eps], steps=args.wm_steps)
        torch.save(wm.state_dict(), wm_path)
        rt = NavWMRuntime(model=wm)

    print("3. training the NCM tokenizer and generators", flush=True)
    cond, cond_wm, plans = build_planner_set(rt, eps)
    del eps
    tok = train_tokenizer(plans, N_TOKENS, steps=args.tok_steps, L=PLAN_L, scales=SCALES)
    acc = tokenizer_accuracy(tok, torch.as_tensor(plans[:5000]))
    report["tokenizer"] = acc
    gen = train_generator(cond, plans, tok, steps=args.gen_steps, B=128)

    for rnd in range(1, args.rounds + 1):
        print(f"4.{rnd} DAgger round {rnd}: the ncm student drives, the teacher labels", flush=True)
        student = NCMNavPolicy(tok, gen, None, with_wm=False)
        eps_d = collect(args.dagger, seed0=500_000 * rnd, student=student)
        c2, cw2, p2 = build_planner_set(rt, eps_d)
        del eps_d
        cond, cond_wm, plans = np.concatenate([cond, c2]), np.concatenate([cond_wm, cw2]), np.concatenate([plans, p2])
        final = rnd == args.rounds
        gen = train_generator(cond, plans, tok, steps=int(args.gen_steps * (1.4 if final else 1.0)), B=128, seed=rnd)
    gens = {"ncm": gen,
            "ncm_wm": train_generator(cond_wm, plans, tok, steps=int(args.gen_steps * 1.4), B=128, seed=99)}
    torch.save({"tokenizer": tok.state_dict(),
                "ncm": gens["ncm"].state_dict(), "ncm_cond_dim": cond.shape[1],
                "ncm_wm": gens["ncm_wm"].state_dict(), "ncm_wm_cond_dim": cond_wm.shape[1]},
               os.path.join(args.out, "nav_ncm.pt"))
    report["planner_samples"] = int(len(plans))
    report["teacher_episodes"] = args.episodes
    report["dagger"] = {"rounds": args.rounds, "episodes_per_round": args.dagger}

    print("5. calibrating the attack monitor on attack-free runs", flush=True)
    calib = calibrate(args.out, rt, args.calib, seed0=900_000)
    with open(os.path.join(args.out, "nav_monitor_calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    report["monitor"] = calib
    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(args.out, "nav_training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"done in {report['minutes']} min -> {args.out}")


if __name__ == "__main__":
    main()
