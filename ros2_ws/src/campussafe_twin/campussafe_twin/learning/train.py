"""Train every learned component and write them to learning/models/.

    python -m campussafe_twin.learning.train                 # full run (~25-40 min on a laptop CPU)
    python -m campussafe_twin.learning.train --quick         # small smoke-test run (~3 min)
    python -m campussafe_twin.learning.train --skip-wm       # reuse world_model.pt, retrain NCM

Steps
  1. collect episodes from randomized scenarios (the 4 named scenarios are held out)
  2. world model (RSSM)                           -> world_model.pt
  3. NCM plan tokenizer + next-concept generator  -> ncm.pt
  4. calibrate the world-model attack monitor on attack-free runs -> monitor_calibration.json
  5. evaluate forecasting + detection thresholds  -> training_report.json
Run from the repo root with the package on the path (run_offline.py does this),
or from ros2_ws/src/campussafe_twin.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..core.building import lamar_cs_floor
from ..core.episode import run_episode
from ..core.policies import RulePolicy
from ..core.scenarios import random_scenario
from . import MODEL_DIR
from .data import collect, load, save
from .features import HAZ_KINDS, Layout, onehot
from .ncm import PLAN_LEN, train_generator, train_tokenizer
from .world_model import RSSM, WorldModelRuntime, obs_input, train_world_model

K = len(HAZ_KINDS)


# ---- NCM dataset: condition vectors (with batched world-model filtering) + expert plans ----

@torch.no_grad()
def build_ncm_dataset(episodes, L: Layout, wm: RSSM, forecast_steps=6, chunk=256):
    eps = [e for e in episodes if (e["expert"] >= 0).all()]
    conds, plans = [], []
    for c0 in range(0, len(eps), chunk):
        group = eps[c0:c0 + chunk]
        E, T = len(group), max(len(e["t"]) for e in group)

        def pad(key, fill=0):
            out = np.full((E, T) + group[0][key].shape[1:], fill, dtype=group[0][key].dtype)
            for i, e in enumerate(group):
                out[i, :len(e[key])] = e[key]
            return out

        khaz, khm, kcnt, kcm = pad("khaz"), pad("khm"), pad("kcnt"), pad("kcm")
        rep, online, loc, guid, tt = pad("rep"), pad("online"), pad("loc"), pad("guid", L.E), pad("t")
        lens = np.array([len(e["t"]) for e in group])
        guid_before = np.concatenate([np.full((E, 1), L.E), guid[:, :-1]], 1)
        h, z = wm.initial(E)
        for k in range(T):
            a = np.zeros((E, L.act_dim), np.float32)
            if k > 0:
                a[np.arange(E), loc[:, k]] = 1
                a[np.arange(E), L.R + guid_before[:, k]] = 1
                a[:, -1] = tt[:, k - 1] / 300.0
            o = obs_input(khaz[:, k], khm[:, k], kcnt[:, k], kcm[:, k])
            h, z, _, _ = wm.step(h, z, torch.as_tensor(a), wm.encode(torch.as_tensor(o)), mode=True)
            # 30 s hazard forecast under the current action
            af = np.zeros((E, L.act_dim), np.float32)
            af[np.arange(E), loc[:, k]] = 1
            af[np.arange(E), L.R + guid_before[:, k]] = 1
            af[:, -1] = tt[:, k] / 300.0
            S = 4
            hh, zz = h.repeat_interleave(S, 0), z.repeat_interleave(S, 0)
            afs = torch.as_tensor(af).repeat_interleave(S, 0)
            risk = torch.zeros(E * S, L.N)
            for _ in range(forecast_steps):
                hh, zz, _, _ = wm.step(hh, zz, afs)
                hz, _, _ = wm.decode(hh, zz)
                risk = torch.maximum(risk, 1 - torch.softmax(hz, -1)[..., 0])
            risk = risk.view(E, S, L.N).mean(1).numpy()
            feat = torch.cat([h, z], -1).numpy()
            for i in np.where(k < lens)[0]:
                gb = np.zeros(L.E + 1, np.float32)
                gb[guid_before[i, k]] = 1
                lc = np.zeros(L.R, np.float32)
                lc[loc[i, k]] = 1
                conds.append(np.concatenate([
                    (onehot(khaz[i, k], K) * khm[i, k][:, None]).ravel(), khm[i, k], kcnt[i, k] * kcm[i, k],
                    kcm[i, k], rep[i, k], np.zeros(L.N, np.float32), lc, gb,
                    [tt[i, k] / 300.0, online[i, k]], feat[i], risk[i]]).astype(np.float32))
                ex = group[i]["expert"]
                p = ex[k:k + PLAN_LEN]
                plans.append(np.concatenate([p, np.zeros(PLAN_LEN - len(p), np.int64)]))
    return np.stack(conds), np.stack(plans).astype(np.int64)


# ---- world-model evaluation: does it forecast new hazards better than chance? ----------

def _auroc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.sum()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * (len(labels) - pos)))


@torch.no_grad()
def evaluate_world_model(episodes, L: Layout, wm: RSSM, horizon=6):
    """30-s forecast of NEW hazards (nodes not yet hazardous), full observations."""
    scores, labels, acc1 = [], [], []
    for e in episodes[:300]:
        T = len(e["t"])
        h, z = wm.initial(1)
        ones = np.ones(L.N, np.float32)
        for k in range(T):
            a = np.zeros(L.act_dim, np.float32)
            if k > 0:
                a[e["loc"][k]] = 1
                a[L.R + e["guid"][k - 1]] = 1
                a[-1] = e["t"][k - 1] / 300.0
            o = obs_input(e["haz"][k][None], ones[None], e["cnt"][k][None], ones[None])
            h, z, _, _ = wm.step(h, z, torch.as_tensor(a)[None], wm.encode(torch.as_tensor(o)), mode=True)
            if k + horizon < T:
                hh, zz = h.repeat(8, 1), z.repeat(8, 1)
                risk = torch.zeros(8, L.N)
                for j in range(1, horizon + 1):
                    aj = np.zeros(L.act_dim, np.float32)
                    aj[e["loc"][k + j]] = 1
                    aj[L.R + e["guid"][k + j - 1]] = 1
                    aj[-1] = e["t"][k + j - 1] / 300.0
                    hh, zz, _, _ = wm.step(hh, zz, torch.as_tensor(aj)[None].repeat(8, 1))
                    if j == 1:
                        hz1, _, _ = wm.decode(hh, zz)
                        pred1 = torch.softmax(hz1, -1).mean(0).argmax(-1).numpy()
                        acc1.append(float((pred1 == e["haz"][k + 1]).mean()))
                hz, _, _ = wm.decode(hh, zz)
                risk = (1 - torch.softmax(hz, -1)[..., 0]).mean(0).numpy()
                now = e["haz"][k] > 0
                fut = e["haz"][k + horizon] > 0
                scores.extend(risk[~now].tolist())
                labels.extend(fut[~now].tolist())
    return {"new_hazard_30s_auroc": _auroc(scores, labels),
            "new_hazard_base_rate": float(np.mean(labels)) if labels else None,
            "one_step_hazard_accuracy": float(np.mean(acc1)) if acc1 else None}


# ---- monitor calibration on attack-free runs -------------------------------------------

def calibrate_monitor(runtime: WorldModelRuntime, n=150, seed0=900_000, log=print):
    from .wm_monitor import DEFAULT_CALIB, WMPlausibilityMonitor
    b = lamar_cs_floor()
    monitors = []

    def factory(calib):
        def f():
            m = WMPlausibilityMonitor(runtime, calib, record=True)
            monitors.append(m)
            return m
        return f

    loose = DEFAULT_CALIB | {"tau": 0.0, "z_th": 1e9, "H": 1e9}
    for i in range(n):
        run_episode(b, random_scenario(b, seed0 + i), RulePolicy(), seed=i, defense="wm", monitor_factory=factory(loose))
    p_new = np.concatenate([m.trace["p_new"] for m in monitors]) if monitors else np.array([])
    z = np.concatenate([m.trace["z"] for m in monitors])
    tau = float(np.clip(np.quantile(p_new, 0.02) if len(p_new) else 0.02, 1e-4, 0.05))
    z_th = float(max(3.0, np.quantile(z, 0.999)))
    monitors.clear()
    mid = DEFAULT_CALIB | {"tau": tau, "z_th": z_th, "H": 1e9}
    for i in range(n):
        run_episode(b, random_scenario(b, seed0 + i), RulePolicy(), seed=i, defense="wm", monitor_factory=factory(mid))
    peaks = np.array([max(m.trace["cusum"] or [0.0]) for m in monitors])
    H = float(max(3.0, np.quantile(peaks, 0.99) + 0.5))
    calib = {"tau": tau, "z_th": z_th, "H": H, "drift": DEFAULT_CALIB["drift"],
             "calibration_runs": n, "legit_new_hazards": int(len(p_new)),
             "legit_new_hazards_flagged_pct": round(100 * float((p_new < tau).mean()), 2) if len(p_new) else None,
             "count_values_flagged_pct": round(100 * float((z > z_th).mean()), 3),
             "runs_alarming_pct": round(100 * float((peaks > H).mean()), 2)}
    log(f"  monitor calibration: {calib}")
    return calib


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--wm-steps", type=int, default=3000)
    p.add_argument("--tok-steps", type=int, default=3000)
    p.add_argument("--gen-steps", type=int, default=5000)
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    p.add_argument("--skip-wm", action="store_true", help="reuse an existing world_model.pt")
    p.add_argument("--calibrate-only", action="store_true", help="only recalibrate the attack monitor")
    p.add_argument("--data", default="", help="cache file for collected episodes (.npz)")
    p.add_argument("--out", default=MODEL_DIR)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    args = p.parse_args()
    if args.quick:
        args.episodes, args.wm_steps, args.tok_steps, args.gen_steps = 200, 200, 300, 300
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    L = Layout(lamar_cs_floor())
    if args.calibrate_only:
        calib = calibrate_monitor(WorldModelRuntime(os.path.join(args.out, "world_model.pt"), layout=L))
        with open(os.path.join(args.out, "monitor_calibration.json"), "w") as f:
            json.dump(calib, f, indent=2)
        rp = os.path.join(args.out, "training_report.json")
        if os.path.exists(rp):
            with open(rp) as f:
                report = json.load(f)
            report["monitor"] = calib
            with open(rp, "w") as f:
                json.dump(report, f, indent=2)
        return
    t0 = time.time()
    report = {"episodes": args.episodes, "torch": torch.__version__}

    print(f"[1/5] collecting {args.episodes} randomized episodes")
    if args.data and os.path.exists(args.data):
        eps = load(args.data)[:args.episodes]
    else:
        eps = collect(args.episodes, 0, args.workers)
        if args.data:
            save(eps, args.data)
    report["ticks"] = int(sum(len(e["t"]) for e in eps))
    print(f"      {report['ticks']} decision ticks ({time.time() - t0:.0f}s)")

    wm_path = os.path.join(args.out, "world_model.pt")
    if args.skip_wm and os.path.exists(wm_path):
        print("[2/5] world model: reusing", wm_path)
        runtime = WorldModelRuntime(wm_path, layout=L)
    else:
        print(f"[2/5] training world model ({args.wm_steps} steps)")
        model = train_world_model(eps, L, steps=args.wm_steps)
        WorldModelRuntime.save(model, wm_path)
        runtime = WorldModelRuntime(model=model.eval(), layout=L)
    held_out = collect(120 if not args.quick else 30, 800_000, args.workers)
    report["world_model"] = evaluate_world_model(held_out, L, runtime.m)
    print(f"      held-out evaluation: {report['world_model']}")

    print("[3/5] NCM dataset (world-model features + expert plans)")
    cond, plans = build_ncm_dataset(eps, L, runtime.m)
    report["ncm_samples"] = int(len(plans))
    print(f"      {len(plans)} samples, condition dim {cond.shape[1]}")
    print(f"[4/5] NCM tokenizer ({args.tok_steps} steps) and next-concept generator ({args.gen_steps} steps)")
    tok = train_tokenizer(plans, L.V, steps=args.tok_steps)
    gen = train_generator(cond, plans, tok, steps=args.gen_steps)
    torch.save({"tok": tok.state_dict(), "gen": gen.state_dict(), "cond_dim": int(cond.shape[1]),
                "tok_config": {"d": tok.d, "codebook": tok.n_codes},
                "gen_config": {"d": gen.d}, "vocab": L.vocab}, os.path.join(args.out, "ncm.pt"))

    print("[5/5] calibrating the world-model attack monitor on attack-free runs")
    calib = calibrate_monitor(runtime, n=150 if not args.quick else 20)
    with open(os.path.join(args.out, "monitor_calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    report["monitor"] = calib
    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(args.out, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"done in {report['minutes']} min -> {args.out}")


if __name__ == "__main__":
    main()
