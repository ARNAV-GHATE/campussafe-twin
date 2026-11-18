#!/usr/bin/env python3
"""Thesis experiment grid: normal operation vs cyberattacks, with and without defenses.

    python run_experiments.py                  # full grid, 5 seeds (~20-40 min on a laptop CPU)
    python run_experiments.py --trials 2       # faster
    python run_experiments.py --quick          # 1 seed, 2 scenarios, smoke test (~2 min)
    python run_experiments.py --policies rule ncm wm_ncm --workers 4   # add the imagination-selection ablation
    python run_experiments.py --policies rule ncm slm --model qwen3:4b  # include the SLM (needs Ollama)

Writes to results/experiments/ (or --out):
    raw.csv        one row per episode
    summary.csv    mean per (policy, attack, defense) over scenarios and seeds
    report.md      the tables below, ready to paste into the thesis
    fig_*.png      figures

Experiments
  E1 normal operation      policies x scenarios, no attack               -> performance
  E2 attack impact         every attack, no defense                      -> vulnerability
  E3 defenses              every attack x {crypto, wm, full}             -> detection, recovery
Resilience ratio R = (evac% attack / evac% normal) x min(1, mean evac time normal / mean evac time attack);
R = 1 means the attack had no effect, lower is worse.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ros2_ws", "src", "campussafe_twin"))

from campussafe_twin.core.attacks import ATTACKS  # noqa: E402
from campussafe_twin.core.building import lamar_cs_floor  # noqa: E402
from campussafe_twin.core.episode import run_episode  # noqa: E402
from campussafe_twin.core.policies import make_policy  # noqa: E402
from campussafe_twin.core.scenarios import SCENARIOS  # noqa: E402

DEFENSES = ("none", "crypto", "wm", "full")
METRICS = ("pct_evacuated", "mean_evacuation_time_s", "people_exposed", "invalid_rate_pct", "mean_latency_ms",
           "detected", "detection_delay_s", "false_alarm", "forged_cmds_executed", "fake_hazards_cleared")


_W: dict = {}


def _init_worker():
    import torch
    torch.set_num_threads(1)
    from campussafe_twin.learning import monitor_factory
    _W["factory"] = monitor_factory()
    _W["b"] = lamar_cs_floor()
    _W["cache"] = {}


def _run_task(task):
    pol, atk, dfn, sc, seed, model = task
    if pol in ("ncm", "wm_ncm"):
        policy = _W["cache"].setdefault(pol, make_policy(pol, model))
        policy.seed = seed
    else:
        policy = make_policy(pol, model, seed=seed)
    r = run_episode(_W["b"], SCENARIOS[sc], policy, seed, attack=atk, defense=dfn, monitor_factory=_W["factory"])
    r["policy"] = pol
    return r


def _progress(done, total, t0):
    if done % 50 == 0 or done == total:
        el = time.time() - t0
        print(f"  {done}/{total} episodes  ({el / 60:.1f} min, ~{el / done * (total - done) / 60:.1f} min left)",
              flush=True)


def mean(vals):
    vals = [v for v in vals if v is not None and v != ""]
    return statistics.mean(float(v) for v in vals) if vals else None


def fmt(v, nd=1, pct=False):
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policies", nargs="+", default=["rule", "ncm"])
    p.add_argument("--attacks", nargs="+", default=list(ATTACKS))
    p.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--model", default="llama3.2:3b", help="Ollama model if 'slm' is in --policies")
    p.add_argument("--out", default="results/experiments")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = p.parse_args()
    if args.quick:
        args.trials, args.scenarios = 1, ["fire_north", "blocked_exit"]
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    jobs = [("none", "none", "none")] + [(pol, atk, d) for pol in args.policies for atk in args.attacks
                                         for d in (["none"] if atk == "none" else args.defenses)]
    jobs += [(pol, "none", d) for pol in args.policies for d in args.defenses if d != "none"]   # false alarms
    tasks = [(pol, atk, dfn, sc, seed, args.model) for pol, atk, dfn in jobs
             for sc in args.scenarios for seed in range(args.trials)]
    print(f"{len(tasks)} episodes on {args.workers} worker(s)", flush=True)
    rows = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker) as pool:
            for r in pool.imap_unordered(_run_task, tasks, chunksize=4):
                rows.append(r)
                _progress(len(rows), len(tasks), t0)
    else:
        _init_worker()
        for task in tasks:
            rows.append(_run_task(task))
            _progress(len(rows), len(tasks), t0)
    rows.sort(key=lambda r: (r["policy"], r["attack"], r["defense"], r["scenario"], r["seed"]))

    with open(os.path.join(args.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def group(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    summary = []
    for key in sorted({(r["policy"], r["attack"], r["defense"]) for r in rows}):
        g = group(policy=key[0], attack=key[1], defense=key[2])
        summary.append({"policy": key[0], "attack": key[1], "defense": key[2], "runs": len(g)}
                       | {m: mean([r[m] for r in g]) for m in METRICS})
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    S = {(s["policy"], s["attack"], s["defense"]): s for s in summary}

    def resilience(pol, atk, dfn):
        n, a = S.get((pol, "none", "none")), S.get((pol, atk, dfn))
        if not n or not a or not a["mean_evacuation_time_s"]:
            return None
        # time factor capped at 1: an attack that traps the slowest people must not look like an improvement
        return (a["pct_evacuated"] / n["pct_evacuated"]) * min(1.0, n["mean_evacuation_time_s"] / a["mean_evacuation_time_s"])

    L = [f"# Experiment results\n\nGenerated by `run_experiments.py` in {(time.time() - t0) / 60:.1f} min: "
         f"{len(rows)} episodes, scenarios {', '.join(args.scenarios)}, {args.trials} seed(s) each.\n"]
    L.append("## E1 – Normal operation\n\n| Policy | Evacuated | Mean evac. time (s) | People exposed | "
             "Invalid actions | Decision latency (ms) |\n|---|---|---|---|---|---|")
    for pol in ["none"] + args.policies:
        s = S.get((pol, "none", "none"))
        if s:
            L.append(f"| {pol} | {fmt(s['pct_evacuated'])}% | {fmt(s['mean_evacuation_time_s'])} | "
                     f"{fmt(s['people_exposed'])} | {fmt(s['invalid_rate_pct'])}% | {fmt(s['mean_latency_ms'])} |")
    L.append("\nPer scenario (mean evacuation time, s):\n\n| Policy | " + " | ".join(args.scenarios)
             + " |\n|---|" + "---|" * len(args.scenarios))
    for pol in ["none"] + args.policies:
        cells = [fmt(mean([r["mean_evacuation_time_s"] for r in group(policy=pol, attack="none", defense="none",
                                                                       scenario=sc)])) for sc in args.scenarios]
        L.append(f"| {pol} | " + " | ".join(cells) + " |")

    atks = [a for a in args.attacks if a != "none"]
    L.append("\n## E2 – Attack impact without defense\n\nResilience ratio R (1.00 = unaffected) and evacuation "
             "outcome.\n\n| Attack | " + " | ".join(f"{p_} R" for p_ in args.policies) + " | "
             + " | ".join(f"{p_} evac / time / exposed" for p_ in args.policies) + " |\n|---|"
             + "---|" * (2 * len(args.policies)))
    for atk in atks:
        rs = [fmt(resilience(p_, atk, "none"), 2) for p_ in args.policies]
        ev = []
        for p_ in args.policies:
            s = S.get((p_, atk, "none"))
            ev.append(f"{fmt(s['pct_evacuated'])}% / {fmt(s['mean_evacuation_time_s'])} s / {fmt(s['people_exposed'])}"
                      if s else "–")
        L.append(f"| {atk} | " + " | ".join(rs) + " | " + " | ".join(ev) + " |")

    for pol in args.policies:
        L.append(f"\n## E3 – Defenses, policy `{pol}`\n\nDetection rate / mean detection delay (s) / resilience "
                 "ratio R.\n\n| Attack | " + " | ".join(args.defenses) + " |\n|---|" + "---|" * len(args.defenses))
        for atk in atks:
            cells = []
            for d in args.defenses:
                s = S.get((pol, atk, d))
                if not s:
                    cells.append("–")
                    continue
                det = "–" if d == "none" else f"{fmt(s['detected'], pct=True)} / {fmt(s['detection_delay_s'])}"
                cells.append(f"{det} / R {fmt(resilience(pol, atk, d), 2)}")
            L.append(f"| {atk} | " + " | ".join(cells) + " |")
        fa = [fmt(S[(pol, "none", d)]["false_alarm"], pct=True) if (pol, "none", d) in S else "–"
              for d in args.defenses]
        L.append("| *false alarms (no attack)* | " + " | ".join(fa) + " |")
        fe = [fmt(S[(pol, "command_injection", d)]["forged_cmds_executed"]) if (pol, "command_injection", d) in S
              else "–" for d in args.defenses]
        L.append("| *forged commands executed (command_injection)* | " + " | ".join(fe) + " |")

    figs = make_figures(S, args, resilience)
    if figs:
        L.append("\n## Figures\n\n" + "\n".join(f"![{f}]({f})" for f in figs))
    L.append("\n## Notes\n\n- Attacks start 10 s after the alarm. `detected` counts only alarms raised after the "
             "attack started; any alarm in an attack-free run counts as a false alarm.\n"
             "- Mean evacuation time covers evacuated people only, so read it together with the evacuated %.\n"
             "- The learned components were trained on randomized scenarios only; the four named scenarios are "
             "held-out test cases.\n")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nSaved {args.out}/raw.csv, summary.csv, report.md" + (", " + ", ".join(figs) if figs else ""))


def make_figures(S, args, resilience) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out = []
    atks = [a for a in args.attacks if a != "none"]
    colors = {"none": "#9aa0a6", "crypto": "#4e79a7", "wm": "#f28e2b", "full": "#59a14f"}
    for pol in args.policies:
        fig, ax = plt.subplots(figsize=(10, 4))
        w = 0.8 / len(args.defenses)
        for i, d in enumerate(args.defenses):
            vals = [resilience(pol, a, d) or 0 for a in atks]
            ax.bar([j + i * w for j in range(len(atks))], vals, w, label=d, color=colors.get(d))
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks([j + w * (len(args.defenses) - 1) / 2 for j in range(len(atks))])
        ax.set_xticklabels(atks, rotation=20, ha="right")
        ax.set_ylabel("Resilience ratio R")
        ax.set_title(f"Resilience under attack by defense – policy {pol}", pad=28)
        ax.set_ylim(0, 1.05)
        ax.legend(title=None, ncol=len(args.defenses), fontsize=8, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        name = f"fig_resilience_{pol}.png"
        fig.savefig(os.path.join(args.out, name), dpi=150)
        plt.close(fig)
        out.append(name)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    pols = ["none"] + args.policies
    vals = [S[(p_, "none", "none")]["mean_evacuation_time_s"] if (p_, "none", "none") in S else 0 for p_ in pols]
    ax.bar(pols, vals, color=["#9aa0a6"] + ["#4e79a7"] * len(args.policies))
    ax.set_ylabel("Mean evacuation time (s)")
    ax.set_title("Normal operation")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_normal.png"), dpi=150)
    plt.close(fig)
    return ["fig_normal.png"] + out


if __name__ == "__main__":
    main()
