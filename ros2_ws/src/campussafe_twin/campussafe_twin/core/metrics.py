"""Evaluation metrics, computed from a world snapshot plus decision logs."""
from __future__ import annotations

import statistics


def summarize(snapshot: dict, decisions: list[dict], replan_times: list[float]) -> dict:
    occ = snapshot["occupants"]
    total = len(occ)
    evac = [o["evac_time"] for o in occ if o["status"] == "evacuated"]
    lat = [d["latency_ms"] for d in decisions]
    invalid = sum(1 for d in decisions if not d["valid"])
    return {
        "scenario": snapshot["scenario"],
        "occupants": total,
        "evacuated": len(evac),
        "pct_evacuated": round(100 * len(evac) / total, 1) if total else 0.0,
        "total_evacuation_time_s": max(evac) if len(evac) == total and evac else None,
        "mean_evacuation_time_s": round(statistics.mean(evac), 1) if evac else None,
        "exposures": sum(o["exposures"] for o in occ),
        "people_exposed": sum(1 for o in occ if o["exposures"] > 0),
        "decisions": len(decisions),
        "invalid_actions": invalid,
        "invalid_rate_pct": round(100 * invalid / len(decisions), 1) if decisions else 0.0,
        "mean_latency_ms": round(statistics.mean(lat), 1) if lat else 0.0,
        "p95_latency_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else 0.0,
        "guidance_replan_time_s": round(statistics.mean(replan_times), 1) if replan_times else None,
        "sim_time_s": snapshot["t"],
    }
