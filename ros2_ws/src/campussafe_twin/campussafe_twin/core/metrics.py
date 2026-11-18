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


def security_metrics(attack, defense: str, gateway, guard, attacker, know, n_verify: int) -> dict:
    """Detection, false alarms and command-channel outcomes for one run."""
    first = gateway.first_alarm
    attacked = attack.name != "none"
    detected = attacked and first is not None and first >= attack.start
    fp = [e for e in gateway.alarms if e.get("false_positive")]
    return {
        "attack": attack.name,
        "defense": defense,
        "detected": int(detected) if attacked else None,
        "detection_delay_s": round(first - attack.start, 1) if detected else None,
        "false_alarm": int(bool(fp) or (first is not None and (not attacked or first < attack.start))),
        "false_quarantines": len(fp),
        "security_alarms": len(gateway.alarms),
        "counts_filtered": len(gateway.events) - len(gateway.alarms),
        "channel_compromised": int(not gateway.channel_trusted),
        "forged_cmds_sent": attacker.forged_sent,
        "forged_cmds_executed": guard.executed_forged,
        "forged_cmds_blocked": guard.blocked,
        "fake_hazards_cleared": len(know.cleared),
        "verify_actions": n_verify,
    }
