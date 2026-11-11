# World Models + Next-Concept Modeling for Secure Emergency Guidance

**Extension of CampusSafe Twin** · Arnav Ghate · Lamar University · November 2026

This document describes how **World Models** (Ha & Schmidhuber 2018; DreamerV3, Hafner et al. 2023/2025) and **Next-Concept Modeling** (Kumbong et al. 2026, arXiv:2610.14162) are incorporated into the CampusSafe Twin robot, and how the robot is evaluated under **normal operation and cyberattacks**. Measured results are in [`docs/RESULTS.md`](RESULTS.md).

## 1. Robotic task

**Task: secure autonomous evacuation guidance.** A mobile (humanoid-form) robot in a simulated campus building must, during a fire/smoke emergency, report hazards, move to where people are, and guide them to a *safe* exit, while its data channels may be under attack.

The task has two coupled sub-problems, and each new technique addresses one:

| Sub-problem | Technique | What it does |
|---|---|---|
| *Understanding and predicting the building* (where is the danger now and in 30 s; what is happening where the robot cannot see; is this sensor data physically plausible?) | **World model** (Dreamer-style RSSM) | belief filtering, hazard forecasting, imagination-based plan evaluation, cyberattack detection |
| *Deciding what to do* (a sequence of actions, not one) | **Next-Concept Modeling** | generates an 8-step plan coarse-to-fine in 4 parallel passes |

Both run in the same loop as the existing SLM and rule policies, behind the same safety validator, so all policies are directly comparable.

```
 Gazebo / offline twin ──truth──▶ building sensors ──signed feed──▶ [ATTACKER: network / compromised sensor]
                                                                            │
                                    onboard sensors (6–8 m, trusted) ──┐    ▼
                                                                       ├─ DEFENSE GATEWAY
                                                                       │   1 HMAC + sequence + freshness
                                                                       │   2 onboard cross-check
                                                                       │   3 world-model plausibility (CUSUM)
                                                                       ▼
                                                              robot knowledge ──▶ state
                                                                                    │
            ┌────────────────────────── policy: rule | slm | ncm ◀────────┘
            │   ncm: world-model belief + 30 s forecast → NCM generates the plan coarse-to-fine
            ▼
       safety validator ──signed command──▶ [ATTACKER: forged commands] ──▶ command guard ──▶ robot
```

## 2. World model

**Architecture** (`learning/world_model.py`): Recurrent State-Space Model (PlaNet / Dreamer family).

- deterministic state `h_t` (GRU, 256 units); stochastic state `z_t` = 32-dim diagonal Gaussian, reparameterized sampling
- encoder input: hazard kind per node (one-hot, 19 nodes × 4 kinds) + people per node, **with observation masks**, so the same model handles full sensor data, sensor failure and robot-only views
- action input: robot node, guidance exit, time
- decoder: hazard-kind distribution and people count (Gaussian) per node
- loss: class-weighted hazard cross-entropy + count NLL + DreamerV3 **KL balancing** (0.5 dynamics / 0.1 representation, 1 free nat), scaled by 0.05

**Design decisions found during development** (worth reporting in the thesis):

1. *DreamerV3's categorical latent did not work at this scale.* With 16×8 or 32×16 straight-through categoricals, the model plateaued at ~88 % per-node hazard accuracy even with the full map visible, never predicting blocked exits. A single-step autoencoder with the same layers reached 99 % in 300 steps, so the network could represent the map; the bottleneck was optimization through the latent.
2. *KL weights must be rescaled for small observations.* DreamerV3's weights assume image reconstruction losses of thousands of nats; ours are ~20 nats per step, so the default KL pressure starved the latent. Scaling KL by 0.05 fixed this.
3. *Rare classes need weighting.* Fires are ~7 % and blocked exits ~1 % of node-steps but are the safety-critical ones (weights 3 and 6).
4. A "prior-decoding" loss (decoding from prior samples) was tried and removed: it made the shared decoder hedge without improving one-step predictions.

**Training data** (`learning/data.py`): randomized emergencies (`core/scenarios.py: random_scenario`): random fire origins, stochastic smoke spread along the corridor graph every 15 s, secondary fires, blocked exits, sensor failures, random occupancy. **The four named scenarios are never used for training; they are the test set.**

**Four uses at run time**

1. *Belief filtering*: fuses whatever the robot knows (possibly partial) into a belief about the whole building.
2. *Forecasting*: imagining forward 6 steps (30 s) gives P(node becomes impassable); this "risk map" conditions the planner.
3. *Plan evaluation in imagination* (ablation `wm_ncm`): candidate plans are rolled out in latent space and scored by predicted people still inside plus predicted people on hazardous nodes. This turned out to be exploitable (see results).
4. *Cyberattack detection*: see §4.

## 3. Next-Concept Modeling planner

NCM generates text as a hierarchy of discrete codes, coarse to fine, predicting all codes of a level in parallel. We apply the same machinery to **robot plans** (`learning/ncm.py`).

**Plans as sequences.** One decision = one token from a 40-token vocabulary (`wait`, `announce`, `guide_to_exit:X`, `navigate_to:X`, `report_hazard:X`). A plan is 8 decisions (40 s).

**Stage 1: multiscale residual VQ-VAE tokenizer.** A transformer encoder maps the plan to `f ∈ R^{8×64}`. For levels k = 1..4 with lengths (1, 2, 4, 8): pool the residual `f − f̂` to length s_k, quantize with codebook k (64 codes), upsample, refine with a small convolution φ_k, and add to `f̂`. A transformer decoder maps `f̂` back to tokens. The level-1 code is the plan's **concept**: a single discrete symbol summarizing the whole 40-s strategy.

**Stage 2: next-concept generator.** A transformer with **block-causal attention** predicts the codes of level k from the condition and all coarser levels (inputs = cumulative reconstruction pooled to the next scale, as in VAR next-scale prediction). All codes within a level are predicted in parallel, so a plan takes **4 forward passes** instead of 8 autoregressive steps.

**Condition.** Robot knowledge (hazards, counts, masks, reported/unverified flags, robot node, guidance, time, sensor status) + world-model state `[h, z]` + 30-s hazard forecast.

**Teacher.** Plans are imitated from a *privileged* expert: the rule policy given true hazards, the hazards 30 s in the future, and true counts. The student only sees what the robot sees plus the world model's forecast, so the world model is what lets it approach the expert's foresight.

**Execution** (`learning/ncm_policy.py`): receding horizon. Only the first decision is proposed; the robot re-plans every 5 s.

- `ncm` (**the proposed method**): one greedy plan
- `wm_ncm` (ablation): 1 greedy + 7 sampled plans (top-k 4), scored in world-model imagination; a sampled plan replaces the greedy one only if better by 5 %. In the experiments this selector exploited the world model's errors and left people behind, so the world model is used for belief, forecasting and attack detection, not as an optimization target (see [RESULTS.md](RESULTS.md#ablation-planning-in-imagination-wm_ncm)).

## 4. Cyberattacks and defenses

### Threat model

| Surface | Attacker capability | Attacks (`core/attacks.py`) |
|---|---|---|
| **Compromised sensor / BMS** | controls a device, **holds its signing key** | `false_hazard` (fake fire at the busiest safe exit), `hazard_suppression` (hide all new hazards, stealthy), `occupancy_spoof` (decoy crowd far away, empty elsewhere) |
| **Network** | on the network, no key | `replay` (freeze the feed), `dos` (drop 95 % of messages) |
| **Command channel** | on the network, no key | `command_injection` (forged *guide to worst exit* / *go to far room*, every 15 s) |
| Combined | | `coordinated` (fake exit fire + suppression + command injection) |

All attacks start 10 s after the alarm. The robot's own onboard sensors (6 m hazard range, 8 m people range) are trusted.

### Defenses (`core/defense.py`, `core/security.py`, `learning/wm_monitor.py`)

| Mode | Layer | Stops |
|---|---|---|
| `crypto` | HMAC-SHA256 + monotonic sequence + timestamp on sensor feed and command channel; stale feed → onboard-only mode | forging, replay, command injection, detects DoS |
| `wm` | onboard cross-check + **world-model plausibility** with CUSUM | validly-signed false data from compromised sensors |
| `full` | both | everything above |

**World-model plausibility monitor.** Each reading is compared with the model's one-step prediction:

- *new hazard* with P(impassable) < τ → **quarantined** as *unverified*. It is still avoided (fail-safe), and the robot is dispatched to a point within its own sensor range; if nothing is there, the claim is cleared as fake.
- *vanished hazard* → strong evidence (hazards persist).
- *people count* with |z| > z_th → that count is ignored.
- evidence accumulates in a CUSUM `S_t = max(0, S_{t−1} + e_t − drift)`; `S_t > H` marks the whole feed compromised and the robot switches to onboard sensing + world-model belief, exactly like a sensor failure.

τ, z_th and H are **calibrated on attack-free randomized runs** (`learning/models/monitor_calibration.json`) to keep false alarms rare. The asymmetric design (quarantine still avoids the hazard) means a false positive costs only a verification detour, never safety.

**Why both layers.** Cryptography cannot detect lies told by a device that holds a valid key; a world model cannot detect a forged motor command. The experiments show each layer covering the other's blind spot.

## 5. Metrics

| Aspect | Metric |
|---|---|
| Performance | % evacuated, mean / total evacuation time, people exposed to fire or smoke |
| Reliability | invalid-action rate (validator rejections), decision latency |
| Resilience | resilience ratio R = (evac %<sub>attack</sub> / evac %<sub>normal</sub>) × (time<sub>normal</sub> / time<sub>attack</sub>); guidance replanning time |
| Security | detection rate, detection delay, false-alarm rate, forged commands executed / blocked, fake hazards cleared |

## 6. Reproducing

```bash
pip install -r requirements.txt                    # PyTorch CPU is enough
python run_offline.py --policy rule ncm --attack none false_hazard --defense none full
python run_experiments.py                          # full thesis grid -> results/experiments/report.md
python -m campussafe_twin.learning.train           # retrain (from ros2_ws/src/campussafe_twin); pre-trained models are included
```

## 7. Limitations

- The world model learns this simulator's simplified physics (graph-based smoke spread), not real fire dynamics.
- The NCM planner imitates a rule-based teacher; it cannot exceed that teacher's strategy except through foresight and plan selection.
- Attacks are scripted, not adaptive; an attacker who knows the world model could craft physically plausible lies (e.g. slow, adjacent "smoke spread"). Onboard verification bounds, but does not eliminate, this risk.
- HMAC with a shared key stands in for SROS2 / DDS-Security; key management is out of scope.
- Simulation results do not certify real-world safety.

## References

- D. Ha, J. Schmidhuber. *World Models.* arXiv:1803.10122, 2018.
- D. Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104; *Nature* 2025.
- H. Kumbong, I. S. Khare et al. *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation.* arXiv:2610.14162, 2026.
- K. Tian et al. *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction* (VAR). NeurIPS 2024.
- E. S. Page. *Continuous Inspection Schemes* (CUSUM). Biometrika, 1954.
