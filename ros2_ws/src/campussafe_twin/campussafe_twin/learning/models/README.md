# Pre-trained models

Trained on CPU with `python -m campussafe_twin.learning.train` from **randomized** scenarios only. The four named scenarios (`fire_north`, `blocked_exit`, `sensor_loss`, `multi_hazard`) were never seen in training and are the test set.

| File | What | Size |
|---|---|---|
| `world_model.pt` | RSSM world model (GRU 256 + 32-dim Gaussian latent) | 3.9 MB |
| `ncm.pt` | NCM plan tokenizer (multiscale residual VQ, scales 1/2/4/8, 64 codes each) + next-concept generator (4-layer transformer) | 2.6 MB |
| `monitor_calibration.json` | thresholds of the world-model attack monitor (τ, z_th, CUSUM H) | – |
| `training_report.json` | dataset size, held-out evaluation, calibration | – |

## Training data

3,000 randomized emergencies (141,910 decision ticks). Teacher for the planner: the rule policy with privileged information (true hazards now and 30 s ahead, true people counts).

## Held-out quality

| Component | Metric | Value |
|---|---|---|
| World model | per-node hazard accuracy, belief with full sensor data | 99.6 % |
| World model | one-step hazard prediction accuracy (120 held-out episodes) | 99.0 % |
| World model | 30-s forecast of **new** hazards, AUROC (base rate 6.7 %) | 0.865 |
| NCM tokenizer | plan-token reconstruction / whole 8-step plan reconstruction | 99.9 % / 99.1 % |
| NCM generator | code accuracy per level (1 / 2 / 4 / 8 codes), validation | 53 / 60 / 68 / 77 % |
| NCM generator | first decision equal to the privileged teacher's | 67 % |
| Attack monitor | legitimate new hazards quarantined (attack-free) | 2.1 % |
| Attack monitor | attack-free runs raising a channel alarm | 1.3 % |

## Training settings

- World model: 4,000 steps, batch 64 × 16 steps, AdamW 1e-3 (one-cycle), KL balancing 0.5/0.1 × 0.05, 1 free nat, hazard class weights (none 1, fire 3, smoke 1.5, blocked 6), count loss × 0.2. ~7 min on 2 CPU cores.
- NCM tokenizer: 3,000 steps, batch 256, dead-code restarts every 200 steps. Generator: 6,000 steps, batch 256, AdamW 5e-4.
- Monitor: τ = 2nd percentile of P(kind) for legitimate new hazards, z_th = 99.9th percentile of count z-scores, H = 99th percentile of CUSUM peaks + 0.5, over 150 attack-free randomized runs.

Retrain after any change to the floor plan (`core/building.py`), since node order defines the input layout.
