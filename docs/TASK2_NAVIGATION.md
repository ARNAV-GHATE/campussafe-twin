# Task 2: Hazard-Aware Robot Navigation with a World Model + NCM, under Cyberattack

**Second robotic task of the CampusSafe Twin thesis extension** · Arnav Ghate · Lamar University · November 2026

Dr. Rana asked for Next-Concept Modeling (NCM) and World Models to be built into **one or two robotic
tasks**, evaluated in normal operation and under cyberattack, as an executable project.
[Task 1](NCM_WORLD_MODEL.md) applies them to the robot's **decisions**: whom to help, which exit to
guide people to, using building-sensor data. Task 2 applies them to the robot's **body**:
driving through the building from LiDAR, odometry and wheel encoders, and producing velocity commands.
Here the attacks hit the robot's own `/odom`, `/scan` and `/cmd_vel` channels, which Task 1
does not cover.

| | Task 1: evacuation guidance | Task 2: hazard-aware navigation |
|---|---|---|
| Decision level | high level, every 5 s | low level, 5 Hz motion commands |
| Inputs | building sensor feed (hazards, occupancy) | 32-beam LiDAR, odometry, wheel encoders, thermal |
| Output | `guide_to_exit`, `navigate_to`, … | `(v, ω)` velocity commands |
| World model | belief over the building, 30 s hazard forecast | dynamics + sensor model of the robot |
| NCM plans | 8 decisions (40 s) in 4 parallel passes | 16 motion tokens (3.2 s) in 5 parallel passes |
| Attacks | false / hidden hazards, decoy crowds, replay, forged decisions | spoofed pose, phantom obstacles, forged velocity commands, replay, DoS |
| Code | `core/`, `learning/` | `nav/` |
| Gazebo | `launch/campussafe.launch.py` | `launch/nav_task.launch.py` |

## 1. The task

The robot drives from a start node to a goal: a room, or `EXIT`, meaning the nearest exit that is
safe right now. It plans a route on the building graph, avoiding active hazards, and re-plans
whenever a new hazard appears. Between waypoints it must stay in the corridors, pass 1.6 m
doorways and keep out of fire and smoke.

**Eight held-out test routes** (`nav/routes.py: TEST_ROUTES`) cover straight corridors,
room-to-exit runs, a 42 m crossing of the building, a blocked exit, and three routes where
smoke appears mid-run and forces a re-plan (`smoke_reroute`, `fire_behind`, `west_room`). The
learned models never train on these routes; they train on `random_route()` only.

**Success** means the robot's **true** position ends within 0.75 m of the goal. A robot that
stops because a spoofed `/odom` claims it has arrived has failed.

## 2. Simulator: the offline twin of the Gazebo robot (`nav/sim.py`)

Fast experiments need thousands of episodes, so `nav/sim.py` reproduces the Gazebo setup in numpy:

- the same walls (`core/building.py: wall_segments`, 0.2 m thick) and doorways
- a unicycle with first-order velocity response: 1.0 m/s, 1.5 rad/s, 1.5 m/s² and 4 rad/s²
- a 0.30 m footprint; a move into a wall is cancelled and counts as a collision
- **LiDAR**: 32 beams over 360°, 8 m range, σ = 2 cm. These are identical to the `gpu_lidar` added to `campus_robot` in
  `tools/generate_world.py`, which uses the same beam angles.
- odometry (σ = 4 mm), noisy wheel encoders (dead reckoning drifts), and an onboard thermal
  sensor (nearest fire/smoke within 6 m)
- fire and smoke are 2 m discs; time inside one is **hazard exposure**. Blocked exits are physical
  barriers the LiDAR sees.

## 3. World model (`nav/learned.py: NavRSSM`)

A Recurrent State-Space Model (PlaNet / Dreamer family):

- observation `o_t` (40-D): LiDAR / 8 m, odometry increment in the previous robot frame, wheel encoders, thermal
- action `a_t`: the controller's **own** velocity command
- `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])` (128 units); prior and posterior Gaussians over `z_t` (16-D);
  Gaussian decoder with a learned per-dimension scale; KL with 1 free nat
- trained on ~394k steps from 1,500 random routes, with random channel dropout so the model can
  fill in a channel it is told not to trust

The world model has three uses:

1. **Attack detection** (`NavMonitor`). Every incoming reading is scored against the model's
   one-step prediction, separately for three channels:
   - *scan*: LiDAR
   - *odom*: odometry increment
   - *cmd*: wheel encoders against the robot's own command

   Each channel runs a CUSUM: `S_c = max(0, S_c + e_c − k_c)`, alarm when `S_c > H_c`. The
   thresholds are calibrated on 200 attack-free random routes (`nav/models/nav_monitor_calibration.json`):
   `k_c` is the 95th percentile of the per-step surprise, and `H_c = max(1.5 × 99th percentile of the
   clean CUSUM peaks, 2 k_c)`.

   A first rule used the 99.5th percentile for `k`. That made the odometry slack so large (k = 15)
   that a 0.25 m/s pose drift never built up evidence, and `odom_spoof` went undetected. The rare
   legitimate spikes (U-turns at re-plans, wall contact) now go into `H` rather than `k`. On the
   attacked test routes, the new rule detects `odom_spoof` 8/8 (mean delay 0.9 s). Clean
   calibration runs crossing `H` went from 0 to 0.5 % (one run in 200, on the LiDAR channel).
2. **Belief for the planner**: `[h, z]` conditions the `ncm_wm` planner.
3. **Imagination as a safety veto**: the greedy NCM plan is rolled out 3.2 s in latent space. If
   the predicted LiDAR range drops below 0.38 m, 6 sampled plans are imagined and the one with the
   largest predicted clearance is used. If none is safe, the robot stops. The world model only
   *vetoes*; it never optimizes progress. That was the lesson of Task 1's `wm_ncm` ablation,
   where optimizing against the model exploited its errors.

## 4. NCM motion planner (`nav/learned.py`, reusing `learning/ncm.py`)

- **Vocabulary**: 35 motion tokens, 5 speeds (0 to 1 m/s) × 7 turn rates (−1.5 to 1.5 rad/s).
- **Plan**: 16 tokens at 5 Hz, i.e. 3.2 s of motion.
- **Stage 1, multiscale residual VQ tokenizer**, scales `(1, 2, 4, 8, 16)`. The level-1 code is the
  plan's *concept*, a single symbol for the whole 3.2 s manoeuvre ("straight down the
  corridor", "swing left into the doorway").
- **Stage 2, next-concept generator**: block-causal transformer that predicts each level from the
  coarser ones, with all codes of a level in parallel. A plan takes **5 forward passes**
  regardless of its length.
- **Condition**: a 44-D egocentric observation: LiDAR, next two waypoints, 1 m look-ahead point,
  velocities, thermal, remaining distance. `ncm_wm` adds the world-model belief.
- **Teacher**: the classical controller given the **true** pose (privileged). Each label is the
  16 tokens the teacher produces when it drives on from a copy of the current world.
- **DAgger**: two rounds of 250 episodes. The student drives, and the teacher labels the states
  the student actually reaches, which fixes the "stuck in a doorway" failure of pure imitation.
- **Execution**: receding horizon with **temporal ensembling** (action chunking, as in ACT, Zhao et
  al. 2023). The robot re-plans every 0.2 s. Each command is the weighted mean of what the last 4
  plans predicted for that moment (weight `exp(−0.5·age)`), so the newest plan dominates. Without
  ensembling, the student flipped between "hard left" and "hard right" plans at room doorways and
  dithered in place. With it, `ncm_wm` went from 3/8 to 5/8 held-out routes. A world-model safety
  stop clears the ensemble so that older plans cannot override it.

Policies compared: `classical` (pure pursuit with LiDAR and thermal reflexes, the baseline),
`ncm` (NCM only), and `ncm_wm` (NCM + world model, **the proposed method**).

## 5. Attacks and defenses (`nav/attacks.py`, `nav/defense.py`)

| Attack | Surface | Has the key? | Effect |
|---|---|---|---|
| `odom_spoof` | compromised localization | yes, signs its lies | reported pose drifts sideways at 0.25 m/s; the controller "corrects" into a wall |
| `lidar_phantom` | compromised LiDAR driver | yes | phantom obstacle 0.5 m ahead; the robot stalls |
| `cmd_injection` | network | no | forged `/cmd_vel` (full speed, hard turns) at 10 Hz |
| `sensor_replay` | network | no | `/odom` + `/scan` from 3 s ago replayed in a loop |
| `dos` | network | no | 90 % of `/odom` and `/scan` dropped |
| `coordinated` | all | mixed | odom drift + LiDAR erased + forged commands |

The wheel encoders and thermal sensor are onboard and trusted, like the robot's own sensors in Task 1.

| Defense | Mechanism | Response |
|---|---|---|
| `none` | naive: last message wins | – |
| `crypto` | HMAC-SHA256 + sequence + freshness on `/odom`, `/scan`, `/cmd_vel` | forged / replayed messages dropped; a starved channel → dead reckoning at ≤ 0.5 m/s |
| `wm` | world-model monitor (per-channel CUSUM) | odom flagged → dead reckoning, **rolled back to the CUSUM onset** so the drift accumulated before the alarm is discarded; scan flagged → **scan rendered from the digital-twin map** at the estimated pose; cmd flagged → hard-wired **motor interlock** (safe stop) |
| `full` | both | |

The two layers cover each other's blind spots, as in Task 1:
- **Crypto** cannot see lies from a device that holds a valid key.
- **The world model** cannot reject a forged command (it has no key); it can only detect it and stop the robot.

## 6. Gazebo / ROS 2 integration

`ros2 launch campussafe_twin nav_task.launch.py route:=<route> nav_policy:=<policy> nav_attack:=<attack> nav_defense:=<defense>`

```
Gazebo (DiffDrive + gpu_lidar) --ros_gz_bridge--> /odom_gz, /scan_gz
   nav_attack_injector   sensor drivers (sign each reading) + the attacker (nav/attacks.py) + building hazard feed
        --> /campussafe/nav/odom_rx, scan_rx (signed, possibly tampered), onboard (trusted), hazards
   nav_controller        NavGuard (defense) + WaypointPath + policy (classical | ncm | ncm_wm), 10 Hz
        --> /campussafe/nav/cmd (signed) --> injector (+ forged) --> /campussafe/nav/cmd_rx
   nav_motor_driver      verifies commands (crypto), motor interlock --> /cmd_vel --> Gazebo
   nav_metrics           scores the run from ground truth --> ~/campussafe_results/nav_*.csv
```

The nodes call exactly the same code as the offline experiments. The robot is moved to the
route's start pose before it drives, and every node uses that pose as its odometry origin.

**Verification without ROS.** `test/ros_stub.py` replaces `rclpy` and the message packages with
in-process fakes and plays Gazebo with the offline simulator. `test/test_nav.py` runs the four
real node classes closed-loop on it: message formats, topic wiring, parameters, the defense
chain and the CSV output. This checks the integration logic; a run in real Gazebo is still the
final check (`nav_task.launch.py`, or the Docker image).

## 7. Metrics

| Aspect | Metric |
|---|---|
| Performance | success (true position), time to goal, path length |
| Safety | runs with a collision, collisions, hazard exposure (s), minimum wall clearance |
| Reliability | world-model vetoes, decision latency |
| Resilience | R = (success under attack / normal) × min(1, time normal / time under attack) |
| Security | detection rate, detection delay, false-alarm rate, forged commands executed |

## 8. Reproducing

```bash
pip install -r requirements.txt
python run_nav_experiments.py                 # full grid -> docs/results/nav/report.md (+ csv, figures)
python run_nav_experiments.py --quick         # smoke test
cd ros2_ws/src/campussafe_twin && python3 -m campussafe_twin.nav.train     # retrain (~1 h CPU)
python3 -m pytest ros2_ws/src/campussafe_twin/test                          # all tests
```

## 9. Limitations

- The offline simulator is kinematic: no wheel slip, people or moving obstacles. Gazebo adds real
  contact physics; the offline/Gazebo gap has not been measured yet.
- The NCM planner imitates a classical teacher, so in normal operation it cannot beat the
  teacher; its value is fast parallel multi-step plans that the world model can check.
- **The learned students are not yet as reliable as the teacher.** In normal operation the classical
  controller finishes 8/8 test routes. `ncm` finishes 3/8 and `ncm_wm` 5/8, with no collisions; the
  failures are timeouts after the student gets stuck at a room doorway, an out-of-distribution
  state. Planned next steps: more DAgger rounds focused on doorway starts, a larger generator, and
  training on the GPU workstation instead of CPU.
- **The world-model monitor does not transfer across controllers.** It was trained on and calibrated
  against the classical controller's driving. With the learned controllers it false-alarms in 75–88 %
  of clean runs (mostly the `cmd` channel), and the interlock stops the robot. Next step: fine-tune the
  world model on the students' own rollouts and recalibrate. See [RESULTS.md](RESULTS.md) T2.4.
- The learned policies are evaluated on 1 seed per configuration (the classical baseline on 3),
  because `ncm_wm` costs about 19 s of CPU per episode. The plans are greedy, so seeds only change
  sensor noise.
- The attacks are scripted, not adaptive. An attacker who knows the monitor could drift the pose
  more slowly than the CUSUM's slack `k`; dead-reckoning rollback bounds, but does not remove, that error.
- Dead reckoning drifts. After a long period on encoders alone, the robot can miss a narrow doorway.
- HMAC with a shared key stands in for SROS2 / DDS-Security; key management is out of scope.
