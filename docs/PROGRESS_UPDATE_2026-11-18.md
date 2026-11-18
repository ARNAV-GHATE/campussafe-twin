# Progress update: World Models + NCM in two robotic tasks

**To:** Dr. Masud Rana · **From:** Arnav Ghate · **Date:** November 18, 2026
**Re:** your email of November 11, 2026 ("advising on Arnav's research")

## What you asked for, and where it is

| Request (Nov 11) | Status | Where |
|---|---|---|
| Incorporate **Next Concept Modeling (NCM)** and **World Models** into **one or two robotic tasks** | ✅ **Two tasks** | Task 1: [NCM_WORLD_MODEL.md](NCM_WORLD_MODEL.md) · Task 2: [TASK2_NAVIGATION.md](TASK2_NAVIGATION.md) |
| Use **NVIDIA Isaac Sim or Gazebo** | ✅ Gazebo Harmonic + ROS 2 Jazzy (already the project's simulator) | `launch/campussafe.launch.py` (Task 1), `launch/nav_task.launch.py` (Task 2) |
| Evaluate **normal operation and cybersecurity attacks**: performance, reliability, resilience | ✅ 7 attacks (Task 1) + 6 attacks (Task 2), 4 defense levels, detection / resilience metrics | [RESULTS.md](RESULTS.md), [results/nav/report.md](results/nav/report.md) |
| A **complete, functional implementation** we can open, execute and test | ✅ One repository; pre-trained models included; offline runners + Gazebo launch files; tests | `run_experiments.py`, `run_nav_experiments.py`, `pytest` |
| Source, robot models, environments, configs, dependencies **organized** | ✅ ROS 2 package with robot + world SDF generator, `config/params.yaml`, `requirements.txt`, Docker image | `ros2_ws/src/campussafe_twin`, `docker/` |
| **Documented**, minimal configuration to run | ✅ README quick start, design docs, Docker image (no local ROS install needed) | `README.md`, `docker/run.sh` |

## The two tasks

**Task 1: secure evacuation guidance (decision level).** Pushed on November 11. A Dreamer-style
world model gives the robot a belief and a 30-s hazard forecast of the building. An NCM planner
(multiscale residual VQ + next-scale transformer, after arXiv:2610.14162) generates 8-decision
plans coarse-to-fine in 4 parallel passes. Seven cyberattacks target the building-sensor feed
and the command channel.

**Task 2: hazard-aware navigation (motion level).** New in this update. The same two techniques
now control the robot's body:

- the robot has a **32-beam LiDAR**, added to the Gazebo robot model;
- a **world model of its own dynamics and sensors** (LiDAR, odometry, wheel encoders) predicts
  every reading one step ahead and imagines 3.2 s into the future;
- an **NCM motion planner** generates 16-token velocity plans in 5 parallel passes, and the
  world model **vetoes** any plan it imagines hitting a wall;
- six attacks hit the robot's own `/odom`, `/scan` and `/cmd_vel` channels: pose spoofing,
  phantom obstacles, forged velocity commands, replay, DoS, and a coordinated attack;
- defenses: HMAC message authentication, plus a **world-model monitor** that detects
  validly-signed lies and recovers. It rolls dead reckoning back to when the attack began, renders
  the LiDAR scan from the digital twin, and engages a hard-wired motor interlock.

## Headline results

**Task 1** (unchanged since Nov 11; [RESULTS.md](RESULTS.md)): the learned planner matches the rule
baseline in normal operation (98.3 % vs 98.0 % evacuated). Only the `full` defense covers both the
network attacks and the compromised-sensor attacks (coordinated attack: R 0.71 → 0.86).

**Task 2** (new; 1,120 episodes on 8 held-out routes; [results/nav/report.md](results/nav/report.md)):

| Normal operation | `classical` (baseline) | `ncm` | `ncm_wm` (proposed) |
|---|---:|---:|---:|
| Success / time to goal | 100 % / 30.9 s | 38 % / 29.1 s | 62 % / 64.8 s |
| Collisions | 0 % | 0 % | 0 % |

| Defenses, classical controller | none | crypto | world model | full |
|---|---:|---:|---:|---:|
| Forged `/cmd_vel` | 0 % | 100 % | 0 % (safe stop) | 100 % |
| Phantom LiDAR obstacle (signed) | 25 % | 25 % | 96 % | 96 % |
| Spoofed odometry (signed) | 0 % | 0 % | 29 % (detected 100 %) | 29 % |
| Coordinated attack | 0 % | 0 % | 0 % | 25 % |
| False alarms, clean runs | – | 0 % | 0 % | 0 % |

What this shows, honestly:

- **The world model makes the NCM planner better** (5/8 routes vs 3/8, no collisions), but the learned
  planner does **not yet match the classical controller** it imitates; it gets stuck at room doorways.
- **Cryptography and the world-model monitor are complementary**, the same result as Task 1: crypto
  stops forged and replayed messages, and the world model catches sensors that sign their own lies.
  Only both together get anything through the coordinated attack.
- **Main open problem:** the monitor is calibrated on the classical controller's driving. With the
  learned controllers it raises false alarms in 75–88 % of clean runs, because their abrupt commands
  are out of distribution for the world model. The motor interlock then stops the robot.

## How to run it

```bash
git clone https://github.com/ARNAV-GHATE/campussafe-twin && cd campussafe-twin
git checkout ncm-world-model-cyber
pip install -r requirements.txt                    # numpy, torch (CPU), matplotlib, pytest
python run_experiments.py --quick                  # Task 1, ~1 min
python run_nav_experiments.py --quick              # Task 2, ~5 min
python -m pytest ros2_ws/src/campussafe_twin/test  # 24 tests
```

**Gazebo**, either on Ubuntu 24.04 with ROS 2 Jazzy, or in the Docker image:
`docker build -t campussafe -f docker/Dockerfile .`

```bash
ros2 launch campussafe_twin campussafe.launch.py policy:=ncm attack:=coordinated defense:=full     # Task 1
ros2 launch campussafe_twin nav_task.launch.py route:=smoke_reroute nav_policy:=ncm_wm \
     nav_attack:=odom_spoof nav_defense:=full                                                      # Task 2
```

## What is verified, and what still needs your machine

- **Verified here:** every offline experiment and all tests. That includes the four Task 2 ROS 2
  nodes running closed-loop on a ROS stub (`test/ros_stub.py`), which checks topic wiring,
  message formats, parameters and the defense chain.
- **Not yet run:** a live Gazebo session of Task 2. My development container cannot install ROS 2
  or Gazebo. The world file, LiDAR, bridge and launch file are ready; the first run on a
  Gazebo-capable machine is the remaining check.
- I chose **Gazebo** over Isaac Sim because the project already runs on Gazebo + ROS 2 and works
  without an NVIDIA GPU. The ROS 2 interface (`/odom`, `/scan`, `/cmd_vel`) would carry over to
  Isaac Sim's ROS 2 bridge if we want it later.

## Known issues and next steps

1. **Retrain the Task 2 world model on the learned planner's own rollouts** (the Dreamer recipe) and
   recalibrate the monitor per policy, to remove the 75–88 % false alarms with learned controllers.
2. Improve the NCM student at room doorways: more DAgger rounds from doorway starts, a larger
   generator, training on the lab GPU instead of my laptop CPU.
3. Recovery after a detected odometry spoof: add LiDAR scan matching against the twin map as a second
   position source (detection is 100 %, but dead reckoning alone finishes only 29 % of routes).
4. Run Task 2 in Gazebo across the 8 routes and compare with the offline numbers (sim-to-sim gap).
5. Task 1's world-model monitor raises false alarms on 25 % of attack-free runs of the held-out
   scenarios, against 1.3 % in its own calibration runs. Recalibrate on a broader scenario mix.
6. An adaptive attacker who knows the monitor (e.g. drifting the pose below the CUSUM slack).
7. A humanoid robot model and a real Lamar floor plan (from the original thesis plan).

## Questions for you

1. Is two tasks at these two levels (decision and motion) the scope you had in mind, or would you
   like a third task (e.g. multi-robot coordination)?
2. Should the end-of-semester presentation for Dr. Tcheslavski focus on Task 2's attack/defense
   results, or cover both tasks?
