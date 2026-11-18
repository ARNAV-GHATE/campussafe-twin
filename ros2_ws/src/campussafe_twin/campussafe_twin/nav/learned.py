"""Task 2 learned components (PyTorch): navigation world model, its attack monitor, NCM planners.

World model (NavRSSM, Dreamer / PlaNet family)
    observation o_t (40-D): LiDAR / 8 m (32) | odometry increment in the previous robot frame,
                            scaled by one step at full speed (3) | wheel encoders (2) | hazard (3)
    action      a_t (2-D) : the controller's OWN command (v / 1 m/s, w / 1.5 rad/s)
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]);  prior p(z_t | h_t);  posterior q(z_t | h_t, o_t)
    decoder p(o_t | h_t, z_t): Gaussian with a learned per-dimension scale
    Uses: (1) predict the next reading on every channel -> attack detection (NavMonitor)
          (2) imagine a candidate motion plan -> safety veto for the NCM planner (ncm_wm)
          (3) its belief [h, z] conditions the NCM planner (ncm_wm)

NCM motion planner (reuses learning/ncm.py)
    A plan is 16 motion tokens at 5 Hz (3.2 s); tokens are (v, w) pairs from controllers.py.
    Multiscale residual VQ tokenizer with scales (1, 2, 4, 8, 16) and a next-concept generator:
    a plan is generated coarse-to-fine in 5 parallel passes. The level-1 code is the plan's
    "concept" (e.g. keep straight down the corridor / swing left into the doorway).
    Executed receding-horizon: the first token is applied and the robot re-plans every 0.2 s.

    ncm     : condition = 44-D observation (controllers.make_obs)
    ncm_wm  : condition = observation + world-model belief, plus an imagination safety veto:
              the greedy plan is rolled out in the world model; if it is predicted to come within
              VETO_CLEAR of a wall, 6 sampled plans are imagined and the one with the largest
              predicted clearance is used. The world model only VETOES (it never optimizes
              progress), the lesson from Task 1's wm_ncm ablation (model exploitation).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..learning.ncm import NextConceptGenerator, PlanTokenizer
from .controllers import N_TOKENS, STOP_TOKEN, from_token, make_obs
from .sim import DT, LIDAR_MAX, ROBOT_R, V_MAX, W_MAX, wrap

MODEL_DIR = os.environ.get("CAMPUSSAFE_NAV_MODEL_DIR",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
WM_OBS = 40
WM_ACT = 2
PLAN_L = 16
SCALES = (1, 2, 4, 8, 16)
CONTROL_PERIOD = 2                       # sim steps per motion token (0.2 s)
CHANNELS = {"scan": slice(0, 32), "odom": slice(32, 35), "cmd": slice(35, 37)}
VETO_CLEAR = ROBOT_R + 0.08              # m, predicted LiDAR range that triggers the veto
ENSEMBLE = 4                             # plans averaged per control step (temporal ensembling)
ENS_DECAY = 0.5                          # weight exp(-decay * plan age)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# ====================================================================== observation encoding
def odom_increment(prev: np.ndarray | None, cur: np.ndarray | None) -> np.ndarray | None:
    if prev is None or cur is None:
        return None
    c, s = math.cos(prev[2]), math.sin(prev[2])
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    return np.array([(c * dx + s * dy) / (V_MAX * DT), (-s * dx + c * dy) / (V_MAX * DT),
                     float(wrap(cur[2] - prev[2])) / (W_MAX * DT)])


def wm_obs(scan, d_odom, proprio, hazard) -> tuple[np.ndarray, np.ndarray]:
    """Observation vector and a mask (1 = this dimension was actually received)."""
    o = np.zeros(WM_OBS, np.float32)
    m = np.ones(WM_OBS, np.float32)
    if scan is None:
        m[CHANNELS["scan"]] = 0
    else:
        o[CHANNELS["scan"]] = np.asarray(scan) / LIDAR_MAX
    if d_odom is None:
        m[CHANNELS["odom"]] = 0
    else:
        o[CHANNELS["odom"]] = np.clip(d_odom, -3, 3)
    o[35:37] = np.asarray(proprio) / np.array([V_MAX, W_MAX])
    o[37:40] = hazard
    return o, m


def act_vec(cmd) -> np.ndarray:
    return np.array([cmd[0] / V_MAX, cmd[1] / W_MAX], np.float32)


# ====================================================================== world model
def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
                         nn.Linear(h, o))


class NavRSSM(nn.Module):
    def __init__(self, obs=WM_OBS, act=WM_ACT, deter=128, stoch=16, hidden=160):
        super().__init__()
        self.deter, self.stoch = deter, stoch
        self.enc = _mlp(obs * 2, hidden, hidden)                  # obs + mask
        self.inp = nn.Sequential(nn.Linear(stoch + act, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior = _mlp(deter, hidden, 2 * stoch)
        self.post = _mlp(deter + hidden, hidden, 2 * stoch)
        self.dec = _mlp(deter + stoch, hidden, obs)
        self.log_std = nn.Parameter(torch.full((obs,), -2.5))

    @staticmethod
    def _dist(raw):
        mu, s = raw.chunk(2, -1)
        return mu, F.softplus(s) + 0.05

    def initial(self, B):
        return torch.zeros(B, self.deter), torch.zeros(B, self.stoch)

    def img_step(self, h, z, a, sample=True):
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        mu, sd = self._dist(self.prior(h))
        z = mu + sd * torch.randn_like(sd) if sample else mu
        return h, z, (mu, sd)

    def obs_step(self, h, z, a, o, m, sample=True):
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        pm, ps = self._dist(self.prior(h))
        qm, qs = self._dist(self.post(torch.cat([h, self.enc(torch.cat([o * m, m], -1))], -1)))
        z = qm + qs * torch.randn_like(qs) if sample else qm
        return h, z, (pm, ps), (qm, qs)

    def decode(self, h, z):
        return self.dec(torch.cat([h, z], -1))


def train_nav_wm(O, M, A, steps=3000, B=32, T=32, lr=1e-3, seed=0, log=print):
    """O, M: [E, L, 40] observations / masks; A: [E, L, 2] actions (a[t] = command applied after o[t])."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NavRSSM()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lens = np.array([len(o) for o in O])
    ok = np.where(lens > T + 1)[0]
    for it in range(1, steps + 1):
        idx = rng.choice(ok, B)
        st = [rng.integers(0, lens[i] - T) for i in idx]
        o = torch.as_tensor(np.stack([O[i][s:s + T] for i, s in zip(idx, st)]))
        m = torch.as_tensor(np.stack([M[i][s:s + T] for i, s in zip(idx, st)]))
        a = torch.as_tensor(np.stack([A[i][s:s + T] for i, s in zip(idx, st)]))
        # random channel dropout teaches the model to fill in missing / distrusted channels
        drop = (torch.rand(B, T, 3) < 0.08).float()
        for c, sl in enumerate(CHANNELS.values()):
            if c < 2:
                m[:, :, sl] = m[:, :, sl] * (1 - drop[:, :, c:c + 1])
        h, z = model.initial(B)
        a_prev = torch.zeros(B, WM_ACT)
        nll = kl = 0.0
        std = model.log_std.exp()
        for t in range(T):
            h, z, (pm, ps), (qm, qs) = model.obs_step(h, z, a_prev, o[:, t], m[:, t])
            mu = model.decode(h, z)
            w = torch.ones_like(m[:, t])
            w[:, CHANNELS["scan"]] = m[:, t, CHANNELS["scan"]]
            w[:, CHANNELS["odom"]] = m[:, t, CHANNELS["odom"]]
            nll = nll + (w * (0.5 * ((o[:, t] - mu) / std) ** 2 + model.log_std)).sum(-1).mean()
            klv = (torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(-1)
            kl = kl + torch.clamp(klv, min=1.0).mean()
            a_prev = a[:, t]
        loss = (nll + 0.5 * kl) / T
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == steps:
            log(f"  nav world model step {it:5d}  nll/step {nll.item() / T:8.2f}  kl/step {kl.item() / T:.2f}")
    return model


class NavWMRuntime:
    """Inference wrapper: one belief per robot, deterministic (mean) latent."""

    def __init__(self, path: str | None = None, model: NavRSSM | None = None):
        if model is None:
            model = NavRSSM()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.m = model.eval()
        self.std = self.m.log_std.detach().exp().numpy()

    @torch.no_grad()
    def init(self):
        return self.m.initial(1)

    @torch.no_grad()
    def predict(self, belief, a_prev):
        """Prior one-step prediction of the next observation (before it arrives)."""
        h, z = belief
        h2, z2, _ = self.m.img_step(h, z, torch.as_tensor(a_prev)[None], sample=False)
        return self.m.decode(h2, z2)[0].numpy()

    @torch.no_grad()
    def observe(self, belief, a_prev, o, mask):
        h, z = belief
        h, z, _, _ = self.m.obs_step(h, z, torch.as_tensor(a_prev)[None], torch.as_tensor(o)[None],
                                     torch.as_tensor(mask)[None], sample=False)
        return h, z

    @torch.no_grad()
    def features(self, belief) -> np.ndarray:
        h, z = belief
        return torch.cat([h, z], -1)[0].numpy()

    @torch.no_grad()
    def imagine_min_range(self, belief, plans: np.ndarray, samples: int = 2) -> np.ndarray:
        """plans [P, 16] tokens -> predicted minimum LiDAR range (m) along each plan."""
        P = len(plans)
        acts = np.array([[act_vec(from_token(int(t))) for t in p for _ in range(CONTROL_PERIOD)] for p in plans])
        h, z = belief
        h = h.repeat(P * samples, 1)
        z = z.repeat(P * samples, 1)
        A = torch.as_tensor(acts, dtype=torch.float32).repeat(samples, 1, 1)
        mins = torch.full((P * samples,), LIDAR_MAX)
        for t in range(A.shape[1]):
            h, z, _ = self.m.img_step(h, z, A[:, t], sample=samples > 1)
            scan = self.m.decode(h, z)[:, CHANNELS["scan"]] * LIDAR_MAX
            mins = torch.minimum(mins, scan.min(-1).values)
        return mins.view(samples, P).min(0).values.numpy()


# ====================================================================== attack monitor
@dataclass
class MonitorVerdict:
    new_alarms: list = field(default_factory=list)
    onset: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)


class NavMonitor:
    """Per-channel surprise of each incoming reading under the world model's one-step prediction.

        e_c(t) = mean over the channel's dims of ((o - mu) / sigma)^2
        S_c(t) = max(0, S_c(t-1) + e_c(t) - k_c)   ->  alarm when S_c > H_c   (CUSUM, Page 1954)

    k_c and H_c are calibrated on attack-free runs (models/nav_monitor_calibration.json).
    A flagged channel is replaced by the world model's prediction when updating the belief,
    so the belief keeps tracking the robot from the trusted channels.
    """

    def __init__(self, runtime: NavWMRuntime, calib: dict):
        self.wm, self.calib = runtime, calib
        self.reset()

    def reset(self):
        self.belief = self.wm.init()
        self.prev_pose = None
        self.prev_cmd = (0.0, 0.0)
        self.S = {c: 0.0 for c in CHANNELS}
        self.since = {c: None for c in CHANNELS}
        self.flagged: set = set()
        self.log_e: dict = {c: [] for c in CHANNELS}
        self.started = False

    def update(self, t, odom_pose, scan, proprio, cmd, hazard) -> MonitorVerdict:
        v = MonitorVerdict()
        d_odom = odom_increment(self.prev_pose, odom_pose)
        if odom_pose is not None:
            self.prev_pose = odom_pose
        o, m = wm_obs(scan, d_odom, proprio, hazard)
        a_prev = act_vec(self.prev_cmd)
        if self.started:
            mu = self.wm.predict(self.belief, a_prev)
            z2 = ((o - mu) / self.wm.std) ** 2
            for c, sl in CHANNELS.items():
                if m[sl].min() == 0:
                    continue
                e = float(z2[sl].mean())
                self.log_e[c].append(e)
                if c in self.flagged:
                    continue
                k, H = self.calib[c]["k"], self.calib[c]["H"]
                if self.S[c] == 0.0:
                    self.since[c] = t
                self.S[c] = max(0.0, self.S[c] + e - k)
                v.scores[c] = self.S[c]
                if self.S[c] > H:
                    self.flagged.add(c)
                    v.new_alarms.append(c)
                    v.onset[c] = self.since[c] if self.since[c] is not None else t
            for c in self.flagged:                       # distrusted channels: impute
                if c != "cmd":
                    m[CHANNELS[c]] = 0
        self.belief = self.wm.observe(self.belief, a_prev, o, m)
        self.prev_cmd = cmd
        self.started = True
        return v


def load_calibration(path):
    with open(path) as f:
        return json.load(f)


def monitor_factory(model_dir: str = MODEL_DIR):
    rt = NavWMRuntime(os.path.join(model_dir, "nav_world_model.pt"))
    calib = load_calibration(os.path.join(model_dir, "nav_monitor_calibration.json"))
    return lambda: NavMonitor(rt, calib)


# ====================================================================== NCM policies
class NCMNavPolicy:
    """Receding-horizon NCM motion planner. with_wm=True -> ncm_wm (belief + imagination veto).

    Called every sim step (10 Hz) so the world-model belief sees every reading, exactly as in
    training; a new plan is generated every CONTROL_PERIOD steps (5 Hz) and its first token held.
    """
    period = 1

    def __init__(self, tok: PlanTokenizer, gen: NextConceptGenerator, wm: NavWMRuntime | None,
                 with_wm: bool, n_alt: int = 6, ensemble: int = ENSEMBLE, ens_decay: float = ENS_DECAY):
        self.tok, self.gen, self.wm = tok.eval(), gen.eval(), wm
        self.with_wm = with_wm
        self.name = "ncm_wm" if with_wm else "ncm"
        self.n_alt = n_alt
        self.ensemble, self.ens_decay = ensemble, ens_decay
        self.reset(None)

    def reset(self, route) -> None:
        self.belief = self.wm.init() if self.wm is not None else None
        self.prev_pose = None
        self.prev_cmd = (0.0, 0.0)
        self.k = 0
        self.held = (0.0, 0.0)
        self.vetoes = 0
        self.plans = 0
        self.recent: list = []                            # (control step, plan tokens) of the last plans

    def _ensembled(self, step: int, plan: np.ndarray) -> tuple[float, float]:
        """Temporal ensembling over overlapping plans (action chunking; Zhao et al. 2023, ACT).
        The command for this control step is the weighted mean of what each of the last `ensemble`
        plans predicted for it, weight exp(-decay * age): the newest plan counts most, so the robot
        still reacts within one step, while older plans damp step-to-step flips between turns."""
        self.recent = [(st, p) for st, p in self.recent if step - st < self.ensemble] + [(step, plan)]
        vs, ws, wt = 0.0, 0.0, 0.0
        for st, p in self.recent:
            age = step - st
            weight = math.exp(-self.ens_decay * age)
            v, w = from_token(int(p[age]))
            vs, ws, wt = vs + weight * v, ws + weight * w, wt + weight
        return vs / wt, ws / wt

    def _track_belief(self, ctx):
        """Belief over the (already defended) pose / scan the controller sees."""
        pose = ctx["pose"]
        o, m = wm_obs(ctx["scan"], odom_increment(self.prev_pose, pose), ctx["proprio"], ctx["hazard"])
        self.belief = self.wm.observe(self.belief, act_vec(self.prev_cmd), o, m)
        self.prev_pose = np.array(pose, dtype=float)

    def act(self, ctx) -> tuple[float, float]:
        if self.with_wm:
            self._track_belief(ctx)
        k, self.k = self.k, self.k + 1
        if k % CONTROL_PERIOD:
            self.prev_cmd = self.held
            return self.held
        if ctx["path"].done or ctx["path"].trapped:
            self.held = self.prev_cmd = (0.0, 0.0)
            return self.held
        obs = make_obs(ctx["scan"], ctx["pose"], ctx["path"], ctx["proprio"], ctx["hazard"])
        cond = np.concatenate([obs, self.wm.features(self.belief)]) if self.with_wm else obs
        c = torch.as_tensor(cond, dtype=torch.float32)[None]
        self.plans += 1
        plan, _ = self.gen.generate(c, self.tok, temperature=0)          # greedy: 5 parallel passes
        chosen = plan[0].numpy()
        if self.with_wm and self.wm.imagine_min_range(self.belief, chosen[None])[0] < VETO_CLEAR:
            # vetoed: imagine sampled alternatives (only now, to keep the common case fast)
            self.vetoes += 1
            alts, _ = self.gen.generate(c.repeat(self.n_alt, 1), self.tok, temperature=1.0, top_k=4,
                                        greedy_first=False)
            cands = np.concatenate([chosen[None], alts.numpy()])
            mins = self.wm.imagine_min_range(self.belief, cands)
            best = int(np.argmax(mins))
            if mins[best] >= VETO_CLEAR:
                chosen = cands[best]
            else:
                chosen = np.full(PLAN_L, STOP_TOKEN)
                self.recent = []                         # a safety stop overrides older plans
        self.held = self._ensembled(self.plans, chosen) if self.ensemble > 1 else from_token(int(chosen[0]))
        self.prev_cmd = self.held
        return self.held


def load_ncm(model_dir: str = MODEL_DIR):
    ck = torch.load(os.path.join(model_dir, "nav_ncm.pt"), map_location="cpu", weights_only=True)
    tok = PlanTokenizer(N_TOKENS, L=PLAN_L, scales=SCALES)
    tok.load_state_dict(ck["tokenizer"])
    gens = {}
    for name in ("ncm", "ncm_wm"):
        g = NextConceptGenerator(ck[f"{name}_cond_dim"], scales=SCALES)
        g.load_state_dict(ck[name])
        gens[name] = g
    return tok, gens


class NavModels:
    """Loads everything once; policies and monitors are cheap per-episode objects."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.dir = model_dir
        self.wm = NavWMRuntime(os.path.join(model_dir, "nav_world_model.pt"))
        self.tok, self.gens = load_ncm(model_dir)
        self.calib = load_calibration(os.path.join(model_dir, "nav_monitor_calibration.json"))

    def monitor(self) -> NavMonitor:
        return NavMonitor(self.wm, self.calib)


def make_learned_policy(name: str, models: NavModels | None = None):
    models = models or NavModels()
    if name == "ncm":
        return NCMNavPolicy(models.tok, models.gens["ncm"], None, with_wm=False)
    if name == "ncm_wm":
        return NCMNavPolicy(models.tok, models.gens["ncm_wm"], models.wm, with_wm=True)
    raise ValueError(f"unknown navigation policy '{name}' (classical | ncm | ncm_wm)")
