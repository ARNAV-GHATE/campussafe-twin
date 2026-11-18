"""The `ncm` and `wm_ncm` decision policies: world-model belief + Next-Concept Modeling planner.

Every decision (5 s):
  1. World model: update the belief about the whole building from what the
     robot knows now (posterior update), then forecast hazards 30 s ahead.
  2. NCM: generate plans coarse-to-fine (4 passes) conditioned on the robot's
     knowledge + world-model belief + forecast.
       ncm     one greedy plan (the proposed method)
       wm_ncm  1 greedy + 7 sampled candidate plans, each rolled out in the
               world model's imagination and scored by predicted people still
               inside and predicted people standing on hazardous nodes. A
               sampled plan replaces the greedy one only if its imagined score
               is better by a margin. ABLATION: in our experiments the selector
               exploited model errors and stranded people (docs/RESULTS.md)
  3. The first decision of the chosen plan is proposed; the safety validator
     still checks it, exactly as for the SLM and rule policies.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from ..core.actions import Decision
from ..core.building import lamar_cs_floor
from ..core.validator import validate
from . import MODEL_DIR
from .features import Layout, advance_robot, onehot
from .ncm import NextConceptGenerator, PlanTokenizer
from .world_model import WorldModelRuntime, hazard_risk

FORECAST_STEPS = 6          # 30 s at 5 s per step
ROBOT_STEP_M = 5.0          # robot walks 1 m/s for one 5 s decision period


def condition_vector(L: Layout, obs, reported, unverified, robot_node, guidance, t, online, wm_feat, risk) -> np.ndarray:
    haz, hm, cnt, cm = obs
    guid = np.zeros(L.E + 1, np.float32)
    guid[L.exits.index(guidance) if guidance in L.exits else L.E] = 1
    loc = np.zeros(L.R, np.float32)
    loc[L.ridx[robot_node]] = 1
    return np.concatenate([
        (onehot(haz, 4) * hm[:, None]).ravel(), hm, cnt * cm, cm, reported, unverified,
        loc, guid, [t / 300.0, float(online)], wm_feat, risk]).astype(np.float32)


def state_condition(L: Layout, state: dict, wm: WorldModelRuntime, belief):
    obs = L.state_obs(state)
    reported = np.zeros(L.N, np.float32)
    unverified = np.zeros(L.N, np.float32)
    for h in state["known_hazards"]:
        reported[L.idx[h["location"]]] = float(h["reported"])
        unverified[L.idx[h["location"]]] = float(not h.get("verified", True))
    a = L.action_vec(state["robot_location"], state["guiding_to_exit"], state["time_s"])
    hp, _ = wm.imagine(belief, np.repeat(a[None, None], FORECAST_STEPS, 1), samples=4)
    risk = hazard_risk(hp[0]).max(0)
    return condition_vector(L, obs, reported, unverified, state["robot_location"], state["guiding_to_exit"],
                            state["time_s"], state["building_sensors"] == "online", wm.features(belief), risk), obs


class NCMPolicy:
    def __init__(self, model_dir: str = MODEL_DIR, candidates: int = 8, select: bool = True,
                 temperature: float = 1.0, top_k: int = 4, seed: int = 0, exposure_weight: float = 5.0,
                 margin: float = 0.05):
        self.b = lamar_cs_floor()
        self.L = Layout(self.b)
        self.wm = WorldModelRuntime(os.path.join(model_dir, "world_model.pt"), layout=self.L)
        ck = torch.load(os.path.join(model_dir, "ncm.pt"), map_location="cpu", weights_only=True)
        self.tok = PlanTokenizer(self.L.V, **ck["tok_config"])
        self.tok.load_state_dict(ck["tok"])
        self.gen = NextConceptGenerator(ck["cond_dim"], **ck["gen_config"])
        self.gen.load_state_dict(ck["gen"])
        self.tok.eval()
        self.gen.eval()
        self.select = select
        self.C = candidates if select else 1
        self.temperature, self.top_k, self.w_exp, self.margin = temperature, top_k, exposure_weight, margin
        self.overrides = 0
        self.name = "wm_ncm" if select else "ncm"
        self.seed = seed
        self.reset()

    def reset(self):
        torch.manual_seed(self.seed)
        self.belief = None
        self.prev_t = 0.0
        self.last_plan: list[str] = []
        self.last_concept: list[int] = []
        self.overrides = 0

    def decide(self, state: dict) -> Decision:
        t0 = time.perf_counter()
        L = self.L
        obs = L.state_obs(state)
        if self.belief is None:
            self.belief = self.wm.init_belief(obs)
        else:
            act = L.action_vec(state["robot_location"], state["guiding_to_exit"], self.prev_t)
            self.belief = self.wm.observe(self.belief, act, obs)
        self.prev_t = state["time_s"]

        cond, _ = state_condition(L, state, self.wm, self.belief)
        with torch.no_grad():
            plans, codes = self.gen.generate(torch.as_tensor(cond)[None].repeat(self.C, 1), self.tok,
                                             temperature=self.temperature if self.C > 1 else 0, top_k=self.top_k)
        plans = plans.numpy()
        best = 0
        if self.C > 1:
            scores = self.score(state, plans)
            cand = int(np.argmax(scores))
            if scores[0] <= -1e8 or scores[cand] > scores[0] + self.margin * abs(scores[0]) + 1e-6:
                best = cand
                self.overrides += int(cand != 0)
        action = L.token_to_action(plans[best, 0])
        self.last_plan = [L.vocab[t] for t in plans[best]]
        self.last_concept = [int(c[best].flatten()[0]) for c in codes[:1]]
        raw = json.dumps({"plan": self.last_plan, "concept_code": self.last_concept[0]})
        return Decision(action, (time.perf_counter() - t0) * 1000, raw)

    def score(self, state: dict, plans: np.ndarray) -> np.ndarray:
        """Imagined outcome of each candidate plan (higher is better)."""
        L, b = self.L, self.b
        avoid = {h["location"] for h in state["known_hazards"]}
        C, H = plans.shape
        acts = np.zeros((C, H, L.act_dim), np.float32)
        valid = np.ones(C, bool)
        for c in range(C):
            node, guid, target = state["robot_location"], state["guiding_to_exit"], None
            for k, tok in enumerate(plans[c]):
                a = L.token_to_action(tok)
                if k == 0 and not validate(a, state, b)[0]:
                    valid[c] = False
                if a.action == "navigate_to":
                    target = a.target
                elif a.action == "guide_to_exit" and a.target not in avoid:
                    guid = a.target
                node = advance_robot(b, node, target, avoid, ROBOT_STEP_M)
                acts[c, k] = L.action_vec(node, guid, state["time_s"] + 5 * k)
        hp, cnt = self.wm.imagine(self.belief, acts, samples=4)
        risk = hazard_risk(hp)                                   # [C, H, N]
        inside = cnt.sum(-1).sum(-1)                             # people-steps still inside (x10)
        exposed = (cnt * risk).sum(-1).sum(-1)
        s = -(inside + self.w_exp * exposed)
        s[~valid] = -1e9
        return s
