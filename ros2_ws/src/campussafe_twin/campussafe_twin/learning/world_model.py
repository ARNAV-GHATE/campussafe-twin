"""Dreamer-style world model of the emergency (a Recurrent State-Space Model).

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])          deterministic memory
    prior      p(z_t | h_t)                         what the model expects next
    posterior  q(z_t | h_t, o_t)                    what it believes after looking
    decoder    p(o_t | h_t, z_t)                    hazard kind per node, people per node

z_t is a 32-dim diagonal Gaussian with reparameterized sampling (PlaNet /
DreamerV1). The DreamerV3 categorical latent (32 x 16, straight-through) is
available with latent="categorical"; on this small dataset it did not learn to
pass the hazard map through the bottleneck within a CPU-sized training budget
(see docs/RESULTS.md), so the Gaussian latent is the default. Training follows
DreamerV3: reconstruction + KL balancing (dyn 0.5 / rep 0.1) with free nats.

Observations may be partial (sensor failure, robot-only view); the encoder
receives the observation and its mask, so the same model fuses whatever the
robot actually knows into a belief about the whole building.

Uses at run time (WorldModelRuntime):
  * belief filtering    observe()   - state estimate of the whole building
  * forecasting         predict() / imagine() - hazard spread, people movement
  * plan evaluation     imagine() with a candidate action sequence (ncm_policy.py)
  * attack detection    predict() vs incoming sensor data (wm_monitor.py)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import HAZ_KINDS, Layout, onehot

K = len(HAZ_KINDS)
# fires (~7 % of node-steps) and blocked exits (~1 %) are rare but safety-critical
HAZARD_CLASS_WEIGHT = torch.tensor([1.0, 3.0, 1.5, 6.0])


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class RSSM(nn.Module):
    def __init__(self, n_nodes: int, act_dim: int, deter=256, stoch=32, classes=16, hidden=256,
                 latent: str = "gaussian"):
        super().__init__()
        self.N, self.act_dim = n_nodes, act_dim
        self.deter, self.stoch, self.classes, self.latent = deter, stoch, classes, latent
        self.gauss = latent == "gaussian"
        self.zdim = stoch if self.gauss else stoch * classes
        stat_dim = 2 * stoch if self.gauss else self.zdim
        obs_in = n_nodes * K + n_nodes + 2 * n_nodes
        self.enc = mlp(obs_in, hidden, hidden)
        self.img_in = nn.Sequential(nn.Linear(self.zdim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior_net = mlp(deter, hidden, stat_dim, 1)
        self.post_net = mlp(deter + hidden, hidden, stat_dim, 1)
        self.dec = mlp(deter + self.zdim, hidden, n_nodes * K + 2 * n_nodes)

    # ---- latent helpers -----------------------------------------------------------
    def _logits(self, raw):
        """Distribution statistics: [mean | std] (Gaussian) or log-probs [.., stoch, classes]."""
        if self.gauss:
            mean, std = raw.chunk(2, -1)
            return torch.cat([mean, F.softplus(std) + 0.1], -1)
        probs = torch.softmax(raw.view(*raw.shape[:-1], self.stoch, self.classes), -1)
        probs = 0.99 * probs + 0.01 / self.classes
        return torch.log(probs)

    def _sample(self, logits, mode=False):
        if self.gauss:
            mean, std = logits.chunk(2, -1)
            return mean if mode else mean + std * torch.randn_like(std)
        probs = logits.exp()
        if mode:
            idx = probs.argmax(-1)
        else:
            idx = torch.distributions.Categorical(probs=probs).sample()
        oh = F.one_hot(idx, self.classes).float()
        return (oh + probs - probs.detach()).flatten(-2)

    def initial(self, B, device="cpu"):
        return torch.zeros(B, self.deter, device=device), torch.zeros(B, self.zdim, device=device)

    def encode(self, obs_in):
        return self.enc(obs_in)

    def step(self, h, z, a, emb=None, mode=False):
        """One transition. With `emb` (encoded observation) returns posterior, else prior only."""
        h = self.gru(self.img_in(torch.cat([z, a], -1)), h)
        prior = self._logits(self.prior_net(h))
        if emb is None:
            return h, self._sample(prior, mode), prior, None
        post = self._logits(self.post_net(torch.cat([h, emb], -1)))
        return h, self._sample(post, mode), prior, post

    def decode(self, h, z):
        out = self.dec(torch.cat([h, z], -1))
        N = self.N
        haz = out[..., :N * K].view(*out.shape[:-1], N, K)
        mu = out[..., N * K:N * K + N]
        logstd = out[..., N * K + N:].clamp(-1.6, 1)
        return haz, mu, logstd


def obs_input(haz_idx, haz_mask, cnt, cnt_mask):
    """Numpy/torch-agnostic encoder input: masked one-hot hazards, masked counts, both masks."""
    if isinstance(haz_idx, np.ndarray):
        oh = onehot(haz_idx, K) * haz_mask[..., None]
        return np.concatenate([oh.reshape(*oh.shape[:-2], -1), cnt * cnt_mask, haz_mask, cnt_mask], -1).astype(np.float32)
    oh = F.one_hot(haz_idx, K).float() * haz_mask[..., None]
    return torch.cat([oh.flatten(-2), cnt * cnt_mask, haz_mask, cnt_mask], -1)


def kl_div(p, q, gauss: bool):
    """KL(p || q) summed over latent dimensions."""
    if gauss:
        mp, sp = p.chunk(2, -1)
        mq, sq = q.chunk(2, -1)
        return (torch.log(sq / sp) + (sp ** 2 + (mp - mq) ** 2) / (2 * sq ** 2) - 0.5).sum(-1)
    return (p.exp() * (p - q)).sum(-1).sum(-1)


# ======================================================================================
# training
# ======================================================================================

def _sequences(episodes, L: Layout, T: int, rng: np.random.Generator, B: int):
    """Random [B, T] training windows with mixed observation masks."""
    batch = {k: [] for k in ("hin", "hm", "cin", "cm", "act", "haz", "cnt", "valid")}
    for _ in range(B):
        ep = episodes[rng.integers(len(episodes))]
        n = len(ep["t"])
        s = rng.integers(0, max(1, n - T + 1))
        idx = np.arange(s, s + T)
        valid = idx < n
        idx = np.minimum(idx, n - 1)
        mode = rng.random()
        haz, cnt = ep["haz"][idx], ep["cnt"][idx]
        if mode < 0.4:                                   # full, true sensor data
            hin, cin = haz, cnt
            hm, cm = np.ones_like(cnt), np.ones_like(cnt)
        elif mode < 0.8:                                 # what the robot actually knew
            hin, hm, cin, cm = ep["khaz"][idx], ep["khm"][idx], ep["kcnt"][idx], ep["kcm"][idx]
        else:                                            # onboard sensors only
            locs = [L.robot_nodes[i] for i in ep["loc"][idx]]
            masks = [L.local_mask(n_) for n_ in locs]
            hm = np.stack([m[0] for m in masks])
            cm = np.stack([m[1] for m in masks])
            hin, cin = haz, cnt
        hm, cm = hm.copy(), cm.copy()
        hm[0], cm[0] = 1, 1                              # the alarm-time snapshot is always complete
        hin = hin.copy()
        hin[0], cin = haz[0], cin.copy()
        cin[0] = cnt[0]
        # action that leads INTO step k: robot node at k, guidance chosen at k-1, time of k-1
        act = np.zeros((T, L.act_dim), np.float32)
        for k in range(1, T):
            j, jp = idx[k], idx[k - 1]
            act[k, ep["loc"][j]] = 1
            act[k, L.R + ep["guid"][jp]] = 1
            act[k, -1] = ep["t"][jp] / 300.0
        for key, v in (("hin", hin), ("hm", hm), ("cin", cin), ("cm", cm), ("act", act),
                       ("haz", haz), ("cnt", cnt), ("valid", valid.astype(np.float32))):
            batch[key].append(v)
    return {k: torch.as_tensor(np.stack(v)) for k, v in batch.items()}


def wm_loss(model: RSSM, b: dict, free_nats=1.0, kl_scale=0.05, count_weight=0.2, pred_weight=0.0):
    B, T = b["valid"].shape
    obs_in = obs_input(b["hin"].long(), b["hm"], b["cin"], b["cm"])
    emb = model.encode(obs_in)
    h, z = model.initial(B)
    stats = {"rec": 0.0, "pred": 0.0, "kl": 0.0, "haz_acc": 0.0}
    loss = 0.0
    for t in range(T):
        h, z, prior, post = model.step(h, z, b["act"][:, t], emb[:, t])
        v = b["valid"][:, t]
        tgt_h, tgt_c = b["haz"][:, t].long(), b["cnt"][:, t]

        def nll(hh, zz):
            haz, mu, ls = model.decode(hh, zz)
            ce = F.cross_entropy(haz.reshape(-1, K), tgt_h.reshape(-1), weight=HAZARD_CLASS_WEIGHT,
                                 reduction="none").view(B, -1).sum(-1)
            g = (0.5 * ((tgt_c - mu) / ls.exp()) ** 2 + ls).sum(-1)
            return ce + count_weight * g, haz

        rec, haz = nll(h, z)
        zp = model._sample(prior)
        pred, _ = nll(h, zp) if t > 0 else (torch.zeros_like(rec), None)
        dyn = kl_div(post.detach(), prior, model.gauss).clamp(min=free_nats)
        rep = kl_div(post, prior.detach(), model.gauss).clamp(min=free_nats)
        step = rec + pred_weight * pred + kl_scale * (0.5 * dyn + 0.1 * rep)
        loss = loss + (step * v).sum() / v.sum().clamp(min=1)
        stats["rec"] += (rec * v).sum().item() / v.sum().clamp(min=1).item()
        stats["pred"] += (pred * v).sum().item() / v.sum().clamp(min=1).item()
        stats["kl"] += (dyn * v).sum().item() / v.sum().clamp(min=1).item()
        stats["haz_acc"] += ((haz.argmax(-1) == tgt_h).float().mean(-1) * v).sum().item() / v.sum().clamp(min=1).item()
    return loss / T, {k: v / T for k, v in stats.items()}


def train_world_model(episodes, L: Layout, steps=4000, B=64, T=16, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_val = max(1, len(episodes) // 10)
    val, tr = episodes[:n_val], episodes[n_val:]
    model = RSSM(L.N, L.act_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for it in range(1, steps + 1):
        model.train()
        loss, st = wm_loss(model, _sequences(tr, L, T, rng, B))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        sched.step()
        if it % 250 == 0 or it == steps:
            model.eval()
            with torch.no_grad():
                _, vs = wm_loss(model, _sequences(val, L, T, np.random.default_rng(1), 128))
            log(f"  wm step {it:5d}  train rec {st['rec']:.2f} pred {st['pred']:.2f} kl {st['kl']:.2f} | "
                f"val rec {vs['rec']:.2f} pred {vs['pred']:.2f} hazard-acc {vs['haz_acc']:.4f}  ({time.time() - t0:.0f}s)")
    return model


# ======================================================================================
# run time
# ======================================================================================

class WorldModelRuntime:
    """Inference wrapper. Beliefs are (h, z) tensors; inputs/outputs are numpy."""

    def __init__(self, path: str | None = None, model: RSSM | None = None, layout: Layout | None = None):
        from ..core.building import lamar_cs_floor
        self.L = layout or Layout(lamar_cs_floor())
        if model is None:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            model = RSSM(self.L.N, self.L.act_dim, **ck["config"])
            model.load_state_dict(ck["state"])
        self.m = model.eval()

    @staticmethod
    def save(model: RSSM, path: str):
        torch.save({"config": {"deter": model.deter, "stoch": model.stoch, "classes": model.classes,
                               "latent": model.latent},
                    "state": model.state_dict()}, path)

    @torch.no_grad()
    def init_belief(self, obs):
        h, z = self.m.initial(1)
        return self.observe((h, z), np.zeros(self.L.act_dim, np.float32), obs)

    @torch.no_grad()
    def observe(self, belief, act_vec, obs):
        """Posterior update. obs = (haz_idx, haz_mask, counts, count_mask)."""
        h, z = belief
        emb = self.m.encode(torch.as_tensor(obs_input(*obs))[None])
        h, z, _, _ = self.m.step(h, z, torch.as_tensor(act_vec)[None], emb, mode=True)
        return h, z

    @torch.no_grad()
    def decode(self, belief):
        haz, mu, ls = self.m.decode(*belief)
        return torch.softmax(haz, -1)[0].numpy(), mu[0].numpy(), ls.exp()[0].numpy()

    @torch.no_grad()
    def predict(self, belief, act_vec, samples: int = 16):
        """Distribution of the NEXT observation: hazard probs [N,K], count mean [N], count std [N]."""
        h, z = belief
        h = h.expand(samples, -1)
        z = z.expand(samples, -1)
        a = torch.as_tensor(act_vec)[None].expand(samples, -1)
        h2, z2, _, _ = self.m.step(h, z, a)
        haz, mu, ls = self.m.decode(h2, z2)
        p = torch.softmax(haz, -1).mean(0)
        m = mu.mean(0)
        var = (ls.exp() ** 2 + mu ** 2).mean(0) - m ** 2
        return p.numpy(), m.numpy(), var.clamp(min=1e-6).sqrt().numpy()

    @torch.no_grad()
    def imagine(self, belief, act_seqs: np.ndarray, samples: int = 4):
        """Roll the prior forward. act_seqs [C, H, act_dim] -> hazard probs [C,H,N,K], counts [C,H,N]."""
        C, H, _ = act_seqs.shape
        h, z = belief
        h = h.expand(C * samples, -1).contiguous()
        z = z.expand(C * samples, -1).contiguous()
        a = torch.as_tensor(act_seqs).repeat_interleave(samples, 0)
        hp, cn = [], []
        for t in range(H):
            h, z, _, _ = self.m.step(h, z, a[:, t])
            haz, mu, _ = self.m.decode(h, z)
            hp.append(torch.softmax(haz, -1))
            cn.append(mu.clamp(min=0))
        hp = torch.stack(hp, 1).view(C, samples, H, self.L.N, K).mean(1)
        cn = torch.stack(cn, 1).view(C, samples, H, self.L.N).mean(1)
        return hp.numpy(), cn.numpy()

    def features(self, belief) -> np.ndarray:
        h, z = belief
        return torch.cat([h, z], -1)[0].numpy()

    @property
    def feat_dim(self) -> int:
        return self.m.deter + self.m.zdim


def hazard_risk(hprobs: np.ndarray) -> np.ndarray:
    """P(node is impassable) from [.., N, K] hazard probabilities."""
    return 1.0 - hprobs[..., 0]

