"""Next-Concept Modeling (NCM) for robot plans.

Follows Kumbong et al., "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine
Text Generation" (arXiv:2610.14162), with robot action plans in place of text:

Stage 1 - plan tokenizer (multiscale residual VQ-VAE)
    A plan is L=8 decisions (40 s at one decision per 5 s), each one token
    from the action vocabulary (wait, announce, guide_to_exit:X,
    navigate_to:X, report_hazard:X). A transformer encoder maps the plan to
    a continuous sequence f [L, d]. Residual quantization then explains f
    with K=4 levels of discrete codes at increasing lengths (1, 2, 4, 8):
        level 1 (1 code)   the plan's "concept" (e.g. go east, guide to main exit)
        level 2 (2 codes)  first / second half of the plan
        level 3 (4 codes)  10-second segments
        level 4 (8 codes)  individual decisions
    At each level the residual is pooled to that length, quantized, upsampled
    and added to the cumulative reconstruction (as in VAR next-scale
    prediction). A transformer decoder maps the reconstruction back to tokens.

Stage 2 - next-concept generator
    A transformer predicts the codes level by level, coarse to fine, all
    codes WITHIN a level in parallel (block-causal attention), conditioned on
    the situation: the robot's knowledge plus the world model's belief and
    hazard forecast. Generating a plan takes K=4 forward passes, regardless of
    plan length, instead of L=8 autoregressive steps.

Plans are executed receding-horizon (MPC): only the first decision is used,
and the robot replans every 5 s.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PLAN_LEN = 8
SCALES = (1, 2, 4, 8)


def _tf(d, layers, heads=4, ff=256, drop=0.1):
    layer = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class PlanTokenizer(nn.Module):
    """Multiscale residual VQ-VAE over action-token plans."""

    def __init__(self, vocab: int, d=64, codebook=64, scales=SCALES, L=PLAN_LEN, beta=0.25):
        super().__init__()
        self.vocab, self.d, self.L, self.scales, self.beta = vocab, d, L, scales, beta
        self.n_codes = codebook
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.encoder = _tf(d, 2)
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(codebook, d) * 0.5) for _ in scales])
        # phi_k: small conv after upsampling, mixed 50/50 with the raw upsample (VAR)
        self.phi = nn.ModuleList([nn.Conv1d(d, d, 3, padding=1) for _ in scales])
        self.dec_pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.decoder = _tf(d, 2)
        self.head = nn.Linear(d, vocab)

    def encode(self, tokens):
        return self.encoder(self.emb(tokens) + self.pos)

    def upsample(self, q, k):
        """[B, s_k, d] -> [B, L, d] contribution of level k."""
        up = F.interpolate(q.transpose(1, 2), size=self.L, mode="nearest")
        return (0.5 * up + 0.5 * self.phi[k](up)).transpose(1, 2)

    def pool(self, x, s):
        return F.adaptive_avg_pool1d(x.transpose(1, 2), s).transpose(1, 2)

    def quantize(self, f):
        """Residual quantization. Returns f_hat, per-level indices, VQ loss, and the
        per-level generator inputs (cumulative reconstruction pooled to the next scale)."""
        f_hat = torch.zeros_like(f)
        idxs, vq, next_inputs = [], 0.0, []
        for k, s in enumerate(self.scales):
            r = self.pool(f - f_hat, s)
            cb = self.codebooks[k]
            d2 = (r.pow(2).sum(-1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(-1))
            idx = d2.argmin(-1)
            q = cb[idx]
            vq = vq + F.mse_loss(q, r.detach()) + self.beta * F.mse_loss(r, q.detach())
            q_st = r + (q - r).detach()
            f_hat = f_hat + self.upsample(q_st, k)
            idxs.append(idx)
            if k + 1 < len(self.scales):
                next_inputs.append(self.pool(f_hat, self.scales[k + 1]))
        return f_hat, idxs, vq, next_inputs

    def decode(self, f_hat):
        return self.head(self.decoder(f_hat + self.dec_pos))

    def forward(self, tokens):
        f = self.encode(tokens)
        f_hat, idxs, vq, _ = self.quantize(f)
        logits = self.decode(f_hat)
        return logits, idxs, vq

    @torch.no_grad()
    def codes(self, tokens):
        f_hat, idxs, _, next_inputs = self.quantize(self.encode(tokens))
        return idxs, next_inputs

    def embed_level(self, k, idx, f_hat):
        """Add level k's codes to the reconstruction (used during generation)."""
        return f_hat + self.upsample(self.codebooks[k][idx], k)


class NextConceptGenerator(nn.Module):
    """Predicts the code hierarchy coarse-to-fine; codes inside a level in parallel."""

    def __init__(self, cond_dim: int, d=64, codebook=64, scales=SCALES, layers=4, hidden_cond=256):
        super().__init__()
        self.scales, self.d = scales, d
        self.P = sum(scales)
        self.cond = nn.Sequential(nn.Linear(cond_dim, hidden_cond), nn.LayerNorm(hidden_cond), nn.SiLU(),
                                  nn.Linear(hidden_cond, d))
        self.sos = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.in_proj = nn.Linear(d, d)
        self.level_emb = nn.Embedding(len(scales), d)
        self.pos_emb = nn.Parameter(torch.randn(1, self.P, d) * 0.02)
        self.body = _tf(d, layers, ff=4 * d)
        self.heads = nn.ModuleList([nn.Linear(d, codebook) for _ in scales])
        level_of = torch.cat([torch.full((s,), k) for k, s in enumerate(scales)])
        self.register_buffer("level_of", level_of, persistent=False)
        # block-causal mask over [cond | level positions]: a level sees the condition,
        # all coarser levels and itself (VAR), never finer levels.
        lv = torch.cat([torch.tensor([-1]), level_of])
        self.register_buffer("mask", lv[None, :] > lv[:, None], persistent=False)

    def _inputs(self, level_inputs):
        """level_inputs: list of [B, s_k, d] for k >= 1 (cumulative f_hat pooled to scale k)."""
        B = level_inputs[0].shape[0] if level_inputs else 1
        parts = [self.sos.expand(B, -1, -1)] + [self.in_proj(x) for x in level_inputs]
        return torch.cat(parts, 1)

    def forward(self, cond, level_inputs):
        B = cond.shape[0]
        x = self._inputs(level_inputs)
        P = x.shape[1]
        x = x + self.level_emb(self.level_of[:P])[None] + self.pos_emb[:, :P]
        c = self.cond(cond)[:, None]
        seq = torch.cat([c, x + c], 1)
        out = self.body(seq, mask=self.mask[:P + 1, :P + 1])[:, 1:]
        logits, start = [], 0
        for k, s in enumerate(self.scales):
            if start >= P:
                break
            logits.append(self.heads[k](out[:, start:start + s]))
            start += s
        assert B == out.shape[0]
        return logits

    @torch.no_grad()
    def generate(self, cond, tokenizer: PlanTokenizer, temperature=1.0, top_k=8, greedy_first=True):
        """K forward passes -> plan tokens [B, L]. Row 0 is greedy when greedy_first."""
        B = cond.shape[0]
        f_hat = torch.zeros(B, tokenizer.L, tokenizer.d)
        level_inputs, all_idx = [], []
        for k, s in enumerate(self.scales):
            logits = self.forward(cond, level_inputs)[k]                     # [B, s, codes]
            if temperature <= 0:
                idx = logits.argmax(-1)
            else:
                lg = logits / temperature
                if top_k:
                    kth = lg.topk(top_k, -1).values[..., -1:]
                    lg = lg.masked_fill(lg < kth, -1e9)
                idx = torch.distributions.Categorical(logits=lg).sample()
                if greedy_first:
                    idx[0] = logits[0].argmax(-1)
            all_idx.append(idx)
            f_hat = tokenizer.embed_level(k, idx, f_hat)
            if k + 1 < len(self.scales):
                level_inputs.append(tokenizer.pool(f_hat, self.scales[k + 1]))
        return tokenizer.decode(f_hat).argmax(-1), all_idx


# ======================================================================================
# training
# ======================================================================================

def train_tokenizer(plans: np.ndarray, vocab: int, steps=3000, B=256, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = PlanTokenizer(vocab)
    opt = torch.optim.AdamW(tok.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X = torch.as_tensor(plans)
    t0 = time.time()
    for it in range(1, steps + 1):
        tok.train()
        xb = X[rng.integers(len(X), size=B)]
        logits, idxs, vq = tok(xb)
        ce = F.cross_entropy(logits.reshape(-1, vocab), xb.reshape(-1))
        loss = ce + vq
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 200 == 0:                                   # restart dead codes from current residuals
            _revive_codes(tok, X[rng.integers(len(X), size=2048)])
        if it % 500 == 0 or it == steps:
            acc = tokenizer_accuracy(tok, X[:4000])
            log(f"  tokenizer step {it:5d}  ce {ce.item():.3f} vq {vq.item():.3f}  "
                f"plan-token acc {acc['token']:.4f}  whole-plan acc {acc['plan']:.4f}  ({time.time() - t0:.0f}s)")
    return tok


@torch.no_grad()
def _revive_codes(tok: PlanTokenizer, xb):
    tok.eval()
    f = tok.encode(xb)
    f_hat = torch.zeros_like(f)
    for k, s in enumerate(tok.scales):
        r = tok.pool(f - f_hat, s).reshape(-1, tok.d)
        cb = tok.codebooks[k]
        d2 = (r.pow(2).sum(-1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(-1))
        idx = d2.argmin(-1)
        used = torch.bincount(idx, minlength=cb.shape[0]) > 0
        dead = (~used).nonzero().flatten()
        if len(dead):
            cb.data[dead] = r[torch.randint(len(r), (len(dead),))] + 0.01 * torch.randn(len(dead), tok.d)
        q = cb[d2.argmin(-1)].view(xb.shape[0], s, tok.d)
        f_hat = f_hat + tok.upsample(q, k)


@torch.no_grad()
def tokenizer_accuracy(tok: PlanTokenizer, X) -> dict:
    tok.eval()
    pred = tok(X)[0].argmax(-1)
    eq = pred == X
    return {"token": eq.float().mean().item(), "plan": eq.all(-1).float().mean().item()}


def train_generator(cond: np.ndarray, plans: np.ndarray, tok: PlanTokenizer, steps=4000, B=256, lr=5e-4,
                    seed=0, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok.eval()
    C = torch.as_tensor(cond, dtype=torch.float32)
    X = torch.as_tensor(plans)
    with torch.no_grad():                                 # precompute targets with the frozen tokenizer
        all_idx, all_in = [], []
        for i in range(0, len(X), 4096):
            idx, nxt = tok.codes(X[i:i + 4096])
            all_idx.append(idx)
            all_in.append(nxt)
        idxs = [torch.cat([a[k] for a in all_idx]) for k in range(len(tok.scales))]
        inputs = [torch.cat([a[k] for a in all_in]) for k in range(len(tok.scales) - 1)]
    n_val = max(256, len(X) // 20)
    perm = torch.as_tensor(rng.permutation(len(X)))
    va, tr = perm[:n_val], perm[n_val:]
    gen = NextConceptGenerator(C.shape[1])
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()

    def batch_loss(ix):
        logits = gen(C[ix], [x[ix] for x in inputs])
        losses = [F.cross_entropy(lg.reshape(-1, lg.shape[-1]), idxs[k][ix].reshape(-1)) for k, lg in enumerate(logits)]
        accs = [(lg.argmax(-1) == idxs[k][ix]).float().mean().item() for k, lg in enumerate(logits)]
        return sum(losses), accs

    for it in range(1, steps + 1):
        gen.train()
        ix = tr[torch.as_tensor(rng.integers(len(tr), size=B))]
        loss, _ = batch_loss(ix)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == steps:
            gen.eval()
            with torch.no_grad():
                vl, accs = batch_loss(va)
                plan, _ = gen.generate(C[va[:1024]], tok, temperature=0)
                first = (plan[:, 0] == X[va[:1024], 0]).float().mean().item()
            log(f"  generator step {it:5d}  train {loss.item():.3f} val {vl.item():.3f}  code acc per level "
                f"{' / '.join(f'{a:.2f}' for a in accs)}  first-action acc {first:.3f}  ({time.time() - t0:.0f}s)")
    return gen
