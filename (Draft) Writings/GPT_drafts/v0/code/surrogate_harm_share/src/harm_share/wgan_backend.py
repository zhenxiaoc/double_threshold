"""A faithful, self-contained Wasserstein-GAN backend for the harm-share
calibration -- a vendored re-implementation of the conditional WGAN used by
Chen & Ritzwoller (2023, App. D.2/D.3) via the ``wgan`` package of Athey,
Imbens, Metzger & Munro (2021, ``ds-wgan``), https://github.com/gsbDBI/ds-wgan.

Why re-implemented rather than pip-installed.  ``ds-wgan`` is not on PyPI and
pins old torch; this file reproduces the same estimator in modern PyTorch and
adds a NumPy export of the trained generator so data generation runs thread-safe
on the CPU (no CUDA-in-threads) during the Monte-Carlo study.

It matches the ds-wgan design element-for-element (verified against the source):

* **Standardization (DataWrapper).**  Each continuous output and context column
  is standardized ``(x-mean)/std``.  The generator learns the standardized map;
  outputs are de-standardized on the way out.
* **Generator.**  MLP ``(noise, context) -> output``; noise ~ N(0,I) with
  ``noise_dim`` = output dim; **ReLU** hidden units + dropout; the final layer's
  continuous outputs are **clamped to the data's [min,max]** (in standardized
  units) -- ds-wgan's ``_transform``.  This clamp is the key stabilizer: it makes
  the generated support bounded, so training cannot run the mean off to infinity.
* **Critic.**  MLP ``(data, context) -> R``; ReLU + dropout; no normalization
  (required for a correct gradient penalty).
* **Gradient penalty (ONE-SIDED).**  ``relu(||grad||_2 - 1).mean()`` -- the
  one-sided ds-wgan penalty, NOT the two-sided ``(||grad||-1)^2`` of Gulrajani.
  The one-sided form leaves the critic free to be flat where it should be, which
  is what keeps the game from limit-cycling.
* **Training.**  Step-based alternation: a generator step every ``critic_steps``
  steps, critic steps otherwise; ``Adam`` with PyTorch-default betas (0.9,0.999);
  critic loss ``-(E[C(real)]-E[C(fake)]) + gp_factor * GP``; generator loss
  ``-E[C(fake)]``.

Only training uses torch; the exported ``NumpyGenerator`` reproduces the forward
pass exactly (unit-tested to 1e-5) for downstream sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# torch is imported lazily inside training so a pure-inference session (loading
# exported NumPy generators) never needs it.


# --------------------------------------------------------------------------- #
# Specification (defaults = ds-wgan; the values Chen & Ritzwoller set explicitly
# are noted in the field comments)
# --------------------------------------------------------------------------- #
@dataclass
class WGANSpec:
    # architecture (ds-wgan defaults: three 128-wide hidden layers, ReLU)
    hidden: tuple[int, ...] = (128, 128, 128)
    noise_dim: int | None = None          # None -> = output dim (ds-wgan default)
    # optimization (ds-wgan: Adam default betas; CR Table: lr 1e-4 both, gp 20, batch 256)
    critic_lr: float = 1e-4
    generator_lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)   # PyTorch/ds-wgan default
    batch_size: int = 256                 # CR Table
    epochs: int = 3000                    # see wgan_calibration on CR's 30000/5000
    critic_steps: int = 15                # ds-wgan default n_critic
    gp_factor: float = 20.0               # CR Table: "Critic Gradient Penalty" = 20
    critic_dropout: float = 0.1           # CR Table (0 for the X-GAN critic)
    generator_dropout: float = 0.1        # CR Table
    device: str | None = None             # None -> cuda if available else cpu
    seed: int = 0


# --------------------------------------------------------------------------- #
# Standardizer (the ds-wgan DataWrapper, continuous-only)
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, A: np.ndarray) -> "Standardizer":
        A = np.atleast_2d(np.asarray(A, float))
        m = A.mean(axis=0)
        s = A.std(axis=0)
        s = np.where(s < 1e-8, 1.0, s)     # constant columns -> no scaling
        return cls(mean=m, scale=s)

    def fwd(self, A):    # raw -> standardized
        return (np.asarray(A, float) - self.mean) / self.scale

    def inv(self, A):    # standardized -> raw
        return np.asarray(A, float) * self.scale + self.mean


# --------------------------------------------------------------------------- #
# NumPy generator (exported; used for all downstream sampling)
# --------------------------------------------------------------------------- #
@dataclass
class NumpyGenerator:
    """Exact NumPy replay of a trained torch Generator (eval mode: dropout off).

    The generator's raw output is a **continuous block** of `p_cont` values
    followed by **categorical groups** (softmax heads) with cardinalities
    `cat_cards` -- ds-wgan's mixed continuous/categorical DataWrapper.  Continuous
    outputs are clamped to `bounds` (std units) and de-standardized; each
    categorical group's logits are softmaxed and sampled.  When `cat_cards` is
    empty this is a pure continuous generator (the S/Y/legacy path, unchanged).

    `generate()` returns the CONTINUOUS block only (what S|X,W and Y|S,X,W need);
    `sample()` returns the full covariate row [continuous..., category indices...].
    """
    weights: list          # [(W0,b0), ..., (WL,bL)] Linear layers in order
    noise_dim: int
    p_cont: int            # number of continuous outputs (== out.mean size)
    bounds: np.ndarray     # (2, p_cont) [lo; hi] for the continuous block (std units)
    ctx: Standardizer | None
    out: Standardizer      # standardizer for the continuous block
    cat_cards: tuple = ()  # cardinalities of the categorical softmax groups

    @property
    def out_dim(self):
        return self.p_cont + int(sum(self.cat_cards))

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def _raw(self, context, noise):
        noise = np.atleast_2d(np.asarray(noise, float))
        if context is None:
            h = noise
        else:
            c = self.ctx.fwd(np.atleast_2d(np.asarray(context, float)))
            h = np.concatenate([noise, c], axis=1)      # order: (noise, context)
        for (W, b) in self.weights[:-1]:
            h = self._relu(h @ W.T + b)
        Wl, bl = self.weights[-1]
        return h @ Wl.T + bl

    def generate(self, context: np.ndarray | None, noise: np.ndarray) -> np.ndarray:
        raw = self._raw(context, noise)
        cont = np.clip(raw[:, :self.p_cont], self.bounds[0], self.bounds[1])
        return self.out.inv(cont)

    def generate_full(self, context: np.ndarray | None, noise: np.ndarray,
                      cat_uniform: np.ndarray | None = None) -> np.ndarray:
        """Full encoded row [continuous | category indices], deterministic in `noise`.

        Like `generate()` for the continuous block, but also emits the categorical
        groups.  Each group is drawn by inverse-CDF from a supplied uniform column
        (`cat_uniform`, shape (n, n_groups)) -- passing the SAME uniforms to two
        arms keeps common random numbers across W=1/W=0 -- or, if `cat_uniform` is
        None, taken as the argmax (deterministic mode).  When there are no
        categorical groups this equals `generate()`.
        """
        raw = self._raw(context, noise)
        cont = self.out.inv(np.clip(raw[:, :self.p_cont], self.bounds[0], self.bounds[1]))
        if not self.cat_cards:
            return cont
        cols = [cont]
        for g, probs in enumerate(self._cat_probs(raw)):
            if cat_uniform is None:
                idx = probs.argmax(axis=1)
            else:
                u = np.asarray(cat_uniform)[:, g:g + 1]
                idx = (u > probs.cumsum(axis=1)).sum(axis=1)
            cols.append(idx.reshape(-1, 1).astype(float))
        return np.concatenate(cols, axis=1)

    def _cat_probs(self, raw):
        probs, off = [], self.p_cont
        for k in self.cat_cards:
            z = raw[:, off:off + k]; off += k
            e = np.exp(z - z.max(axis=1, keepdims=True))
            probs.append(e / e.sum(axis=1, keepdims=True))
        return probs

    def sample(self, context: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
        if context is not None:
            context = np.atleast_2d(np.asarray(context, float))
            n = context.shape[0]
        z = rng.standard_normal((n, self.noise_dim))
        raw = self._raw(context, z)
        cont = self.out.inv(np.clip(raw[:, :self.p_cont], self.bounds[0], self.bounds[1]))
        if not self.cat_cards:
            return cont
        cols = [cont]
        for probs in self._cat_probs(raw):
            u = rng.random((probs.shape[0], 1))
            idx = (u > probs.cumsum(axis=1)).sum(axis=1)     # inverse-CDF category draw
            cols.append(idx.reshape(-1, 1).astype(float))
        return np.concatenate(cols, axis=1)

    # --- (de)serialization to a flat dict of numpy arrays (npz-friendly) ----- #
    def to_state(self, prefix: str) -> dict:
        st = {
            f"{prefix}/noise_dim": np.array(self.noise_dim),
            f"{prefix}/p_cont": np.array(self.p_cont),
            f"{prefix}/cat_cards": np.array(list(self.cat_cards), dtype=int),
            f"{prefix}/n_layers": np.array(len(self.weights)),
            f"{prefix}/has_ctx": np.array(1 if self.ctx is not None else 0),
            f"{prefix}/bounds": self.bounds,
            f"{prefix}/out_mean": self.out.mean,
            f"{prefix}/out_scale": self.out.scale,
        }
        if self.ctx is not None:
            st[f"{prefix}/ctx_mean"] = self.ctx.mean
            st[f"{prefix}/ctx_scale"] = self.ctx.scale
        for i, (W, b) in enumerate(self.weights):
            st[f"{prefix}/W{i}"] = W
            st[f"{prefix}/b{i}"] = b
        return st

    @classmethod
    def from_state(cls, prefix: str, st: dict) -> "NumpyGenerator":
        n_layers = int(st[f"{prefix}/n_layers"])
        weights = [(st[f"{prefix}/W{i}"], st[f"{prefix}/b{i}"]) for i in range(n_layers)]
        ctx = None
        if int(st[f"{prefix}/has_ctx"]):
            ctx = Standardizer(mean=st[f"{prefix}/ctx_mean"], scale=st[f"{prefix}/ctx_scale"])
        out = Standardizer(mean=st[f"{prefix}/out_mean"], scale=st[f"{prefix}/out_scale"])
        cat_cards = tuple(int(c) for c in st[f"{prefix}/cat_cards"]) if f"{prefix}/cat_cards" in st else ()
        return cls(
            weights=weights, noise_dim=int(st[f"{prefix}/noise_dim"]),
            p_cont=int(st[f"{prefix}/p_cont"]), bounds=st[f"{prefix}/bounds"],
            ctx=ctx, out=out, cat_cards=cat_cards,
        )


# --------------------------------------------------------------------------- #
# Training (torch)
# --------------------------------------------------------------------------- #
def train_conditional_wgan(
    Y: np.ndarray,
    context: np.ndarray | None,
    spec: WGANSpec,
    log_every: int = 0,
    cat: np.ndarray | None = None,
    cat_cards: list | None = None,
) -> tuple[NumpyGenerator, dict]:
    """Train a conditional ds-wgan for  (Y, cat) | context  and return a NumpyGenerator.

    Y         : (n, p_cont) continuous targets.
    context   : (n, q) conditioning variables, or None (unconditional, e.g. GAN1).
    cat       : (n, n_cat) integer-coded categorical targets, or None.
    cat_cards : cardinalities of the categorical columns (softmax groups).

    The critic sees [standardized continuous | one-hot categoricals]; the generator
    emits a clamped continuous block + softmax logits per categorical group
    (ds-wgan's mixed DataWrapper).  Returns (exported NumpyGenerator, diagnostics).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(spec.seed)
    np_rng = np.random.default_rng(spec.seed)

    Y = np.atleast_2d(np.asarray(Y, float))
    n, p_cont = Y.shape
    cat_cards = list(cat_cards) if cat_cards else []
    p_cat = int(sum(cat_cards))
    noise_dim = spec.noise_dim if spec.noise_dim is not None else max(p_cont, 2)

    out_std = Standardizer.fit(Y)
    Ys = out_std.fwd(Y)
    bounds = np.stack([Ys.min(axis=0), Ys.max(axis=0)])   # (2, p_cont) standardized units
    # one-hot the categoricals for the critic's "real" input
    if cat_cards:
        cat = np.atleast_2d(np.asarray(cat, int))
        onehot = np.concatenate(
            [np.eye(k)[cat[:, j]] for j, k in enumerate(cat_cards)], axis=1)
        real_np = np.concatenate([Ys, onehot], axis=1)
    else:
        real_np = Ys
    p_total = p_cont + p_cat

    if context is not None:
        context = np.atleast_2d(np.asarray(context, float))
        ctx_std = Standardizer.fit(context)
        Cs = ctx_std.fwd(context)
        q = Cs.shape[1]
    else:
        ctx_std, Cs, q = None, None, 0

    Rt = torch.tensor(real_np, dtype=torch.float32, device=dev)     # critic "real" rows
    Ct = torch.tensor(Cs, dtype=torch.float32, device=dev) if q else None
    bounds_t = torch.tensor(bounds, dtype=torch.float32, device=dev)
    group_sizes = cat_cards

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            di = [noise_dim + q] + list(spec.hidden)
            do = list(spec.hidden) + [p_total]
            self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(di, do)])
            self.drop = nn.Dropout(spec.generator_dropout)

        def forward(self, ctx_batch, m):
            z = torch.randn(m, noise_dim, device=dev)
            x = z if ctx_batch is None else torch.cat([z, ctx_batch], dim=1)
            for layer in self.layers[:-1]:
                x = self.drop(F.relu(layer(x)))
            x = self.layers[-1](x)
            cont = torch.min(torch.max(x[:, :p_cont], bounds_t[0:1]), bounds_t[1:2])  # clamp
            if not group_sizes:
                return cont
            parts, off = [cont], p_cont
            for k in group_sizes:
                parts.append(F.softmax(x[:, off:off + k], dim=1)); off += k
            return torch.cat(parts, dim=1)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            di = [p_total + q] + list(spec.hidden)
            do = list(spec.hidden) + [1]
            self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(di, do)])
            self.drop = nn.Dropout(spec.critic_dropout)

        def forward(self, x, ctx_batch):
            h = x if ctx_batch is None else torch.cat([x, ctx_batch], dim=1)
            for layer in self.layers[:-1]:
                h = self.drop(F.relu(layer(h)))
            return self.layers[-1](h)

        def gradient_penalty(self, real, fake, ctx_batch):
            a = torch.rand(real.size(0), 1, device=dev)
            inter = (a * real + (1 - a) * fake).requires_grad_(True)
            score = self.forward(inter, ctx_batch)
            grads = torch.autograd.grad(
                score, inter, torch.ones_like(score),
                retain_graph=True, create_graph=True)[0]
            return F.relu(grads.norm(2, dim=1) - 1.0).mean()   # one-sided (ds-wgan)

    gen = Generator().to(dev)
    crit = Critic().to(dev)
    opt_g = torch.optim.Adam(gen.parameters(), lr=spec.generator_lr, betas=spec.betas)
    opt_c = torch.optim.Adam(crit.parameters(), lr=spec.critic_lr, betas=spec.betas)

    bs = min(spec.batch_size, n)
    steps_per_epoch = max(1, n // bs)
    hist = {"critic_loss": [], "w_est": []}
    step = 0
    for epoch in range(spec.epochs):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, n, (bs,), device=dev)
            real = Rt[idx]
            cb = Ct[idx] if Ct is not None else None
            if step % spec.critic_steps == 0:
                fake = gen(cb, bs)
                loss_g = -crit(fake, cb).mean()
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()
            else:
                fake = gen(cb, bs).detach()
                wd = crit(real, cb).mean() - crit(fake, cb).mean()
                loss_c = -wd + spec.gp_factor * crit.gradient_penalty(real, fake, cb)
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()
            step += 1
        if log_every and (epoch % log_every == 0 or epoch == spec.epochs - 1):
            with torch.no_grad():
                m = min(n, 4096)
                sub = torch.randint(0, n, (m,), device=dev)
                cbs = Ct[sub] if Ct is not None else None
                w = (crit(Rt[sub], cbs).mean() - crit(gen(cbs, m), cbs).mean()).item()
            hist["w_est"].append(float(w))
            print(f"      epoch {epoch:5d}/{spec.epochs}  W~{w:+.4f}")

    # ---- export generator to NumPy (eval mode: dropout is identity) ----
    gen.eval()
    weights = []
    for layer in gen.layers:
        weights.append((layer.weight.detach().cpu().numpy().astype(np.float64),
                        layer.bias.detach().cpu().numpy().astype(np.float64)))
    ng = NumpyGenerator(
        weights=weights, noise_dim=noise_dim, p_cont=p_cont, bounds=bounds,
        ctx=ctx_std, out=out_std, cat_cards=tuple(cat_cards),
    )

    # ---- consistency check torch-vs-numpy on a small batch ----
    with torch.no_grad():
        m = min(64, n)
        z = np_rng.standard_normal((m, noise_dim))
        if q:
            csub = Cs[:m]
            x = torch.tensor(np.concatenate([z, csub], axis=1), dtype=torch.float32, device=dev)
            raw_ctx = ctx_std.inv(csub)
        else:
            x = torch.tensor(z, dtype=torch.float32, device=dev)
            raw_ctx = None
        for layer in gen.layers[:-1]:
            x = F.relu(layer(x))
        x = gen.layers[-1](x)
        cont = torch.min(torch.max(x[:, :p_cont], bounds_t[0:1]), bounds_t[1:2])
        t_cont = out_std.inv(cont.cpu().numpy())
        n_cont = ng.generate(raw_ctx, z)                    # continuous block only
        max_err = float(np.max(np.abs(t_cont - n_cont)))
        if group_sizes:                                     # also check the softmax probs
            raw_np = ng._raw(raw_ctx, z)
            for pt, pn in zip(
                    [F.softmax(x[:, p_cont:][:, sum(group_sizes[:j]):sum(group_sizes[:j+1])], dim=1).cpu().numpy()
                     for j in range(len(group_sizes))],
                    ng._cat_probs(raw_np)):
                max_err = max(max_err, float(np.max(np.abs(pt - pn))))
    hist["numpy_torch_max_err"] = max_err
    return ng, hist
