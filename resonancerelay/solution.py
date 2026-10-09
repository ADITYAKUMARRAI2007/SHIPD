"""ResonanceRelay solution v2: frontier-transfer operator.

1. Exact label rule: 19x19 Gauss-Hermite marginalisation of the shared gain/offset,
   ideal Bayes decoder, and the accuracy of each of the 5 policies. With the true
   curves this reproduces 100% of the training labels.
2. Verified symmetry: the curve does not change under the 4 lattice-translation
   orderings of the elements {id, (01)(23), (02)(13), (03)(12)}. Training uses all four
   orderings and predictions average over them.
3. Interpolator: an MLP ensemble mapping geometry to the 128-sample curve. It is
   reliable only inside the training region (channel A <= 44).
4. Height operator (the main gain): a 1-D conv net along the spectral axis. It takes
   the interpolated curve at the design with channel A clipped to a level T, plus
   per-element log(A_target/A_clipped), B and A_clipped, and predicts the residual
   to the curve of the unclipped design. It is trained on in-range pairs (random
   T below each design's maximum). At test time T=44, so the interpolator is
   used only inside its domain and the operator carries the frontier step.
   Local frontier check (train on designs with max A <= 40, predict the 44
   frontier): 0.545 for the direct MLP vs 0.688 with this operator.
5. Decision: average each member's policy accuracies and take the argmax.
Fixed plan: fixed seeds, epochs and members, with no time-based choices.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import ndtr

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KLEIN = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]
LEVELS = np.array([30, 32, 34, 36, 38, 40, 42.5, 44.0])
INTERP_SEEDS = (0, 1, 2)
INTERP_EPOCHS = 100
OP_SEEDS = (0, 1, 2)
OP_EPOCHS = 10
OP_NT = 6


def load(split):
    df = pd.read_csv(public_dir / f"{split}.csv")
    d = np.load(public_dir / f"{split}_inputs.npz", allow_pickle=False)
    idx = {k: i for i, k in enumerate(d["keys"])}
    ii = np.array([idx[k] for k in df.tensor_key])
    out = dict(df=df, G=d["geometry"][ii].astype(np.float64), P=d["policy_bands"][ii],
               T=d["second_thresholds"][ii].astype(np.float64),
               cal=d["response_calibration"].astype(np.float64))
    if split == "train":
        s = np.load(public_dir / "train_spectra.npz", allow_pickle=False)
        idx2 = {k: i for i, k in enumerate(s["keys"])}
        out["S"] = s["spectra"][np.array([idx2[k] for k in df.tensor_key])].astype(np.float64)
    return out


# ---------------- label rule ----------------
_t, _w = np.polynomial.hermite.hermgauss(19)
GX = _t * np.sqrt(2.0)
GW = _w / np.sqrt(np.pi)


def spec2z(S, cal):
    b = S.reshape(*S.shape[:-1], 8, 16).mean(-1)
    return (b - cal[:, 0]) / cal[:, 1]


def policy_acc(z, df, P, T):
    N = z.shape[0]
    ar = np.arange(N)
    rb = df.root_band.values
    rt = df.root_threshold.values
    gs = df.gain_sigma.values
    os_ = df.offset_sigma.values
    ns = df.noise_sigma.values[:, None, None, None]
    g = np.exp(gs[:, None] * GX[None] - 0.5 * gs[:, None] ** 2)
    o = os_[:, None] * GX[None]
    W = GW[:, None] * GW[None, :]

    def mean(zz):
        return g[:, :, None, None] * zz[:, None, None, :] + o[:, None, :, None]

    pu0 = ndtr((rt[:, None, None, None] - mean(z[ar, :, rb])) / ns)
    pu = [pu0, 1 - pu0]
    acc = np.zeros((N, 5))
    for p in range(5):
        for u in range(2):
            b = P[:, p, u]
            m = mean(z[ar, :, b])
            c0 = ndtr((T[ar, b, 0][:, None, None, None] - m) / ns)
            c1 = ndtr((T[ar, b, 1][:, None, None, None] - m) / ns)
            for pv in (c0, c1 - c0, 1 - c1):
                j = np.einsum("ngoh,go->nh", pu[u] * pv, W)
                acc[:, p] += 0.25 * j.max(1)
    return acc


# ---------------- helpers ----------------
def kperm(Xd, p):
    return Xd.reshape(-1, 4, 2)[:, list(p)].reshape(-1, 8)


def kaug(Xd, *Ys):
    return (np.concatenate([kperm(Xd, p) for p in KLEIN]),) + tuple(np.concatenate([Y] * 4) for Y in Ys)


def feat(X):
    A, B = X[:, 0::2], X[:, 1::2]
    return np.concatenate([(A - 40) / 5, (B - 47) / 3], 1)


def mlp(i, o, h=256, L=4):
    layers, d = [], i
    for _ in range(L):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def run_batched(fn, x, bs=4096):
    return torch.cat([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


# ---------------- interpolator ----------------
def train_interp(Xtr, Str):
    Xa, Sa = kaug(Xtr, Str)
    nets = []
    for s in INTERP_SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        xt = torch.tensor(feat(Xa), dtype=torch.float32, device=DEV)
        yt = torch.tensor(Sa, dtype=torch.float32, device=DEV)
        ym, ys = yt.mean(0), yt.std(0) + 1e-6
        yn = (yt - ym) / ys
        m = mlp(8, 128).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        n, bs = len(xt), 128
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=INTERP_EPOCHS * ((n + bs - 1) // bs))
        gen = torch.Generator().manual_seed(s)
        for _ in range(INTERP_EPOCHS):
            perm = torch.randperm(n, generator=gen).to(DEV)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                loss = ((m(xt[b]) - yn[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
        m.eval()
        nets.append((m, ym, ys))

    def pred(Xq):
        out = 0
        with torch.no_grad():
            for m, ym, ys in nets:
                for p in KLEIN:
                    xq = torch.tensor(feat(kperm(Xq, p)), dtype=torch.float32, device=DEV)
                    out = out + (run_batched(m, xq) * ys + ym).cpu().numpy()
        return out / (len(nets) * len(KLEIN))
    return pred


# ---------------- height operator ----------------
class Block(nn.Module):
    def __init__(s, c, dil):
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        s.c2 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        s.a = nn.SiLU()

    def forward(s, x):
        return x + s.c2(s.a(s.c1(s.a(x))))


class OpNet(nn.Module):
    def __init__(s, cin, c=64, dils=(1, 2, 4, 8, 16, 1, 2, 4)):
        super().__init__()
        s.inp = nn.Conv1d(cin, c, 1)
        s.blocks = nn.Sequential(*[Block(c, d) for d in dils])
        s.out = nn.Conv1d(c, 1, 1)

    def forward(s, x):
        return s.out(s.blocks(s.inp(x))).squeeze(1)


def op_inputs(Sp, Xp, Xt):
    N = len(Sp)
    Hp, Ht, R = Xp[:, 0::2], Xt[:, 0::2], Xt[:, 1::2]
    C = np.concatenate([np.log(Ht / Hp) * 10, (R - 47) / 3, (Hp - 40) / 5], 1)
    j = np.broadcast_to(np.linspace(-1, 1, 128), (N, 128))
    x = np.concatenate([np.stack([(Sp - 0.5) / 0.25, j], 1), np.repeat(C[:, :, None], 128, 2)], 1)
    return x.astype(np.float32)


def make_pairs(Xd, rng):
    H = Xd[:, 0::2]
    hm = H.max(1)
    Xt_l, Xp_l, idx_l = [], [], []
    for _ in range(OP_NT):
        T = np.array([rng.choice(LEVELS[LEVELS < m]) if (LEVELS < m).any() else m for m in hm])
        Xp = Xd.copy()
        Xp[:, 0::2] = np.minimum(H, T[:, None])
        Xt_l.append(Xd)
        Xp_l.append(Xp)
        idx_l.append(np.arange(len(Xd)))
    return np.concatenate(Xt_l), np.concatenate(Xp_l), np.concatenate(idx_l)


def train_op(Xd, Sd, interp, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xa, Sa = kaug(Xd, Sd)
    Xt, Xp, idx = make_pairs(Xa, rng)
    Sp = interp(Xp)
    xin = torch.tensor(op_inputs(Sp, Xp, Xt), device=DEV)
    y = torch.tensor(((Sa[idx] - Sp) / 0.05).astype(np.float32), device=DEV)
    m = OpNet(xin.shape[1]).to(DEV)
    lr, bs = 2e-3, 64
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    n = len(xin)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=OP_EPOCHS * ((n + bs - 1) // bs))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(OP_EPOCHS):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = ((m(xin[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    m.eval()

    def apply(Xq, clip):
        Xp = Xq.copy()
        Xp[:, 0::2] = np.minimum(Xq[:, 0::2], clip)
        out = 0
        with torch.no_grad():
            for p in KLEIN:
                xq, xp = kperm(Xq, p), kperm(Xp, p)
                sp = interp(xp)
                xi = torch.tensor(op_inputs(sp, xp, xq), device=DEV)
                out = out + sp + 0.05 * run_batched(m, xi, 1024).cpu().numpy()
        return out / len(KLEIN)
    return apply


def main():
    tr = load("train")
    te = load("test")
    Ntr, Nte = len(tr["df"]), len(te["df"])
    Xtr = (tr["G"] * 217).reshape(Ntr * 4, 8)
    Str = tr["S"].reshape(Ntr * 4, 128)
    Xte = (te["G"] * 217).reshape(Nte * 4, 8)
    cal = te["cal"]
    clip = float(Xtr[:, 0::2].max())  # training frontier (44)

    interp = train_interp(Xtr, Str)
    print("interpolator trained", flush=True)
    acc_sum = np.zeros((Nte, 5))
    for s in OP_SEEDS:
        apply = train_op(Xtr, Str, interp, seed=s)
        pred = apply(Xte, clip)
        z = spec2z(pred, cal).reshape(Nte, 4, 8)
        acc_sum += policy_acc(z, te["df"], te["P"], te["T"])
        print(f"operator seed {s} done", flush=True)

    sub = pd.DataFrame({"id": te["df"].id.values, "prediction": acc_sum.argmax(1).astype(int)})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)
    print(sub.prediction.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
