"""ResonanceRelay solution.

Pipeline (fixed plan, no time-based choices):
1. Forward surrogate: geometry (4 elements x (A,B)) -> 128-sample response curve.
   * Verified data symmetry: the response is invariant under the 4 lattice-translation
     permutations of the element axis {id, (01)(23), (02)(13), (03)(12)} (held-out MSE
     for those permutations equals identity; all other permutations are ~30x worse).
     Training uses 4x augmentation with them; prediction averages over them.
   * Ensemble of MLPs with mixed input parameterisations (raw / inverse-size) and
     activations, fixed seeds and epochs.
2. Exact label rule: 19x19 Gauss-Hermite marginalisation of shared gain/offset,
   ideal Bayes decoder, policy accuracy for all 5 candidates (reproduces 100% of
   training labels from the true curves).
3. Decision under surrogate uncertainty: average each member's policy accuracy and
   take argmax (Bayes-optimal w.r.t. the ensemble's predictive distribution).
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
torch.set_num_threads(max(1, torch.get_num_threads()))

KLEIN = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]


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
    """z: (N,4,8) standardized band responses -> (N,5) policy accuracy."""
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


# ---------------- surrogate ----------------
def feat_raw(X):
    A, B = X[:, 0::2], X[:, 1::2]
    return np.concatenate([(A - 40) / 5, (B - 47) / 3], 1)


def feat_inv(X):
    A, B = X[:, 0::2], X[:, 1::2]
    return np.concatenate([(40 / A - 1) * 10, (47 / B - 1) * 10], 1)


def klein_aug(Xd):
    return np.concatenate([Xd.reshape(-1, 4, 2)[:, list(p)].reshape(-1, 8) for p in KLEIN])


def make_mlp(i, o, h, L, act):
    layers, d = [], i
    for _ in range(L):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def fit_predict(Xtr, Ytr, Xte_list, featf, act, seed, epochs=100, h=256, L=4,
                lr=2e-3, wd=1e-4, bs=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xa = klein_aug(Xtr)
    Ya = np.concatenate([Ytr] * 4)
    xt = torch.tensor(featf(Xa), dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ya, dtype=torch.float32, device=DEV)
    ym, ys = yt.mean(0), yt.std(0) + 1e-6
    yn = (yt - ym) / ys
    m = make_mlp(xt.shape[1], yt.shape[1], h, L, act).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(xt)
    steps = epochs * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = ((m(xt[b]) - yn[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for Xte in Xte_list:
            preds = []
            for p in KLEIN:  # symmetry averaging at prediction time
                xp = Xte.reshape(-1, 4, 2)[:, list(p)].reshape(-1, 8)
                xq = torch.tensor(featf(xp), dtype=torch.float32, device=DEV)
                preds.append((m(xq) * ys + ym).cpu().numpy())
            outs.append(np.mean(preds, 0))
    return outs


# Fixed ensemble plan: (feature map, activation, seed)
MEMBERS = []
for s in range(4):
    MEMBERS.append((feat_inv, nn.GELU, 100 + s))
    MEMBERS.append((feat_raw, nn.GELU, 200 + s))
    MEMBERS.append((feat_raw, nn.SiLU, 300 + s))


def main():
    tr = load("train")
    te = load("test")
    Ntr, Nte = len(tr["df"]), len(te["df"])
    Xtr = (tr["G"] * 217).reshape(Ntr * 4, 8)
    Str = tr["S"].reshape(Ntr * 4, 128)
    Xte = (te["G"] * 217).reshape(Nte * 4, 8)
    cal = te["cal"]

    acc_sum = np.zeros((Nte, 5))
    for k, (ff, act, seed) in enumerate(MEMBERS):
        (pred,) = fit_predict(Xtr, Str, [Xte], ff, act, seed)
        z = spec2z(pred, cal).reshape(Nte, 4, 8)
        acc_sum += policy_acc(z, te["df"], te["P"], te["T"])
        print(f"member {k + 1}/{len(MEMBERS)} done", flush=True)

    pred = acc_sum.argmax(1).astype(int)
    sub = pd.DataFrame({"id": te["df"].id.values, "prediction": pred})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)
    print(sub.prediction.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
