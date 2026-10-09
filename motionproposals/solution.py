"""Federal Council proposals on Swiss parliamentary motions (Project Eris).

Run:  python3 solution.py <public_dir> <submission_out>

Pipeline (everything is trained inside this script, from the provided data only):

1. Per-motion multinomial logit models with *board fixed effects*.
   Each training board gets its own free class intercepts (restricted to the classes
   present in its counts), so the shared text weights only learn what separates the
   proposals *within* a board -- exactly what the metric rewards (the counts are given,
   so era/session/council priors carry no information).  Weights are L2-regularised and
   fitted with L-BFGS on sparse TF-IDF views of the texts (fitted on train texts only).
   Seven views (German/French words and characters, era interactions, a class-balanced
   pooled model) give a diverse family.
2. In-script hyperparameter search: the L2 strength of every view is chosen by grouped
   5-fold cross-validation (folds = boards) on within-board one-vs-rest AUC.
3. Optional transformer branch (only when a GPU, torch and transformers are present and
   the pretrained backbone downloads): a multilingual encoder is fine-tuned with the
   board's log class shares as a logit offset (the same fixed-effect idea), 5 folds,
   with a wall-clock guard.  It is dropped automatically if it fails or runs late.
4. In-script ensemble selection: greedy forward selection (with replacement) of the
   out-of-fold log-probabilities, maximising the within-board AUC.
5. Decoding per board: Sinkhorn scaling of the probabilities to the board's counts
   (this plays the role of the unknown test-board intercepts), then an exact
   assignment (Hungarian) that maximises the expected macro recall
   sum_k 1/n_k * sum_{i->k} q_ik subject to using each code exactly n_k times.
   Whether Sinkhorn is applied is itself chosen on the out-of-fold board scores.
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

T0 = time.time()
TIME_BUDGET = 50 * 60          # stop optional work and write the submission by ~50 min
NA = "(not available)"
NCLS = 5
SEED = 0
N_JOBS = max(1, min(4, os.cpu_count() or 1))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- data
def load(pub):
    tr = pd.read_csv(pub / "train.csv")
    te = pd.read_csv(pub / "test.csv")
    ti = pd.read_csv(pub / "train_items.csv", keep_default_na=False)
    si = pd.read_csv(pub / "test_items.csv", keep_default_na=False)
    tg = pd.read_csv(pub / "train_targets.csv")
    lab = {}
    for bid, props in zip(tg.item_id, tg.proposals):
        for i, y in enumerate(json.loads(props)):
            lab[(bid, i + 1)] = int(y)
    ti["y"] = [lab[(a, b)] for a, b in zip(ti.item_id, ti.item_no)]
    ti = ti.merge(tr[["item_id", "council", "counts"]], on="item_id", how="left")
    si = si.merge(te[["item_id", "council", "counts"]], on="item_id", how="left")
    ti = ti.sort_values(["item_id", "item_no"]).reset_index(drop=True)
    si = si.sort_values(["item_id", "item_no"]).reset_index(drop=True)
    return te, ti, si


def field(d, lang, parts):
    s = d[f"{parts[0]}_{lang}"].astype(str)
    for p in parts[1:]:
        s = s + " \n " + d[f"{p}_{lang}"].astype(str)
    return s.str.replace(NA, " ", regex=False)


def meta(d):
    yr = d.submitted.str[:4].astype(int).values
    cols = [((yr >= lo) & (yr < lo + 3)).astype(float) for lo in range(1995, 2027, 3)]
    cols.append((d.council == "Council of States").values.astype(float))
    return sp.csr_matrix(np.stack(cols, 1))


def era_onehot(d):
    yr = d.submitted.str[:4].astype(int).values
    return [(yr < 2004), (yr >= 2004) & (yr < 2012), (yr >= 2012) & (yr < 2020), yr >= 2020]


def build_views(ti, si):
    """Return {name: (X_train, X_test, fixed_effects, balanced)}; vectorisers fit on train only."""
    def tfidf(kind, lang, parts):
        if kind == "w":
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=150000,
                                sublinear_tf=True)
        else:
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150000,
                                sublinear_tf=True)
        a = v.fit_transform(field(ti, lang, parts))
        return a.tocsr(), v.transform(field(si, lang, parts)).tocsr()

    full, demand = ["title", "text", "reason"], ["title", "text"]
    w_de = tfidf("w", "de", full)
    c_de = tfidf("c", "de", demand)
    w_fr = tfidf("w", "fr", full)
    c_fr = tfidf("c", "fr", demand)
    Mtr, Mte = meta(ti), meta(si)

    def cat(*mats):
        return sp.hstack(mats).tocsr()

    def era(Xtr, Xte):
        etr, ete = era_onehot(ti), era_onehot(si)
        return (cat(Xtr, *[sp.diags(e.astype(float)) @ Xtr for e in etr]),
                cat(Xte, *[sp.diags(e.astype(float)) @ Xte for e in ete]))

    e_tr, e_te = era(*w_de)
    views = {
        "w_de": (cat(w_de[0], Mtr), cat(w_de[1], Mte), True, False),
        "wc_de": (cat(w_de[0], c_de[0], Mtr), cat(w_de[1], c_de[1], Mte), True, False),
        "c_de": (cat(c_de[0], Mtr), cat(c_de[1], Mte), True, False),
        "w_fr": (cat(w_fr[0], Mtr), cat(w_fr[1], Mte), True, False),
        "c_fr": (cat(c_fr[0], Mtr), cat(c_fr[1], Mte), True, False),
        "era_de": (cat(e_tr, Mtr), cat(e_te, Mte), True, False),
        "bal_de": (cat(w_de[0], Mtr * 3), cat(w_de[1], Mte * 3), False, True),
    }
    return views


# --------------------------------------------------------------------------- model
def fit_fe_logit(X, y, groups, w, lam, mask, iters=300):
    """Multinomial logit; free (group x class) intercepts, L2 on the text weights only.
    mask[g, k] False removes class k from group g (absent from the board's counts)."""
    n, D = X.shape
    ug, gi = np.unique(groups, return_inverse=True)
    nb = len(ug)
    Y = np.zeros((n, NCLS))
    Y[np.arange(n), y] = 1
    neg = np.where(mask, 0.0, -1e9)[gi]
    Xt = X.T.tocsr()

    def f(theta):
        W = theta[:D * NCLS].reshape(D, NCLS)
        B = theta[D * NCLS:].reshape(nb, NCLS)
        Z = X @ W + B[gi] + neg
        lse = logsumexp(Z, 1, keepdims=True)
        P = np.exp(Z - lse)
        loss = -(w * (Z - lse)[np.arange(n), y]).sum() + 0.5 * lam * (W ** 2).sum()
        G = (P - Y) * w[:, None]
        gW = Xt @ G + lam * W
        gB = np.zeros((nb, NCLS))
        np.add.at(gB, gi, G)
        return loss, np.concatenate([gW.ravel(), gB.ravel()])

    r = minimize(f, np.zeros(D * NCLS + nb * NCLS), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=iters))
    return r.x[:D * NCLS].reshape(D, NCLS)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def run_fold(Xtr, Xte, y, groups, counts_of, a, b, lam, fe, balanced):
    if fe:
        g = groups[a]
        ug = np.unique(g)
        mask = np.array([[counts_of[x][k] > 0 for k in range(NCLS)] for x in ug])
        w = np.ones(len(a))
    else:
        g = np.zeros(len(a), dtype=object)
        mask = np.ones((1, NCLS), bool)
        cw = len(a) / (NCLS * np.maximum(np.bincount(y[a], minlength=NCLS), 1))
        w = cw[y[a]] if balanced else np.ones(len(a))
        w = w * len(a) / w.sum()
    W = fit_fe_logit(Xtr[a], y[a], g, w, lam, mask)
    return b, softmax(Xtr[b] @ W), softmax(Xte @ W)


# --------------------------------------------------------------------------- decoding / metrics
def sinkhorn(P, counts, it=60):
    Q = np.clip(P, 1e-12, None).copy()
    cnt = np.asarray(counts, float)
    Q[:, cnt == 0] = 0
    m = cnt > 0
    for _ in range(it):
        Q /= Q.sum(1, keepdims=True)
        Q[:, m] *= cnt[m] / Q[:, m].sum(0)
    return Q / Q.sum(1, keepdims=True)


def assign(P, counts):
    """Exact maximiser of the expected macro recall given the counts."""
    cols, wcols = [], []
    for k, c in enumerate(counts):
        for _ in range(int(c)):
            cols.append(k)
            wcols.append(P[:, k] / c)
    W = np.stack(wcols, 1)
    r, c = linear_sum_assignment(-W)
    out = np.empty(len(P), int)
    out[r] = np.array(cols)[c]
    return out


def board_score(pred, y, counts):
    present = [k for k in range(NCLS) if counts[k] > 0]
    K = len(present)
    if K < 2:
        return None
    rec = np.mean([np.mean(pred[y == k] == k) for k in present])
    return (rec - 1 / K) / (1 - 1 / K)


class Evaluator:
    def __init__(self, ti):
        self.boards = []
        for _, g in ti.groupby("item_id", sort=False):
            cnt = json.loads(g.counts.iloc[0])
            if sum(c > 0 for c in cnt) >= 2:
                self.boards.append((g.index.values, cnt, g.y.values))

    def auc(self, P):
        res = []
        for idx, cnt, y in self.boards:
            Q = sinkhorn(P[idx], cnt)
            a = []
            for k in range(NCLS):
                pos = y == k
                npos = pos.sum()
                if 0 < npos < len(y):
                    r = rankdata(Q[:, k])
                    a.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))
            res.append(np.mean(a))
        return float(np.mean(res))

    def score(self, P, use_sk=True):
        s = []
        for idx, cnt, y in self.boards:
            Q = sinkhorn(P[idx], cnt) if use_sk else P[idx]
            s.append(board_score(assign(Q, cnt), y, cnt))
        return 100 * float(np.mean(s))


# --------------------------------------------------------------------------- transformer branch
def transformer_branch(ti, si, folds, deadline):
    """Fine-tune a pretrained multilingual encoder (GPU only). Returns (oof, test) or None."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception as e:  # noqa: BLE001
        log("transformer branch skipped (import):", repr(e)[:120])
        return None
    if not torch.cuda.is_available():
        log("transformer branch skipped: no GPU")
        return None
    name = os.environ.get("ERIS_BACKBONE", "xlm-roberta-base")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        AutoModelForSequenceClassification.from_pretrained(name, num_labels=NCLS)
    except Exception as e:  # noqa: BLE001
        log("transformer branch skipped (download):", repr(e)[:120])
        return None

    dev = torch.device("cuda")
    MAXLEN, BS, EPOCHS, LR = 384, 16, 3, 2e-5

    def texts(d):
        return (d.title_de.astype(str) + ". " + d.text_de.astype(str) + " " + d.reason_de.astype(str)
                ).str.replace(NA, " ", regex=False).tolist()

    enc_tr = tok(texts(ti), truncation=True, max_length=MAXLEN)
    enc_te = tok(texts(si), truncation=True, max_length=MAXLEN)

    def offsets(d):
        c = np.array([json.loads(x) for x in d.counts], float)
        return np.log((c + 0.1) / (c.sum(1, keepdims=True) + 0.5)).astype(np.float32)

    off_tr = offsets(ti)
    y = ti.y.values

    def batches(enc, idx, shuffle, rng=None):
        idx = np.array(idx)
        if shuffle:
            idx = idx[rng.permutation(len(idx))]
        else:  # length-sorted for fast inference
            idx = idx[np.argsort([len(enc["input_ids"][i]) for i in idx])]
        for s in range(0, len(idx), BS):
            bi = idx[s:s + BS]
            L = max(len(enc["input_ids"][i]) for i in bi)
            ids = np.full((len(bi), L), tok.pad_token_id, np.int64)
            att = np.zeros((len(bi), L), np.int64)
            for j, i in enumerate(bi):
                t = enc["input_ids"][i]
                ids[j, :len(t)] = t
                att[j, :len(t)] = 1
            yield bi, torch.from_numpy(ids).to(dev), torch.from_numpy(att).to(dev)

    def predict(model, enc, idx):
        model.eval()
        out = np.zeros((len(enc["input_ids"]), NCLS), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for bi, ids, att in batches(enc, idx, False):
                out[bi] = model(input_ids=ids, attention_mask=att).logits.float().cpu().numpy()
        return out

    oof = np.zeros((len(ti), NCLS))
    test = np.zeros((len(si), NCLS))
    t_fold = None
    for f, (a, b) in enumerate(folds):
        t1 = time.time()
        if t_fold is not None and t1 + t_fold > deadline:
            log("transformer branch stopped by time guard")
            return None
        torch.manual_seed(SEED + f)
        rng = np.random.default_rng(SEED + f)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=NCLS).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * int(np.ceil(len(a) / BS))
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda")
        yt = torch.from_numpy(y).to(dev)
        ot = torch.from_numpy(off_tr).to(dev)
        for ep in range(EPOCHS):
            model.train()
            for bi, ids, att in batches(enc_tr, a, True, rng):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ids, attention_mask=att).logits.float()
                bt = torch.from_numpy(bi).to(dev)
                loss = torch.nn.functional.cross_entropy(logits + ot[bt], yt[bt])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
            if time.time() > deadline:
                log("transformer branch stopped by time guard (mid-fold)")
                return None
        oof[b] = softmax(predict(model, enc_tr, b)[b])
        test += softmax(predict(model, enc_te, np.arange(len(si)))) / len(folds)
        del model, opt
        torch.cuda.empty_cache()
        t_fold = time.time() - t1
        log(f"transformer fold {f} done in {t_fold:.0f}s")
    return oof, test


# --------------------------------------------------------------------------- main
def main():
    pub = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    te, ti, si = load(pub)
    log(f"train motions {len(ti)}, test motions {len(si)}, test boards {len(te)}")
    y = ti.y.values
    groups = ti.item_id.values
    counts_of = {b: json.loads(c) for b, c in zip(ti.item_id, ti.counts)}
    folds = list(GroupKFold(5, shuffle=True, random_state=SEED).split(ti, groups=groups))
    ev = Evaluator(ti)

    views = build_views(ti, si)
    log("views built:", {k: v[0].shape[1] for k, v in views.items()})

    # ---- in-script HPO of the L2 strength per view (grouped CV on within-board AUC)
    lam_grid = [0.5, 1.0, 2.0]
    jobs = [(name, lam, fi) for name in views for lam in lam_grid for fi in range(len(folds))]
    res = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_fold)(views[n][0], views[n][1], y, groups, counts_of, folds[fi][0], folds[fi][1],
                          lam, views[n][2], views[n][3]) for n, lam, fi in jobs)
    cand = {}
    for (name, lam, fi), (b, pb, pt) in zip(jobs, res):
        o, t = cand.setdefault((name, lam), (np.zeros((len(ti), NCLS)), np.zeros((len(si), NCLS))))
        o[b] = pb
        t += pt / len(folds)
    members = {}
    for name in views:
        aucs = {lam: ev.auc(cand[(name, lam)][0]) for lam in lam_grid}
        best = max(aucs, key=aucs.get)
        members[name] = cand[(name, best)]
        log(f"view {name:7s} AUC by lambda {({k: round(100 * v, 2) for k, v in aucs.items()})} -> {best}"
            f" | board score {ev.score(members[name][0]):.2f}")

    # ---- optional fine-tuned transformer
    tb = transformer_branch(ti, si, folds, T0 + TIME_BUDGET)
    if tb is not None:
        members["transformer"] = tb
        log(f"view transformer AUC {100 * ev.auc(tb[0]):.2f} | board score {ev.score(tb[0]):.2f}")

    # ---- in-script greedy ensemble selection (with replacement) on OOF log-probs
    names = list(members)
    L_oof = {k: np.log(np.clip(members[k][0], 1e-9, 1)) for k in names}
    L_te = {k: np.log(np.clip(members[k][1], 1e-9, 1)) for k in names}
    # Greedy runs a fixed number of steps (no early stop, so it can pass through a
    # locally worse blend) and keeps the best prefix; the uniform blend of all views
    # is a competing candidate.
    chosen, cur, path = [], None, []
    for step in range(12):
        trial = {}
        for k in names:
            Z = L_oof[k] if cur is None else (cur * len(chosen) + L_oof[k]) / (len(chosen) + 1)
            trial[k] = ev.auc(softmax(Z))
        k = max(trial, key=trial.get)
        cur = L_oof[k] if cur is None else (cur * len(chosen) + L_oof[k]) / (len(chosen) + 1)
        chosen.append(k)
        path.append((trial[k], list(chosen)))
        log(f"ensemble step {step}: +{k} -> AUC {100 * trial[k]:.2f}")
    best_auc, best_set = max(path, key=lambda t: t[0])
    weights = {k: best_set.count(k) / len(best_set) for k in set(best_set)}
    uni_auc = ev.auc(softmax(sum(L_oof[k] for k in names) / len(names)))
    log(f"greedy best AUC {100 * best_auc:.2f} vs uniform blend AUC {100 * uni_auc:.2f}")
    if uni_auc > best_auc:
        weights = {k: 1 / len(names) for k in names}
    P_oof = softmax(sum(w * L_oof[k] for k, w in weights.items()))
    P_te = softmax(sum(w * L_te[k] for k, w in weights.items()))
    s_sk, s_raw = ev.score(P_oof, True), ev.score(P_oof, False)
    use_sk = s_sk >= s_raw
    log(f"ensemble weights {weights}")
    log(f"OOF board score: sinkhorn {s_sk:.2f} | raw {s_raw:.2f} -> use_sinkhorn={use_sk}")

    # ---- decode test boards
    si = si.reset_index(drop=True)
    rows = []
    for bid, cstr in zip(te.item_id, te.counts):
        cnt = json.loads(cstr)
        g = si[si.item_id == bid].sort_values("item_no")
        P = P_te[g.index.values]
        Q = sinkhorn(P, cnt) if use_sk else P
        pred = assign(Q, cnt)
        assert len(pred) == sum(cnt) and all(np.bincount(pred, minlength=NCLS) == np.array(cnt))
        rows.append((bid, json.dumps([int(v) for v in pred], separators=(",", ":"))))
    sub = pd.DataFrame(rows, columns=["item_id", "proposals"])
    assert len(sub) == len(te) and sub.item_id.is_unique
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(sub)} boards)")


if __name__ == "__main__":
    main()
