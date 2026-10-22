#!/usr/bin/env python3
"""Reproducible feature-selection benchmark for IoT IDS (UNSW-NB15, CICIoT2023).

Full run:   python benchmark.py --unsw data/unsw --cic data/ciciot2023 --out results
Smoke test: python benchmark.py --unsw data/unsw --cic data/ciciot2023 --out results_quick --quick
"""
import argparse
import glob
import json
import os
import pickle
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================== data loading
WEB = {"BrowserHijacking", "CommandInjection", "SqlInjection", "XSS",
       "Backdoor_Malware", "Uploading_Attack"}
PREFIX = [("Benign", "Benign"), ("DDoS", "DDoS"), ("DoS", "DoS"), ("Mirai", "Mirai"),
          ("Recon", "Recon"), ("VulnerabilityScan", "Recon"), ("MITM", "Spoofing"),
          ("DNS_Spoofing", "Spoofing"), ("DictionaryBruteForce", "BruteForce")]


def cic_category(lbl):
    if lbl in WEB:
        return "Web"
    for p, c in PREFIX:
        if lbl.startswith(p):
            return c
    raise ValueError(f"Unmapped CICIoT2023 label: {lbl}")


def load_unsw(d):
    """Official UNSW-NB15 train/test partition. Returns X_tr, X_te, cat_tr, cat_te."""
    out = []
    for f in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
        df = pd.read_csv(os.path.join(d, f))
        df = df.drop(columns=[c for c in ("id",) if c in df.columns])
        cat = df["attack_cat"].astype(str).str.strip().replace({"Normal": "Benign"})
        X = df.drop(columns=["label", "attack_cat"])
        out.append((X, cat.reset_index(drop=True)))
    (Xtr, ctr), (Xte, cte) = out
    for c in ("proto", "service", "state"):
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats).codes
        Xte[c] = pd.Categorical(Xte[c], categories=cats).codes
    return Xtr.reset_index(drop=True), Xte.reset_index(drop=True), ctr, cte


def load_cic(d, per_label, cache, benign_cap):
    """Per-label capped sample of CICIoT2023 (benign capped separately), stratified 70/30 split."""
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        from collections import Counter
        files = sorted(f for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
                       if "MERGED" not in f.upper())
        if not files:
            raise FileNotFoundError(f"No CSV files under {d}")
        n_in_folder = Counter(os.path.dirname(f) for f in files)

        def cap(lbl):
            return benign_cap if cic_category(lbl) == "Benign" else per_label

        parts = []
        for i, f in enumerate(files, 1):
            part = pd.read_csv(f, low_memory=False)
            part.columns = [c.strip() for c in part.columns]
            if "Label" in part.columns and "label" not in part.columns:
                part = part.rename(columns={"Label": "label"})
            if "label" not in part.columns:  # per-class folder layout: folder name is the label
                part["label"] = os.path.basename(os.path.dirname(f))
            part = part.sample(frac=1, random_state=0)
            n = n_in_folder[os.path.dirname(f)]
            for lbl, g in part.groupby("label"):
                parts.append(g.head(max(1, int(np.ceil(cap(lbl) * 3 / n)))))
            print(f"  CICIoT2023 file {i}/{len(files)}", end="\r", flush=True)
        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=0)
        df = pd.concat([g.head(cap(lbl)) for lbl, g in df.groupby("label")], ignore_index=True)
        df.to_pickle(cache)
    print("\n  CICIoT2023 records per label:\n" + df["label"].value_counts().to_string())
    cat = df["label"].map(cic_category)
    X = df.drop(columns=["label"])
    Xtr, Xte, ctr, cte = train_test_split(X, cat, test_size=0.3, stratify=cat, random_state=0)
    return (Xtr.reset_index(drop=True), Xte.reset_index(drop=True),
            ctr.reset_index(drop=True), cte.reset_index(drop=True))


def task_labels(cat, task):
    return cat if task == "multi" else cat.where(cat == "Benign", "Attack")


def prepare(Xtr, Xte, ytr, yte):
    """Train-only fitting of imputation, deduplication and scaling (no leakage)."""
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan)
    Xte = Xte.replace([np.inf, -np.inf], np.nan)
    med = Xtr.median(numeric_only=True)
    Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    dup = pd.concat([Xtr, ytr.rename("_y")], axis=1).duplicated().values
    Xtr, ytr = Xtr[~dup], ytr[~dup]
    names = list(Xtr.columns)
    sc = MinMaxScaler().fit(Xtr)
    le = LabelEncoder().fit(pd.concat([ytr, yte]))
    return (sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32),
            le.transform(ytr), le.transform(yte), names, le, int(dup.sum()))


def safe_split(X, y, size, seed):
    try:
        return train_test_split(X, y, test_size=size, stratify=y, random_state=seed)
    except ValueError:
        return train_test_split(X, y, test_size=size, random_state=seed)


def subsample(X, y, n, seed):
    if len(y) <= n:
        return X, y
    _, Xs, _, ys = safe_split(X, y, n, seed)
    return Xs, ys


# ============================================================ FS methods
K_GRID = (5, 10, 15, 20, 25, 30)


def pick_k(X, y, ranking, seed):
    Xa, Xb, ya, yb = safe_split(X, y, 0.3, seed)
    best_s, best_k = -1.0, K_GRID[0]
    for k in K_GRID:
        if k > X.shape[1]:
            break
        idx = ranking[:k]
        m = DecisionTreeClassifier(random_state=seed).fit(Xa[:, idx], ya)
        s = f1_score(yb, m.predict(Xb[:, idx]), average="macro")
        if s > best_s + 1e-4:
            best_s, best_k = s, k
    return np.sort(ranking[:best_k])


def fs_none(X, y, seed, a):
    return np.arange(X.shape[1])


def fs_mi(X, y, seed, a):
    s = mutual_info_classif(X, y, random_state=seed)
    return pick_k(X, y, np.argsort(s)[::-1], seed)


def fs_rf(X, y, seed, a):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(X, y)
    return pick_k(X, y, np.argsort(rf.feature_importances_)[::-1], seed)


def make_fitness(X, y, seed, alpha=0.99):
    Xa, Xb, ya, yb = safe_split(X, y, 0.3, seed)
    cache = {}

    def fit(mask):
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        if mask.sum() == 0:
            return 1.0
        idx = np.flatnonzero(mask)
        m = DecisionTreeClassifier(random_state=seed).fit(Xa[:, idx], ya)
        err = 1 - f1_score(yb, m.predict(Xb[:, idx]), average="macro")
        v = alpha * err + (1 - alpha) * mask.sum() / len(mask)
        cache[key] = v
        return v
    return fit


def fs_bpso(X, y, seed, a):
    rng = np.random.default_rng(seed)
    n, d, fit = a.pop, X.shape[1], make_fitness(X, y, seed)
    pos = (rng.random((n, d)) < 0.5).astype(np.uint8)
    vel = rng.uniform(-1, 1, (n, d))
    pb, pbf = pos.copy(), np.array([fit(p) for p in pos])
    g, gf = pb[pbf.argmin()].copy(), pbf.min()
    for t in range(a.iters):
        w = 0.9 - 0.5 * t / a.iters
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        vel = np.clip(w * vel + 2 * r1 * (pb - pos) + 2 * r2 * (g - pos), -6, 6)
        pos = (rng.random((n, d)) < 1 / (1 + np.exp(-vel))).astype(np.uint8)
        f = np.array([fit(p) for p in pos])
        imp = f < pbf
        pb[imp], pbf[imp] = pos[imp], f[imp]
        if pbf.min() < gf:
            gf, g = pbf.min(), pb[pbf.argmin()].copy()
    return np.flatnonzero(g) if g.sum() else np.arange(d)


def fs_bgwo(X, y, seed, a):
    rng = np.random.default_rng(seed)
    n, d, fit = a.pop, X.shape[1], make_fitness(X, y, seed)

    def binarize(p):
        return (rng.random(p.shape) < 1 / (1 + np.exp(-10 * (p - 0.5)))).astype(np.uint8)

    pos = rng.random((n, d))
    B = binarize(pos)
    f = np.array([fit(b) for b in B])
    best, bestf = B[f.argmin()].copy(), f.min()
    for t in range(a.iters):
        o = np.argsort(f)
        leaders = (pos[o[0]], pos[o[1]], pos[o[2]])
        aa = 2 - 2 * t / a.iters
        new = np.zeros_like(pos)
        for L in leaders:
            r1, r2 = rng.random((n, d)), rng.random((n, d))
            A, C = 2 * aa * r1 - aa, 2 * r2
            new += L - A * np.abs(C * L - pos)
        pos = np.clip(new / 3, 0, 1)
        B = binarize(pos)
        f = np.array([fit(b) for b in B])
        if f.min() < bestf:
            bestf, best = f.min(), B[f.argmin()].copy()
    return np.flatnonzero(best) if best.sum() else np.arange(d)


def fs_pca(X, y, seed, a):
    return PCA(n_components=0.95, random_state=seed).fit(X)


METHODS = {"None": fs_none, "Filter-MI": fs_mi, "Embedded-RF": fs_rf,
           "Wrapper-BPSO": fs_bpso, "Wrapper-BGWO": fs_bgwo, "PCA": fs_pca}
SELECTORS = ["Filter-MI", "Embedded-RF", "Wrapper-BPSO", "Wrapper-BGWO"]


def apply_sel(sel, X):
    return sel.transform(X).astype(np.float32) if isinstance(sel, PCA) else X[:, sel]


# ============================================================ classifiers
CLFS = {
    "DT": lambda s: DecisionTreeClassifier(random_state=s),
    "RF": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s),
    "XGB": lambda s: XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   tree_method="hist", n_jobs=-1, random_state=s),
    "kNN": lambda s: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}


def evaluate(yt, yp, benign):
    b = yt == benign
    return dict(accuracy=accuracy_score(yt, yp),
                macro_f1=f1_score(yt, yp, average="macro"),
                mcc=matthews_corrcoef(yt, yp),
                far=float(np.mean(yp[b] != benign)) if b.any() else np.nan,
                dr=float(np.mean(yp[~b] != benign)) if (~b).any() else np.nan)


def single_latency_us(m, X, n=200):
    try:
        m.set_params(n_jobs=1)
    except (ValueError, AttributeError):
        pass
    idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
    m.predict(X[idx[:5]])
    ts = []
    for i in idx:
        x = X[i:i + 1]
        t0 = time.perf_counter()
        m.predict(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6)


# ============================================================ reporting
def holm(p):
    p = np.asarray(p, float)
    o, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def slug(s):
    return s.lower().replace("-", "").replace(" ", "")


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def report(df, sel, out):
    order = list(METHODS)
    num = ["accuracy", "macro_f1", "mcc", "far", "dr", "n_features", "fs_time_s",
           "train_time_s", "batch_us", "latency_us", "model_kb"]
    summ = df.groupby(["dataset", "task", "method", "clf"])[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(out, "summary.csv"), index=False)

    stat_rows, stab_rows = [], []
    for (ds, task), d in df.groupby(["dataset", "task"]):
        tag = f"{slug(ds)}_{task}"
        clfs = [c for c in CLFS if c in d.clf.unique()]
        # macro-F1 table
        mu = d.pivot_table(index="method", columns="clf", values="macro_f1", aggfunc="mean")
        sd = d.pivot_table(index="method", columns="clf", values="macro_f1", aggfunc="std").fillna(0)
        lines = [r"\begin{table}[htbp]\centering\small",
                 rf"\caption{{Macro-F1 (mean $\pm$ s.d. over seeds) on {ds}, {task} task.}}",
                 rf"\label{{tab:bench_f1_{tag}}}",
                 r"\begin{tabular}{l" + "c" * len(clfs) + "}", r"\toprule",
                 "Method & " + " & ".join(clfs) + r" \\", r"\midrule"]
        for mth in order:
            if mth in mu.index:
                best = {c: mu[c].max() for c in clfs}
                cells = []
                for c in clfs:
                    s = f"{mu.loc[mth, c]:.4f} $\\pm$ {sd.loc[mth, c]:.4f}"
                    cells.append(rf"\textbf{{{s}}}" if np.isclose(mu.loc[mth, c], best[c]) else s)
                lines.append(f"{mth} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        write(os.path.join(out, f"tab_f1_{tag}.tex"), "\n".join(lines))

        # Friedman + Wilcoxon vs None
        piv = d.pivot_table(index=["clf", "seed"], columns="method", values="macro_f1").dropna(axis=1)
        piv = piv[[m for m in order if m in piv.columns]]
        ranks = piv.rank(axis=1, ascending=False).mean()
        try:
            fp = friedmanchisquare(*[piv[c].values for c in piv.columns]).pvalue
        except ValueError:
            fp = np.nan
        others = [m for m in piv.columns if m != "None"]
        raw = []
        for m in others:
            try:
                raw.append(wilcoxon(piv[m], piv["None"]).pvalue)
            except ValueError:
                raw.append(1.0)
        adj = dict(zip(others, holm(raw))) if others else {}
        for m in piv.columns:
            sets = [set(v) for k, v in sel.items() if k.startswith(f"{ds}|{task}|{m}|")]
            jac = np.nan
            if m in SELECTORS and len(sets) > 1:
                pairs = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
                jac = float(np.mean(pairs))
            stat_rows.append(dict(dataset=ds, task=task, method=m, mean_rank=ranks[m],
                                  friedman_p=fp, holm_p_vs_none=adj.get(m, np.nan), jaccard=jac))

        # Pareto figure (XGB)
        x = d[d.clf == "XGB"].groupby("method")[["latency_us", "macro_f1"]].mean()
        if len(x):
            fig, ax = plt.subplots(figsize=(4.5, 3.4))
            ax.scatter(x.latency_us, x.macro_f1)
            for mth, r in x.iterrows():
                ax.annotate(mth, (r.latency_us, r.macro_f1), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(r"Single-sample latency ($\mu$s)")
            ax.set_ylabel("Macro-F1")
            ax.set_title(f"{ds} ({task}), XGBoost", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"fig_pareto_{tag}.pdf"))
            plt.close(fig)

    st = pd.DataFrame(stat_rows)
    st.to_csv(os.path.join(out, "stats.csv"), index=False)

    # Cost / operational table (XGB)
    x = df[df.clf == "XGB"].groupby(["dataset", "task", "method"])[
        ["n_features", "fs_time_s", "far", "dr", "latency_us", "model_kb"]].mean().reset_index()
    x = x.merge(st[["dataset", "task", "method", "mean_rank", "holm_p_vs_none", "jaccard"]],
                on=["dataset", "task", "method"], how="left")
    lines = [r"\begin{table*}[htbp]\centering\scriptsize",
             r"\caption{Operational metrics with XGBoost (means over seeds). Rank: mean Friedman rank of macro-F1 across classifiers and seeds (lower is better); $p$: Holm-adjusted Wilcoxon test versus no FS; $J$: mean pairwise Jaccard stability of selected subsets.}",
             r"\label{tab:bench_cost}",
             r"\begin{tabular}{llcrrrrrrrr}", r"\toprule",
             r"Dataset & Task & Method & \#Feat. & FS time (s) & FAR & DR & Lat. ($\mu$s) & Size (KB) & Rank & $p$ / $J$ \\",
             r"\midrule"]
    for _, r in x.iterrows():
        pj = ("--" if np.isnan(r.holm_p_vs_none) else f"{r.holm_p_vs_none:.3f}") + " / " + \
             ("--" if np.isnan(r.jaccard) else f"{r.jaccard:.2f}")
        lines.append(f"{r.dataset} & {r.task} & {r.method} & {r.n_features:.1f} & {r.fs_time_s:.2f} & "
                     f"{r.far:.4f} & {r.dr:.4f} & {r.latency_us:.0f} & {r.model_kb:.0f} & "
                     f"{r.mean_rank:.2f} & {pj} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(os.path.join(out, "tab_cost.tex"), "\n".join(lines))


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsw", help="folder with UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv")
    ap.add_argument("--cic", help="folder with CICIoT2023 CSV files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fs-cap", type=int, default=30000)
    ap.add_argument("--knn-cap", type=int, default=50000)
    ap.add_argument("--cic-per-label", type=int, default=20000)
    ap.add_argument("--cic-benign-cap", type=int, default=100000)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.fs_cap, a.knn_cap, a.cic_per_label, a.pop, a.iters = 2, 3000, 5000, 1000, 6, 4
        a.cic_benign_cap = 5000
    os.makedirs(os.path.join(a.out, "edge"), exist_ok=True)

    data = {}
    if a.unsw:
        print("Loading UNSW-NB15"); data["UNSW-NB15"] = load_unsw(a.unsw)
    if a.cic:
        print("Loading CICIoT2023")
        data["CICIoT2023"] = load_cic(
            a.cic, a.cic_per_label,
            os.path.join(a.out, f"cache_cic_{a.cic_per_label}_{a.cic_benign_cap}.pkl"),
            a.cic_benign_cap)
    if not data:
        ap.error("give --unsw and/or --cic")

    rows, sel, info = [], {}, []
    for ds, (Xtr0, Xte0, ctr, cte) in data.items():
        for task in ("binary", "multi"):
            Xtr, Xte, ytr, yte, names, le, ndup = prepare(
                Xtr0, Xte0, task_labels(ctr, task), task_labels(cte, task))
            benign = int(le.transform(["Benign"])[0])
            info.append(dict(dataset=ds, task=task, n_train=len(ytr), n_test=len(yte),
                             n_features=len(names), n_classes=len(le.classes_),
                             train_duplicates_removed=ndup,
                             classes=";".join(le.classes_)))
            if task == "binary":
                np.save(os.path.join(a.out, "edge", f"sample_{slug(ds)}.npy"), Xte[:1000])
            for seed in range(a.seeds):
                Xfs, yfs = subsample(Xtr, ytr, a.fs_cap, seed)
                for mth, fs in METHODS.items():
                    t0 = time.perf_counter()
                    s = fs(Xfs, yfs, seed, a)
                    fs_time = time.perf_counter() - t0
                    Atr, Ate = apply_sel(s, Xtr), apply_sel(s, Xte)
                    if not isinstance(s, PCA):
                        sel[f"{ds}|{task}|{mth}|{seed}"] = [names[i] for i in s]
                    for cn, mk in CLFS.items():
                        if cn == "kNN" and a.no_knn:
                            continue
                        Xc, yc = subsample(Atr, ytr, a.knn_cap, seed) if cn == "kNN" else (Atr, ytr)
                        m = mk(seed)
                        t0 = time.perf_counter(); m.fit(Xc, yc); tr_t = time.perf_counter() - t0
                        t0 = time.perf_counter(); yp = m.predict(Ate); pr_t = time.perf_counter() - t0
                        lat = single_latency_us(m, Ate)
                        rows.append(dict(dataset=ds, task=task, seed=seed, method=mth, clf=cn,
                                         n_features=Atr.shape[1], fs_time_s=fs_time,
                                         train_time_s=tr_t, batch_us=pr_t / len(yte) * 1e6,
                                         latency_us=lat, model_kb=len(pickle.dumps(m)) / 1024,
                                         **evaluate(yte, yp, benign)))
                        if seed == 0 and task == "binary" and cn in ("RF", "XGB"):
                            joblib.dump(dict(selector=s, model=m, dataset=ds, method=mth, clf=cn,
                                             sample=f"sample_{slug(ds)}.npy"),
                                        os.path.join(a.out, "edge", f"{slug(ds)}_{slug(mth)}_{cn}.joblib"))
                    print(f"{ds} | {task} | seed {seed} | {mth}: {Atr.shape[1]} features, "
                          f"FS {fs_time:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "raw_results.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "raw_results.csv"), index=False)
    pd.DataFrame(info).to_csv(os.path.join(a.out, "dataset_info.csv"), index=False)
    with open(os.path.join(a.out, "selected_features.json"), "w") as fh:
        json.dump(sel, fh, indent=1)
    report(df, sel, a.out)
    print(f"Done. Results in {a.out}/")


if __name__ == "__main__":
    main()
