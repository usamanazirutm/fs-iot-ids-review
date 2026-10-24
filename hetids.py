#!/usr/bin/env python3
"""Objective 2: unified deep-learning intrusion detection across heterogeneous IoT networks.

HetIDS = domain-specific feature-gated adapters + shared residual backbone + unified attack
taxonomy head + domain-specific fine heads + domain-adversarial alignment (gradient reversal).
Feature subsets come from Objective 1 (stable consensus of Filter-MI and Embedded-RF).

Needs benchmark.py (Objective 1) in the same folder.
Run (from the Feature Selection folder):
  python hetids.py --unsw "data\\unsw\\Training and Testing Sets" --cic data\\ciciot2023 --edge data\\edgeiiot --out results_obj2
  add --quick for a smoke test
"""
import argparse, collections, copy, glob, json, os, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

import benchmark as B

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, os.cpu_count() or 1))

UNIFIED = ["Benign", "DoS/DDoS", "Reconnaissance", "Exploitation", "Malware/Botnet",
           "Spoofing/MITM", "BruteForce"]
UIDX = {c: i for i, c in enumerate(UNIFIED)}
UNSW_MAP = {"Benign": "Benign", "DoS": "DoS/DDoS", "Reconnaissance": "Reconnaissance",
            "Analysis": "Reconnaissance", "Fuzzers": "Exploitation", "Exploits": "Exploitation",
            "Shellcode": "Exploitation", "Generic": "Exploitation", "Backdoor": "Malware/Botnet",
            "Backdoors": "Malware/Botnet", "Worms": "Malware/Botnet"}
CIC_MAP = {"Benign": "Benign", "DDoS": "DoS/DDoS", "DoS": "DoS/DDoS", "Mirai": "Malware/Botnet",
           "Recon": "Reconnaissance", "Spoofing": "Spoofing/MITM", "Web": "Exploitation",
           "BruteForce": "BruteForce"}


def edge_unified(t):
    t = str(t).lower()
    for keys, u in [(("normal",), "Benign"), (("ddos", "dos"), "DoS/DDoS"),
                    (("port", "finger", "vulnerab", "scan"), "Reconnaissance"),
                    (("sql", "xss", "upload"), "Exploitation"),
                    (("backdoor", "ransom"), "Malware/Botnet"), (("mitm",), "Spoofing/MITM"),
                    (("password",), "BruteForce")]:
        if any(k in t for k in keys):
            return u
    raise ValueError(f"Unmapped Edge-IIoTset label: {t}")


# ====================================================================== data
def load_domains(a):
    doms = []
    if a.unsw:
        Xtr, Xte, ctr, cte = B.load_unsw(a.unsw)
        doms.append(dict(name="UNSW-NB15", Xtr=Xtr, Xte=Xte, ftr=ctr, fte=cte,
                         umap=lambda s: s.map(UNSW_MAP)))
    if a.cic:
        cache = a.cic_cache
        if not os.path.exists(cache):
            B.load_cic(a.cic, 20000, cache, 100000)
        df = pd.read_pickle(cache)
        fine = df["label"]
        cat = fine.map(B.cic_category)
        X = df.drop(columns=["label"])
        Xtr, Xte, ctr, cte, ftr, fte = train_test_split(X, cat, fine, test_size=0.3,
                                                        stratify=cat, random_state=0)
        # fine level = 8 CICIoT2023 categories (as in Objective 1); Backdoor_Malware -> Malware
        uf = lambda f: pd.Series(np.where(f.values == "Backdoor_Malware", "Malware/Botnet",
                                          f.map(lambda l: CIC_MAP[B.cic_category(l)]).values))
        doms.append(dict(name="CICIoT2023", Xtr=Xtr.reset_index(drop=True), Xte=Xte.reset_index(drop=True),
                         ftr=ctr.reset_index(drop=True), fte=cte.reset_index(drop=True),
                         utr=uf(ftr), ute=uf(fte)))
    if a.edge:
        files = sorted(glob.glob(os.path.join(a.edge, "**", "*.csv"), recursive=True))
        ml = [f for f in files if "ML-EdgeIIoT" in os.path.basename(f)] or \
             [f for f in files if "edge" in os.path.basename(f).lower()]
        if not ml:
            raise FileNotFoundError("Put ML-EdgeIIoT-dataset.csv in the --edge folder")
        df = pd.read_csv(ml[0], low_memory=False)
        y = df["Attack_type"].astype(str).str.strip()
        leak = ("Attack_type", "Attack_label", "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
                "udp.port", "mqtt.msg")  # identifiers, timestamps, ports and payloads (dataset authors' preprocessing)
        X = df.drop(columns=[c for c in leak if c in df.columns])
        for c in list(X.columns):
            if not pd.api.types.is_numeric_dtype(X[c]):
                num = pd.to_numeric(X[c], errors="coerce")
                if num.notna().mean() > 0.95:
                    X[c] = num
                elif X[c].nunique() <= 50:
                    X[c] = pd.Categorical(X[c]).codes
                else:
                    X = X.drop(columns=[c])   # identifiers, timestamps, payloads
        X = X.loc[:, X.nunique() > 1]
        Xtr, Xte, ftr, fte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        doms.append(dict(name="Edge-IIoTset", Xtr=Xtr.reset_index(drop=True), Xte=Xte.reset_index(drop=True),
                         ftr=ftr.reset_index(drop=True), fte=fte.reset_index(drop=True),
                         umap=lambda s: s.map(edge_unified)))
    return doms


def objective1_features(d, sel_json, seeds=3):
    """Stable consensus of Objective-1 Filter-MI and Embedded-RF selections (multi-class task)."""
    names = list(d["Xtr"].columns)
    if sel_json and os.path.exists(sel_json):
        sel = json.load(open(sel_json))
        keys = [k for k in sel if k.startswith(f"{d['name']}|multi|")]
        if keys:
            chosen = set()
            for m in ("Filter-MI", "Embedded-RF"):
                c = collections.Counter(f for k in keys if f"|{m}|" in k for f in sel[k])
                n = len([k for k in keys if f"|{m}|" in k])
                chosen |= {f for f, v in c.items() if v >= max(1, (n + 1) // 2)}
            feats = [f for f in names if f in chosen]
            if feats:
                return feats, "Objective-1 results file"
    # compute with the Objective-1 procedure
    X = d["Xtr"].replace([np.inf, -np.inf], np.nan).fillna(d["Xtr"].median(numeric_only=True)).fillna(0)
    Xs = MinMaxScaler().fit_transform(X).astype(np.float32)
    y = LabelEncoder().fit_transform(d["ftr"])
    chosen = set()
    for fs in (B.fs_mi, B.fs_rf):
        c = collections.Counter()
        for s in range(seeds):
            Xa, ya = B.subsample(Xs, y, 30000, s)
            c.update(fs(Xa, ya, s, None))
        chosen |= {names[i] for i, v in c.items() if v >= (seeds + 1) // 2}
    return [f for f in names if f in chosen], "Objective-1 procedure (recomputed)"


def prepare(d, feats, quick):
    Xtr, Xte = d["Xtr"][feats].copy(), d["Xte"][feats].copy()
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan); Xte = Xte.replace([np.inf, -np.inf], np.nan)
    med = Xtr.median(numeric_only=True)
    Xtr, Xte = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    ftr, fte = d["ftr"].astype(str), d["fte"].astype(str)
    utr = d["utr"] if "utr" in d else d["umap"](ftr)
    ute = d["ute"] if "ute" in d else d["umap"](fte)
    dup = pd.concat([Xtr, ftr.rename("_y")], axis=1).duplicated().values
    Xtr, ftr, utr = Xtr[~dup], ftr[~dup], pd.Series(utr)[~dup]
    tf = lambda A: np.sign(A) * np.log1p(np.abs(A))
    sc = StandardScaler().fit(tf(Xtr.values.astype(np.float64)))
    Xtr = sc.transform(tf(Xtr.values.astype(np.float64))).astype(np.float32)
    Xte = sc.transform(tf(Xte.values.astype(np.float64))).astype(np.float32)
    Xtr, Xte = np.clip(Xtr, -10, 10), np.clip(Xte, -10, 10)
    le = LabelEncoder().fit(pd.concat([ftr, fte]))
    yf_tr, yf_te = le.transform(ftr), le.transform(fte)
    yu_tr = np.array([UIDX[u] for u in utr]); yu_te = np.array([UIDX[u] for u in ute])
    if quick:
        Xtr, yf_tr, yu_tr = sub(Xtr, yf_tr, yu_tr, 6000, 0)
        Xte, yf_te, yu_te = sub(Xte, yf_te, yu_te, 3000, 0)
    Xa, Xv, fa, fv, ua, uv = safe_split(Xtr, yf_tr, yu_tr, 0.1)
    return dict(name=d["name"], feats=feats, k=len(feats), fine=list(le.classes_), n_dup=int(dup.sum()),
                Xtr=Xa, yf_tr=fa, yu_tr=ua, Xva=Xv, yf_va=fv, yu_va=uv, Xte=Xte, yf_te=yf_te, yu_te=yu_te,
                fine2u=np.array([UIDX[u] for u in (pd.Series(le.classes_).map(
                    lambda f: dict(zip(ftr, utr)).get(f, dict(zip(fte, ute)).get(f))))]))


def safe_split(X, yf, yu, size, seed=0):
    try:
        return train_test_split(X, yf, yu, test_size=size, stratify=yf, random_state=seed)
    except ValueError:
        return train_test_split(X, yf, yu, test_size=size, random_state=seed)


def sub(X, yf, yu, n, seed):
    if len(yf) <= n:
        return X, yf, yu
    _, Xs, _, fs_, _, us = safe_split(X, yf, yu, n, seed)
    return Xs, fs_, us


def few(X, yf, yu, frac, seed):
    """Stratified fraction with at least 2 records per fine class."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(yf):
        ii = np.flatnonzero(yf == c)
        n = max(2, int(round(frac * len(ii))))
        idx.extend(rng.choice(ii, min(n, len(ii)), replace=False))
    idx = np.array(idx)
    return X[idx], yf[idx], yu[idx]


# ===================================================================== models
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def ple_edges(X, T=16):
    """Quantile bin edges per feature for piecewise-linear encoding (Gorishniy et al., 2022)."""
    k = X.shape[1]
    L = np.zeros((k, T), np.float32); W = np.full((k, T), np.inf, np.float32)
    for j in range(k):
        e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, T + 1)))
        if len(e) < 2:
            e = np.array([X[:, j].min(), X[:, j].min() + 1.0])
        n = len(e) - 1
        L[j, :n] = e[:-1]; W[j, :n] = np.maximum(e[1:] - e[:-1], 1e-6)
    return torch.from_numpy(L), torch.from_numpy(W)


class Adapter(nn.Module):
    def __init__(self, k, d=128, p=0.1, gate=True, edges=None):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(k, k), nn.Sigmoid()) if gate else None
        self.ple = edges is not None
        if self.ple:
            self.register_buffer("L", edges[0]); self.register_buffer("W", edges[1])
            T = edges[0].shape[1]
            self.emb = nn.Parameter(torch.randn(k, T, 16) * 0.05)   # per-feature embedding of the PLE code
            fin = k * 16
        else:
            fin = k
        self.net = nn.Sequential(nn.Linear(fin, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))

    def forward(self, x):
        g = None
        if self.gate is not None:
            g = self.gate(x)
        if self.ple:
            c = torch.clamp((x.unsqueeze(-1) - self.L) / self.W, 0, 1)        # (B,k,T)
            e = torch.einsum("bkt,ktd->bkd", c, self.emb)                        # (B,k,16)
            if g is not None:
                e = 2 * e * g.unsqueeze(-1)
            h = e.flatten(1)
        else:
            h = 2 * x * g if g is not None else x
        return self.net(h), g


class ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))

    def forward(self, x):
        return x + self.f(x)


class HetIDS(nn.Module):
    def __init__(self, in_dims, n_fine, d=128, blocks=3, gate=True, dann=True, fine=True, edges=None):
        super().__init__()
        self.adapters = nn.ModuleList([Adapter(k, d, gate=gate, edges=edges[i] if edges else None)
                                       for i, k in enumerate(in_dims)])
        self.backbone = nn.Sequential(*[ResBlock(d) for _ in range(blocks)], nn.LayerNorm(d))
        self.uhead = nn.Linear(d, len(UNIFIED))
        self.fheads = nn.ModuleList([nn.Linear(d, n) for n in n_fine]) if fine else None
        self.disc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, len(in_dims))) \
            if (dann and len(in_dims) > 1) else None

    def forward(self, x, dom, lam=0.0):
        h, g = self.adapters[dom](x)
        z = self.backbone(h)
        out = {"u": self.uhead(z), "z": z, "g": g}
        if self.fheads is not None:
            out["f"] = self.fheads[dom](z)
        if self.disc is not None:
            out["d"] = self.disc(GRL.apply(z, lam))
        return out


class Pooled(nn.Module):
    """Naive unified baseline: zero-padded union of all feature spaces + domain one-hot."""
    def __init__(self, in_dims, n_fine, d=128, blocks=3):
        super().__init__()
        self.off = np.cumsum([0] + list(in_dims))
        self.n = len(in_dims)
        self.core = HetIDS([int(self.off[-1]) + self.n], n_fine, d, blocks, gate=False, dann=False, fine=False)
        self.fheads = nn.ModuleList([nn.Linear(d, n) for n in n_fine])

    def forward(self, x, dom, lam=0.0):
        B_ = x.shape[0]
        pad = torch.zeros(B_, int(self.off[-1]) + self.n)
        pad[:, self.off[dom]:self.off[dom + 1]] = x
        pad[:, int(self.off[-1]) + dom] = 1.0
        o = self.core(pad, 0)
        o["f"] = self.fheads[dom](o["z"])
        return o


class CNN1D(nn.Module):
    def __init__(self, k, n_fine):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1),
                               nn.ReLU(), nn.AdaptiveMaxPool1d(4), nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
                               nn.Dropout(0.1))
        self.uhead = nn.Linear(128, len(UNIFIED))
        self.fheads = nn.ModuleList([nn.Linear(128, n_fine)])
        self.disc = None

    def forward(self, x, dom, lam=0.0):
        z = self.c(x.unsqueeze(1))
        return {"u": self.uhead(z), "f": self.fheads[0](z), "z": z, "g": None}


# =================================================================== training
def class_w(y, n):
    c = np.bincount(y, minlength=n).astype(float)
    w = np.where(c > 0, 1 / np.sqrt(np.maximum(c, 1)), 0)
    return torch.tensor(w / w[c > 0].mean(), dtype=torch.float32)


def predict(model, X, dom, bs=8192):
    model.eval()
    U, Fp, Z, G = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(torch.from_numpy(X[i:i + bs]), dom)
            U.append(o["u"].argmax(1).numpy())
            if "f" in o:
                Fp.append(o["f"].argmax(1).numpy())
            Z.append(o["z"].numpy())
            if o["g"] is not None:
                G.append(o["g"].numpy())
    return (np.concatenate(U), np.concatenate(Fp) if Fp else None, np.concatenate(Z),
            np.concatenate(G) if G else None)


def train(model, data, slots, a, seed, params=None, epochs=None, val_slots=None):
    """data: list of prepared domains; slots: list of (data_index, model_domain_index)."""
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=a.lr, weight_decay=1e-4)
    T = [(torch.from_numpy(data[i]["Xtr"]), torch.from_numpy(data[i]["yu_tr"]).long(),
          torch.from_numpy(data[i]["yf_tr"]).long(), class_w(data[i]["yu_tr"], len(UNIFIED)),
          class_w(data[i]["yf_tr"], len(data[i]["fine"])), m) for i, m in slots]
    steps = min(a.max_steps, max(len(t[0]) for t in T) // a.bs + 1)
    E = epochs or a.epochs
    best, best_state, bad = -1, None, 0
    for ep in range(E):
        model.train()
        p = ep / max(1, E - 1)
        lam = 2 / (1 + np.exp(-10 * p)) - 1
        for _ in range(steps):
            loss = 0
            for X, yu, yf, wu, wf, m in T:
                idx = torch.randint(0, len(X), (min(a.bs, len(X)),))
                o = model(X[idx], m, lam)
                loss = loss + F.cross_entropy(o["u"], yu[idx], weight=wu)
                if "f" in o:
                    loss = loss + F.cross_entropy(o["f"], yf[idx], weight=wf)
                if o.get("d") is not None:
                    loss = loss + a.dann_w * F.cross_entropy(o["d"], torch.full((len(idx),), m, dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step()
        vs = []
        for i, m in (val_slots or slots):
            up, _, _, _ = predict(model, data[i]["Xva"], m)
            vs.append(f1_score(data[i]["yu_va"], up, average="macro"))
        v = float(np.mean(vs))
        if v > best + 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model


def metrics(yu, up, yf=None, fp=None):
    b = yu == 0
    r = dict(u_macro_f1=f1_score(yu, up, average="macro"), accuracy=accuracy_score(yu, up),
             mcc=matthews_corrcoef(yu, up),
             far=float(np.mean(up[b] != 0)) if b.any() else np.nan,
             dr=float(np.mean(up[~b] != 0)) if (~b).any() else np.nan)
    r["f_macro_f1"] = f1_score(yf, fp, average="macro") if fp is not None else np.nan
    return r


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def latency_us(model, X, dom, n=20000):
    X = X[:n]
    t0 = time.perf_counter(); predict(model, X, dom); return (time.perf_counter() - t0) / len(X) * 1e6


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsw"); ap.add_argument("--cic"); ap.add_argument("--edge")
    ap.add_argument("--cic-cache", default=os.path.join("results", "cache_cic_20000_100000.pkl"))
    ap.add_argument("--sel", default=os.path.join("results", "selected_features.json"))
    ap.add_argument("--out", default="results_obj2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dann-w", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--prev", help="raw_results.csv of the first run; its baseline rows are reused")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.max_steps = 1, 3, 30
    os.makedirs(a.out, exist_ok=True)
    fracs = [0.05] if a.quick else [0.01, 0.05, 0.10]

    print("Loading domains ...", flush=True)
    raw = load_domains(a)
    if len(raw) < 2:
        ap.error("need at least two domains")
    data, info = [], []
    for d in raw:
        feats, src = objective1_features(d, a.sel)
        P = prepare(d, feats, a.quick)
        data.append(P)
        info.append(dict(domain=P["name"], n_features_all=d["Xtr"].shape[1], n_features_selected=P["k"],
                         feature_source=src, features=";".join(feats), n_train=len(P["yf_tr"]),
                         n_val=len(P["yf_va"]), n_test=len(P["yf_te"]), fine_classes=";".join(P["fine"]),
                         unified_classes=";".join(UNIFIED[i] for i in np.unique(P["yu_te"])),
                         train_duplicates_removed=P["n_dup"]))
        print(f"  {P['name']}: {P['k']} features ({src}), train {len(P['yf_tr'])}, test {len(P['yf_te'])}", flush=True)
    pd.DataFrame(info).to_csv(os.path.join(a.out, "domain_info.csv"), index=False)
    nd = len(data)
    in_dims = [d["k"] for d in data]
    n_fine = [len(d["fine"]) for d in data]
    allslots = [(i, i) for i in range(nd)]
    rows, fs_rows, keep = [], [], {}

    def log(model_name, seed, i, up, fp, lat=np.nan, npar=np.nan):
        d = data[i]
        rows.append(dict(model=model_name, seed=seed, domain=d["name"], latency_us=lat, params=npar,
                         **metrics(d["yu_te"], up, d["yf_te"], fp)))

    edges = [ple_edges(d["Xtr"]) for d in data]
    prev = pd.read_csv(a.prev) if a.prev and os.path.exists(a.prev) else None
    reuse = ["RF (per-domain)", "XGBoost (per-domain)", "MLP (per-domain)", "CNN-1D (per-domain)",
             "Pooled zero-padded MLP", "HetIDS w/o adversarial alignment", "HetIDS w/o feature gate",
             "HetIDS w/o fine heads"]
    rename = {"HetIDS w/o adversarial alignment": "HetIDS w/o PLE encoding",
              "HetIDS w/o feature gate": "HetIDS w/o PLE, w/o feature gate",
              "HetIDS w/o fine heads": "HetIDS w/o PLE, w/o fine heads"}
    if prev is not None:
        pr = prev[prev.model.isin(reuse) & (prev.seed < a.seeds)].copy()
        pr["model"] = pr.model.replace(rename)
        rows.extend(pr.to_dict("records"))
        print(f"Reusing {len(pr)} baseline rows from {a.prev}", flush=True)
    for seed in range(a.seeds):
        if prev is None:
            # ---------------- classical per-domain baselines (fine labels -> unified)
            for i, d in enumerate(data):
                for name, mk in [("RF (per-domain)", lambda: RandomForestClassifier(200, n_jobs=-1, random_state=seed)),
                                 ("XGBoost (per-domain)", lambda: XGBClassifier(n_estimators=300, max_depth=8,
                                  learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=seed))]:
                    m = mk().fit(d["Xtr"], d["yf_tr"])
                    t0 = time.perf_counter(); fp = m.predict(d["Xte"]); lat = (time.perf_counter() - t0) / len(fp) * 1e6
                    log(name, seed, i, d["fine2u"][fp], fp, lat)
                print(f"seed {seed} | {d['name']} | classical baselines done", flush=True)
            # ---------------- per-domain deep baselines
            for i, d in enumerate(data):
                for name, mk in [("MLP (per-domain)", lambda: HetIDS([d["k"]], [len(d["fine"])], gate=False, dann=False)),
                                 ("CNN-1D (per-domain)", lambda: CNN1D(d["k"], len(d["fine"])))]:
                    m = train(mk(), data, [(i, 0)], a, seed)
                    up, fp, _, _ = predict(m, d["Xte"], 0)
                    log(name, seed, i, up, fp, latency_us(m, d["Xte"], 0), nparams(m))
                print(f"seed {seed} | {d['name']} | deep per-domain baselines done", flush=True)
            # ---------------- naive pooled unified model
            m = train(Pooled(in_dims, n_fine), data, allslots, a, seed)
            for i, d in enumerate(data):
                up, fp, _, _ = predict(m, d["Xte"], i)
                log("Pooled zero-padded MLP", seed, i, up, fp, latency_us(m, d["Xte"], i), nparams(m))
            print(f"seed {seed} | pooled baseline done", flush=True)
        # ---------------- HetIDS v2 (PLE encoder, gated adapters, shared backbone, unified + fine heads)
        variants = {"HetIDS (neural)": {"dann": False, "edges": edges},
                    "HetIDS + adversarial alignment": {"dann": True, "edges": edges}}
        for name, kw in variants.items():
            m = train(HetIDS(in_dims, n_fine, **kw), data, allslots, a, seed)
            for i, d in enumerate(data):
                up, fp, Z, G = predict(m, d["Xte"], i)
                log(name, seed, i, up, fp, latency_us(m, d["Xte"], i), nparams(m))
                if seed == 0:
                    keep[(name, i)] = dict(up=up, Z=Z, G=G)
            print(f"seed {seed} | {name} done", flush=True)
            if name == "HetIDS (neural)":
                # HetIDS-X: unified deep embedding + selected features -> per-domain gradient-boosted heads
                for i, d in enumerate(data):
                    _, _, Ztr, _ = predict(m, d["Xtr"], i)
                    _, _, Zte, _ = predict(m, d["Xte"], i)
                    xg = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                                       n_jobs=-1, random_state=seed).fit(np.hstack([d["Xtr"], Ztr]), d["yf_tr"])
                    t0 = time.perf_counter()
                    _, _, Zl, _ = predict(m, d["Xte"], i); fp = xg.predict(np.hstack([d["Xte"], Zl]))
                    lat = (time.perf_counter() - t0) / len(fp) * 1e6
                    log("HetIDS-X (proposed)", seed, i, d["fine2u"][fp], fp, lat, nparams(m))
                    if seed == 0:
                        keep[("HetIDS-X (proposed)", i)] = dict(up=d["fine2u"][fp], Z=Zte, G=keep[(name, i)]["G"])
                print(f"seed {seed} | HetIDS-X done", flush=True)
        # ---------------- HetIDS without Objective-1 feature selection (all features)
        if seed == 0 or not a.quick:
            data_all = []
            for d in raw:
                P = prepare(d, list(d["Xtr"].columns), a.quick); data_all.append(P)
            m = train(HetIDS([p["k"] for p in data_all], [len(p["fine"]) for p in data_all], dann=False,
                             edges=[ple_edges(p["Xtr"]) for p in data_all]), data_all, allslots, a, seed)
            for i, p in enumerate(data_all):
                up, fp, _, _ = predict(m, p["Xte"], i)
                rows.append(dict(model="HetIDS (neural) w/o Objective-1 FS", seed=seed, domain=p["name"],
                                 latency_us=latency_us(m, p["Xte"], i), params=nparams(m),
                                 **metrics(p["yu_te"], up, p["yf_te"], fp)))
            print(f"seed {seed} | no-FS ablation done", flush=True)
        # ---------------- few-shot onboarding of an unseen domain
        for t in range(nd):
            src = [(i, i) for i in range(nd) if i != t]
            base = train(HetIDS(in_dims, n_fine, dann=False, edges=edges), data, src, a, seed)
            for fr in fracs:
                Xf, yff, yuf = few(data[t]["Xtr"], data[t]["yf_tr"], data[t]["yu_tr"], fr, seed)
                shot = dict(data[t]); shot.update(Xtr=Xf, yf_tr=yff, yu_tr=yuf)
                tmp = list(data); tmp[t] = shot
                for mname, freeze in [("HetIDS transfer (frozen backbone)", True), ("HetIDS transfer (full fine-tuning)", False)]:
                    m = copy.deepcopy(base)
                    if freeze:
                        for p_ in m.parameters():
                            p_.requires_grad = False
                        params = list(m.adapters[t].parameters()) + list(m.fheads[t].parameters())
                        for p_ in params:
                            p_.requires_grad = True
                    else:
                        params = None
                    m = train(m, tmp, [(t, t)], a, seed, params=params)
                    up, fp, _, _ = predict(m, data[t]["Xte"], t)
                    fs_rows.append(dict(method=mname, seed=seed, target=data[t]["name"], fraction=fr,
                                        n_labelled=len(yff), **metrics(data[t]["yu_te"], up, data[t]["yf_te"], fp)))
                sc = train(HetIDS([data[t]["k"]], [n_fine[t]], dann=False, edges=[edges[t]]), tmp, [(t, 0)], a, seed)
                up, fp, _, _ = predict(sc, data[t]["Xte"], 0)
                fs_rows.append(dict(method="Scratch (same architecture)", seed=seed, target=data[t]["name"],
                                    fraction=fr, n_labelled=len(yff),
                                    **metrics(data[t]["yu_te"], up, data[t]["yf_te"], fp)))
            print(f"seed {seed} | few-shot target {data[t]['name']} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "raw_results.csv"), index=False)
        pd.DataFrame(fs_rows).to_csv(os.path.join(a.out, "fewshot_results.csv"), index=False)

    report(pd.DataFrame(rows), pd.DataFrame(fs_rows), data, keep, a.out)
    print(f"Done. Results in {a.out}/")


# ===================================================================== report
def tex_table(path, caption, label, header, lines, star=False):
    env = "table*" if star else "table"
    t = [rf"\begin{{{env}}}[htbp]\centering\scriptsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"] + lines + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    open(path, "w").write("\n".join(t))


def report(df, fs, data, keep, out):
    doms = [d["name"] for d in data]
    order = ["RF (per-domain)", "XGBoost (per-domain)", "MLP (per-domain)", "CNN-1D (per-domain)",
             "Pooled zero-padded MLP", "HetIDS (neural)", "HetIDS-X (proposed)"]
    abl = ["HetIDS-X (proposed)", "HetIDS (neural)", "HetIDS + adversarial alignment", "HetIDS w/o PLE encoding",
           "HetIDS w/o PLE, w/o feature gate", "HetIDS w/o PLE, w/o fine heads", "HetIDS (neural) w/o Objective-1 FS"]
    abl = [m for m in abl if m in set(df.model)]
    g = df.groupby(["model", "domain"])
    mu, sd = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0)
    df.groupby(["model", "domain"]).agg(["mean", "std"]).to_csv(os.path.join(out, "summary.csv"))

    def cell(m, d, c, f="{:.3f}"):
        if (m, d) not in mu.index:
            return "--"
        return f.format(mu.loc[(m, d), c]) + r" $\pm$ " + f.format(sd.loc[(m, d), c])

    def block(models, col, fname, cap, lab):
        best = {d: max(mu.loc[(m, d), col] for m in models if (m, d) in mu.index) for d in doms}
        lines = []
        for m in models:
            cells = []
            for d in doms:
                c = cell(m, d, col)
                cells.append(r"\textbf{" + c + "}" if (m, d) in mu.index and np.isclose(mu.loc[(m, d), col], best[d]) else c)
            avg = np.mean([mu.loc[(m, d), col] for d in doms if (m, d) in mu.index])
            lines.append(f"{m} & " + " & ".join(cells) + f" & {avg:.3f} \\\\")
        tex_table(os.path.join(out, fname), cap, lab, ["Model"] + doms + ["Mean"], lines, star=True)

    block(order, "u_macro_f1", "tab_main.tex", "Unified-taxonomy macro-F1 on each domain's test set (mean $\\pm$ s.d. over seeds).", "tab:main")
    block(order, "f_macro_f1", "tab_fine.tex", "Domain-specific (fine-grained) macro-F1 (mean $\\pm$ s.d. over seeds).", "tab:fine")
    block(abl, "u_macro_f1", "tab_ablation.tex", "Ablation of HetIDS components: unified macro-F1 (mean $\\pm$ s.d. over seeds).", "tab:ablation")
    # FAR / DR / MCC / cost table for main models
    lines = []
    for m in order:
        for d in doms:
            if (m, d) not in mu.index:
                continue
            r = mu.loc[(m, d)]
            par = "--" if np.isnan(r.params) else f"{r.params / 1e3:.0f}k"
            lines.append(f"{m} & {d} & {r.far:.3f} & {r.dr:.3f} & {r.mcc:.3f} & {r.latency_us:.1f} & {par} \\\\")
    tex_table(os.path.join(out, "tab_ops.tex"), "False alarm rate, detection rate, MCC, per-record batch latency, and model size (means over seeds).",
              "tab:ops", ["Model", "Domain", "FAR", "DR", "MCC", "Lat. ($\\mu$s)", "Params"], lines, star=True)
    # statistics: HetIDS vs each baseline over (domain, seed)
    st = []
    pv = df.pivot_table(index=["domain", "seed"], columns="model", values="u_macro_f1")
    raw = []
    for m in list(dict.fromkeys(order[:-1] + abl[1:])):
        if m in pv.columns:
            x = pv[["HetIDS-X (proposed)", m]].dropna()
            try:
                p = wilcoxon(x.iloc[:, 0], x.iloc[:, 1]).pvalue
            except ValueError:
                p = 1.0
            raw.append(p); st.append(dict(baseline=m, mean_diff=(x.iloc[:, 0] - x.iloc[:, 1]).mean(),
                                          wins=int((x.iloc[:, 0] > x.iloc[:, 1]).sum()), pairs=len(x), p=p))
    if st:
        adj = B.holm(raw)
        for s_, a_ in zip(st, adj):
            s_["p_holm"] = a_
    pd.DataFrame(st).to_csv(os.path.join(out, "stats.csv"), index=False)
    # few-shot
    if len(fs):
        f = fs.groupby(["target", "fraction", "method"])["u_macro_f1"].agg(["mean", "std"]).reset_index()
        f.to_csv(os.path.join(out, "fewshot_summary.csv"), index=False)
        lines = []
        for (t, fr), gg in f.groupby(["target", "fraction"]):
            vals = {r.method: f"{r['mean']:.3f} $\\pm$ {r['std'] if not np.isnan(r['std']) else 0:.3f}" for _, r in gg.iterrows()}
            n = fs[(fs.target == t) & (fs.fraction == fr)].n_labelled.iloc[0]
            lines.append(f"{t} & {fr * 100:.0f}\\% ({n}) & {vals.get('Scratch (same architecture)', '--')} & "
                         f"{vals.get('HetIDS transfer (frozen backbone)', '--')} & {vals.get('HetIDS transfer (full fine-tuning)', '--')} \\\\")
        tex_table(os.path.join(out, "tab_fewshot.tex"), "Onboarding an unseen domain with few labelled records (unified macro-F1, mean $\\pm$ s.d.): training from scratch versus transferring HetIDS pre-trained on the other domains, either with the shared backbone frozen or with full fine-tuning.",
                  "tab:fewshot", ["Target domain", "Labelled (n)", "Scratch", "Transfer (frozen)", "Transfer (fine-tuned)"], lines)
        fig, axes = plt.subplots(1, len(doms), figsize=(3.3 * len(doms), 2.8), sharey=True)
        for ax, t in zip(np.atleast_1d(axes), doms):
            for mth, mk in [("Scratch (same architecture)", "o--"), ("HetIDS transfer (frozen backbone)", "s:"),
                            ("HetIDS transfer (full fine-tuning)", "^-")]:
                gg = f[(f.target == t) & (f.method == mth)]
                ax.errorbar(gg.fraction * 100, gg["mean"], yerr=gg["std"].fillna(0), fmt=mk, capsize=3,
                            label=mth.replace("HetIDS transfer ", "Transfer ").replace(" (same architecture)", ""))
            ax.set_title(t, fontsize=9); ax.set_xlabel("Labelled target data (%)"); ax.set_xscale("log")
        np.atleast_1d(axes)[0].set_ylabel("Unified macro-F1"); np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_fewshot.pdf")); plt.close(fig)
    # feature-gate importance (seed 0)
    lines = []
    for i, d in enumerate(data):
        G = keep.get(("HetIDS (neural)", i), {}).get("G")
        if G is None:
            continue
        imp = pd.Series(G.mean(0), index=d["feats"]).sort_values(ascending=False).head(8)
        lines.append(f"{d['name']} & " + ", ".join(f"{k.replace('_', chr(92) + '_')} ({v:.2f})" for k, v in imp.items()) + r" \\")
    if lines:
        tex_table(os.path.join(out, "tab_gate.tex"), "Highest mean feature-gate activations of HetIDS per domain (seed 0).",
                  "tab:gate", ["Domain", "Top gated features (mean gate)"], lines)
    # alignment: silhouette by domain and by class on shared embedding (seed 0)
    rng = np.random.default_rng(0); al = []
    emb = {}
    for name in ["HetIDS (neural)", "HetIDS + adversarial alignment"]:
        Zs, ds_, cs = [], [], []
        for i, d in enumerate(data):
            k = keep.get((name, i))
            if k is None:
                continue
            idx = rng.choice(len(k["Z"]), min(1500, len(k["Z"])), replace=False)
            Zs.append(k["Z"][idx]); ds_ += [i] * len(idx); cs.append(d["yu_te"][idx])
        if not Zs:
            continue
        Z, ds_, cs = np.vstack(Zs), np.array(ds_), np.concatenate(cs)
        emb[name] = (Z, ds_, cs)
        al.append(dict(model=name, silhouette_domain=silhouette_score(Z, ds_), silhouette_class=silhouette_score(Z, cs)))
    if al:
        pd.DataFrame(al).to_csv(os.path.join(out, "alignment.csv"), index=False)
        tex_table(os.path.join(out, "tab_align.tex"), "Silhouette of the shared embedding grouped by domain (lower = better aligned across domains) and by unified attack class (higher = better separated).",
                  "tab:align", ["Model", "Silhouette (domain)", "Silhouette (class)"],
                  [f"{r['model']} & {r['silhouette_domain']:.3f} & {r['silhouette_class']:.3f} \\\\" for r in al])
    if "HetIDS (neural)" in emb:
        Z, ds_, cs = emb["HetIDS (neural)"]
        E = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(Z)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        for i, d in enumerate(doms):
            ax[0].scatter(E[ds_ == i, 0], E[ds_ == i, 1], s=3, label=d)
        for c in np.unique(cs):
            ax[1].scatter(E[cs == c, 0], E[cs == c, 1], s=3, label=UNIFIED[c])
        ax[0].set_title("Coloured by domain", fontsize=9); ax[1].set_title("Coloured by unified attack class", fontsize=9)
        for x in ax:
            x.set_xticks([]); x.set_yticks([]); x.legend(fontsize=6, markerscale=3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_tsne.pdf")); plt.close(fig)
    # confusion matrices (seed 0)
    fig, axes = plt.subplots(1, len(doms), figsize=(4 * len(doms), 3.8))
    for ax, (i, d) in zip(np.atleast_1d(axes), enumerate(data)):
        k = keep.get(("HetIDS-X (proposed)", i))
        if k is None:
            continue
        labs = np.unique(np.concatenate([d["yu_te"], k["up"]]))
        cm = confusion_matrix(d["yu_te"], k["up"], labels=labs, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels([UNIFIED[l][:10] for l in labs], rotation=60, fontsize=6)
        ax.set_yticklabels([UNIFIED[l][:10] for l in labs], fontsize=6)
        for r_ in range(len(labs)):
            for c_ in range(len(labs)):
                ax.text(c_, r_, f"{cm[r_, c_]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if cm[r_, c_] > 0.5 else "black")
        ax.set_title(d["name"], fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_confusion.pdf")); plt.close(fig)
    # per-class F1: HetIDS vs per-domain MLP (seed-averaged over test with seed 0 predictions only)
    pcs = []
    for i, d in enumerate(data):
        k = keep.get(("HetIDS-X (proposed)", i))
        if k is None:
            continue
        for c in np.unique(d["yu_te"]):
            pcs.append(dict(domain=d["name"], cls=UNIFIED[c], n_test=int((d["yu_te"] == c).sum()),
                            f1=f1_score(d["yu_te"] == c, k["up"] == c)))
    pd.DataFrame(pcs).to_csv(os.path.join(out, "perclass_hetids.csv"), index=False)


if __name__ == "__main__":
    main()
