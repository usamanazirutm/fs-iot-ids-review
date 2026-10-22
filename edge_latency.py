#!/usr/bin/env python3
"""Measure single-sample inference latency on an edge device (e.g., Raspberry Pi).

python edge_latency.py --models results/edge --out edge_results.csv
"""
import argparse
import glob
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="results/edge")
ap.add_argument("--out", default="edge_results.csv")
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.models, "*.joblib"))):
    b = joblib.load(f)
    X = np.load(os.path.join(a.models, b["sample"]))[: a.n]
    sel, m = b["selector"], b["model"]
    try:
        m.set_params(n_jobs=1)
    except (ValueError, AttributeError):
        pass

    def run(x):
        z = sel.transform(x) if hasattr(sel, "transform") else x[:, sel]
        return m.predict(z.astype(np.float32))

    for i in range(20):
        run(X[i:i + 1])
    ts = []
    for i in range(len(X)):
        t0 = time.perf_counter()
        run(X[i:i + 1])
        ts.append((time.perf_counter() - t0) * 1e6)
    rows.append(dict(device=platform.machine(), python=platform.python_version(),
                     dataset=b["dataset"], method=b["method"], clf=b["clf"],
                     median_us=np.median(ts), p95_us=np.percentile(ts, 95),
                     model_kb=os.path.getsize(f) / 1024))
    print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"Saved {a.out}")
