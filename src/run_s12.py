"""Прогон S12 трендовых линий. out/s12.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import trendlines as tl

B = int(sys.argv[1]) if len(sys.argv) > 1 else 50
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None

SETS = [("EURUSD_H1", "mt5", "data/raw/EURUSD_PERIOD_H1.csv", None),
        ("GBPUSD_H1", "mt5", "data/raw/GBPUSD_PERIOD_H1.csv", None),
        ("XAUUSD_H1", "mt5", "data/raw/XAUUSD_PERIOD_H1.csv", None),
        ("USTEC_H1", "mt5", "data/raw/USTEC_PERIOD_H1.csv", None),
        ("NQ_1h", "csv", "data/raw/NQ_1h.csv", None),
        ("ES_1h", "csv", "data/raw/ES_1h.csv", None),
        ("EURUSD_M15", "mt5", "data/raw/EURUSD_PERIOD_M5.csv", "15min"),
        ("GBPUSD_M15", "mt5", "data/raw/GBPUSD_PERIOD_M5.csv", "15min"),
        ("XAUUSD_M15", "mt5", "data/raw/XAUUSD_PERIOD_M5.csv", "15min"),
        ("USTEC_M15", "mt5", "data/raw/USTEC_PERIOD_M5.csv", "15min")]
KEYS = tl.METRICS
rows = []
for name, kind, path, rule in SETS:
    if ONLY and name not in ONLY:
        continue
    df = engine.load_mt5_csv(path) if kind == "mt5" else engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    bpd = int(round(len(df) / df.index.normalize().nunique()))
    real, z = tl.s12_stats(df, bpd, B=B, R=R)
    print(f"\n===== {name}: баров {len(df)}, bpd {bpd}, линий {int(real['n_lines'])}, касаний {int(real['n_touch'])} "
          f"(up {int(real.get('up_n_touch', 0))}, dn {int(real.get('dn_n_touch', 0))}, aligned {int(real.get('al_n_touch', 0))})")
    for pref in ("", "al_"):
        for k in KEYS:
            kk = pref + k
            if kk not in z:
                continue
            r = z[kk]
            s = f"  {kk:18s} real={r['real']:+.3f}"
            for nm in ("n1", "n2"):
                if nm + "_mean" in r:
                    s += f" | {nm}={r[nm + '_mean']:+.3f}±{r[nm + '_sd']:.3f} z{r[nm + '_z']:+.1f}"
            print(s)
    for k, r in z.items():
        rows.append({"market": name, "metric": k, **r})
    pd.DataFrame(rows).to_csv("out/s12.csv", index=False)
print("\nзаписано out/s12.csv")
