"""Прогон T10 SD-проекций. out/t10.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import sd_projections as sp
import sessions

B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SETS = [("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", 60), ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", 60),
        ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", 60), ("USTEC_H1", "data/raw/USTEC_PERIOD_H1.csv", 60),
        ("EURUSD_M5", "data/raw/EURUSD_PERIOD_M5.csv", 5), ("GBPUSD_M5", "data/raw/GBPUSD_PERIOD_M5.csv", 5),
        ("XAUUSD_M5", "data/raw/XAUUSD_PERIOD_M5.csv", 5)]
rows = []
for name, path, mb in SETS:
    df = sessions.to_et(engine.load_mt5_csv(path))
    real, z = sp.sd_stats(df, mb, B=B)
    print(f"\n===== {name}: дней {int(real['cbdr_n'])}")
    for ref in ("cbdr", "cbdrPOS", "asian", "asianPOS", "float", "floatPOS"):
        for sub in ("all", "narrow", "wide"):
            if f"{ref}_{sub}_n" not in real:
                continue
            print(f"  {ref:5s} {sub:6s} n={int(real[f'{ref}_{sub}_n']):4d} | " + " ".join(
                f"{m}={z[f'{ref}_{sub}_{m}']['real']:.3f}/{z[f'{ref}_{sub}_{m}']['null_mean']:.3f}(z{z[f'{ref}_{sub}_{m}']['z']:+.1f})"
                for m in ("median_d", "p_inside", "p_0_2", "p_0_3", "p_gt3", "p_on_integer", "p_london_0_2") if f"{ref}_{sub}_{m}" in z))
    print(f"  pivot n={int(real['pivot_n'])} | " + " ".join(
        f"{m}={z[f'pivot_{m}']['real']:.3f}/{z[f'pivot_{m}']['null_mean']:.3f}(z{z[f'pivot_{m}']['z']:+.1f})"
        for m in ("mean_count", "p_count_eq4", "p_count_3_5", "p_ext_within_5pct", "p_ext_within_10pct")))
    for k, r in z.items():
        rows.append({"market": name, "metric": k, **r})
pd.DataFrame(rows).to_csv("out/t10.csv", index=False)
print("\nзаписано out/t10.csv")
