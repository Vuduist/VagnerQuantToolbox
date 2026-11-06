"""Прогон HAR-предсказуемости волатильности по всем рынкам."""

import sys

import numpy as np
import pandas as pd

import engine
from volatility import daily_rv, har_oos, rv_bootstrap

SETS = [
    ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5"),
    ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5"),
    ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5"),
    ("USTEC_H1",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5"),
    ("BTC_1h",    "data/raw/BTCUSDT_1m.csv",       "binance"),
    ("EURUSD_D",  "data/raw/EURUSD_PERIOD_D1.csv", "mt5"),
    ("GBPUSD_D",  "data/raw/GBPUSD_PERIOD_D1.csv", "mt5"),
    ("XAUUSD_D",  "data/raw/XAUUSD_PERIOD_D1.csv", "mt5"),
    ("NQ_D",      "data/raw/NQ_1d.csv",            "binance"),
]

B_NULL = 30

for name, path, fmt in SETS:
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    if name == "BTC_1h":
        df = engine.resample(df, "1h")
    rv = daily_rv(df)
    if len(rv) < 800:
        print(f"{name}: мало дней ({len(rv)})")
        continue
    real = har_oos(rv)

    # устойчивость по годам
    yrs = pd.Series(real["idx"].year, index=real["idx"])
    per_year = {}
    for y in sorted(yrs.unique()):
        m = (yrs == y).to_numpy()
        if m.sum() < 100:
            continue
        a, p = real["actual"][m], real["preds"][m]
        r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
        per_year[y] = r2

    # нулл
    nulls = []
    for b in range(B_NULL):
        r0 = har_oos(rv_bootstrap(rv, block=30, seed=9000 + b))
        nulls.append(r0["R2_vs_mean"])
    nulls = np.array(nulls)
    z = (real["R2_vs_mean"] - nulls.mean()) / nulls.std()

    pos = np.mean([v > 0 for v in per_year.values()])
    print(f"{name:10s} дней={len(rv):5d} | R2_oos={real['R2_vs_mean']:.3f} "
          f"(vs наивный {real['R2_vs_naive']:+.3f}) corr={real['corr']:.3f} | "
          f"нулл R2={nulls.mean():.3f}±{nulls.std():.3f} z={z:+.1f} | "
          f"лет в плюсе: {pos:.0%}")
    print("   по годам:", " ".join(f"{y}:{v:+.2f}" for y, v in per_year.items()))
