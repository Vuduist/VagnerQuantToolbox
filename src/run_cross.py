"""Прогон межрыночных тестов по всем доступным парам."""

import sys

import numpy as np
import pandas as pd

import engine
import cross

PAIRS = [
    ("EURUSD/GBPUSD H1", "data/raw/EURUSD_PERIOD_H1.csv", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5"),
    ("EURUSD/GBPUSD D1", "data/raw/EURUSD_PERIOD_D1.csv", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5"),
    ("NQ/ES D1",         "data/raw/NQ_1d.csv",            "data/raw/ES_1d.csv",            "y"),
    ("XAU/GC D1",        "data/raw/XAUUSD_PERIOD_D1.csv", "data/raw/GC_1d.csv",            "y"),
    ("USTEC/NQ H1 (кросс-лента)", "data/raw/USTEC_PERIOD_H1.csv", "data/raw/NQ_1h.csv", "y"),
]


def load(path, fmt):
    if fmt == "mt5":
        return engine.load_mt5_csv(path)
    return engine.load_csv(path)


for name, pa, pb, fmt in PAIRS:
    j = cross.align(load(pa, fmt), load(pb, fmt))
    if len(j) < 2000:
        print(f"{name}: мало общих баров ({len(j)})")
        continue
    cA, cB = j["close_a"].to_numpy(), j["close_b"].to_numpy()
    hA, lA = j["high_a"].to_numpy(), j["low_a"].to_numpy()
    hB, lB = j["high_b"].to_numpy(), j["low_b"].to_numpy()
    print(f"\n===== {name} ({len(j)} общих баров, {j.index[0].date()}..{j.index[-1].date()}) =====")
    ll = cross.leadlag(cA, cB)
    best = max(ll, key=lambda k: abs(ll[k]))
    print(f"  лид-лаг: максимум corr={ll[best]:.3f} при k={best:+d} (k>0: B ведёт)")
    idx, d = cross.smt_events(hA, lA, cA, hB, lB, cB)
    print(f"  SMT-дивергенций: {len(idx)}")
    st = cross.smt_forward_stats(cA, cB, idx, d)
    for H, v in st.items():
        print(f"    h={H:3d}: A p_win={v['A_p_win']:.3f} mean={v['A_mean_bp']:+6.1f}бп | "
              f"B p_win={v['B_p_win']:.3f} mean={v['B_mean_bp']:+6.1f}бп (n={v['n']})")
    sr = cross.spread_reversion(cA, cB)
    if sr:
        print(f"  спред-реверсия (z>2, h=24): n={sr['n']} mean={sr['mean_bp']:+.1f}бп p_win={sr['p_win']:.3f}")
