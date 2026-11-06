"""Симуляция счёта с параметрами: Donchian-пробой, риск на сделку.

argv: csv risk_pct stop_k hold_days donchian_n [tf_rule]
Пример: sim_account.py data/raw/EURUSD_PERIOD_D1.csv 1 2.0 20 20
"""

import sys

import numpy as np
import pandas as pd

import engine


def run(path, risk, stop_k, hold, don_n, rule=None, seed=None, boot=False):
    if "PERIOD" in path:
        df = engine.load_mt5_csv(path)
    else:
        df = engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    if boot:
        o, h, l, c = engine.block_bootstrap_ohlc(o, h, l, c, block=100,
                                                 seed=seed)
    dt = df.index

    atr = pd.Series(
        np.maximum(h[1:] - l[1:],
                   np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    ).rolling(20).mean().to_numpy()
    atr = np.concatenate([[np.nan], atr])
    hh = pd.Series(h).rolling(don_n).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(don_n).min().shift(1).to_numpy()

    start_i = int(np.searchsorted(dt, dt[-1] - pd.Timedelta(days=365)))
    equity = 1000.0
    pos = None
    peak = equity
    maxdd = 0.0
    n_trades = 0
    wins = 0
    for i in range(start_i, len(c)):
        if pos is not None:
            d, e, s, size, i0 = pos
            exit_px = None
            if (d > 0 and l[i] <= s) or (d < 0 and h[i] >= s):
                exit_px = s
            elif (d > 0 and c[i] < ll[i]) or (d < 0 and c[i] > hh[i]):
                exit_px = c[i]
            elif i - i0 >= hold:
                exit_px = c[i]
            if exit_px is not None:
                cost = 1.5e-4 * (e * size + exit_px * size)
                equity += d * (exit_px - e) * size - cost
                n_trades += 1
                wins += int(d * (exit_px - e) > 0)
                pos = None
        if pos is None and np.isfinite(atr[i]):
            if c[i] > hh[i]:
                pos = (1.0, c[i], c[i] - stop_k * atr[i],
                       equity * risk / (stop_k * atr[i]), i)
            elif c[i] < ll[i]:
                pos = (-1.0, c[i], c[i] + stop_k * atr[i],
                       equity * risk / (stop_k * atr[i]), i)
        peak = max(peak, equity)
        maxdd = min(maxdd, equity / peak - 1)
    return {"final": equity, "ret_pct": (equity / 1000 - 1) * 100,
            "maxdd_pct": maxdd * 100, "n": n_trades,
            "winrate": wins / n_trades if n_trades else np.nan}


if __name__ == "__main__":
    path = sys.argv[1]
    risk = float(sys.argv[2]) / 100
    stop_k = float(sys.argv[3])
    hold = int(sys.argv[4])
    don_n = int(sys.argv[5])
    rule = sys.argv[6] if len(sys.argv) > 6 else None
    r = run(path, risk, stop_k, hold, don_n, rule)
    print(f"{path.split('/')[-1]} риск={risk*100:.0f}% | "
          f"сделок {r['n']} | винрейт {r['winrate']:.2f} | "
          f"год: {r['ret_pct']:+.1f}% | макс.DD {r['maxdd_pct']:.1f}%")
