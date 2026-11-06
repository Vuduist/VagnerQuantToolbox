"""Портфельный тренд-фолловинг: Donchian(20) D1 на корзине рынков.

Капитал делится на равные sleeve по рынкам, внутри каждого — риск 1%
на дистанцию стопа. Портфель = сумма sleeve-кривых (ежедневный MTM).
Сравнение одиночных рынков и портфеля на общей истории.
"""

import sys

import numpy as np
import pandas as pd

import engine

RISK = 0.01
STOP_K = 2.0
HOLD = 20
DON = 20


def equity_curve(df, cost, start_dt=None):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    dt = df.index
    atr = pd.Series(
        np.maximum(h[1:] - l[1:],
                   np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    ).rolling(20).mean().to_numpy()
    atr = np.concatenate([[np.nan], atr])
    hh = pd.Series(h).rolling(DON).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(DON).min().shift(1).to_numpy()
    cash = 1000.0
    pos = None
    eq = np.full(len(c), np.nan)
    for i in range(len(c)):
        cur = cash
        if pos is not None:
            d, e, s, size, i0 = pos
            cur = cash + d * (c[i] - e) * size
            exit_px = None
            if (d > 0 and l[i] <= s) or (d < 0 and h[i] >= s):
                exit_px = s
            elif (d > 0 and c[i] < ll[i]) or (d < 0 and c[i] > hh[i]):
                exit_px = c[i]
            elif i - i0 >= HOLD:
                exit_px = c[i]
            if exit_px is not None:
                cash += d * (exit_px - e) * size - cost * (e * size + exit_px * size)
                pos = None
                cur = cash
        if pos is None and np.isfinite(atr[i]) and np.isfinite(hh[i]):
            if c[i] > hh[i]:
                pos = (1.0, c[i], c[i] - STOP_K * atr[i],
                       cash * RISK / (STOP_K * atr[i]), i)
            elif c[i] < ll[i]:
                pos = (-1.0, c[i], c[i] + STOP_K * atr[i],
                       cash * RISK / (STOP_K * atr[i]), i)
        eq[i] = cur
    s = pd.Series(eq, index=dt).ffill()
    if start_dt is not None:
        s = s[s.index >= start_dt]
        s = s / s.iloc[0] * 1000.0
    return s


def stats(eq: pd.Series):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    return cagr * 100, dd * 100, sh, (cagr * 100) / abs(dd * 100) if dd else np.nan


SETS4 = [
    ("EURUSD", "data/raw/EURUSD_PERIOD_D1.csv", 1.5e-4),
    ("GBPUSD", "data/raw/GBPUSD_PERIOD_D1.csv", 1.5e-4),
    ("XAUUSD", "data/raw/XAUUSD_PERIOD_D1.csv", 1.0e-4),
    ("NQ",     "data/raw/NQ_1d.csv",            1.0e-4),
]

print("=== Портфель из 4 рынков (FX2 + золото + индекс), общая история с 2004 ===")
curves = {}
for name, path, cost in SETS4:
    df = engine.load_mt5_csv(path) if "PERIOD" in path else engine.load_csv(path)
    curves[name] = equity_curve(df, cost, start_dt="2005-01-01")

pf = pd.DataFrame(curves).ffill().dropna()
pf_eq = pf.mean(axis=1)  # равные sleeve
hdr = f"{'':10s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>7s} {'Calmar':>7s}"
print(hdr)
for name in curves:
    c, d, s, cl = stats(pf[name])
    print(f"{name:10s} {c:6.1f}% {d:6.1f}% {s:7.2f} {cl:7.2f}")
c, d, s, cl = stats(pf_eq)
print(f"{'ПОРТФЕЛЬ':10s} {c:6.1f}% {d:6.1f}% {s:7.2f} {cl:7.2f}")

print("\n=== + BTC (с 2021) ===")
btc = engine.resample(engine.load_csv("data/raw/BTCUSDT_1m.csv"), "1D")
curves5 = dict(curves)
curves5["BTC"] = equity_curve(btc, 10e-4, start_dt="2021-01-01")
pf5 = pd.DataFrame({k: v for k, v in curves5.items()}).ffill().dropna()
pf5 = pf5[pf5.index >= "2021-01-01"]
hdr = f"{'':10s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>7s} {'Calmar':>7s}"
print(hdr)
for name in pf5.columns:
    c, d, s, cl = stats(pf5[name])
    print(f"{name:10s} {c:6.1f}% {d:6.1f}% {s:7.2f} {cl:7.2f}")
c, d, s, cl = stats(pf5.mean(axis=1))
print(f"{'ПОРТФЕЛЬ5':10s} {c:6.1f}% {d:6.1f}% {s:7.2f} {cl:7.2f}")

pf_eq.to_csv("out/portfolio_eq.csv")
