"""Вайкофф на дневках с объёмом (18 фьючерсов Yahoo). Доходности в единицах ATR20, пул; нулл — перестановка объёма внутри
инструмента-года (контроль: то же событие без условия на объём)."""
import sys
import numpy as np
import pandas as pd
import engine

rng = np.random.default_rng(1)
SYMS = ["ES", "NQ", "ZB", "ZN", "GC", "SI", "HG", "CL", "NG", "ZC", "ZS", "ZW", "KC", "SB", "CT", "CC", "LE", "HE"]
rows = []
for s in SYMS:
    D = engine.load_csv(f"data/raw/{s}_1d.csv"); D.index = D.index.normalize(); D = D[~D.index.duplicated()]
    D = D[(D.volume > 1) & (D.index.weekday < 5)]
    if len(D) < 500: continue
    tr = pd.concat([D.high - D.low, (D.high - D.close.shift(1)).abs(), (D.low - D.close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().shift(1); vavg = D.volume.rolling(20).mean().shift(1)
    r = D.close.diff(); fwd5 = (D.close.shift(-5) - D.close) / atr; back5 = (D.close - D.close.shift(5)) / atr
    vr = D.volume / vavg; rr = (D.high - D.low) / atr
    lo20 = D.low.rolling(20).min().shift(1); hi20 = D.high.rolling(20).max().shift(1)
    df = pd.DataFrame({"sym": s, "year": D.index.year, "fwd5": fwd5, "back5": back5, "vr": vr, "rr": rr, "sign": np.sign(r),
                       "break_lo": D.close < lo20, "break_hi": D.close > hi20, "up3": (r > 0) & (r.shift(1) > 0) & (r.shift(2) > 0) & (r < r.shift(1)) & (r.shift(1) < r.shift(2))}, index=D.index).dropna()
    rows.append(df)
A = pd.concat(rows)
st = lambda x: f"{x.mean():+.3f} ATR (t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}, n {len(x)})"
def perm(ev_mask, cond_vol, target, n=500):
    """разница target[cond_vol] − target[~cond_vol] среди событий ev_mask против перестановки метки объёма внутри sym-year."""
    E = A[ev_mask]; x = target[ev_mask]; c = cond_vol[ev_mask].values
    obs = x[c].mean() - x[~c].mean(); null = []
    g = (E.sym + E.year.astype(str)).values
    for _ in range(n):
        c2 = c.copy()
        for k in np.unique(g):
            m = g == k; c2[m] = rng.permutation(c[m])
        null.append(x[c2].mean() - x[~c2].mean())
    return obs, np.mean(np.abs(null) >= abs(obs))
halves = lambda x: f"до 2013 {x[x.index < '2013'].mean():+.3f}, с 2013 {x[x.index >= '2013'].mean():+.3f}"
# (а) климакс
ev = (A.rr > 2) ; vol = A.vr > 2
y = -A.sign * A.fwd5                                       # против направления дня
print(f"(а) климакс: диапазон > 2 ATR и объём > 2×: 5 дн. против дня {st(y[ev & vol])}; {halves(y[ev & vol])} | тот же диапазон без объёма {st(y[ev & ~vol])}", file=sys.stderr)
o, p = perm(ev, vol, y); print(f"    объём добавляет {o:+.3f} ATR, перестановочный p {p:.3f}", file=sys.stderr)
# (б) spring
ev = A.break_lo; low = A.vr < 0.8; high = A.vr > 1.2
print(f"(б) spring: пробой 20-дн. минимума на объёме < 0.8× → 5 дн. {st(A.fwd5[ev & low])}; {halves(A.fwd5[ev & low])} | на объёме > 1.2× {st(A.fwd5[ev & high])} | все пробои {st(A.fwd5[ev])}", file=sys.stderr)
o, p = perm(ev, low, A.fwd5); print(f"    низкий объём добавляет {o:+.3f} ATR, p {p:.3f}", file=sys.stderr)
ev = A.break_hi
print(f"    зеркально, upthrust: пробой максимума на объёме < 0.8× → 5 дн. {st(A.fwd5[ev & low])} | > 1.2× {st(A.fwd5[ev & high])}", file=sys.stderr)
# (в) усилие без результата
ev = (A.rr < 0.7); vol = A.vr > 1.5; y = -np.sign(A.back5) * A.fwd5
print(f"(в) усилие без результата: объём > 1.5×, диапазон < 0.7 ATR → разворот 5-дн. тренда {st(y[ev & vol])}; {halves(y[ev & vol])} | узкий день без объёма {st(y[ev & ~vol])}", file=sys.stderr)
o, p = perm(ev, vol, y); print(f"    объём добавляет {o:+.3f} ATR, p {p:.3f}", file=sys.stderr)
# (г) shortening of thrust
ev = A.up3; print(f"(г) shortening of thrust (3 роста с убывающим приростом) → 5 дн. {st(A.fwd5[ev])}; {halves(A.fwd5[ev])} | все дни {st(A.fwd5)}", file=sys.stderr)
