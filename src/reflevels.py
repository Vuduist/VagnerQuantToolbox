"""W03: опорные уровни открытия (weekly open, Friday close, midnight open, monthly open)
как магнит/уровень реакции. H1 в ET.

Событие: от уровня (цена открытия периода) цена ушла на ≥ d·ATR(100), затем впервые
вернулась к нему (бар накрыл уровень) до конца периода. Метрики:
  p_revisit    — доля периодов с уходом, где уровень затем достигнут;
  p_bounce     — при первом возврате close через K баров на той стороне, откуда пришли;
  mean_ret_K_bp — средняя доходность в сторону отбоя через K баров.
Нулл: «псевдопериоды» — те же длины периодов, но старт в случайном баре (календарная
привязка разрушена, механика «уровень стартует на расстоянии 0» сохранена).
Если реал ≡ нулл, open недели/дня/месяца — просто ещё один прошлый уровень.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings


def _periods(idx: pd.DatetimeIndex, kind: str):
    t = idx.tz_localize(None)
    if kind == "week":
        iso = t.isocalendar()
        return (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    if kind == "day":
        return t.normalize().map(pd.Timestamp.toordinal).to_numpy()
    if kind == "month":
        return (t.year * 100 + t.month).to_numpy()
    raise ValueError(kind)


def _run(o, h, l, c, a, starts, ends, level_fn, d, K):
    n_dep = n_rev = n_bounce = 0
    rets = []
    for s, e in zip(starts, ends):
        if s == 0 or e - s < 10 or not np.isfinite(a[s]):
            continue
        L = level_fn(s)
        thr = d * a[s]
        away = np.flatnonzero(np.abs(c[s:e] - L) >= thr)
        if len(away) == 0:
            continue
        t0 = s + away[0]
        side = np.sign(c[t0] - L)
        n_dep += 1
        back = np.flatnonzero((l[t0 + 1:e] <= L) & (h[t0 + 1:e] >= L))
        if len(back) == 0:
            continue
        t1 = t0 + 1 + back[0]
        n_rev += 1
        if t1 + K < len(c):
            r = side * (c[t1 + K] - L) / L * 1e4
            rets.append(r)
            n_bounce += r > 0
    return {"n_dep": float(n_dep), "p_revisit": n_rev / max(n_dep, 1),
            "n_rev": float(n_rev), "p_bounce": n_bounce / max(len(rets), 1),
            "mean_ret_K_bp": float(np.mean(rets)) if rets else np.nan}


def level_test(df_et: pd.DataFrame, kind: str, level: str = "open", d=1.0, K=4,
               B=50, seed=0) -> tuple[dict, dict]:
    """kind: week/day/month; level: 'open' (первый бар периода) или 'prev_close'
    (close последнего бара предыдущего периода, напр. Friday close для недели)."""
    o, h, l, c = (df_et[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = swings.atr(h, l, c, 100)
    per = _periods(df_et.index, kind)
    starts = np.flatnonzero(np.r_[True, per[1:] != per[:-1]])
    ends = np.r_[starts[1:], len(c)]
    lengths = ends - starts
    lf = (lambda s: o[s]) if level == "open" else (lambda s: c[s - 1])
    real = _run(o, h, l, c, a, starts, ends, lf, d, K)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(c)
    for _ in range(B):
        rs = np.sort(rng.integers(1, n - 1, size=len(starts)))
        re = np.minimum(rs + rng.permutation(lengths), n)
        boots.append(_run(o, h, l, c, a, rs, re, lf, d, K))
    z = {}
    for k in ("p_revisit", "p_bounce", "mean_ret_K_bp"):
        vals = np.array([b[k] for b in boots], float)
        vals = vals[np.isfinite(vals)]
        sd = vals.std(ddof=1)
        z[k] = {"real": real[k], "null_mean": vals.mean(), "null_sd": sd,
                "z": (real[k] - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z
