"""NDOG/NWOG: гэпы открытия дня/недели как магниты и уровни.

NDOG: open первого бара дня vs close последнего бара прошлого дня.
NWOG: то же для недели (первый бар недели против close пятницы).
Гипотезы ICT: гэп -> магнит (заполнение) и/или уровень (отбой от краёв).
Контроль: дни без гэпа (внутри тех же данных).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def gap_events(df: pd.DataFrame, weekly=False, thr_atr=0.5):
    """Гэпы открытия периода (день/неделя). Возвращает таблицу событий."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = _atr(h, l, c)
    idx = df.index
    if weekly:
        iso = idx.isocalendar()
        per = (iso.year * 100 + iso.week).to_numpy()
    else:
        per = idx.normalize().map(pd.Timestamp.toordinal).to_numpy()
    first = np.zeros(len(idx), dtype=bool)
    first[0] = True
    first[1:] = per[1:] != per[:-1]
    events = []
    for i in np.where(first)[0]:
        if i == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        gap = (o[i] - c[i - 1]) / a[i]
        events.append({"i": i, "gap_atr": gap, "dir": np.sign(gap),
                       "prev_close": c[i - 1]})
    return pd.DataFrame(events)


def fill_stats(df: pd.DataFrame, ev: pd.DataFrame, thr_atr=0.5):
    """NDOG: гэп >= thr*ATR -> вероятность заполнения за день.
    Контроль: дни без гэпа; событие = день прошёл дистанцию G (в ATR,
    из распределения гэп-дней) в случайную сторону от открытия."""
    h, l, o = (df[x].to_numpy() for x in ("high", "low", "open"))
    a = _atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy())
    idx = df.index
    day_ord = idx.normalize().map(pd.Timestamp.toordinal).to_numpy()
    rows = []
    for _, e in ev.iterrows():
        i = int(e.i)
        end = min(int(np.searchsorted(day_ord, day_ord[i] + 1)), len(h))
        g = abs(e.gap_atr)
        if end <= i + 1 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        touched = bool((l[i:end] <= e.prev_close).any()) if e.gap_atr > 0 else \
                  bool((h[i:end] >= e.prev_close).any())
        trav_up = float((h[i:end].max() - o[i]) / a[i])
        trav_dn = float((o[i] - l[i:end].min()) / a[i])
        rows.append({"gap": g, "touched": touched, "trav_up": trav_up,
                     "trav_dn": trav_dn, "has_gap": g >= thr_atr,
                     "no_gap": g < 0.25})
    r = pd.DataFrame(rows)
    out = {}
    gsub, csub = r[r.has_gap], r[r.no_gap]
    if len(gsub) > 20:
        out["gap"] = {"n": len(gsub), "p_fill": float(gsub.touched.mean())}
    if len(csub) > 20 and len(gsub) > 20:
        rng = np.random.default_rng(0)
        gsizes = rng.choice(gsub.gap.to_numpy(), size=len(csub))
        sides = rng.choice([1, -1], size=len(csub))
        trav = np.where(sides > 0, csub.trav_up.to_numpy(),
                        csub.trav_dn.to_numpy())
        out["ctrl"] = {"n": len(csub),
                       "p_travel": float(np.mean(trav >= gsizes))}
    return out, r
