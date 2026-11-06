"""Dealing Range: диапазон между последними подтверждёнными структурными
экстремумами. Проверки:
1) Снятие одной границы -> вероятность достижения противоположной за K.
2) FVG-зоны в дисконте (бычьи ниже 50% диапазона) против премии:
   отскок и движение к противоположной границе.
Всё каузально (swings.py), нулл — блочный бутстреп.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod


def _last_confirmed(h, l, c, k=3.0):
    _, px, d, conf = swmod.causal_swings(h, l, c, k=k)
    order = np.argsort(conf)
    conf_s, px_s, dir_s = conf[order], px[order], d[order]
    n = len(c)
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    j = 0
    chi, clo = np.nan, np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            if dir_s[j] > 0:
                chi = px_s[j]
            else:
                clo = px_s[j]
            j += 1
        last_hi[t], last_lo[t] = chi, clo
    return last_hi, last_lo


def range_traversal(o, h, l, c, K=96, k_swing=3.0):
    """Касание/снятие границы диапазона -> достижение противоположной."""
    n = len(c)
    last_hi, last_lo = _last_confirmed(h, l, c, k_swing)
    res = {"sweep_hi": [0, 0], "sweep_lo": [0, 0]}  # [n, reached_opposite]
    touched_hi = False
    touched_lo = False
    t_touch = 0
    for t in range(n):
        hi_, lo_ = last_hi[t], last_lo[t]
        if not (np.isfinite(hi_) and np.isfinite(lo_)) or hi_ <= lo_:
            continue
        if not touched_hi and h[t] >= hi_:
            touched_hi, t_touch = True, t
            continue
        if not touched_lo and l[t] <= lo_:
            touched_lo, t_touch = True, t
            continue
        if touched_hi and (t - t_touch) <= K and l[t] <= lo_:
            res["sweep_hi"][0] += 1
            res["sweep_hi"][1] += 1
            touched_hi = False
            continue
        if touched_lo and (t - t_touch) <= K and h[t] >= hi_:
            res["sweep_lo"][0] += 1
            res["sweep_lo"][1] += 1
            touched_lo = False
            continue
        if touched_hi and (t - t_touch) > K:
            res["sweep_hi"][0] += 1
            touched_hi = False
        if touched_lo and (t - t_touch) > K:
            res["sweep_lo"][0] += 1
            touched_lo = False
    return {k: (v[1] / v[0] if v[0] else np.nan, v[0])
            for k, v in res.items()}


def fvg_in_range(o, h, l, c, K=96, M=48, k_swing=3.0):
    """Бычий FVG в дисконте диапазона / медвежий в премии:
    отскок и достижение противоположной границы."""
    n = len(c)
    last_hi, last_lo = _last_confirmed(h, l, c, k_swing)
    out = {"good": [0, 0], "bad": [0, 0]}   # [n, bounce]
    reach = {"good": [0, 0], "bad": [0, 0]}  # [n, reached far boundary]
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:
            d, bottom, top = 1, h[t - 1], l[t + 1]
        elif h[t + 1] < l[t - 1]:
            d, bottom, top = -1, h[t + 1], l[t - 1]
        else:
            continue
        born = t + 1
        hi_, lo_ = last_hi[born], last_lo[born]
        if not (np.isfinite(hi_) and np.isfinite(lo_)) or hi_ <= lo_:
            continue
        mid = 0.5 * (top + bottom)
        pos = (mid - lo_) / (hi_ - lo_)
        if not (0 <= pos <= 1):
            continue                      # зона вне диапазона
        good = (d > 0 and pos < 0.5) or (d < 0 and pos > 0.5)
        grp = "good" if good else "bad"
        start = born + 1
        end = min(start + K, n)
        tm = (l[start:end] <= top) if d > 0 else (h[start:end] >= bottom)
        if not tm.any():
            continue
        tt = start + int(np.argmax(tm))
        s2, e2 = tt, min(tt + M, n)
        if d > 0:
            fm, bm = l[s2:e2] <= bottom, c[s2:e2] > top
        else:
            fm, bm = h[s2:e2] >= top, c[s2:e2] < bottom
        fb = np.argmax(fm) if fm.any() else M + 1
        bb = np.argmax(bm) if bm.any() else M + 1
        if fb > M and bb > M:
            continue
        out[grp][0] += 1
        out[grp][1] += int(bb < fb)
        # движение к противоположной границе после касания
        s3, e3 = tt, min(tt + K, n)
        reached = ((h[s3:e3] >= hi_).any() if d > 0 else (l[s3:e3] <= lo_).any())
        reach[grp][0] += 1
        reach[grp][1] += int(reached)
    bounce = {k: (v[1] / v[0] if v[0] else np.nan, v[0]) for k, v in out.items()}
    reachd = {k: (v[1] / v[0] if v[0] else np.nan, v[0]) for k, v in reach.items()}
    return bounce, reachd
