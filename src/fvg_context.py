"""Контекст рождения FVG-зоны: консолидация / у уровня / экспансия.

Категории на момент рождения зоны (бар t+1, каузально):
- at_level:    |close - ближайший подтверждённый структурный экстремум|
               < 0.75 * ATR(100)
- expansion:   Kaufman ER(20) > 0.5 (и не у уровня)
- consolidation: ER(20) < 0.25 (и не у уровня)
- middle:      остальное
Для каждой категории: n, касание, медиана баров до касания, отскок.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod


def _er(c, W=20):
    net = np.abs(c[W:] - c[:-W])
    path = pd.Series(np.abs(np.diff(c))).rolling(W).sum().to_numpy()[W - 1:]
    er = net / np.where(path > 0, path, np.nan)
    return np.concatenate([np.full(W, np.nan), er])


def _atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def zones_with_context(o, h, l, c, K=96, M=48):
    n = len(c)
    er = _er(c)
    a = _atr(h, l, c)
    _, px_all, _, conf_all = swmod.causal_swings(h, l, c, k=3.0)
    order = np.argsort(conf_all)
    conf_s, px_s = conf_all[order], px_all[order]

    # для каждого бара — последний подтверждённый уровень (любой стороны)
    last_px = np.full(n, np.nan)
    j = 0
    cur = np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            cur = px_s[j]
            j += 1
        last_px[t] = cur

    out = []
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:
            d, bottom, top = 1, h[t - 1], l[t + 1]
        elif h[t + 1] < l[t - 1]:
            d, bottom, top = -1, h[t + 1], l[t - 1]
        else:
            continue
        born = t + 1
        if not np.isfinite(er[born]) or not np.isfinite(a[born]) or a[born] <= 0:
            continue
        dist_lvl = (abs(c[born] - last_px[born]) / a[born]
                    if np.isfinite(last_px[born]) else np.inf)
        if dist_lvl < 0.75:
            cat = "at_level"
        elif er[born] > 0.5:
            cat = "expansion"
        elif er[born] < 0.25:
            cat = "consolidation"
        else:
            cat = "middle"

        start = born + 1
        end = min(start + K, n)
        if start >= n:
            continue
        tm = (l[start:end] <= top) if d > 0 else (h[start:end] >= bottom)
        if not tm.any():
            continue
        t_touch = start + int(np.argmax(tm))
        s2, e2 = t_touch, min(t_touch + M, n)
        if d > 0:
            fm = l[s2:e2] <= bottom
            bm = c[s2:e2] > top
        else:
            fm = h[s2:e2] >= top
            bm = c[s2:e2] < bottom
        fb = np.argmax(fm) if fm.any() else M + 1
        bb = np.argmax(bm) if bm.any() else M + 1
        if fb > M and bb > M:
            continue
        out.append({"cat": cat, "bounced": bool(bb < fb)})
    df = pd.DataFrame(out)
    res = {}
    for cat in ("at_level", "expansion", "consolidation", "middle"):
        sub = df[df.cat == cat]
        if len(sub) < 30:
            continue
        res[cat] = {"n": len(sub), "p_bounce": float(sub.bounced.mean())}
    return res
