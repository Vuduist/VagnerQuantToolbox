"""Анатомия движения цены: поиск эпох-инвариантов (последние 15-55 лет).

Никаких сетапов. Только измерения структуры пути цены:
Hurst/вариационное отношение, серии, растяжение-откат, возвраты к уровням,
эффективность пути (Kaufman ER), автокорреляции. Каждая метрика — по эпохам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rets(c):
    return np.diff(np.log(c))


def variance_ratio(c, lags=(2, 4, 8, 16, 32, 64)):
    """VR(q) = Var(r_q) / (q * Var(r_1)). 1.0 = блуждание;
    >1 — трендовость на горизонте q, <1 — реверсия."""
    r = _rets(c)
    v1 = np.var(r)
    out = {}
    for q in lags:
        rq = np.log(c[q:] / c[:-q])
        out[q] = float(np.var(rq) / (q * v1))
    return out


def streak_survival(c, max_k=10):
    """Доля серий длины >= k среди всех серий знака доходности.
    Для монетки: 2^-(k-1)."""
    r = _rets(c)
    s = np.sign(r)
    s = s[s != 0]
    change = np.concatenate([[True], s[1:] != s[:-1]])
    starts = np.where(change)[0]
    lens = np.diff(np.concatenate([starts, [len(s)]]))
    out = {}
    for k in range(1, max_k + 1):
        out[k] = float(np.mean(lens >= k))
    return out


def stretch_snap(o, h, l, c, W=5, buckets=(1, 2, 3, 4)):
    """После движения за W баров в m*ATR единиц без сильного отката:
    вероятность продолжения на следующие W баров (по знаку)."""
    a = atr(h, l, c, 100)
    move = (c[W:] - c[:-W]) / np.where(a[W:] > 0, a[W:], np.nan)
    fwd = (c[2 * W:] - c[W:-W]) / np.where(a[W:-W] > 0, a[W:-W], np.nan)
    move = move[:len(fwd)]
    out = {}
    for m0 in buckets:
        sel = np.abs(move) >= m0
        if sel.sum() < 100:
            continue
        cont = np.sign(fwd[sel]) == np.sign(move[sel])
        out[m0] = {"n": int(sel.sum()), "p_cont": float(np.mean(cont))}
    return out


def revisit_rate(h, l, c, K=100, grid_atr=1.0):
    """Цена проходит уровень (сетка в ATR) — вероятность повторного
    касания в течение K баров. 'Рынок проходит по уровню не раз'."""
    a = atr(h, l, c, 100)
    n = len(c)
    grid = np.round(c / np.where(a > 0, a, np.nan) / grid_atr)
    # событие: смена ячейки сетки
    cross = np.concatenate([[True], grid[1:] != grid[:-1]])
    idx = np.where(cross & np.isfinite(grid))[0]
    idx = idx[idx + K < n]
    if len(idx) < 200:
        return None
    # для подвыборки событий: возврат в ту же ячейку
    sel = idx[::max(1, len(idx) // 4000)]
    hits = []
    for t in sel:
        future = grid[t + 1:t + K + 1]
        hits.append(np.any(future == grid[t]))
    return float(np.mean(hits))


def efficiency_ratio(c, W=20):
    """Kaufman ER = |нетто-движение| / длина пути за W баров."""
    net = np.abs(c[W:] - c[:-W])
    path = np.sum(np.abs(np.diff(c)), axis=0)  # placeholder
    diffs = np.abs(np.diff(c))
    path = pd.Series(diffs).rolling(W).sum().to_numpy()[W - 1:]
    er = net / np.where(path > 0, path, np.nan)
    er = er[np.isfinite(er)]
    return {"median": float(np.median(er)), "q25": float(np.quantile(er, 0.25)),
            "q75": float(np.quantile(er, 0.75))}


def acf_profile(c, lags=(1, 2, 5, 10, 20)):
    """Автокорреляция знаковых доходностей и |доходностей|."""
    r = _rets(c)
    out = {}
    for q in lags:
        out[q] = (float(np.corrcoef(r[q:], r[:-q])[0, 1]),
                  float(np.corrcoef(np.abs(r[q:]), np.abs(r[:-q]))[0, 1]))
    return out


def atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


ERAS = [(1971, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2030)]


def anatomy(df: pd.DataFrame, name: str):
    rows = []
    years = df.index.year.to_numpy()
    for y0, y1 in ERAS:
        sub = df[(years >= y0) & (years < y1)]
        if len(sub) < 1500:
            continue
        o, h, l, c = (sub[x].to_numpy() for x in ("open", "high", "low", "close"))
        vr = variance_ratio(c)
        st = streak_survival(c)
        ss = stretch_snap(o, h, l, c)
        rv = revisit_rate(h, l, c)
        er = efficiency_ratio(c)
        acf = acf_profile(c)
        rows.append({
            "market": name, "era": f"{y0}-{y1}", "n": len(sub),
            "VR2": vr[2], "VR8": vr[8], "VR32": vr[32],
            "streak3": st[3], "streak5": st[5],
            "stretch2_p_cont": ss.get(2, {}).get("p_cont", np.nan),
            "stretch3_p_cont": ss.get(3, {}).get("p_cont", np.nan),
            "revisit100": rv,
            "ER_med": er["median"],
            "acf1": acf[1][0], "acf1_abs": acf[1][1], "acf20_abs": acf[20][1],
        })
    return rows
