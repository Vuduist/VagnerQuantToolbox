"""Оставшиеся проблемные стороны FVG — пакет 2.

A. Порог размера гэпа: ширина зоны в ATR -> отскок.
B. Склейка кластеров: соседние однонаправленные зоны (окно 10 баров,
   пересекающиеся) как одна широкая зона против одиночных.
C. Вложенность ТФ: зона LTF, коснувшаяся внутри активной HTF-зоны
   того же направления, против вне её.
D. Сессионные дыры: зоны, рождённые в первые 3 бара торгового дня
   брокера (CFD), против остальных.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def _zones(o, h, l, c):
    """Все FVG: (born, bottom, top, dir)."""
    n = len(c)
    z = []
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:
            z.append((t + 1, h[t - 1], l[t + 1], 1))
        elif h[t + 1] < l[t - 1]:
            z.append((t + 1, h[t + 1], l[t - 1], -1))
    return z


def _outcome(l, h, c, born, bottom, top, d, K=96, M=48):
    """(touched, bounced) свежей зоны."""
    n = len(c)
    start = born + 1
    end = min(start + K, n)
    if start >= end:
        return None
    tm = (l[start:end] <= top) if d > 0 else (h[start:end] >= bottom)
    if not tm.any():
        return None
    tt = start + int(np.argmax(tm))
    s2, e2 = tt, min(tt + M, n)
    if d > 0:
        fm, bm = l[s2:e2] <= bottom, c[s2:e2] > top
    else:
        fm, bm = h[s2:e2] >= top, c[s2:e2] < bottom
    fb = np.argmax(fm) if fm.any() else M + 1
    bb = np.argmax(bm) if bm.any() else M + 1
    if fb > M and bb > M:
        return None
    return bool(bb < fb)


def _rate(df, col=None):
    if col is None:
        return float(np.mean(df)) if len(df) else np.nan
    out = {}
    for k in sorted(df[col].unique()):
        out[k] = float(df[df[col] == col if False else df[col] == k]["bounce"].mean())
    return out


def test_all(o, h, l, c, hours=None):
    zones = _zones(o, h, l, c)
    a = _atr(h, l, c)
    n = len(c)
    res = {"size": {}, "cluster": {}, "session": {}}
    recs = []
    for born, bottom, top, d in zones:
        if not np.isfinite(a[born]) or a[born] <= 0:
            continue
        bnc = _outcome(l, h, c, born, bottom, top, d)
        if bnc is None:
            continue
        recs.append({"born": born, "bottom": bottom, "top": top, "d": d,
                     "w_atr": (top - bottom) / a[born], "bounce": bnc,
                     "hour": hours[born] if hours is not None else -1})
    df = pd.DataFrame(recs)
    if df.empty:
        return res

    # A. размер в ATR
    df["size_bucket"] = pd.cut(df.w_atr, [0, 0.3, 0.8, np.inf],
                               labels=["small", "mid", "big"])
    for b in ("small", "mid", "big"):
        sub = df[df.size_bucket == b]
        if len(sub) > 30:
            res["size"][b] = (float(sub.bounce.mean()), len(sub))

    # B. склейка: зона vs кластер (та же сторона, рождение в 10 баров,
    #    пересечение диапазонов)
    clust = np.zeros(len(df), dtype=bool)
    b_arr, t_arr, d_arr = df.bottom.to_numpy(), df.top.to_numpy(), df.d.to_numpy()
    born_arr = df.born.to_numpy()
    for i in range(len(df)):
        lo, hi = b_arr[i], t_arr[i]
        for j in range(i + 1, len(df)):
            if born_arr[j] - born_arr[i] > 10 or d_arr[j] != d_arr[i]:
                break
            if b_arr[j] <= hi and t_arr[j] >= lo:
                lo, hi = min(lo, b_arr[j]), max(hi, t_arr[j])
                clust[i] = True
                clust[j] = True
    for grp, mask in (("single", ~clust), ("cluster", clust)):
        sub = df[mask]
        if len(sub) > 30:
            res["cluster"][grp] = (float(sub.bounce.mean()), len(sub))

    # D. сессионные дыры (только CFD): зона рождена в первые 3 бара дня
    if hours is not None:
        first3 = df.hour <= 2
        for grp, mask in (("day_open", first3), ("rest", ~first3)):
            sub = df[mask]
            if len(sub) > 30:
                res["session"][grp] = (float(sub.bounce.mean()), len(sub))
    return res


def test_nested(o_l, h_l, l_l, c_l, o_h, h_h, l_h, c_h, t_l, t_h):
    """C. Вложенность: LTF-зона касается внутри активной HTF-зоны."""
    z_htf = _zones(o_h, h_h, l_h, c_h)
    z_ltf = _zones(o_l, h_l, l_l, c_l)
    n_l = len(c_l)
    # активность HTF-зоны на LTF-шкале: born_htf <= t_now, ещё не заполнена
    t_l = np.asarray(t_l)
    t_h = np.asarray(t_h)
    res = {"inside": [0, 0], "outside": [0, 0]}
    for born, bottom, top, d in z_ltf:
        bnc = _outcome(l_l, h_l, c_l, born, bottom, top, d)
        if bnc is None:
            continue
        tb = t_l[born]
        inside = False
        for hb, hbot, htop, hd in z_htf:
            if t_h[hb] > tb:
                break
            if hd != d:
                continue
            # HTF-зона активна и цена внутри неё
            if hbot <= c_l[born] <= htop:
                inside = True
                break
        grp = "inside" if inside else "outside"
        res[grp][0] += 1
        res[grp][1] += int(bnc)
    return {k: (v[1] / v[0] if v[0] else np.nan, v[0])
            for k, v in res.items()}
