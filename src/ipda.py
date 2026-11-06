"""X05 quarterly shift и X06 IPDA data ranges 20/40/60 на D1.

X05: интервалы между подтверждёнными структурными разворотами (swings.causal_swings,
k·ATR100), автокорреляция 3-месячных доходностей, календарь разворотов.
X06: достижение экстремумов окон 20/40/60 дней, каскад после пробоя 20-дневного лоу,
реакция у экстремума окна, «cast forward» 20/40/60 дней от разворота.
Нулл для всего: блочный бутстреп D1 (block=250) через ict_checks.bootstrap_d1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings

QS_MONTHS = {12, 1, 3, 4, 6, 7, 9, 10}    # окна quarterly shift по ICT: Dec–Jan, Mar–Apr, Jun–Jul, Sep–Oct


def quarterly_stats(df: pd.DataFrame, k=4.0) -> dict:
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    idx, px, d, conf = swings.causal_swings(h, l, c, k=k)
    out = {"n_swings": float(len(idx))}
    if len(idx) > 20:
        gaps = np.diff(idx)
        out["gap_median"] = float(np.median(gaps))
        out["gap_mean"] = float(gaps.mean())
        out["gap_p_50_90"] = float(((gaps >= 50) & (gaps <= 90)).mean())
        out["gap_p_lt30"] = float((gaps < 30).mean())
        out["gap_cv"] = float(gaps.std() / gaps.mean())
        months = df.index[idx].month
        out["turn_p_qs_months"] = float(np.isin(months, list(QS_MONTHS)).mean())
        days = df.index[idx].day
        out["turn_p_first5days"] = float((days <= 5).mean())
    # 3-месячная автокорреляция (63 дня), выборка каждые 21 день
    lc = np.log(c)
    t = np.arange(63, len(c) - 63, 21)
    r_prev = lc[t] - lc[t - 63]
    r_next = lc[t + 63] - lc[t]
    out["n_3m"] = float(len(t))
    out["corr_3m"] = float(np.corrcoef(r_prev, r_next)[0, 1])
    big = np.abs(r_prev) > r_prev.std()
    out["p_cont_after_big3m"] = float((np.sign(r_prev[big]) == np.sign(r_next[big])).mean())
    out["p_cont_3m"] = float((np.sign(r_prev) == np.sign(r_next)).mean())
    return out


def ipda_stats(df: pd.DataFrame, k=4.0) -> dict:
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    n = len(c)
    out = {}
    H = {w: pd.Series(h).rolling(w).max().shift(1).to_numpy() for w in (20, 40, 60)}
    L = {w: pd.Series(l).rolling(w).min().shift(1).to_numpy() for w in (20, 40, 60)}
    fh20 = pd.Series(h[::-1]).rolling(20).max().to_numpy()[::-1]   # max high t..t+19
    fl20 = pd.Series(l[::-1]).rolling(20).min().to_numpy()[::-1]
    fh10 = pd.Series(h[::-1]).rolling(10).max().to_numpy()[::-1]
    fl10 = pd.Series(l[::-1]).rolling(10).min().to_numpy()[::-1]
    ok = np.isfinite(H[60]) & np.isfinite(fh20)
    for w in (20, 40, 60):
        touch20 = ((fh20 >= H[w]) | (fl20 <= L[w]))[ok]
        touch10 = ((fh10 >= H[w]) | (fl10 <= L[w]))[ok]
        out[f"p_touch{w}_in20"] = float(touch20.mean())
        out[f"p_touch{w}_in10"] = float(touch10.mean())
    # каскад: пробой 20-дневного лоу закрытием, когда 40-дневный лоу ниже -> достигнут ли он за 20 дней
    brk = (c < L[20]) & (L[40] < L[20] * 0.999) & ok
    out["n_break20"] = float(brk.sum())
    if brk.sum() > 30:
        out["p_reach40_after_break20"] = float((fl20[brk] <= L[40][brk]).mean())
        far = brk & (L[60] < L[40] * 0.999)
        if far.sum() > 30:
            out["p_reach60_after_break20"] = float((fl20[far] <= L[60][far]).mean())
    # контроль каскада: та же дистанция до 40-дневного лоу, но без пробоя 20-дневного
    nobrk = (c >= L[20]) & (L[40] < L[20] * 0.999) & ok
    dist_b = (c[brk] - L[40][brk]) / c[brk] if brk.sum() else np.array([])
    dist_n = (c[nobrk] - L[40][nobrk]) / c[nobrk]
    if brk.sum() > 30 and nobrk.sum() > 30:
        # сопоставление по дистанции: берём дни без пробоя с дистанцией не больше медианы пробойных
        m = dist_n <= np.median(dist_b)
        if m.sum() > 30:
            out["p_reach40_ctrl_samedist"] = float((fl20[nobrk][m] <= L[40][nobrk][m]).mean())
    # реакция у экстремума окна: лоу дня коснулся LL_w (отдельный уровень), close выше -> рост через 3 дня?
    c3 = np.roll(c, -3)
    for w, deeper in ((20, 40), (40, 60)):
        lvl = L[w]
        distinct = lvl > L[deeper] * 1.001
        touch = (l <= lvl) & (c > lvl) & distinct & ok
        touch[-3:] = False
        out[f"n_touchL{w}"] = float(touch.sum())
        if touch.sum() > 30:
            out[f"p_up3_after_touchL{w}"] = float((c3[touch] > c[touch]).mean())
        lvlh = H[w]
        distinct = lvlh < H[deeper] * 0.999
        touch = (h >= lvlh) & (c < lvlh) & distinct & ok
        touch[-3:] = False
        if touch.sum() > 30:
            out[f"p_dn3_after_touchH{w}"] = float((c3[touch] < c[touch]).mean())
    # cast forward: от подтверждённого разворота через 20/40/60 дней (±2) новый разворот?
    idx, px, d, conf = swings.causal_swings(h, l, c, k=k)
    sidx = np.zeros(n, bool)
    sidx[idx] = True
    out["n_swings"] = float(len(idx))
    base = sidx.mean() * 5
    out["p_turn_any5"] = float(base)
    for off in (10, 20, 30, 40, 50, 60, 70):
        hits = []
        for s in idx:
            t0 = s + off
            if t0 + 2 < n:
                hits.append(sidx[t0 - 2:t0 + 3].any())
        out[f"p_turn_at{off}"] = float(np.mean(hits)) if hits else np.nan
    return out
