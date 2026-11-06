"""S05 проекции, S04 длительность консолидации, S07 immediate rebalance, S11 осцилляторы.
Все функции возвращают плоский dict; нулл — ict_checks.bootstrap_d1 (блочный бутстреп).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings


# ---------------------------------------------------------------- S05 проекции

def projection_stats(df: pd.DataFrame, k=3.0, H=200) -> dict:
    """После подтверждённой ноги A→B и ретрейсмента C (следующий подтверждённый свинг):
    цели C + m·(B−A) для m в (0.5, 1, 1.27, 1.5, 1.68, 2) — доля достигнутых за H баров
    до слома C. Плюс «точность»: доля терминальных свингов в ±5% ноги от какого-либо уровня."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    idx, px, d, conf = swings.causal_swings(h, l, c, k=k)
    n = len(c)
    out = {"n_legs": 0.0}
    mults = (0.5, 1.0, 1.27, 1.5, 1.68, 2.0)
    hits = {m: [] for m in mults}
    term_err = []
    for i in range(len(idx) - 3):
        A, B, C = px[i], px[i + 1], px[i + 2]
        dirn = d[i + 1]                      # +1: B хай (нога вверх), проекция вверх от C
        leg = abs(B - A)
        if leg <= 0:
            continue
        start = conf[i + 2]                  # C подтверждён
        if start >= n:
            continue
        end = min(start + H, n)
        out["n_legs"] += 1
        for m in mults:
            target = C + dirn * m * leg
            reached = (h[start:end] >= target).any() if dirn > 0 else (l[start:end] <= target).any()
            # цель должна быть достигнута раньше слома C
            broke = (l[start:end] < C).any() if dirn > 0 else (h[start:end] > C).any()
            if reached and broke:
                t_r = np.argmax(h[start:end] >= target) if dirn > 0 else np.argmax(l[start:end] <= target)
                t_b = np.argmax(l[start:end] < C) if dirn > 0 else np.argmax(h[start:end] > C)
                reached = t_r < t_b
            hits[m].append(reached)
        # точность: следующий терминал D (px[i+3]) относительно проекций
        D = px[i + 3]
        ext = dirn * (D - C) / leg
        term_err.append(min(abs(ext - m) for m in mults))
    for m in mults:
        out[f"p_hit_{m}"] = float(np.mean(hits[m])) if hits[m] else np.nan
    te = np.array(term_err)
    out["p_term_within_0.05_of_level"] = float((te <= 0.05).mean()) if len(te) else np.nan
    out["p_term_within_0.10_of_level"] = float((te <= 0.10).mean()) if len(te) else np.nan
    return out


# ---------------------------------------------------------------- S04 консолидация

def consolidation_stats(df: pd.DataFrame, width=2.0, H=20) -> dict:
    """Коробка: от бара t диапазон удерживается в пределах width·ATR(100); длина = число баров
    до первого закрытия вне коробки. Исход: макс. экскурсия в сторону пробоя за H баров / ATR."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    a = swings.atr(h, l, c, 100)
    n = len(c)
    lengths, moves = [], []
    t = 100
    while t < n - H - 1:
        if not np.isfinite(a[t]) or a[t] <= 0:
            t += 1
            continue
        top, bot = h[t], l[t]
        j = t + 1
        while j < n - H - 1:
            top, bot = max(top, h[j]), min(bot, l[j])
            if top - bot > width * a[t]:
                break
            j += 1
        if j >= n - H - 1:
            break
        length = j - t
        if length >= 3:
            dirn = 1 if c[j] > top - (top - bot) / 2 else -1
            # пробой: close вне коробки предыдущего состояния
            exc = (h[j + 1:j + 1 + H].max() - c[j]) / a[t] if dirn > 0 else (c[j] - l[j + 1:j + 1 + H].min()) / a[t]
            lengths.append(length)
            moves.append(exc)
        t = j + 1
    L, M = np.array(lengths), np.array(moves)
    out = {"n_boxes": float(len(L))}
    if len(L) < 50:
        return out
    out["corr_len_move"] = float(np.corrcoef(L, M)[0, 1])
    q = np.quantile(L, [0.33, 0.67])
    out["move_short"] = float(M[L <= q[0]].mean())
    out["move_mid"] = float(M[(L > q[0]) & (L <= q[1])].mean())
    out["move_long"] = float(M[L > q[1]].mean())
    out["ratio_long_short"] = out["move_long"] / out["move_short"]
    out["len_median"] = float(np.median(L))
    return out


# ---------------------------------------------------------------- S07 immediate rebalance

def immediate_rebalance_stats(df: pd.DataFrame, K=12) -> dict:
    """После дисплейсмента (range > 2·mean200, тело ≥ 60%) следующий бар либо касается
    диапазона дисплейсмента (immediate rebalance, FVG не образован), либо оставляет гэп (FVG).
    Исход: продолжение за K баров в сторону дисплейсмента (знак и величина / ATR)."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    rng = h - l
    mean = pd.Series(rng).rolling(200).mean().shift(1).to_numpy()
    body = np.abs(c - o) / np.where(rng > 0, rng, np.nan)
    disp = (rng > 2 * mean) & (body >= 0.6)
    d = np.sign(c - o)
    a = swings.atr(h, l, c, 100)
    n = len(c)
    rows = {"imm": [], "gap": []}
    for t in np.flatnonzero(disp):
        if t + 1 + K >= n or not np.isfinite(a[t]) or a[t] <= 0:
            continue
        if d[t] > 0:
            imm = l[t + 1] <= h[t - 1] if t > 0 else False   # вернулся в диапазон до дисплейсмента (нет FVG)
        else:
            imm = h[t + 1] >= l[t - 1] if t > 0 else False
        fwd = d[t] * (c[t + 1 + K] - c[t + 1]) / a[t]
        rows["imm" if imm else "gap"].append(fwd)
    out = {}
    for k_, v in rows.items():
        v = np.array(v)
        out[f"n_{k_}"] = float(len(v))
        if len(v) > 30:
            out[f"{k_}_p_cont"] = float((v > 0).mean())
            out[f"{k_}_mean_atr"] = float(v.mean())
    if "imm_p_cont" in out and "gap_p_cont" in out:
        out["diff_p_cont"] = out["imm_p_cont"] - out["gap_p_cont"]
        out["diff_mean_atr"] = out["imm_mean_atr"] - out["gap_mean_atr"]
    return out


# ---------------------------------------------------------------- S11 осцилляторы

def oscillator_stats(df: pd.DataFrame, n_r=14, H=5) -> dict:
    """Williams %R(14): экстремум (<−80 / >−20) -> доходность через H баров;
    stochastic-дивергенция (цена higher low, %K lower low — hidden bullish; классическая:
    цена lower low, %K higher low) -> доходность через 2H."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    hh = pd.Series(h).rolling(n_r).max().to_numpy()
    ll = pd.Series(l).rolling(n_r).min().to_numpy()
    wr = -100 * (hh - c) / np.where(hh - ll > 0, hh - ll, np.nan)
    n = len(c)
    fwd = np.full(n, np.nan)
    fwd[:-H] = np.log(c[H:] / c[:-H])
    fwd2 = np.full(n, np.nan)
    fwd2[:-2 * H] = np.log(c[2 * H:] / c[:-2 * H])
    ok = np.isfinite(wr) & np.isfinite(fwd)
    out = {"all_fwd_bp": float(np.nanmean(fwd[ok]) * 1e4), "all_p_up": float((fwd[ok] > 0).mean())}
    for name, m in (("os", wr < -80), ("ob", wr > -20)):
        mm = m & ok
        out[f"n_{name}"] = float(mm.sum())
        out[f"{name}_fwd_bp"] = float(fwd[mm].mean() * 1e4)
        out[f"{name}_p_up"] = float((fwd[mm] > 0).mean())
    # дивергенции на локальных минимумах %K (окно 5) с шагом ≥ 5 баров
    k = wr + 100   # 0..100 как %K
    loc_min = np.zeros(n, bool)
    loc_min[2:-2] = (k[2:-2] < k[:-4]) & (k[2:-2] < k[1:-3]) & (k[2:-2] <= k[3:-1]) & (k[2:-2] <= k[4:])
    mins = np.flatnonzero(loc_min & np.isfinite(k))
    hid, cls = [], []
    for i in range(1, len(mins)):
        p, q = mins[i - 1], mins[i]
        if q - p < 5 or q - p > 40 or q + 2 * H >= n:
            continue
        price_hl, k_ll = l[q] > l[p], k[q] < k[p]
        price_ll, k_hl = l[q] < l[p], k[q] > k[p]
        if price_hl and k_ll:
            hid.append(fwd2[q + 2])          # минимум %K подтверждён через 2 бара
        elif price_ll and k_hl:
            cls.append(fwd2[q + 2])
    for name, v in (("hidden_bull", hid), ("classic_bull", cls)):
        v = np.array(v)
        out[f"n_{name}"] = float(len(v))
        if len(v) > 30:
            out[f"{name}_fwd2_bp"] = float(np.nanmean(v) * 1e4)
            out[f"{name}_p_up"] = float((v > 0).mean())
    out["all_fwd2_bp"] = float(np.nanmean(fwd2) * 1e4)
    return out
