"""Межрыночные взаимодействия: SMT-дивергенции, лид-лаг, спред-реверсия.

SMT (ICT): инструмент A снимает подтверждённый структурный хай, а
коррелированный инструмент B НЕ делает нового хая -> дивергенция,
сигнал против движения A. Всё каузально (swings.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod


def align(dfA: pd.DataFrame, dfB: pd.DataFrame):
    """Пересечение по временным меткам (для D1 — по датам)."""
    a, b = dfA.copy(), dfB.copy()
    if (a.index.hour == 0).all() or (b.index.hour == 0).all():
        a.index = a.index.normalize()
        b.index = b.index.normalize()
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    return j.dropna()


def leadlag(cA, cB, max_lag=5):
    """corr(rA_t, rB_{t+k}) для k в -max_lag..max_lag.
    k>0: B ведёт A; k<0: A ведёт B."""
    rA = np.diff(np.log(cA))
    rB = np.diff(np.log(cB))
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            x, y = rA[:-k], rB[k:]
        elif k < 0:
            x, y = rA[-k:], rB[:k]
        else:
            x, y = rA, rB
        out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def _last_confirmed_swings(h, l, c, k=3.0):
    """Массивы: последний подтверждённый хай/лоу структуры на момент t."""
    _, sw_px, sw_dir, sw_conf = swmod.causal_swings(h, l, c, k=k)
    order = np.argsort(sw_conf)
    conf_s, px_s, dir_s = sw_conf[order], sw_px[order], sw_dir[order]
    n = len(c)
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    j = 0
    cur_hi, cur_lo = np.nan, np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            if dir_s[j] > 0:
                cur_hi = px_s[j]
            else:
                cur_lo = px_s[j]
            j += 1
        last_hi[t], last_lo[t] = cur_hi, cur_lo
    return last_hi, last_lo


def smt_events(hA, lA, cA, hB, lB, cB, k=3.0):
    """Каузальная SMT-дивергенция на подтверждённой структуре.
    Возвращает (idx, dir): dir=-1 -> медвежья для A (A снял хай, B нет)."""
    hiA, loA = _last_confirmed_swings(hA, lA, cA, k)
    hiB, loB = _last_confirmed_swings(hB, lB, cB, k)
    bear = (hA > hiA) & np.isfinite(hiA) & (hB <= hiB) & np.isfinite(hiB)
    bull = (lA < loA) & np.isfinite(loA) & (lB >= loB) & np.isfinite(loB)
    idx = np.where(bear | bull)[0]
    d = np.where(bull[idx], 1.0, -1.0)
    # дедупликация эпизодов: минимум 10 баров между событиями
    if len(idx) == 0:
        return idx, d
    keep = np.concatenate([[True], np.diff(idx) > 10])
    return idx[keep], d[keep]


def smt_forward_stats(cA, cB, idx, d, horizons=(12, 24, 48)):
    """Знаковая доходность A и B после дивергенции."""
    out = {}
    n = len(cA)
    for H in horizons:
        ok = idx + H < n
        i2 = idx[ok]
        if len(i2) < 30:
            continue
        d2 = d[ok]
        rA = d2 * (cA[i2 + H] - cA[i2]) / cA[i2]
        rB = d2 * (cB[i2 + H] - cB[i2]) / cB[i2]
        out[H] = {
            "n": len(i2),
            "A_p_win": float(np.mean(rA > 0)),
            "A_mean_bp": float(np.mean(rA) * 1e4),
            "B_p_win": float(np.mean(rB > 0)),
            "B_mean_bp": float(np.mean(rB) * 1e4),
        }
    return out


def spread_reversion(cA, cB, win=200, z_in=2.0, H=24):
    """Парная реверсия: лог-спред, z-скор против скользящего окна;
    вход против отклонения, выход через H баров."""
    la, lb = np.log(cA), np.log(cB)
    s = la - lb
    m = pd.Series(s).rolling(win).mean().to_numpy()
    sd = pd.Series(s).rolling(win).std().to_numpy()
    z = (s - m) / sd
    sig = np.where(z > z_in, -1.0, np.where(z < -z_in, 1.0, 0.0))
    idx = np.where(sig != 0)[0]
    idx = idx[np.isfinite(z[idx])]
    if len(idx) == 0:
        return None
    # дедуп
    keep = np.concatenate([[True], np.diff(idx) > 10])
    idx = idx[keep]
    idx = idx[idx + H < len(cA)]
    if len(idx) < 30:
        return None
    d = sig[idx]
    pnl = d * ((la[idx + H] - la[idx]) - (lb[idx + H] - lb[idx]))
    return {"n": len(idx), "mean_bp": float(np.mean(pnl) * 1e4),
            "p_win": float(np.mean(pnl > 0))}
