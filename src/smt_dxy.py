"""X07 (вариант DXY): SMT-дивергенция DXY vs пара (EURUSD/GBPUSD) на D1 и H1.

Событие в баре t (окно L баров): пара делает новый L-лоу (low < min low прошлых L), DXY делает
новый L-хай — «подтверждение» (symmetry, ICT: продолжение); если одно из двух есть, а второго
нет — «дивергенция» (ICT: разворот в сторону того, кто не подтвердил: пара вверх). Зеркально для
новых хаев пары / лоу DXY. Берём первое событие в серии (предыдущие gap баров без событий).
Исход: доходность пары за h баров в ATR-единицах, со знаком ожидания ICT (дивергенция — разворот
пары против её нового экстремума; подтверждение — продолжение), и P(> 0).
Нулл: циркулярный сдвиг ряда DXY на случайное смещение ≥ 60 баров (B раз) — разрушает связь
DXY↔пара, сохраняя обе динамики; события и метрики пересчитываются.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from swings import atr


def _newext(h, l, L):
    hi_prev = pd.Series(h).rolling(L).max().shift(1).to_numpy()
    lo_prev = pd.Series(l).rolling(L).min().shift(1).to_numpy()
    return h > hi_prev, l < lo_prev


def smt_metrics(ph, pl, pc, dh, dl, L=20, hs=(5, 10, 20), gap=5):
    """ph/pl/pc — пара; dh/dl — DXY (выровнены). Возвращает dict метрик."""
    n = len(pc)
    a = atr(ph, pl, pc, 20)
    p_hh, p_ll = _newext(ph, pl, L)
    d_hh, d_ll = _newext(dh, dl, L)
    # дивергенция bullish для пары: (p_ll xor d_hh); подтверждение: p_ll & d_hh
    div_bull = (p_ll & ~d_hh) | (d_hh & ~p_ll)
    conf_bear = p_ll & d_hh
    div_bear = (p_hh & ~d_ll) | (d_ll & ~p_hh)
    conf_bull = p_hh & d_ll
    out = {}

    def first_in_run(mask):
        idx = np.flatnonzero(mask)
        keep, last = [], -10**9
        for i in idx:
            if i - last > gap:
                keep.append(i)
            last = i
        return np.array(keep, int)

    for name, mask, sign in (("div_bull", div_bull, +1), ("conf_bear", conf_bear, -1),
                             ("div_bear", div_bear, -1), ("conf_bull", conf_bull, +1)):
        idx = first_in_run(mask)
        idx = idx[(idx > L + 20) & (idx < n - max(hs) - 1)]
        out[f"{name}_n"] = float(len(idx))
        for h in hs:
            r = sign * (pc[idx + h] - pc[idx]) / a[idx]
            r = r[np.isfinite(r)]
            out[f"{name}_ret{h}"] = float(r.mean()) if len(r) else np.nan
            out[f"{name}_p{h}"] = float((r > 0).mean()) if len(r) else np.nan
    # сводные: дивергенция vs подтверждение (обе стороны), в единицах «ожидание ICT»
    for h in hs:
        for kind in ("div", "conf"):
            parts = [(out[f"{kind}_bull_n"], out[f"{kind}_bull_ret{h}"], out[f"{kind}_bull_p{h}"]),
                     (out[f"{kind}_bear_n"], out[f"{kind}_bear_ret{h}"], out[f"{kind}_bear_p{h}"])]
            w = np.array([p[0] for p in parts]);
            if w.sum() > 0:
                out[f"{kind}_ret{h}"] = float(np.nansum([p[0] * p[1] for p in parts]) / w.sum())
                out[f"{kind}_p{h}"] = float(np.nansum([p[0] * p[2] for p in parts]) / w.sum())
    return out


def smt_test(pair: pd.DataFrame, dxy: pd.DataFrame, L=20, hs=(5, 10, 20), B=300, seed=0):
    j = pair.join(dxy[["high", "low", "close"]].rename(columns=lambda c: "d_" + c), how="inner").dropna()
    ph, pl, pc = (j[k].to_numpy(float) for k in ("high", "low", "close"))
    dh, dl = j["d_high"].to_numpy(float), j["d_low"].to_numpy(float)
    real = smt_metrics(ph, pl, pc, dh, dl, L, hs)
    rng = np.random.default_rng(seed)
    n = len(pc)
    boots = []
    for _ in range(B):
        s = int(rng.integers(60, n - 60))
        boots.append(smt_metrics(ph, pl, pc, np.roll(dh, s), np.roll(dl, s), L, hs))
    z = {}
    for k, v in real.items():
        vals = np.array([b.get(k, np.nan) for b in boots], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10 or not np.isfinite(v):
            continue
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z, len(j)
