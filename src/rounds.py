"""S06: круглые числа. Знаковое расстояние high/low бара до ближайшего круглого уровня,
в единицах (пип / $0.5 / 5 пт). Плотность в полосе |d| ≤ 2 ед. против локальной базы
6–10 ед. (устойчиво к неравномерной занятости ценовых уровней). Контроль — смещённые
«некруглые» уровни (25/75 пипсов и т.п.). «Реакция»: экстремум в полосе круглого уровня →
разворот через K баров чаще, чем экстремум вне полосы?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _signed_units(ticks: np.ndarray, period: int, offset: int, unit: int) -> np.ndarray:
    r = (ticks - offset) % period
    signed = np.where(r > period // 2, r - period, r)
    return np.floor_divide(signed, unit)


def digit_stats(df: pd.DataFrame, tick: float, unit: int, levels: dict, K=3) -> dict:
    """levels: имя -> (период в тиках, смещение). unit — размер корзины в тиках
    (FX: пип = 100 тиков). Полоса near: корзины −2..+1 (|d| ≤ 2 ед.); база: 6..10 ед. с обеих сторон."""
    h, l, c = (df[x].to_numpy() for x in ("high", "low", "close"))
    th = np.round(h / tick).astype(np.int64)
    tl = np.round(l / tick).astype(np.int64)
    n = len(h)
    out = {"n_bars": float(n)}
    cK = np.roll(c, -K)
    valid = np.ones(n, bool); valid[-K:] = False
    for name, (period, offset) in levels.items():
        res = {}
        for lab, t in (("high", th), ("low", tl)):
            u = _signed_units(t, period, offset, unit)
            near = (u >= -2) & (u <= 1)                     # [-2, +2) единиц
            base = ((u >= 6) & (u <= 9)) | ((u >= -10) & (u <= -7))
            dens_near = near.sum() / 4
            dens_base = base.sum() / 8
            out[f"{lab}_{name}_ratio"] = float(dens_near / dens_base) if dens_base > 0 else np.nan
            out[f"{lab}_{name}_n_near"] = float(near.sum())
            # z по биномиальной модели: доля near среди near+base против 4/12
            m = near.sum() + base.sum()
            p = near.sum() / m if m else np.nan
            out[f"{lab}_{name}_z"] = (p - 1 / 3) / np.sqrt((1 / 3) * (2 / 3) / m) if m else np.nan
            res[lab] = near
        # реакция
        for lab, rev in (("high", lambda a: cK[a] < c[a]), ("low", lambda a: cK[a] > c[a])):
            a = res[lab] & valid
            b = ~res[lab] & valid
            pa, pb = float(rev(a).mean()), float(rev(b).mean())
            n1, n2 = a.sum(), b.sum()
            pp = (pa * n1 + pb * n2) / (n1 + n2)
            out[f"rev_{lab}_{name}_near"] = pa
            out[f"rev_{lab}_{name}_other"] = pb
            out[f"rev_{lab}_{name}_z"] = (pa - pb) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return out
