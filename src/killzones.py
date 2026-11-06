"""T01: формируют ли killzones ICT экстремум дня (H1/M5 в ET, true day 00:00–16:00).

Утверждения: London open 02–05 ET, NY open 07–10, opening range 09:30–10:30,
London close 10–12 формируют high/low дня «чаще случайного»; хай трендового дня до 14:00.

Нулл «часы волатильности без расписания»: внутри каждого дня бары перемешиваются,
но каждая компонента бара (gap, o→h, o→l, o→c) стандартизуется шкалой своего
исходного часа и масштабируется шкалой нового часа. Сохраняется U-профиль волы
и распределение движений дня; разрушается любая иная структура времени суток.
Если реал ≡ нулл, «killzone формирует экстремум» = профиль волатильности +
краевой эффект (арксинус), а не расписание алгоритма.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_HOURS = range(0, 16)  # true day 00:00–16:00 ET

WINDOWS_H1 = {
    "london_2_5": (2, 3, 4),
    "ny_open_7_10": (7, 8, 9),
    "or_9_11": (9, 10),
    "london_close_10_12": (10, 11),
    "pm_13_16": (13, 14, 15),
    "asia_0_2": (0, 1),
}


def day_matrix(df_et: pd.DataFrame, bar_minutes=60):
    """Полные дни -> массивы (days × slots) компонент и часов слота."""
    d = df_et.copy()
    idx = d.index.tz_localize(None)
    d["day"] = idx.normalize()
    d["slot"] = (idx.hour * 60 + idx.minute) // bar_minutes
    n_slots = 16 * 60 // bar_minutes
    d = d[(idx.hour >= 0) & (idx.hour < 16) & (idx.dayofweek < 5)]
    cnt = d.groupby("day").slot.nunique()
    full = cnt[cnt == n_slots].index
    d = d[d.day.isin(full)]
    d = d[~d.duplicated(["day", "slot"])]
    piv = {k: d.pivot(index="day", columns="slot", values=k).to_numpy()
           for k in ("open", "high", "low", "close")}
    o, h, l, c = piv["open"], piv["high"], piv["low"], piv["close"]
    gap = np.zeros_like(o)
    gap[:, 1:] = np.log(o[:, 1:] / c[:, :-1])
    comp = np.stack([gap, np.log(h / o), np.log(l / o), np.log(c / o)], axis=0)  # 4×days×slots
    hours = (np.arange(n_slots) * bar_minutes) // 60
    return comp, o[:, 0], hours, np.asarray(full)


def rebuild(comp, open0):
    """Компоненты -> high/low по слотам (цепочка внутри дня)."""
    gap, oh, ol, oc = comp
    logc = np.log(open0)[:, None] + np.cumsum(gap + oc, axis=1)
    logo = logc - oc
    return np.exp(logo + oh), np.exp(logo + ol)


def extreme_slots(comp, open0):
    h, l = rebuild(comp, open0)
    return h.argmax(axis=1), l.argmin(axis=1)


def window_stats(hi_slot, lo_slot, hours, windows) -> dict:
    out = {"n_days": float(len(hi_slot))}
    hi_h, lo_h = hours[hi_slot], hours[lo_slot]
    for name, hrs in windows.items():
        m_hi, m_lo = np.isin(hi_h, hrs), np.isin(lo_h, hrs)
        out[f"{name}_p_high"] = float(m_hi.mean())
        out[f"{name}_p_low"] = float(m_lo.mean())
        out[f"{name}_p_any"] = float((m_hi | m_lo).mean())
    for hr in range(16):
        out[f"hour{hr:02d}_p_ext"] = float(((hi_h == hr) | (lo_h == hr)).mean() / 2)
    out["p_high_before_14"] = float((hi_h < 14).mean())
    out["p_low_before_14"] = float((lo_h < 14).mean())
    return out


def shuffled(comp, hours, rng):
    """Перестановка слотов внутри дня с пересчётом шкалы часа."""
    scale = np.nanmean(comp[1] - comp[2], axis=0)          # средний диапазон слота
    scale = np.maximum(scale, 1e-9)
    std = comp / scale[None, None, :]
    n_days, n_slots = comp.shape[1], comp.shape[2]
    perm = np.argsort(rng.random((n_days, n_slots)), axis=1)
    rows = np.arange(n_days)[:, None]
    out = std[:, rows, perm] * scale[None, None, :]
    out[0, :, 0] = 0.0                                     # первый бар дня без гэпа
    return out


def killzone_test(df_et: pd.DataFrame, bar_minutes=60, windows=None, B=200, seed=0):
    windows = windows or WINDOWS_H1
    comp, open0, hours, days = day_matrix(df_et, bar_minutes)
    hi, lo = extreme_slots(comp, open0)
    real = window_stats(hi, lo, hours, windows)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        hi_b, lo_b = extreme_slots(shuffled(comp, hours, rng), open0)
        boots.append(window_stats(hi_b, lo_b, hours, windows))
    z = {}
    for k, v in real.items():
        if k.startswith("n_"):
            continue
        vals = np.array([b[k] for b in boots])
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd,
                "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z
