"""AMD / Power of Three: накопление (азия 00-03 ET) -> манипуляция
(снятие границы азии в 03-05 ET) -> дистрибуция (движение дня в обратную
сторону до закрытия).

Гипотеза ICT: после снятия лоя азии день расширяется ВВЕРХ за хай азии
(и наоборот). Контроль: дни без манипуляции + бутстреп (дневной блок).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sessions


def amd_days(df_et: pd.DataFrame):
    """Разбивает по торговым дням ET, возвращает события по дням."""
    o, h, l, c = (df_et[x].to_numpy() for x in ("open", "high", "low", "close"))
    hm = df_et.index.hour * 60 + df_et.index.minute
    day = df_et.index.normalize()
    res = []
    for d, grp in df_et.groupby(day):
        hms = grp.index.hour * 60 + grp.index.minute
        asia = grp[(hms >= 0) & (hms < 180)]
        ldn = grp[(hms >= 180) & (hms < 300)]
        rest = grp[(hms >= 300) & (hms <= 1380)]
        if len(asia) < 5 or len(rest) < 5:
            continue
        a_hi, a_lo = asia.high.max(), asia.low.min()
        rng = a_hi - a_lo
        if rng <= 0:
            continue
        swept_hi = bool((ldn.high > a_hi).any()) if len(ldn) else False
        swept_lo = bool((ldn.low < a_lo).any()) if len(ldn) else False
        # дистрибуция: максимальное расширение за противоположную границу
        # в остаток дня (в единицах диапазона азии)
        ext_beyond_hi = (rest.high.max() - a_hi) / rng if len(rest) else 0
        ext_beyond_lo = (a_lo - rest.low.min()) / rng if len(rest) else 0
        res.append({
            "day": d, "swept_hi": swept_hi, "swept_lo": swept_lo,
            "ext_hi": float(ext_beyond_hi), "ext_lo": float(ext_beyond_lo),
            "day_ret": float((rest.close.iloc[-1] - rest.open.iloc[0])
                             / rest.open.iloc[0]),
            "asia_range": float(rng),
        })
    return pd.DataFrame(res)


def amd_stats(days: pd.DataFrame):
    out = {}
    # манипуляция вниз (сняли лой азии) -> ждём расширение вверх
    for name, mask, ext_col, sign in (
            ("manip_lo->up", (days.swept_lo & ~days.swept_hi), "ext_hi", 1),
            ("manip_hi->dn", (days.swept_hi & ~days.swept_lo), "ext_lo", -1),
            ("no_manip", ~(days.swept_hi | days.swept_lo), "ext_hi", 0)):
        sub = days[mask]
        if len(sub) < 30:
            continue
        out[name] = {
            "n": len(sub),
            "p_expand": float(np.mean(sub[ext_col] > 0.5)),  # >0.5 диапазона
            "mean_ext": float(sub[ext_col].mean()),
            "mean_day_ret_bp": float(sub.day_ret.mean() * sign * 1e4)
                               if sign else float(sub.day_ret.mean() * 1e4),
        }
    return out
