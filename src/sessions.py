"""Сессионный слой: поведение цены по времени суток (ET).

Брокер = EET/EEST -> конвертируем в America/New_York через tz-базу
(корректная обработка недель расхождения DST).

Тесты: почасовой профиль (вола/дрейф/частота событий), то же событие
внутри окна ICT vs снаружи, Opening Range Breakout, Judas swing
(снятие азиатского диапазона на открытии Лондона).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine

BROKER_TZ = "Europe/Helsinki"   # EET/EEST
ET = "America/New_York"

# окна ICT в ET
WINDOWS = {
    "asia":      (0, 3),      # 00:00-03:00 ET (тихая азия)
    "london":    (3, 5),      # 03:00-05:00 ET открытие Лондона
    "ny_open":   (9, 11),     # 09:30-11:00 ET (часы 9 и 10)
    "sb_pm":     (14, 15),    # 14:00-15:00 ET
    "london_cl": (11, 12),    # 11:00-12:00 ET закрытие Лондона
}


def to_et(df: pd.DataFrame, src_tz: str = BROKER_TZ) -> pd.DataFrame:
    """MT5-брокер = EET/EEST; Binance/Yahoo = UTC."""
    out = df.copy()
    out.index = (out.index.tz_localize(src_tz, nonexistent="shift_forward")
                 .tz_convert(ET))
    return out


def hourly_profile(df: pd.DataFrame):
    """По каждому часу ET: индекс волатильности (доля диапазона),
    частота displacement, средняя доходность и её t-стат."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    ev, _ = engine.displacement_events(o, h, l, c)
    hours = df.index.hour.to_numpy()
    rng = (h - l) / c
    ret = np.diff(np.log(c), prepend=np.nan)
    rows = []
    for hr in range(24):
        m = hours == hr
        if m.sum() < 50:
            continue
        r = ret[m]
        r = r[np.isfinite(r)]
        t = float(r.mean() / (r.std() / np.sqrt(len(r)))) if len(r) > 30 else np.nan
        rows.append({
            "hour_et": hr, "n": int(m.sum()),
            "vol_index": float(rng[m].mean() / rng.mean()),
            "disp_rate": float(ev[m].mean() / ev.mean()),
            "ret_bp": float(np.nanmean(r) * 1e4), "t_stat": t,
        })
    return pd.DataFrame(rows)


def window_event_test(df: pd.DataFrame, win_name: str, K=24):
    """Displacement внутри окна vs снаружи: продолжение и скелет сделки."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    ev, d = engine.displacement_events(o, h, l, c)
    idx = np.where(ev)[0]
    hours = df.index.hour.to_numpy()
    h0, h1 = WINDOWS[win_name]
    in_win = (hours[idx] >= h0) & (hours[idx] < h1)
    out = {}
    for nm, mask in (("in", in_win), ("out", ~in_win)):
        i2 = idx[mask]
        if len(i2) < 30:
            out[nm] = None
            continue
        fwd = engine.forward_stats(c, i2, d, horizons=(K,))
        tr = engine.trade_stats(h, l, c, i2, d, K=K)
        out[nm] = {
            "n": len(i2),
            "p_cont": fwd[K]["p_cont"] if K in fwd else np.nan,
            "exp_R": tr["expectancy_R"] if tr else np.nan,
            "win_rate": tr["win_rate"] if tr else np.nan,
        }
    return out


def orb(df: pd.DataFrame, range_minutes=30, minutes_per_bar=5):
    """Opening Range Breakout: диапазон 09:30-10:00 ET; пробой после 10:00 —
    доходность от пробоя до конца сессии (16:00 ET) в сторону пробоя."""
    bars_in_range = range_minutes // minutes_per_bar
    et = df.index
    day = et.normalize()
    hm = et.hour * 60 + et.minute
    is_open_bar = (hm >= 570) & (hm < 570 + range_minutes)   # 09:30-10:00
    is_trade = (hm >= 570 + range_minutes) & (hm < 960)      # до 16:00

    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    orb_hi = pd.Series(h).where(is_open_bar).groupby(day).max()
    orb_lo = pd.Series(l).where(is_open_bar).groupby(day).min()
    day_arr = day.to_numpy()
    hi_map = orb_hi.reindex(day).to_numpy()
    lo_map = orb_lo.reindex(day).to_numpy()

    brk_up = is_trade & (c > hi_map) & np.isfinite(hi_map)
    brk_dn = is_trade & (c < lo_map) & np.isfinite(lo_map)
    idx = np.where(brk_up | brk_dn)[0]
    # первый пробой дня
    if len(idx) == 0:
        return None
    day_of = day_arr[idx]
    first = np.concatenate([[True], day_of[1:] != day_of[:-1]])
    idx = idx[first]
    d = np.where(brk_up[idx], 1.0, -1.0)
    # выход: последний бар дня
    close_day = pd.Series(c).groupby(day).last()
    eod = close_day.reindex(day_arr[idx]).to_numpy()
    ret = d * (eod - c[idx]) / c[idx]
    return {"n": len(idx), "mean_bp": float(np.mean(ret) * 1e4),
            "p_win": float(np.mean(ret > 0))}


def judas(df: pd.DataFrame, minutes_per_bar=5):
    """Снятие high/low азиатского диапазона (00:00-03:00 ET) в окне
    03:00-05:00 ET -> реверсия обратно в диапазон к 12:00 ET."""
    et = df.index
    hm = et.hour * 60 + et.minute
    day = et.normalize()
    is_asia = (hm >= 0) & (hm < 180)
    is_ldn = (hm >= 180) & (hm < 300)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a_hi = pd.Series(h).where(is_asia).groupby(day).max()
    a_lo = pd.Series(l).where(is_asia).groupby(day).min()
    day_arr = day.to_numpy()
    hi_map = a_hi.reindex(day).to_numpy()
    lo_map = a_lo.reindex(day).to_numpy()
    sweep_hi = is_ldn & (h > hi_map) & np.isfinite(hi_map)
    sweep_lo = is_ldn & (l < lo_map) & np.isfinite(lo_map)
    idx = np.where(sweep_hi | sweep_lo)[0]
    if len(idx) == 0:
        return None
    day_of = day_arr[idx]
    first = np.concatenate([[True], day_of[1:] != day_of[:-1]])
    idx = idx[first]
    d = np.where(sweep_lo[idx], 1.0, -1.0)   # реверсия против снятия
    # выход: бар 12:00 ET того же дня (или ближайший <= 720 мин)
    noon_mask = (hm == 720)
    noon_px = pd.Series(c).where(noon_mask).groupby(day).first()
    tgt = noon_px.reindex(day_arr[idx]).to_numpy()
    ok = np.isfinite(tgt)
    idx, d, tgt = idx[ok], d[ok], tgt[ok]
    ret = d * (tgt - c[idx]) / c[idx]
    return {"n": len(idx), "mean_bp": float(np.mean(ret) * 1e4),
            "p_win": float(np.mean(ret > 0))}
