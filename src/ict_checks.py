"""Проверки ICT уровня дня/недели: дни недели экстремума (W01), TGIF (W02),
правило London/ADR (T07). Нуллы под уровень утверждения:

- W01/W02 (дневные бары): блочный бутстреп OHLC D1 (block=250) — рушит привязку
  к дню недели, сохраняет кластеризацию волы и дрейф. Метки дней остаются от
  исходного индекса, данные приходят из случайных позиций.
- T07 (часовые бары в ET): перестановка сессий — London дня i спаривается с NY
  дня i+k (k случайный ±1..10). Рушит внутридневную связь London→NY,
  сохраняет режим волы и профиль часа.

Все метрики — плоские dict, z = (real - mean(null)) / std(null).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ------------------------------------------------------------------ недели

def _weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Дневные бары -> таблица недель (≥4 торговых дня, пн..пт). Векторно."""
    d = df.copy()
    d.index = d.index.normalize()
    d = d[d.index.dayofweek < 5]
    d = d[~d.index.duplicated()]
    iso = d.index.isocalendar()
    d["wk"] = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    d["dow"] = d.index.dayofweek
    g = d.groupby("wk")
    w = g.agg(open=("open", "first"), close=("close", "last"), high=("high", "max"),
              low=("low", "min"), n_days=("dow", "size"))
    w["hi_dow"] = d.loc[g["high"].idxmax().to_numpy(), "dow"].to_numpy()
    w["lo_dow"] = d.loc[g["low"].idxmin().to_numpy(), "dow"].to_numpy()
    w = w[w.n_days >= 4]
    w["ret"] = w.close / w.open - 1
    w["rng"] = w.high - w.low
    pre = d[d.dow <= 3].groupby("wk").agg(pre_o=("open", "first"), pre_c=("close", "last"),
                                          pre_hi=("high", "max"), pre_lo=("low", "min"),
                                          n_pre=("dow", "size"))
    fri = d[d.dow == 4].set_index("wk")[["open", "high", "low", "close"]]
    fri.columns = ["f_open", "f_high", "f_low", "f_close"]
    w = w.join(pre).join(fri)
    w["has_fri"] = w.f_close.notna() & (w.n_pre >= 3)
    w["pre_rng"] = w.pre_hi - w.pre_lo
    w["pre_eff"] = (w.pre_c - w.pre_o).abs() / w.pre_rng.replace(0, np.nan)
    w["dir"] = np.sign(w.pre_c - w.pre_o)
    # откат пятницы против направления недели, от close четверга, в долях диапазона пн..чт
    against = np.where(w.dir > 0, w.pre_c - w.f_low, w.f_high - w.pre_c)
    w["fri_retrace"] = against / w.pre_rng.replace(0, np.nan)
    w["fri_ret_dir"] = w.dir * (w.f_close - w.f_open) / w.f_open
    w["fri_close_against"] = (w.dir * (w.f_close - w.pre_c) < 0).astype(float)
    return w.reset_index()


def weekday_extremes(df: pd.DataFrame) -> dict:
    """W01: какой день недели даёт хай/лоу недели; безусловно и при условии
    направления недели (close недели vs open)."""
    w = _weeks(df)
    out = {"n_weeks": float(len(w))}
    for cond, sub in (("all", w), ("up", w[w.ret > 0]), ("dn", w[w.ret < 0])):
        n = len(sub)
        out[f"n_{cond}"] = float(n)
        if n < 30:
            continue
        for k in range(5):
            out[f"{cond}_lo_{DOW[k]}"] = float((sub.lo_dow == k).mean())
            out[f"{cond}_hi_{DOW[k]}"] = float((sub.hi_dow == k).mean())
        out[f"{cond}_lo_MonTue"] = float(sub.lo_dow.isin([0, 1]).mean())
        out[f"{cond}_hi_MonTue"] = float(sub.hi_dow.isin([0, 1]).mean())
        out[f"{cond}_lo_MonWed"] = float(sub.lo_dow.isin([0, 1, 2]).mean())
        out[f"{cond}_hi_MonWed"] = float(sub.hi_dow.isin([0, 1, 2]).mean())
    return out


def tgif(df: pd.DataFrame, eff_min=0.5) -> dict:
    """W02: после направленной недели (пн..чт: эффективность ≥ eff_min и диапазон
    выше медианы) пятница откатывает 20–30% диапазона недели."""
    w = _weeks(df)
    w = w[w.has_fri & np.isfinite(w.pre_eff)]
    med = w.pre_rng.median()
    strong = w[(w.pre_eff >= eff_min) & (w.pre_rng >= med)]
    weak = w[(w.pre_eff < eff_min)]
    out = {"n_weeks": float(len(w)), "n_strong": float(len(strong)), "n_weak": float(len(weak))}
    for name, sub in (("strong", strong), ("weak", weak)):
        if len(sub) < 30:
            continue
        r = sub.fri_retrace.to_numpy()
        out[f"{name}_retr_median"] = float(np.nanmedian(r))
        out[f"{name}_p_retr_ge20"] = float(np.nanmean(r >= 0.2))
        out[f"{name}_p_retr_20_30"] = float(np.nanmean((r >= 0.2) & (r <= 0.3)))
        out[f"{name}_p_retr_ge50"] = float(np.nanmean(r >= 0.5))
        out[f"{name}_p_fri_close_against"] = float(sub.fri_close_against.mean())
        out[f"{name}_fri_ret_dir_bp"] = float(sub.fri_ret_dir.mean() * 1e4)
    return out


# ------------------------------------------------------------------ London / NY

LONDON = (2, 8)     # 02:00–08:00 ET (часы 2..7)
NEWYORK = (8, 16)   # 08:00–16:00 ET (часы 8..15)


def _sessions(df_et: pd.DataFrame) -> pd.DataFrame:
    """Часовые бары в ET -> по дням: диапазон/нетто Лондона и NY, ADR(5) прошлых дней."""
    d = df_et.copy()
    d["day"] = d.index.tz_localize(None).normalize()
    d["hr"] = d.index.hour
    rows = []
    for day, g in d.groupby("day"):
        if day.dayofweek >= 5:
            continue
        lon = g[(g.hr >= LONDON[0]) & (g.hr < LONDON[1])]
        ny = g[(g.hr >= NEWYORK[0]) & (g.hr < NEWYORK[1])]
        if len(lon) < 4 or len(ny) < 5:
            continue
        rows.append({
            "day": day, "day_rng": g.high.max() - g.low.min(),
            "lon_hi": lon.high.max(), "lon_lo": lon.low.min(),
            "lon_rng": lon.high.max() - lon.low.min(),
            "lon_net": lon.close.iloc[-1] - lon.open.iloc[0],
            "ny_rng": ny.high.max() - ny.low.min(),
            "ny_net": ny.close.iloc[-1] - ny.open.iloc[0],
            "ny_hi": ny.high.max(), "ny_lo": ny.low.min(),
        })
    s = pd.DataFrame(rows).set_index("day")
    s["adr5"] = s.day_rng.rolling(5).mean().shift(1)
    return s.dropna()


def _london_stats(lon: pd.DataFrame, ny: pd.DataFrame) -> dict:
    """lon и ny — выровненные по строкам таблицы (одинаковой длины)."""
    share = lon.lon_rng.to_numpy() / lon.adr5.to_numpy()
    dom = share >= 0.8
    ny_rng = ny.ny_rng.to_numpy() / lon.adr5.to_numpy()
    ny_abs = np.abs(ny.ny_net.to_numpy()) / lon.adr5.to_numpy()
    # NY вышел за границы Лондона (расширение дневного диапазона)
    expand = (ny.ny_hi.to_numpy() > lon.lon_hi.to_numpy()) | (ny.ny_lo.to_numpy() < lon.lon_lo.to_numpy())
    reverse = np.sign(ny.ny_net.to_numpy()) == -np.sign(lon.lon_net.to_numpy())
    out = {"n_days": float(len(lon)), "n_dom": float(dom.sum()), "p_dom": float(dom.mean())}
    for name, m in (("dom", dom), ("nodom", ~dom)):
        if m.sum() < 30:
            continue
        out[f"{name}_ny_rng_adr"] = float(np.mean(ny_rng[m]))
        out[f"{name}_ny_absnet_adr"] = float(np.mean(ny_abs[m]))
        out[f"{name}_p_ny_expand"] = float(np.mean(expand[m]))
        out[f"{name}_p_ny_reverse"] = float(np.mean(reverse[m]))
    if "dom_ny_rng_adr" in out and "nodom_ny_rng_adr" in out:
        out["diff_ny_rng_adr"] = out["dom_ny_rng_adr"] - out["nodom_ny_rng_adr"]
        out["diff_p_ny_expand"] = out["dom_p_ny_expand"] - out["nodom_p_ny_expand"]
        out["diff_p_ny_reverse"] = out["dom_p_ny_reverse"] - out["nodom_p_ny_reverse"]
    return out


def london_adr_rule(df_et: pd.DataFrame, B=200, seed=0) -> tuple[dict, dict]:
    """T07: London ≥ 80% ADR(5) -> NY без значимого хода (узкий диапазон, не выходит
    за Лондон, разворот). Нулл: NY-сессия взята из дня i+k, k ∈ ±1..10."""
    s = _sessions(df_et)
    real = _london_stats(s, s)
    rng = np.random.default_rng(seed)
    n = len(s)
    boots = []
    for _ in range(B):
        k = rng.integers(1, 11, size=n) * rng.choice([-1, 1], size=n)
        j = np.clip(np.arange(n) + k, 0, n - 1)
        boots.append(_london_stats(s, s.iloc[j]))
    return real, _zscores(real, boots)


# ------------------------------------------------------------------ нулл для D1

def bootstrap_d1(df: pd.DataFrame, func, B=100, block=250, seed=0) -> tuple[dict, dict]:
    """Прогон func на реальных D1 и на B бутстреп-копиях (block=250 дней)."""
    real = func(df)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    boots = []
    for b in range(B):
        so, sh, sl, sc = engine.block_bootstrap_ohlc(o, h, l, c, block=block, seed=seed + b)
        sdf = pd.DataFrame({"open": so, "high": sh, "low": sl, "close": sc,
                            "volume": 1.0}, index=df.index)
        boots.append(func(sdf))
    return real, _zscores(real, boots)


def _zscores(real: dict, boots: list[dict]) -> dict:
    """Для каждой метрики: mean/std по нуллу и z."""
    out = {}
    for k, v in real.items():
        vals = np.array([b.get(k, np.nan) for b in boots], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10 or k.startswith("n_"):
            continue
        mu, sd = vals.mean(), vals.std(ddof=1)
        out[k] = {"real": v, "null_mean": mu, "null_sd": sd,
                  "z": (v - mu) / sd if sd > 0 else np.nan}
    return out
