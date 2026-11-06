"""X11: новостные дни (NFP 08:30 ET, FOMC 14:00 ET) на H1 в ET.

Дневные метрики (день = календарный день ET, бары 00–23):
  rng_adr        диапазон дня / ADR20 прошлых дней
  first_false    первый ход новостного бара (08:00 или 14:00) противоположен остатку дня
                 (close 16:00-бара − close новостного бара)
  pre_comp       диапазон окна перед новостью (05–07 для NFP, 10–13 для FOMC) / ADR20
  ext_news_hour  экстремум дня (хай или лоу) лежит в новостном баре
  eff            |close дня − open дня| / диапазон дня (направленность)
Сравнение: NFP-дни vs прочие пятницы; FOMC-дни vs прочие дни того же дня недели (среда);
NFP-неделя: Чт/Пт недели NFP vs прочие Чт/Пт (rng_adr, eff, first_false по 08:00).
z — перестановка меток события среди дней контрольной группы (B раз).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS = {"NFP": {"news": 8, "pre": (5, 7)}, "FOMC": {"news": 14, "pre": (10, 13)}}


def day_table(df_et: pd.DataFrame) -> pd.DataFrame:
    d = df_et.copy()
    idx = d.index.tz_localize(None)
    d["day"] = idx.normalize(); d["hour"] = idx.hour
    g = d.groupby("day")
    t = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "open": g.open.first(), "close": g.close.last(),
                      "nbars": g.size()})
    t["range"] = t.high - t.low
    t["adr"] = t["range"].rolling(20).mean().shift(1)
    t["rng_adr"] = t["range"] / t["adr"]
    t["eff"] = (t.close - t.open).abs() / t["range"]
    c16 = d[d.hour == 16].set_index("day").close
    t["c16"] = c16[~c16.index.duplicated()].reindex(t.index)
    for ev, cfg in HOURS.items():
        hb = d[d.hour == cfg["news"]].set_index("day")
        hb = hb[~hb.index.duplicated()].reindex(t.index)
        t[f"{ev}_fm"] = np.sign(hb.close - hb.open)
        t[f"{ev}_rest"] = np.sign(t["c16"] - hb.close)
        t[f"{ev}_first_false"] = np.where(t[f"{ev}_fm"].notna() & t[f"{ev}_rest"].notna(),
                                         (t[f"{ev}_fm"] * t[f"{ev}_rest"] < 0).astype(float), np.nan)
        t[f"{ev}_ext"] = np.where(hb.high.notna(), ((hb.high >= t.high) | (hb.low <= t.low)).astype(float), np.nan)
        pre = d[(d.hour >= cfg["pre"][0]) & (d.hour <= cfg["pre"][1])].groupby("day")
        t[f"{ev}_pre_comp"] = (pre.high.max() - pre.low.min()) / t["adr"]
    t = t[(t.nbars >= 12) & t.adr.notna()]
    t["dow"] = t.index.dayofweek
    return t


def _perm_z(x_ev, x_ctl, B, rng):
    x_ev = x_ev[np.isfinite(x_ev)]; x_ctl = x_ctl[np.isfinite(x_ctl)]
    if len(x_ev) < 10 or len(x_ctl) < 30:
        return np.nan, np.nan, np.nan, np.nan
    pool = np.concatenate([x_ev, x_ctl]); k = len(x_ev)
    real = x_ev.mean()
    null = np.array([rng.permutation(pool)[:k].mean() for _ in range(B)])
    sd = null.std(ddof=1)
    return real, x_ctl.mean(), null.std(ddof=1), (real - null.mean()) / sd if sd > 0 else np.nan


def news_test(df_et: pd.DataFrame, cal: pd.DataFrame, B=1000, seed=0) -> dict:
    t = day_table(df_et)
    rng = np.random.default_rng(seed)
    out = {}
    nfp = pd.DatetimeIndex(cal[cal.event == "NFP"].date)
    fomc = pd.DatetimeIndex(cal[cal.event == "FOMC"].date)
    # NFP vs прочие пятницы
    ev = t[t.index.isin(nfp)]; ctl = t[(t.dow == 4) & ~t.index.isin(nfp)]
    out["NFP_n"] = float(len(ev)); out["NFP_ctl_n"] = float(len(ctl))
    for m in ("rng_adr", "eff", "NFP_first_false", "NFP_ext", "NFP_pre_comp"):
        r, c, sd, z = _perm_z(ev[m].to_numpy(float), ctl[m].to_numpy(float), B, rng)
        out[f"NFP_{m}"] = {"real": r, "null_mean": c, "null_sd": sd, "z": z}
    # FOMC vs прочие дни того же дня недели
    ev = t[t.index.isin(fomc)]
    dows = ev.dow.unique()
    ctl = t[t.dow.isin(dows) & ~t.index.isin(fomc) & ~t.index.isin(nfp)]
    out["FOMC_n"] = float(len(ev)); out["FOMC_ctl_n"] = float(len(ctl))
    for m in ("rng_adr", "eff", "FOMC_first_false", "FOMC_ext", "FOMC_pre_comp"):
        r, c, sd, z = _perm_z(ev[m].to_numpy(float), ctl[m].to_numpy(float), B, rng)
        out[f"FOMC_{m}"] = {"real": r, "null_mean": c, "null_sd": sd, "z": z}
    # NFP-неделя: Чт и Пт недели NFP (сама пятница NFP — Пт) vs прочие Чт/Пт
    wk = set(d.to_period("W-SUN") for d in nfp)
    inwk = pd.Index([d.to_period("W-SUN") in wk for d in t.index])
    ev = t[inwk & t.dow.isin([3, 4])]; ctl = t[~inwk & t.dow.isin([3, 4])]
    out["NFPWK_n"] = float(len(ev))
    for m in ("rng_adr", "eff", "NFP_first_false"):
        r, c, sd, z = _perm_z(ev[m].to_numpy(float), ctl[m].to_numpy(float), B, rng)
        out[f"NFPWK_{m}"] = {"real": r, "null_mean": c, "null_sd": sd, "z": z}
    # Чт/Пт NFP-недели без самой NFP-пятницы (чистый «sloppy Thu/Fri»)
    ev2 = ev[~ev.index.isin(nfp)]
    for m in ("rng_adr", "eff"):
        r, c, sd, z = _perm_z(ev2[m].to_numpy(float), ctl[m].to_numpy(float), B, rng)
        out[f"NFPWKx_{m}"] = {"real": r, "null_mean": c, "null_sd": sd, "z": z}
    return out
