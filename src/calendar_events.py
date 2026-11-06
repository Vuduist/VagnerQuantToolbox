"""Календарь событий: FOMC (из страниц ФРС) и NFP (правило «первая пятница» + проверка по спайку
волатильности 08:30 ET на EURUSD H1). Результат: data/calendar_events.csv (date, event, note).

FOMC: даты statement берутся из ссылок press/monetary/YYYYMMDD и monetaryYYYYMMDDa на
fomccalendars.htm и fomchistorical<year>.htm; оставляем Вт–Чт (внеплановые воскресные/пятничные
пресс-релизы отбрасываются); 2020-03-03 — внеплановое снижение в 10:00 (помечено).
NFP: первая пятница месяца; 1 января → следующая пятница; 4 июля (пт) → четверг; если на EURUSD
H1 бар 08:00 ET первой пятницы не выделяется (spike < 1.5× медианы), а второй пятницы
выделяется (> 2.5×), берём вторую пятницу (сдвиги BLS, шатдауны).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

import engine
import sessions

HTML_DIR = Path("data/raw/fed_html")
OUT = Path("data/calendar_events.csv")


def fomc_dates() -> pd.DatetimeIndex:
    dates = set()
    for f in glob.glob(str(HTML_DIR / "fomc_*.htm")):
        h = open(f, encoding="utf-8", errors="ignore").read()
        for m in re.findall(r"press/monetary/(20\d{6})|monetary(20\d{6})a", h):
            dates.add(m[0] or m[1])
    idx = pd.to_datetime(sorted(dates), format="%Y%m%d")
    idx = idx[idx.dayofweek.isin([1, 2, 3])]
    return idx


def nfp_dates(df_h1_et: pd.DataFrame, start="2009-01-01", end="2026-12-31") -> tuple[pd.DatetimeIndex, dict]:
    months = pd.period_range(start, end, freq="M")
    # спайк 08:00-бара ET: диапазон / медиана диапазона 08:00-баров за ±30 дней
    d = df_h1_et.copy()
    d["day"] = d.index.tz_localize(None).normalize()
    b8 = d[d.index.hour == 8]
    rng8 = (b8.high - b8.low).to_frame("r").set_index(b8["day"])["r"]
    med = rng8.rolling(60, center=True, min_periods=20).median()
    spike = (rng8 / med).to_dict()
    out, notes = [], {"rule": 0, "second_friday": 0, "holiday": 0, "no_data": 0}
    for p in months:
        d1 = p.start_time
        first_fri = d1 + pd.Timedelta(days=(4 - d1.dayofweek) % 7)
        cand = first_fri
        if cand.month == 1 and cand.day == 1:
            cand = cand + pd.Timedelta(days=7); notes["holiday"] += 1
        elif cand.month == 7 and cand.day == 4:
            cand = cand - pd.Timedelta(days=1); notes["holiday"] += 1
        elif cand.month == 7 and cand.day == 3:
            cand = cand - pd.Timedelta(days=1); notes["holiday"] += 1
        second = first_fri + pd.Timedelta(days=7)
        s1, s2 = spike.get(cand, np.nan), spike.get(second, np.nan)
        if np.isfinite(s1) and np.isfinite(s2) and s1 < 1.5 and s2 > 2.5:
            cand = second; notes["second_friday"] += 1
        elif not np.isfinite(s1):
            notes["no_data"] += 1
        else:
            notes["rule"] += 1
        out.append(cand)
    return pd.DatetimeIndex(out), notes


if __name__ == "__main__":
    eur = sessions.to_et(engine.load_mt5_csv("data/raw/EURUSD_PERIOD_H1.csv"))
    f = fomc_dates()
    n, notes = nfp_dates(eur)
    rows = [{"date": d.date(), "event": "FOMC", "note": "10:00 emergency" if str(d.date()) == "2020-03-03" else ""} for d in f]
    rows += [{"date": d.date(), "event": "NFP", "note": ""} for d in n]
    cal = pd.DataFrame(rows).sort_values(["date", "event"])
    cal.to_csv(OUT, index=False)
    print(f"FOMC: {len(f)} дат {f.min().date()}…{f.max().date()}; по годам:",
          f.year.value_counts().sort_index().to_dict())
    print(f"NFP: {len(n)} дат; {notes}")
    print("записано", OUT)
