"""Поток агрессора BTCUSDT (Binance spot klines 1m: объём, объём агрессивных покупок, число сделок), 2020-01 → 2025-09.
ofi = (2·taker_buy − volume) / volume ∈ [−1, 1] — доля чистых агрессивных покупок; svol = (2·taker_buy − volume) — в BTC.
Прогноз будущей доходности на 1/5/15/60 минут сверх собственной доходности минуты."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd


def load():
    parts = []
    for f in sorted(glob.glob("data/raw/BTCUSDT_1m/BTCUSDT-1m-*.csv")):
        d = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4, 5, 8, 9],
                        names=["t", "o", "h", "l", "c", "v", "n", "tb"])
        if d.t.iloc[0] > 10**14:            # микросекунды (новые файлы)
            d["t"] = d.t // 1000
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates("t").sort_values("t")
    d.index = pd.to_datetime(d.t, unit="ms")
    d["r"] = np.log(d.c).diff()
    d["ofi"] = ((2 * d.tb - d.v) / d.v.replace(0, np.nan)).fillna(0)
    d["svol"] = 2 * d.tb - d.v
    for k in (1, 5, 15, 60):
        d[f"f{k}"] = np.log(d.c.shift(-k) / d.c)
    return d
