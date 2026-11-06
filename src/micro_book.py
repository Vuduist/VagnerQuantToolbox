"""Дисбаланс лучших котировок BTCUSDT perpetual (Binance bookTicker) → движение mid через 1/5/30 с.
Файл дня читается кусками; на сетке 1 с берётся последнее состояние (best bid/ask, объёмы). Дисбаланс
I = (bidQty − askQty)/(bidQty + askQty). Исход — изменение mid в тиках (0.1 $) и в бп."""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd


def day_grid(path, step_ms=1000, chunk=2_000_000):
    z = zipfile.ZipFile(path); name = z.namelist()[0]
    parts = []
    reader = pd.read_csv(z.open(name), chunksize=chunk,
                         usecols=["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time"],
                         dtype={"best_bid_price": "float64", "best_bid_qty": "float32", "best_ask_price": "float64", "best_ask_qty": "float32"})
    for c in reader:
        c["g"] = c.transaction_time // step_ms
        parts.append(c.groupby("g").last())
    g = pd.concat(parts)
    g = g[~g.index.duplicated(keep="last")].sort_index()
    full = np.arange(g.index.min(), g.index.max() + 1)
    g = g.reindex(full).ffill()
    g["mid"] = (g.best_bid_price + g.best_ask_price) / 2
    g["spr_ticks"] = ((g.best_ask_price - g.best_bid_price) / 0.1).round()
    g["imb"] = (g.best_bid_qty - g.best_ask_qty) / (g.best_bid_qty + g.best_ask_qty)
    return g[["mid", "spr_ticks", "imb", "best_bid_qty", "best_ask_qty"]]


def imbalance_table(g, hz=(1, 5, 30)):
    out = []
    mid = g.mid.to_numpy()
    bins = [-1.01, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.01]
    b = pd.cut(g.imb, bins)
    for h in hz:
        dm = np.r_[mid[h:] - mid[:-h], np.full(h, np.nan)]
        df = pd.DataFrame({"b": b, "dm_ticks": dm / 0.1, "dm_bp": dm / mid * 1e4})
        t = df.groupby("b", observed=True).agg(n=("dm_ticks", "size"), up=("dm_ticks", lambda x: (x > 0).mean()),
                                               dn=("dm_ticks", lambda x: (x < 0).mean()), ticks=("dm_ticks", "mean"), bp=("dm_bp", "mean"))
        t["h"] = h
        out.append(t)
    return pd.concat(out)
