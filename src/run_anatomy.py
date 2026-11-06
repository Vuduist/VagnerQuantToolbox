"""Атлас анатомии движения цены по рынкам и эпохам."""

import sys

import pandas as pd

import engine
from anatomy import anatomy

SETS = [
    ("EURUSD_D", "data/raw/EURUSD_PERIOD_D1.csv", "mt5", None),
    ("GBPUSD_D", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5", None),
    ("XAUUSD_D", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5", None),
    ("NQ_D",     "data/raw/NQ_1d.csv",            "binance", None),
    ("EURUSD_H", "data/raw/EURUSD_PERIOD_H1.csv", "mt5", None),
    ("XAUUSD_H", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5", None),
    ("USTEC_H",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5", None),
    ("BTC",      "data/raw/BTCUSDT_1m.csv",       "binance", "1h"),
]

rows = []
for name, path, fmt, rule in SETS:
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    rows.extend(anatomy(df, name))
    print(f"{name}: ok", file=sys.stderr)

out = pd.DataFrame(rows)
out.to_csv("out/anatomy.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string(index=False))
