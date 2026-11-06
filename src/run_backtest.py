"""Сканер: все рынки x все ТФ x стратегии -> нетто-ожидание в bp по годам."""

import sys

import pandas as pd

import engine
from backtest import run_market

# (имя, файл, формат, список ТФ, издержки bp на круг)
MARKETS = [
    ("BTC",     "data/raw/BTCUSDT_1m.csv",       "binance", ["5m", "15m", "1h"], 10.0),
    ("USTEC",   "data/raw/USTEC_PERIOD_M5.csv",  "mt5",     ["5m", "15m", "1h", "4h"], 1.0),
    ("USTEC_H", "data/raw/USTEC_PERIOD_H1.csv",  "mt5",     ["1h", "4h", "1d"], 1.0),
    ("EURUSD",   "data/raw/EURUSD_PERIOD_M5.csv", "mt5",     ["5m", "15m", "1h"], 1.5),
    ("EURUSD_H", "data/raw/EURUSD_PERIOD_H1.csv", "mt5",     ["1h", "4h", "1d"], 1.5),
    ("EURUSD_D", "data/raw/EURUSD_PERIOD_D1.csv", "mt5",     ["1d"], 1.5),
    ("GBPUSD",   "data/raw/GBPUSD_PERIOD_M5.csv", "mt5",     ["5m", "15m", "1h"], 1.5),
    ("GBPUSD_H", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5",     ["1h", "4h", "1d"], 1.5),
    ("GBPUSD_D", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5",     ["1d"], 1.5),
    ("XAUUSD",   "data/raw/XAUUSD_PERIOD_M5.csv", "mt5",     ["5m", "15m", "1h"], 1.0),
    ("XAUUSD_H", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5",     ["1h", "4h", "1d"], 1.0),
    ("XAUUSD_D", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5",     ["1d"], 1.0),
    ("NQ_h",    "data/raw/NQ_1h.csv",            "binance", ["1h"], 1.0),
    ("NQ_d",    "data/raw/NQ_1d.csv",            "binance", ["1d"], 1.0),
]

TF_RULE = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

rows = []
for name, path, fmt, tfs, cost in MARKETS:
    base = (engine.load_mt5_csv(path) if fmt == "mt5"
            else engine.load_csv(path))
    for tf in tfs:
        df = engine.resample(base, TF_RULE[tf])
        if len(df) < 1500:
            continue
        for r in run_market(df, cost, f"{name}_{tf}"):
            rows.append(r)
        print(f"{name}_{tf}: ok ({len(df)} баров)", file=sys.stderr)

out = pd.DataFrame(rows)
out.to_csv("out/backtest_scan.csv", index=False)

ycols = sorted(c for c in out.columns if c.startswith("y"))
show = ["market", "strategy", "n", "exp_bp"] + ycols
pd.set_option("display.width", 250)
print(out[show].round(2).to_string(index=False))
