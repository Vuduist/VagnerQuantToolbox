"""Сканер субъективных моделей: models.py x рынки x ТФ -> нетто бп по годам."""

import sys

import numpy as np
import pandas as pd

import engine
import models
from backtest import sim_time

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

SIGNALS = [
    ("donchian20", lambda df: models.sig_donchian(
        *(_a(df, "high", "low", "close")), n=20)),
    ("rsi2", lambda df: models.sig_rsi(_a(df, "close")[0])),
    ("ma_pullback", lambda df: models.sig_ma_pullback(
        *_a(df, "open", "high", "low", "close"))),
    ("engulfing", lambda df: models.sig_engulfing(
        *_a(df, "open", "high", "low", "close"))),
    ("inside_brk", lambda df: models.sig_inside_breakout(
        *_a(df, "high", "low", "close"))),
    ("pdhl_sweep", lambda df: models.sig_pdhl_sweep(
        df.index, *_a(df, "high", "low", "close"))),
    ("gap_fade", lambda df: models.sig_gap_fade(
        df.index, *_a(df, "open", "high", "low", "close"))),
]


def _a(df, *cols):
    return [df[x].to_numpy() for x in cols]


def main():
    rows = []
    for name, path, fmt, tfs, cost in MARKETS:
        base = (engine.load_mt5_csv(path) if fmt == "mt5"
                else engine.load_csv(path))
        for tf in tfs:
            df = engine.resample(base, TF_RULE[tf])
            if len(df) < 1500:
                continue
            c = df["close"].to_numpy()
            years = df.index.year.to_numpy()
            for sig_name, fn in SIGNALS:
                try:
                    idx, d = fn(df)
                except Exception as e:
                    print(f"{name}_{tf} {sig_name}: ERR {e}", file=sys.stderr)
                    continue
                if len(idx) < 50:
                    continue
                for h_ in (5, 20):
                    r = sim_time(c, idx, d, h_, cost)
                    if r is None or r["n"] < 50:
                        continue
                    row = {"market": f"{name}_{tf}", "signal": sig_name,
                           "h": h_, "n": r["n"],
                           "exp_bp": float(np.mean(r["net_bp"])),
                           "p_win": float(np.mean(r["net_bp"] > 0))}
                    for y in np.unique(years):
                        m = years[r["idx"]] == y
                        if m.sum() >= 20:
                            row[f"y{y}"] = float(np.mean(r["net_bp"][m]))
                    rows.append(row)
            print(f"{name}_{tf}: ok", file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv("out/models_scan.csv", index=False)
    ycols = sorted(c for c in out.columns if c.startswith("y"))

    def pos_share(r):
        vals = [r[c] for c in ycols if pd.notna(r[c])]
        return sum(v > 0 for v in vals) / len(vals) if vals else np.nan

    out["pos_years"] = out.apply(pos_share, axis=1)
    out["n_years"] = out[ycols].notna().sum(axis=1)
    top = out[(out.n_years >= 3)].sort_values("exp_bp", ascending=False)
    pd.set_option("display.width", 250)
    print("\n=== Топ по нетто-ожиданию (>=3 лет в данных) ===")
    print(top[["market", "signal", "h", "n", "exp_bp", "p_win",
               "pos_years", "n_years"]].head(25).round(2).to_string(index=False))
    print("\n=== Устойчивые: pos_years==1.0 и n_years>=4, exp_bp>0 ===")
    st = out[(out.pos_years == 1.0) & (out.n_years >= 4) & (out.exp_bp > 0)]
    print(st[["market", "signal", "h", "n", "exp_bp", "p_win",
              "n_years"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
