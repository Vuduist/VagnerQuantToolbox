"""Парный тест выходов: случайные входы -> разные конструкции выхода.

python run_exits.py
"""

import sys

import numpy as np
import pandas as pd

import engine
import exits
import swings as sw

MARKETS = [
    ("BTC_15m",   "data/raw/BTCUSDT_1m.csv",       "binance", "15min", 10.0),
    ("BTC_1h",    "data/raw/BTCUSDT_1m.csv",       "binance", "1h",    10.0),
    ("USTEC_1h",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5",     None,     1.0),
    ("USTEC_15m", "data/raw/USTEC_PERIOD_M5.csv",  "mt5",     "15min",  1.0),
    ("EURUSD_1h", "data/raw/EURUSD_PERIOD_H1.csv", "mt5",     None,     1.5),
    ("EURUSD_1d", "data/raw/EURUSD_PERIOD_D1.csv", "mt5",     None,     1.5),
    ("GBPUSD_1h", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5",     None,     1.5),
    ("XAUUSD_1h", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5",     None,     1.0),
    ("NQ_1d",     "data/raw/NQ_1d.csv",            "binance", None,     1.0),
]

EXITS = [
    ("time_10",      lambda **k: exits.exit_time(K=10, **k)),
    ("time_30",      lambda **k: exits.exit_time(K=30, **k)),
    ("atr_1.5r1",    lambda **k: exits.exit_atr_rr(K=60, rr=1.0, **k)),
    ("atr_1.5r2",    lambda **k: exits.exit_atr_rr(K=120, rr=2.0, **k)),
    ("atr_1.5r3",    lambda **k: exits.exit_atr_rr(K=200, rr=3.0, **k)),
    ("chandelier3",  lambda **k: exits.exit_chandelier(K=200, **k)),
    ("swing_trail",  lambda **k: exits.exit_swing(K=200, **k)),
]

ENTR_STEP = 20          # вход каждые 20 баров
SEED = 12345


def main():
    rows = []
    for name, path, fmt, rule, cost in MARKETS:
        df = (engine.load_mt5_csv(path) if fmt == "mt5"
              else engine.load_csv(path))
        if rule:
            df = engine.resample(df, rule)
        if len(df) < 3000:
            continue
        o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
        years = df.index.year.to_numpy()
        atr_arr = exits.atr(h, l, c)
        _, _, sw_dir, sw_conf = sw.causal_swings(h, l, c, k=3.0)

        rng = np.random.default_rng(SEED)
        idx = np.arange(200, len(df) - 300, ENTR_STEP)
        d = rng.choice([-1.0, 1.0], size=len(idx))   # случайное направление

        for ename, fn in EXITS:
            net, used = fn(h=h, l=l, c=c, idx=idx, d=d, cost_bp=cost,
                           atr_arr=atr_arr, sw_conf=sw_conf, sw_dir=sw_dir)
            row = {"market": name, "exit": ename, "n": len(net),
                   "exp_bp": float(np.mean(net)),
                   "p_win": float(np.mean(net > 0))}
            for y in np.unique(years):
                m = years[used] == y
                if m.sum() >= 20:
                    row[f"y{y}"] = float(np.mean(net[m]))
            rows.append(row)
        print(f"{name}: ok ({len(df)} баров, {len(idx)} входов)",
              file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv("out/exits_scan.csv", index=False)
    ycols = sorted(c for c in out.columns if c.startswith("y"))
    out["pos_years"] = (out[ycols].gt(0).sum(axis=1)
                        / out[ycols].notna().sum(axis=1))

    print("\n=== Среднее по рынкам для каждого выхода ===")
    agg = out.groupby("exit").agg(mean_bp=("exp_bp", "mean"),
                                  med_bp=("exp_bp", "median"),
                                  pos_mk=("exp_bp", lambda s: (s > 0).mean()),
                                  pos_yr=("pos_years", "mean"))
    print(agg.round(2).to_string())
    print("\n=== Детально ===")
    print(out.pivot_table(index="market", columns="exit",
                          values="exp_bp").round(1).to_string())


if __name__ == "__main__":
    main()
