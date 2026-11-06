"""Прогон W03 опорные уровни, T02 macros, T09 NY open, S06 круглые числа. out/w03_t02_t09_s06.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import macros
import reflevels
import rounds
import sessions

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SRC = {"mt5": "Europe/Helsinki", "utc": "UTC"}
rows = []


def collect(market, test, metric, r):
    rows.append({"market": market, "test": test, "metric": metric, "real": r["real"],
                 "null_mean": r.get("null_mean", np.nan), "null_sd": r.get("null_sd", np.nan), "z": r.get("z", np.nan)})


def load(path, fmt, rule=None):
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    return sessions.to_et(df, SRC[fmt])


H1 = [("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5"), ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5"),
      ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5"), ("USTEC_H1", "data/raw/USTEC_PERIOD_H1.csv", "mt5"),
      ("BTC_1h", "data/raw/BTCUSDT_1m.csv", "utc")]

print("#################### W03 опорные уровни (нулл = случайный close предыдущего периода)")
for name, path, fmt in H1:
    df = load(path, fmt, "1h" if fmt == "utc" else None)
    print(f"\n--- {name}: {len(df)} баров")
    for kind, lvl, label in (("week", "open", "weekly_open"), ("week", "prev_close", "friday_close"),
                             ("day", "open", "midnight_open"), ("month", "open", "monthly_open")):
        real, z = reflevels.level_test(df, kind, lvl, d=1.0, K=4, B=B // 4)
        print(f"  {label:14s} n_dep={int(real['n_dep']):5d} n_rev={int(real['n_rev']):5d} | " +
              " ".join(f"{k}={z[k]['real']:.3f}(нулл {z[k]['null_mean']:.3f}±{z[k]['null_sd']:.3f} z={z[k]['z']:+.2f})"
                       for k in ("p_revisit", "p_bounce", "mean_ret_K_bp")))
        for k in z:
            collect(name, f"W03_{label}", k, z[k])

print("\n#################### T02 macros / T09 NY open (M5, нулл = перестановка слотов с профилем волы)")
M5 = [("USTEC_M5", "data/raw/USTEC_PERIOD_M5.csv", "mt5", None), ("BTC_5m", "data/raw/BTCUSDT_1m.csv", "utc", "5min"),
      ("EURUSD_M5", "data/raw/EURUSD_PERIOD_M5.csv", "mt5", None), ("XAUUSD_M5", "data/raw/XAUUSD_PERIOD_M5.csv", "mt5", None)]
for name, path, fmt, rule in M5:
    df = load(path, fmt, rule)
    real, z = macros.macro_test(df, B=B)
    print(f"\n--- {name}: полных дней {int(real['n_days'])}")
    for w in macros.WINDOWS:
        print(f"  {w:16s} " + " ".join(f"{m}={z[f'{w}_{m}']['real']:.3f}/{z[f'{w}_{m}']['null_mean']:.3f}(z{z[f'{w}_{m}']['z']:+.1f})"
                                       for m in ("rng_share", "absnet_share", "er", "p_break1h")))
    for k in [k for k in real if k.startswith("t09")]:
        if k in z:
            print(f"  {k:30s} real={z[k]['real']:.3f} null={z[k]['null_mean']:.3f}±{z[k]['null_sd']:.3f} z={z[k]['z']:+.2f}")
        else:
            print(f"  {k:30s} {real[k]:.3f}")
    for k in z:
        collect(name, "T02" if not k.startswith("t09") else "T09", k, z[k])

print("\n#################### S06 круглые числа (плотность экстремумов в ±2 ед. от уровня vs 6–10 ед.)")
FX = {"50pip": (5000, 0), "10pip": (1000, 0), "ctrl_25pip": (5000, 2500)}
AU = {"10usd": (1000, 0), "ctrl_5usd_off": (1000, 500)}
IX = {"100pt": (1000, 0), "ctrl_50pt_off": (1000, 500)}
RND = [("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", 1e-5, 100, FX, "пип"),
       ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", 1e-5, 100, FX, "пип"),
       ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", 1e-2, 50, AU, "$0.5"),
       ("USTEC_H1", "data/raw/USTEC_PERIOD_H1.csv", 1e-1, 50, IX, "5пт"),
       ("USTEC_M5", "data/raw/USTEC_PERIOD_M5.csv", 1e-1, 50, IX, "5пт")]
for name, path, tick, unit, levels, ulab in RND:
    df = engine.load_mt5_csv(path)
    r = rounds.digit_stats(df, tick, unit, levels, K=3)
    print(f"\n--- {name}: n={int(r['n_bars'])}, единица={ulab}")
    for g in levels:
        print(f"  {g:14s} high: ratio={r[f'high_{g}_ratio']:.2f} z={r[f'high_{g}_z']:+6.1f} (n_near={int(r[f'high_{g}_n_near'])}) | "
              f"low: ratio={r[f'low_{g}_ratio']:.2f} z={r[f'low_{g}_z']:+6.1f} | реакция K=3: high {r[f'rev_high_{g}_near']:.3f} vs {r[f'rev_high_{g}_other']:.3f} (z{r[f'rev_high_{g}_z']:+.1f}), "
              f"low {r[f'rev_low_{g}_near']:.3f} vs {r[f'rev_low_{g}_other']:.3f} (z{r[f'rev_low_{g}_z']:+.1f})")
    for k, v in r.items():
        rows.append({"market": name, "test": "S06", "metric": k, "real": v, "null_mean": np.nan, "null_sd": np.nan, "z": np.nan})

pd.DataFrame(rows).to_csv("out/w03_t02_t09_s06.csv", index=False)
print("\nзаписано out/w03_t02_t09_s06.csv")
