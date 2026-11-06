"""Прогон X12 дифференциал ставок. out/x12.csv"""

import sys

import pandas as pd

import engine
import rates as rt

B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
cb = pd.read_csv("data/raw/cb_rates.csv", index_col=0, parse_dates=True)
PAIRS = {"EURUSD": ("mt5", "data/raw/EURUSD_PERIOD_D1.csv", "EUR", "USD"), "GBPUSD": ("mt5", "data/raw/GBPUSD_PERIOD_D1.csv", "GBP", "USD"),
         "USDJPY": ("csv", "data/raw/USDJPY_1d.csv", "USD", "JPY"), "AUDUSD": ("csv", "data/raw/AUDUSD_1d.csv", "AUD", "USD"),
         "NZDUSD": ("csv", "data/raw/NZDUSD_1d.csv", "NZD", "USD"), "USDCAD": ("csv", "data/raw/USDCAD_1d.csv", "USD", "CAD"),
         "USDCHF": ("csv", "data/raw/USDCHF_1d.csv", "USD", "CHF")}
KEYS = ["lvl_ret3", "lvl_p3", "lvl_ret6", "lvl_p6", "chg_ret3", "chg_p3", "chg_ret6", "chg_p6", "rho_chg3", "rho_chg6", "rho_lvl3", "rho_lvl6", "rho_same3"]
rows = []
pooled = []
for name, (kind, path, b, q) in PAIRS.items():
    px = engine.load_mt5_csv(path) if kind == "mt5" else engine.load_csv(path)
    px.index = px.index.normalize(); px = px[~px.index.duplicated()].close
    px = px[px.index >= "1999-01-01"]
    real, z, n = rt.rates_test(px, cb[b], cb[q], B=B)
    print(f"\n===== {name} ({b}−{q}): месяцев {n}, событий Δdiff {int(real['chg_n3'])}")
    for k in KEYS:
        if k in z:
            r = z[k]; print(f"  {k:10s} real={r['real']:+.3f} | null={r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
    for k, r in z.items():
        rows.append({"market": name, "metric": k, **r})
pd.DataFrame(rows).to_csv("out/x12.csv", index=False)
print("\nзаписано out/x12.csv")
