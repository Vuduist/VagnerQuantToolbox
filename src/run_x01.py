"""Прогон X01 COT. out/x01.csv"""

import sys

import pandas as pd

import cot as ct
import engine

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cotd = pd.read_csv("data/raw/cot_legacy.csv", parse_dates=["date"])
PRICES = {"EUR": ("mt5", "data/raw/EURUSD_PERIOD_D1.csv", False), "GBP": ("mt5", "data/raw/GBPUSD_PERIOD_D1.csv", False),
          "JPY": ("csv", "data/raw/USDJPY_1d.csv", True), "GOLD": ("mt5", "data/raw/XAUUSD_PERIOD_D1.csv", False),
          "ES": ("csv", "data/raw/ES_1d.csv", False)}
KEYS = ["ext_ret4", "ext_p4", "ext_ret13", "ext_p13", "ext_ret26", "ext_p26", "bull_ret13", "bear_ret13",
        "chg_ret4", "chg_p4", "chg_ret13", "chg_p13", "rho4", "rho13", "rho26"]
rows = []
for mk, (kind, path, inv) in PRICES.items():
    px = engine.load_mt5_csv(path) if kind == "mt5" else engine.load_csv(path)
    px.index = px.index.normalize(); px = px[~px.index.duplicated()].close
    cm = cotd[cotd.market == mk]
    for W, hi, lo in ((52, 0.9, 0.1), (156, 0.9, 0.1), (52, 1.0, 0.0)):
        real, z, n = ct.cot_test(cm, px, invert=inv, W=W, B=B, hi=hi, lo=lo)
        tag = f"{mk}_W{W}_{int(hi*100)}"
        print(f"\n===== {tag}: недель {n}, экстремумов bull {int(real['n_bull'])} bear {int(real['n_bear'])}")
        for k in KEYS:
            if k in z:
                r = z[k]; print(f"  {k:10s} real={r['real']:+.3f} | null={r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
        for k, r in z.items():
            rows.append({"market": tag, "metric": k, **r})
pd.DataFrame(rows).to_csv("out/x01.csv", index=False)
print("\nзаписано out/x01.csv")
