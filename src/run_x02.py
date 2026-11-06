"""Прогон X02 open interest. out/x02.csv"""

import sys

import pandas as pd

import cot_oi as co
import engine

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
cotd = pd.read_csv("data/raw/cot_legacy.csv", parse_dates=["date"])
PRICES = {"EUR": ("mt5", "data/raw/EURUSD_PERIOD_D1.csv", False), "GBP": ("mt5", "data/raw/GBPUSD_PERIOD_D1.csv", False),
          "JPY": ("csv", "data/raw/USDJPY_1d.csv", True), "GOLD": ("mt5", "data/raw/XAUUSD_PERIOD_D1.csv", False),
          "ES": ("csv", "data/raw/ES_1d.csv", False)}
rows = []
for mk, (kind, path, inv) in PRICES.items():
    px = engine.load_mt5_csv(path) if kind == "mt5" else engine.load_csv(path)
    px.index = px.index.normalize(); px = px[~px.index.duplicated()].close
    cm = cotd[cotd.market == mk]
    for k, thr in ((4, 0.15), (4, 0.10), (8, 0.15)):
        real, z, n = co.oi_test(cm, px, invert=inv, B=B, k=k, thr=thr)
        tag = f"{mk}_k{k}_thr{int(thr*100)}"
        print(f"\n===== {tag}: недель {n}; событий drop {int(real['drop_n'])} rise {int(real['rise_n'])} "
              f"drop&pdown {int(real['drop_pdown_n'])} rise&pup {int(real['rise_pup_n'])} drop&pup {int(real['drop_pup_n'])} rise&pdown {int(real['rise_pdown_n'])} "
              f"dev_low&pdown {int(real['dev_low_pdown_n'])} dev_high&pup {int(real['dev_high_pup_n'])}")
        for key in ("drop_pdown_ret4", "drop_pdown_p4", "drop_pdown_ret13", "drop_pdown_p13", "rise_pup_ret13", "rise_pup_p13",
                    "drop_pup_ret13", "rise_pdown_ret13", "dev_low_pdown_ret13", "dev_low_pdown_p13", "dev_high_pup_ret13",
                    "rho4_all", "rho13_all", "oi_max_price_pos"):
            if key in z:
                r = z[key]; print(f"  {key:22s} real={r['real']:+.3f} | null={r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
        for key, r in z.items():
            rows.append({"market": tag, "metric": key, **r})
pd.DataFrame(rows).to_csv("out/x02.csv", index=False)
print("\nзаписано out/x02.csv")
