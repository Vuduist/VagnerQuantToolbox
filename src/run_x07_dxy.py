"""Прогон X07 DXY-SMT. out/x07_dxy.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import sessions
import smt_dxy as sm

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rows = []


def daily(path):
    df = engine.load_mt5_csv(path)
    df.index = df.index.normalize()
    return df[~df.index.duplicated()]


dxy_d = engine.load_csv("data/raw/DXY_1d.csv"); dxy_d.index = dxy_d.index.normalize()
dxy_h = engine.load_csv("data/raw/DXY_1h.csv")
dxy_h.index = dxy_h.index.tz_localize("UTC")
dxy_h = dxy_h[~dxy_h.index.duplicated()]

for name, path in (("EURUSD", "data/raw/EURUSD_PERIOD_D1.csv"), ("GBPUSD", "data/raw/GBPUSD_PERIOD_D1.csv")):
    pair = daily(path)
    pair = pair[pair.index >= "1990-01-01"]
    for L in (20, 60):
        real, z, n = sm.smt_test(pair, dxy_d, L=L, B=B)
        print(f"\n===== {name} D1 vs DXY, L={L}, дней {n}: события div_bull {int(real['div_bull_n'])} div_bear {int(real['div_bear_n'])} "
              f"conf_bear {int(real['conf_bear_n'])} conf_bull {int(real['conf_bull_n'])}")
        for k in ("div_ret5", "div_p5", "div_ret10", "div_p10", "div_ret20", "div_p20",
                  "conf_ret5", "conf_p5", "conf_ret10", "conf_p10", "conf_ret20", "conf_p20"):
            if k in z:
                r = z[k]; print(f"  {k:12s} real={r['real']:+.3f} | null={r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
        for k, r in z.items():
            rows.append({"market": f"{name}_D1_L{L}", "metric": k, **r})
    # H1: последние ~2 года (DXY_1h с Yahoo)
    h1 = sessions.to_et(engine.load_mt5_csv(path.replace("_D1", "_H1"))).tz_convert("UTC")
    h1.index = h1.index.floor("h")
    h1 = h1[~h1.index.duplicated()]
    real, z, n = sm.smt_test(h1, dxy_h, L=24, hs=(4, 12, 24), B=B)
    print(f"\n===== {name} H1 vs DXY_1h, L=24, баров {n}: события div_bull {int(real['div_bull_n'])} div_bear {int(real['div_bear_n'])} "
          f"conf_bear {int(real['conf_bear_n'])} conf_bull {int(real['conf_bull_n'])}")
    for k in ("div_ret4", "div_p4", "div_ret12", "div_p12", "div_ret24", "div_p24",
              "conf_ret4", "conf_p4", "conf_ret12", "conf_p12", "conf_ret24", "conf_p24"):
        if k in z:
            r = z[k]; print(f"  {k:12s} real={r['real']:+.3f} | null={r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
    for k, r in z.items():
        rows.append({"market": f"{name}_H1_L24", "metric": k, **r})
pd.DataFrame(rows).to_csv("out/x07_dxy.csv", index=False)
print("\nзаписано out/x07_dxy.csv")
