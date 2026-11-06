"""Прогон W04 NWOG как S/R. out/w04_sr.csv"""

import sys

import pandas as pd

import engine
import nwog_sr as nw
import sessions

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
B = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SETS = [("EURUSD_H1", "mt5", "data/raw/EURUSD_PERIOD_H1.csv"), ("GBPUSD_H1", "mt5", "data/raw/GBPUSD_PERIOD_H1.csv"),
        ("XAUUSD_H1", "mt5", "data/raw/XAUUSD_PERIOD_H1.csv"), ("USTEC_H1", "mt5", "data/raw/USTEC_PERIOD_H1.csv"),
        ("NQ_1h", "yahoo", "data/raw/NQ_1h.csv"), ("ES_1h", "yahoo", "data/raw/ES_1h.csv")]
KEYS = ["touch_per_week", "all_p_bounce", "all_p_through", "all_p_bounce_dec", "filled_p_bounce_dec", "unfilled_p_bounce_dec",
        "support_p_bounce_dec", "resist_p_bounce_dec", "first_p_bounce_dec", "first_p_bounce"]
rows = []
for name, kind, path in SETS:
    if kind == "mt5":
        df = sessions.to_et(engine.load_mt5_csv(path))
    else:
        df = engine.load_csv(path); df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        df = df[~df.index.duplicated()]
    real, z = nw.nwog_test(df, R=R, B=B)
    print(f"\n===== {name}: недель с зоной {int(real['n_weeks'])}, касаний {int(real['n_touch'])} "
          f"(filled {int(real.get('filled_n', 0))}, unfilled {int(real.get('unfilled_n', 0))}, support {int(real.get('support_n', 0))}, resist {int(real.get('resist_n', 0))})")
    for k in KEYS:
        if k in z and "n1_mean" in z[k]:
            r = z[k]
            s = f"  {k:22s} real={r['real']:+.3f} | n1={r['n1_mean']:+.3f}±{r['n1_sd']:.3f} z{r['n1_z']:+.1f}"
            for nm in ("n3", "n2"):
                if nm + "_mean" in r:
                    s += f" | {nm}={r[nm + '_mean']:+.3f}±{r[nm + '_sd']:.3f} z{r[nm + '_z']:+.1f}"
            print(s)
    for k, r in z.items():
        rows.append({"market": name, "metric": k, **r})
    pd.DataFrame(rows).to_csv("out/w04_sr.csv", index=False)
print("\nзаписано out/w04_sr.csv")
