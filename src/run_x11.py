"""Прогон X11 новостные дни. out/x11.csv"""

import sys

import pandas as pd

import engine
import news_days as nd
import sessions

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
cal = pd.read_csv("data/calendar_events.csv", parse_dates=["date"])
SETS = [("EURUSD_H1", "mt5", "data/raw/EURUSD_PERIOD_H1.csv"), ("GBPUSD_H1", "mt5", "data/raw/GBPUSD_PERIOD_H1.csv"),
        ("XAUUSD_H1", "mt5", "data/raw/XAUUSD_PERIOD_H1.csv"), ("USTEC_H1", "mt5", "data/raw/USTEC_PERIOD_H1.csv"),
        ("NQ_1h", "yahoo", "data/raw/NQ_1h.csv"), ("ES_1h", "yahoo", "data/raw/ES_1h.csv")]
rows = []
for name, kind, path in SETS:
    if kind == "mt5":
        df = sessions.to_et(engine.load_mt5_csv(path))
    else:
        df = engine.load_csv(path); df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    out = nd.news_test(df, cal, B=B)
    print(f"\n===== {name}: NFP {int(out['NFP_n'])} vs пятниц {int(out['NFP_ctl_n'])}; FOMC {int(out['FOMC_n'])} vs {int(out['FOMC_ctl_n'])}; NFP-неделя Чт/Пт {int(out['NFPWK_n'])}")
    for k, r in out.items():
        if isinstance(r, dict):
            print(f"  {k:22s} real={r['real']:+.3f} | ctl={r['null_mean']:+.3f} z{r['z']:+.1f}")
            rows.append({"market": name, "metric": k, **r})
pd.DataFrame(rows).to_csv("out/x11.csv", index=False)
print("\nзаписано out/x11.csv")
