"""Прогон X04 сезонность, X05 quarterly shift, X06 IPDA 20/40/60. out/x04_x06.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import ict_checks as ic
import ipda
import seasonality as se

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
D1 = [
    ("EURUSD", "data/raw/EURUSD_PERIOD_D1.csv", "mt5"),
    ("GBPUSD", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5"),
    ("XAUUSD", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5"),
    ("GC",     "data/raw/GC_1d.csv",            "utc"),
    ("NQ",     "data/raw/NQ_1d.csv",            "utc"),
    ("ES",     "data/raw/ES_1d.csv",            "utc"),
    ("USTEC",  "data/raw/USTEC_PERIOD_D1.csv",  "mt5"),
]
rows = []


def load(path, fmt):
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    df.index = df.index.normalize()
    return df[df.index.dayofweek < 5]


def collect(market, test, metric, real, null_mean=np.nan, null_sd=np.nan, z=np.nan, extra=""):
    rows.append({"market": market, "test": test, "metric": metric, "real": real,
                 "null_mean": null_mean, "null_sd": null_sd, "z": z, "extra": extra})


print("#################### X04 сезонность: месяцы (нулл = перестановка месяцев в году)")
data = {}
for name, path, fmt in D1:
    df = load(path, fmt)
    data[name] = df
    if len(df) < 2500:
        continue
    r = se.monthly_test(df.close, B=1000)
    print(f"\n--- {name}: {r['n_years']} полных лет; max|t| по месяцам = {r['max_abs_t']:.2f}, "
          f"p(нулл ≥) = {r['p_maxt']:.3f}")
    print("   мес:  " + " ".join(f"{m:>6d}" for m in r["months"]))
    print("   mean% " + " ".join(f"{v:6.2f}" for v in r["mean_pct"]))
    print("   t     " + " ".join(f"{v:6.2f}" for v in r["t"]))
    print("   p_pos " + " ".join(f"{v:6.2f}" for v in r["p_pos"]))
    print("   z     " + " ".join(f"{v:6.2f}" for v in r["z"]))
    for m, mu, t, z in zip(r["months"], r["mean_pct"], r["t"], r["z"]):
        collect(name, "X04_month", f"m{m:02d}_mean_pct", mu, z=z, extra=f"t={t:.2f}")
    collect(name, "X04_month", "max_abs_t", r["max_abs_t"], extra=f"p_null={r['p_maxt']:.3f}")

print("\n#################### X04 сезонность: окна ICT (нулл = случайное окно той же длины)")
for wname, mkts, start, end, sign in se.WINDOWS:
    for name in data:
        if "*" not in mkts and name not in mkts:
            continue
        if len(data[name]) < 2500:
            continue
        r = se.window_test(data[name].close, start, end, sign, B=500)
        tag = "ER" if sign == 0 else "ret×sign"
        print(f"  {wname[:52]:52s} {name:7s} n={r['n_years']:2d} {tag}={r['mean']*100:+6.2f}% "
              f"t={r['t']:+5.2f} p_pos={r['p_pos']:.2f} h1/h2={r['h1']*100:+.1f}/{r['h2']*100:+.1f} "
              f"| нулл {r['null_mean']*100:+.2f}±{r['null_sd']*100:.2f} z={r['z']:+5.2f}")
        collect(name, "X04_window", wname, r["mean"], r["null_mean"], r["null_sd"], r["z"],
                extra=f"t={r['t']:.2f} p_pos={r['p_pos']:.2f} n={r['n_years']}")

print("\n#################### X05 quarterly shift (нулл = бутстреп D1 block=250)")
for name, df in data.items():
  for kk in (4.0, 8.0):
    real, z = ic.bootstrap_d1(df, lambda d, kk=kk: ipda.quarterly_stats(d, k=kk), B=B, block=250)
    print(f"\n--- {name}: свингов {int(real['n_swings'])} (k={kk:.0f}·ATR100), 3м-пар {int(real['n_3m'])}")
    for k in ["gap_median", "gap_mean", "gap_p_50_90", "gap_p_lt30", "gap_cv", "turn_p_qs_months",
              "turn_p_first5days", "corr_3m", "p_cont_3m", "p_cont_after_big3m"]:
        if k in z:
            r = z[k]
            print(f"  {k:22s} real={r['real']:7.3f}  null={r['null_mean']:7.3f}±{r['null_sd']:.3f}  z={r['z']:+6.2f}")
            collect(name, "X05", k, r["real"], r["null_mean"], r["null_sd"], r["z"], extra=f"k={kk:.0f}")

print("\n#################### X06 IPDA 20/40/60 (нулл = бутстреп D1 block=250)")
for name, df in data.items():
    real, z = ic.bootstrap_d1(df, ipda.ipda_stats, B=B, block=250)
    print(f"\n--- {name}: break20={int(real.get('n_break20', 0))} touchL20={int(real.get('n_touchL20', 0))} "
          f"touchL40={int(real.get('n_touchL40', 0))} swings={int(real['n_swings'])}")
    for k in ["p_touch20_in10", "p_touch20_in20", "p_touch40_in20", "p_touch60_in20",
              "p_reach40_after_break20", "p_reach40_ctrl_samedist", "p_reach60_after_break20",
              "p_up3_after_touchL20", "p_up3_after_touchL40", "p_dn3_after_touchH20", "p_dn3_after_touchH40",
              "p_turn_any5", "p_turn_at10", "p_turn_at20", "p_turn_at30", "p_turn_at40", "p_turn_at50",
              "p_turn_at60", "p_turn_at70"]:
        if k in z:
            r = z[k]
            print(f"  {k:24s} real={r['real']:7.3f}  null={r['null_mean']:7.3f}±{r['null_sd']:.3f}  z={r['z']:+6.2f}")
            collect(name, "X06", k, r["real"], r["null_mean"], r["null_sd"], r["z"])
        elif k in real:
            print(f"  {k:24s} real={real[k]:7.3f}")

pd.DataFrame(rows).to_csv("out/x04_x06.csv", index=False)
print("\nзаписано out/x04_x06.csv")
