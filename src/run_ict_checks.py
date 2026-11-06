"""Прогон ict_checks: W01 дни недели, W02 TGIF (D1, бутстреп block=250),
T07 London/ADR (H1 в ET, перестановка сессий). Результат: out/ict_checks.csv."""

import sys

import numpy as np
import pandas as pd

import engine
import ict_checks as ic
import sessions

D1 = [
    ("EURUSD_D1", "data/raw/EURUSD_PERIOD_D1.csv", "mt5"),
    ("GBPUSD_D1", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5"),
    ("XAUUSD_D1", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5"),
    ("USTEC_D1",  "data/raw/USTEC_PERIOD_D1.csv",  "mt5"),
    ("NQ_1d",     "data/raw/NQ_1d.csv",            "utc"),
    ("ES_1d",     "data/raw/ES_1d.csv",            "utc"),
]
H1 = [
    ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5"),
    ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5"),
    ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5"),
    ("USTEC_H1",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5"),
    ("NQ_1h",     "data/raw/NQ_1h.csv",            "utc"),
    ("ES_1h",     "data/raw/ES_1h.csv",            "utc"),
]
SRC_TZ = {"mt5": "Europe/Helsinki", "utc": "UTC"}


def load(path, fmt):
    return engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)


def show(name, test, real, z, keys):
    print(f"\n--- {name} / {test}  (" +
          ", ".join(f"{k}={int(v)}" for k, v in real.items() if k.startswith("n_")) + ")")
    for k in keys:
        if k in z:
            r = z[k]
            print(f"  {k:24s} real={r['real']:7.3f}  null={r['null_mean']:7.3f}±{r['null_sd']:.3f}  z={r['z']:+6.2f}")
        elif k in real:
            print(f"  {k:24s} real={real[k]:7.3f}  (нулл: n/a)")


rows = []


def collect(name, test, real, z):
    for k, v in real.items():
        r = z.get(k, {})
        rows.append({"market": name, "test": test, "metric": k, "real": v,
                     "null_mean": r.get("null_mean", np.nan),
                     "null_sd": r.get("null_sd", np.nan), "z": r.get("z", np.nan)})


B = int(sys.argv[1]) if len(sys.argv) > 1 else 100

for name, path, fmt in D1:
    df = load(path, fmt)
    df.index = df.index.normalize()
    print(f"\n===== {name}: {len(df)} дней, {df.index[0].date()} .. {df.index[-1].date()} =====")
    real, z = ic.bootstrap_d1(df, ic.weekday_extremes, B=B)
    show(name, "W01 weekday", real, z,
         ["all_lo_Mon", "all_lo_Tue", "all_lo_Wed", "all_lo_Thu", "all_lo_Fri",
          "all_hi_Mon", "all_hi_Fri", "all_lo_MonWed",
          "up_lo_Tue", "up_lo_MonTue", "up_lo_MonWed", "dn_hi_Tue", "dn_hi_MonTue", "dn_hi_MonWed"])
    collect(name, "W01", real, z)
    real, z = ic.bootstrap_d1(df, ic.tgif, B=B)
    show(name, "W02 TGIF", real, z,
         ["strong_retr_median", "strong_p_retr_ge20", "strong_p_retr_20_30", "strong_p_retr_ge50",
          "strong_p_fri_close_against", "strong_fri_ret_dir_bp",
          "weak_retr_median", "weak_p_retr_ge20", "weak_p_fri_close_against"])
    collect(name, "W02", real, z)

for name, path, fmt in H1:
    df = sessions.to_et(load(path, fmt), SRC_TZ[fmt])
    print(f"\n===== {name}: {len(df)} баров, {df.index[0]} .. {df.index[-1]} =====")
    real, z = ic.london_adr_rule(df, B=2 * B)
    show(name, "T07 London/ADR", real, z,
         ["p_dom", "dom_ny_rng_adr", "nodom_ny_rng_adr", "diff_ny_rng_adr",
          "dom_ny_absnet_adr", "nodom_ny_absnet_adr",
          "dom_p_ny_expand", "nodom_p_ny_expand", "diff_p_ny_expand",
          "dom_p_ny_reverse", "nodom_p_ny_reverse", "diff_p_ny_reverse"])
    collect(name, "T07", real, z)

pd.DataFrame(rows).to_csv("out/ict_checks.csv", index=False)
print("\nзаписано out/ict_checks.csv")
