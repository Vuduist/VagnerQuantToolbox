"""Прогон T01 (killzones → экстремум дня) и S03 (range expansion). out/t01_s03.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import ict_checks as ic
import killzones as kz
import range_expansion as rx
import sessions

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SRC_TZ = {"mt5": "Europe/Helsinki", "utc": "UTC"}
H1 = [
    ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5"),
    ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5"),
    ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5"),
    ("USTEC_H1",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5"),
    ("NQ_1h",     "data/raw/NQ_1h.csv",            "utc"),
    ("ES_1h",     "data/raw/ES_1h.csv",            "utc"),
]
M5 = [("USTEC_M5", "data/raw/USTEC_PERIOD_M5.csv", "mt5")]
D1 = [
    ("EURUSD_D1", "data/raw/EURUSD_PERIOD_D1.csv", "mt5"),
    ("GBPUSD_D1", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5"),
    ("XAUUSD_D1", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5"),
    ("USTEC_D1",  "data/raw/USTEC_PERIOD_D1.csv",  "mt5"),
    ("NQ_1d",     "data/raw/NQ_1d.csv",            "utc"),
    ("ES_1d",     "data/raw/ES_1d.csv",            "utc"),
]
WIN_M5 = {
    "london_2_5": tuple(range(2, 5)), "ny_open_7_10": tuple(range(7, 10)),
    "or_930_1030": ("930",), "london_close_10_12": (10, 11),
    "pm_13_16": (13, 14, 15), "asia_0_2": (0, 1),
}

rows = []


def load(path, fmt):
    return engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)


def collect(name, test, real, z):
    for k, v in real.items():
        r = z.get(k, {})
        rows.append({"market": name, "test": test, "metric": k, "real": v,
                     "null_mean": r.get("null_mean", np.nan),
                     "null_sd": r.get("null_sd", np.nan), "z": r.get("z", np.nan)})


def show(real, z, keys):
    for k in keys:
        if k in z:
            r = z[k]
            print(f"  {k:26s} real={r['real']:6.3f}  null={r['null_mean']:6.3f}±{r['null_sd']:.3f}  z={r['z']:+6.2f}")
        elif k in real:
            print(f"  {k:26s} real={real[k]:6.3f}")


print("#################### T01 killzones -> экстремум дня")
for name, path, fmt in H1:
    df = sessions.to_et(load(path, fmt), SRC_TZ[fmt])
    real, z = kz.killzone_test(df, 60, B=B)
    print(f"\n--- {name}: полных дней {int(real['n_days'])}")
    show(real, z, [f"{w}_{s}" for w in kz.WINDOWS_H1 for s in ("p_high", "p_low", "p_any")]
         + ["p_high_before_14", "p_low_before_14"])
    print("  часовой профиль P(экстремум в часе), реал / нулл:")
    print("   " + " ".join(f"{hr:02d}:{real[f'hour{hr:02d}_p_ext']*100:4.1f}/{z[f'hour{hr:02d}_p_ext']['null_mean']*100:4.1f}"
                           for hr in range(16)))
    collect(name, "T01", real, z)

# M5: окно opening range 9:30–10:30 точно
for name, path, fmt in M5:
    df = sessions.to_et(load(path, fmt), SRC_TZ[fmt])
    comp, open0, hours, days = kz.day_matrix(df, 5)
    slots = np.arange(comp.shape[2])
    minutes = slots * 5
    win = {
        "london_2_5": tuple(slots[(minutes >= 120) & (minutes < 300)]),
        "ny_open_7_10": tuple(slots[(minutes >= 420) & (minutes < 600)]),
        "or_930_1030": tuple(slots[(minutes >= 570) & (minutes < 630)]),
        "or_930_1000": tuple(slots[(minutes >= 570) & (minutes < 600)]),
        "london_close_10_12": tuple(slots[(minutes >= 600) & (minutes < 720)]),
        "pm_13_16": tuple(slots[(minutes >= 780) & (minutes < 960)]),
    }
    # hours-массив для M5 = сам слот, окна заданы в слотах
    hi, lo = kz.extreme_slots(comp, open0)
    real = kz.window_stats(hi, lo, slots, win)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(B):
        hb, lb = kz.extreme_slots(kz.shuffled(comp, slots, rng), open0)
        boots.append(kz.window_stats(hb, lb, slots, win))
    z = {}
    for k, v in real.items():
        if k.startswith("n_") or k.startswith("hour"):
            continue
        vals = np.array([b[k] for b in boots]); sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    print(f"\n--- {name} (M5, слоты 5 мин): полных дней {int(real['n_days'])}")
    show(real, z, [f"{w}_{s}" for w in win for s in ("p_high", "p_low", "p_any")])
    collect(name, "T01_M5", {k: v for k, v in real.items() if not k.startswith("hour")}, z)

print("\n#################### S03 range expansion после сжатия (D1 и недели)")
for name, path, fmt in D1:
    df = load(path, fmt)
    df.index = df.index.normalize()
    df = df[df.index.dayofweek < 5]
    real, z = ic.bootstrap_d1(df, rx.expansion_stats, B=B, block=250)
    print(f"\n--- {name} D1: n={int(real['n_bars'])} inside={int(real['n_inside'])} nr7={int(real['n_nr7'])} small={int(real['n_small'])}")
    show(real, z, ["all_next_mean", "all_p_next_gt15",
                   "inside_next_mean", "inside_ratio_vs_all", "inside_p_next_gt15", "inside_p_next2_gt15",
                   "nr4_next_mean", "nr4_ratio_vs_all",
                   "nr7_next_mean", "nr7_ratio_vs_all", "nr7_p_next_gt15",
                   "small_next_mean", "small_ratio_vs_all", "small_p_next_gt15",
                   "big_next_mean", "big_ratio_vs_all"])
    collect(name, "S03_D1", real, z)
    w = rx.weekly(df)
    if len(w) > 400:
        real, z = ic.bootstrap_d1(w, rx.expansion_stats, B=B, block=52)
        print(f"--- {name} недели: n={int(real['n_bars'])} inside={int(real['n_inside'])} nr4={int(real['n_nr4'])}")
        show(real, z, ["all_next_mean", "inside_next_mean", "inside_ratio_vs_all", "inside_p_next2_gt15",
                       "nr4_next_mean", "nr4_ratio_vs_all", "small_ratio_vs_all", "big_ratio_vs_all"])
        collect(name, "S03_W", real, z)

pd.DataFrame(rows).to_csv("out/t01_s03.csv", index=False)
print("\nзаписано out/t01_s03.csv")
