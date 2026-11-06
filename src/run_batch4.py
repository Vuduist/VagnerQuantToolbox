"""Прогон T05/T04 (M5), B01/F01 (зоны), S05/S04/S07/S11 (D1/H1). out/batch4.csv"""

import sys

import numpy as np
import pandas as pd

import am_pm
import engine
import geometry as ge
import ict_checks as ic
import sessions
import zones2

B = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SRC = {"mt5": "Europe/Helsinki", "utc": "UTC"}
rows = []


def collect(market, test, z, real=None):
    for k, r in z.items():
        rows.append({"market": market, "test": test, "metric": k, "real": r["real"],
                     "null_mean": r["null_mean"], "null_sd": r["null_sd"], "z": r["z"]})
    if real:
        for k, v in real.items():
            if k.startswith("n_"):
                rows.append({"market": market, "test": test, "metric": k, "real": v,
                             "null_mean": np.nan, "null_sd": np.nan, "z": np.nan})


def show(real, z, keys):
    for k in keys:
        if k in z and np.isfinite(z[k]["null_sd"]):
            r = z[k]
            print(f"  {k:34s} real={r['real']:8.3f}  null={r['null_mean']:8.3f}±{r['null_sd']:.3f}  z={r['z']:+6.2f}")
        elif k in real:
            print(f"  {k:34s} real={real[k]:8.3f}")


def load(path, fmt, rule=None, et=False):
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    return sessions.to_et(df, SRC[fmt]) if et else df


print("#################### T05 AM/PM и T04 lunch (M5, нулл = перестановка слотов с профилем волы)")
M5 = [("USTEC_M5", "data/raw/USTEC_PERIOD_M5.csv", "mt5", None), ("BTC_5m", "data/raw/BTCUSDT_1m.csv", "utc", "5min"),
      ("EURUSD_M5", "data/raw/EURUSD_PERIOD_M5.csv", "mt5", None), ("XAUUSD_M5", "data/raw/XAUUSD_PERIOD_M5.csv", "mt5", None)]
for name, path, fmt, rule in M5:
    df = load(path, fmt, rule, et=True)
    real, z = am_pm.am_pm_test(df, B=B)
    print(f"\n--- {name}: дней {int(real['n_days'])}, сильных AM {int(real['n_strong_am'])}, lunch-свипов против AM {int(real['n_lunch_sweep_against'])}")
    show(real, z, ["p_pm_same_sign_as_am", "corr_am_pm_net", "p_pm_same_sign_given_strong_am",
                   "pm_over_am_absnet_median", "pm_over_am_rng_median", "day_over_am_rng_median", "p_day_rng_ge_1.8am",
                   "pm_net_in_am_dir_over_amrng", "lunch_rng_share", "p_lunch_sweeps_am_against", "p_lunch_sweeps_am_with",
                   "p_pm_cont_am_after_lunch_sweep", "p_pm_cont_am_no_lunch_sweep", "p_lunchrun_breaks_prior50"])
    collect(name, "T05_T04", z, real)

print("\n#################### B01 propulsion / F01 volume imbalance (нулл = бутстреп block=500)")
Z = [("BTC_15m", "data/raw/BTCUSDT_1m.csv", "utc", "15min"), ("USTEC_M5", "data/raw/USTEC_PERIOD_M5.csv", "mt5", None),
     ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5", None)]
for name, path, fmt, rule in Z:
    df = load(path, fmt, rule)
    real, z = ic.bootstrap_d1(df, zones2.propulsion_stats, B=B // 2, block=500)
    print(f"\n--- {name} propulsion: OB {int(real['n_ob'])} (касаний {int(real['n_ob_touch'])}), propulsion {int(real['n_prop'])} (касаний {int(real['n_prop_touch'])})")
    show(real, z, ["p_bounce_ob", "p_bounce_prop"])
    collect(name, "B01", z, real)
    real, z = ic.bootstrap_d1(df, zones2.volume_imbalance_stats, B=B // 2, block=500)
    print(f"--- {name} volume imbalance vs FVG по ширине (ATR): " + ", ".join(f"{k[2:]}={int(v)}" for k, v in real.items() if k.startswith("n_")))
    show(real, z, [k for k in real if k.startswith("p_bounce_")])
    collect(name, "F01", z, real)

print("\n#################### S05 проекции / S04 консолидация / S07 immediate rebalance / S11 осцилляторы")
D = [("EURUSD_D1", "data/raw/EURUSD_PERIOD_D1.csv", "mt5", None, 250), ("GBPUSD_D1", "data/raw/GBPUSD_PERIOD_D1.csv", "mt5", None, 250),
     ("XAUUSD_D1", "data/raw/XAUUSD_PERIOD_D1.csv", "mt5", None, 250), ("NQ_1d", "data/raw/NQ_1d.csv", "utc", None, 250),
     ("ES_1d", "data/raw/ES_1d.csv", "utc", None, 250),
     ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5", None, 500), ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5", None, 500),
     ("USTEC_H1", "data/raw/USTEC_PERIOD_H1.csv", "mt5", None, 500), ("BTC_1h", "data/raw/BTCUSDT_1m.csv", "utc", "1h", 500)]
for name, path, fmt, rule, block in D:
    df = load(path, fmt, rule)
    if fmt == "utc" and not rule:
        df.index = df.index.normalize()
    print(f"\n===== {name}: {len(df)} баров")
    real, z = ic.bootstrap_d1(df, ge.projection_stats, B=B // 2, block=block)
    print(f"--- S05 проекции: ног {int(real['n_legs'])}")
    show(real, z, ["p_hit_0.5", "p_hit_1.0", "p_hit_1.27", "p_hit_1.5", "p_hit_1.68", "p_hit_2.0",
                   "p_term_within_0.05_of_level", "p_term_within_0.10_of_level"])
    collect(name, "S05", z, real)
    real, z = ic.bootstrap_d1(df, ge.consolidation_stats, B=B // 2, block=block)
    print(f"--- S04 консолидация: коробок {int(real['n_boxes'])}")
    show(real, z, ["len_median", "corr_len_move", "move_short", "move_mid", "move_long", "ratio_long_short"])
    collect(name, "S04", z, real)
    real, z = ic.bootstrap_d1(df, ge.immediate_rebalance_stats, B=B // 2, block=block)
    print(f"--- S07 immediate rebalance: imm {int(real.get('n_imm', 0))}, gap {int(real.get('n_gap', 0))}")
    show(real, z, ["imm_p_cont", "gap_p_cont", "diff_p_cont", "imm_mean_atr", "gap_mean_atr", "diff_mean_atr"])
    collect(name, "S07", z, real)
    if block == 250:
        real, z = ic.bootstrap_d1(df, ge.oscillator_stats, B=B // 2, block=block)
        print(f"--- S11 осцилляторы: os {int(real['n_os'])}, ob {int(real['n_ob'])}, hidden {int(real.get('n_hidden_bull', 0))}, classic {int(real.get('n_classic_bull', 0))}")
        show(real, z, ["all_fwd_bp", "all_p_up", "os_fwd_bp", "os_p_up", "ob_fwd_bp", "ob_p_up",
                       "all_fwd2_bp", "hidden_bull_fwd2_bp", "hidden_bull_p_up", "classic_bull_fwd2_bp", "classic_bull_p_up"])
        collect(name, "S11", z, real)

pd.DataFrame(rows).to_csv("out/batch4.csv", index=False)
print("\nзаписано out/batch4.csv")
