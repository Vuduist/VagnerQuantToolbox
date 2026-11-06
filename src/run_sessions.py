"""Прогон сессионного слоя."""

import sys

import pandas as pd

import engine
import sessions

SETS = [
    ("USTEC_M5",  "data/raw/USTEC_PERIOD_M5.csv",  "mt5", None, 5),
    ("USTEC_H1",  "data/raw/USTEC_PERIOD_H1.csv",  "mt5", None, 60),
    ("EURUSD_H1", "data/raw/EURUSD_PERIOD_H1.csv", "mt5", None, 60),
    ("GBPUSD_H1", "data/raw/GBPUSD_PERIOD_H1.csv", "mt5", None, 60),
    ("XAUUSD_H1", "data/raw/XAUUSD_PERIOD_H1.csv", "mt5", None, 60),
    ("BTC_1h",    "data/raw/BTCUSDT_1m.csv",       "utc", "1h", 60),
]
SRC_TZ = {"mt5": "Europe/Helsinki", "utc": "UTC", "binance": "UTC"}

for name, path, fmt, rule, mpb in SETS:
    df = engine.load_mt5_csv(path) if fmt == "mt5" else engine.load_csv(path)
    if rule:
        df = engine.resample(df, rule)
    df = sessions.to_et(df, SRC_TZ[fmt])
    print(f"\n===== {name} ({len(df)} баров, {df.index[0]} .. {df.index[-1]}) =====")

    prof = sessions.hourly_profile(df)
    top_vol = prof.nlargest(4, "vol_index")
    print("пики волы (час ET): " +
          ", ".join(f"{r.hour_et}:00({r.vol_index:.2f}x, t={r.t_stat:+.1f})"
                    for r in top_vol.itertuples()))
    sig_t = prof[prof.t_stat.abs() > 2.6]
    if len(sig_t):
        print("часы со значимым дрейфом:")
        print(sig_t.round(3).to_string(index=False))
    else:
        print("значимого часового дрейфа нет (|t|>2.6)")

    for w in ("london", "ny_open", "sb_pm"):
        r = sessions.window_event_test(df, w)
        if r["in"] and r["out"]:
            di, do = r["in"], r["out"]
            print(f"  окно {w}: in n={di['n']} p_cont={di['p_cont']:.3f} "
                  f"exp_R={di['exp_R']:+.3f} | out n={do['n']} "
                  f"p_cont={do['p_cont']:.3f} exp_R={do['exp_R']:+.3f}")

    if mpb == 5:
        orb = sessions.orb(df, 30, mpb)
        jud = sessions.judas(df, mpb)
        if orb:
            print(f"  ORB(9:30-10:00): n={orb['n']} mean={orb['mean_bp']:+.1f}бп "
                  f"p_win={orb['p_win']:.3f}")
        if jud:
            print(f"  Judas(азия->лондон): n={jud['n']} mean={jud['mean_bp']:+.1f}бп "
                  f"p_win={jud['p_win']:.3f}")
