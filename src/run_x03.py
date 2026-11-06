"""Прогон X03 сырьё. out/x03.csv"""

import sys

import numpy as np
import pandas as pd

import commod as cm
import engine
import seasonality as se

B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NAMES = ["ZW", "ZS", "ZC", "CL", "NG", "HG", "SI", "GC", "KC", "CT", "SB", "CC", "LE", "HE", "ZB", "DXY"]
data = {}
for n in NAMES:
    df = engine.load_csv(f"data/raw/{n}_1d.csv"); df.index = df.index.normalize()
    data[n] = df[~df.index.duplicated()].close
rows = []
print("#################### X03 сезонность: месяцы (нулл = перестановка месяцев в году)")
for n in NAMES[:-2]:
    r = se.monthly_test(data[n], B=1000)
    best = int(np.argmax(np.abs(r["t"])))
    print(f"  {n:3s} лет={r['n_years']:2d} max|t|={r['max_abs_t']:.2f} p={r['p_maxt']:.3f}  сильнейший месяц {r['months'][best]}: {r['mean_pct'][best]:+.2f}% (z {r['z'][best]:+.1f})")
    rows.append({"market": n, "test": "month", "metric": "max_abs_t", "real": r["max_abs_t"], "p": r["p_maxt"]})
print("\n#################### X03 сезонность: окна ICT (нулл = случайное окно той же длины)")
hits = 0
for n, label, s, e, sign in cm.WINDOWS:
    r = se.window_test(data[n], s, e, sign, B=500)
    hits += abs(r["z"]) >= 2 and r["mean"] > 0
    print(f"  {n:3s} {label:52s} лет={r['n_years']:2d} mean={r['mean']*100:+.2f}% t={r['t']:+.2f} p_pos={r['p_pos']:.2f} | нулл {r['null_mean']*100:+.2f}%±{r['null_sd']*100:.2f} z{r['z']:+.1f}")
    rows.append({"market": n, "test": "window", "metric": label, "real": r["mean"], "t": r["t"], "p_pos": r["p_pos"],
                 "null_mean": r["null_mean"], "null_sd": r["null_sd"], "z": r["z"]})
print(f"  окон с z ≥ +2 в сторону ICT: {hits} из {len(cm.WINDOWS)}")
print("\n#################### X03 лидер сектора по failure swing (нулл = перестановка ролей)")
for L in (20, 60):
    for sec, members in cm.SECTORS.items():
        out = cm.leader_test(data, members, L=L, B=B)
        print(f"\n  --- {sec} L={L}: событий bull {int(out.get('bull_n', 0))} bear {int(out.get('bear_n', 0))}")
        for k, r in out.items():
            if isinstance(r, dict):
                print(f"    {k:8s} лидер−соседи={r['real']:+.3f} sd-ед. (P>0 {r['p_pos']:.2f}; лидер {r['lead']:+.2f}, соседи {r['lag']:+.2f}) | нулл {r['null_mean']:+.3f}±{r['null_sd']:.3f} z{r['z']:+.1f}")
                rows.append({"market": f"{sec}_L{L}", "test": "leader", "metric": k, **r})
# DXY ↔ сырьё: корреляция недельных доходностей
print("\n#################### DXY ↔ сырьё, corr недельных доходностей")
w = pd.concat({n: data[n].resample("W-FRI").last().pct_change() for n in NAMES}, axis=1).dropna()
for n in NAMES[:-2]:
    c = w[n].corr(w["DXY"]); print(f"  {n:3s} {c:+.2f}", end="")
    rows.append({"market": n, "test": "dxy_corr", "metric": "corr_weekly", "real": c})
print()
pd.DataFrame(rows).to_csv("out/x03.csv", index=False)
print("\nзаписано out/x03.csv")
