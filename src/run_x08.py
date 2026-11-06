"""Прогон X08 интермаркет. out/x08.csv"""

import sys

import numpy as np
import pandas as pd

import engine
import intermarket as im

B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SRC = {"DXY": "data/raw/DXY_1d.csv", "ZN": "data/raw/ZN_1d.csv", "ES": "data/raw/ES_1d.csv", "GC": "data/raw/GC_1d.csv",
       "CL": "data/raw/CL_1d.csv", "HG": "data/raw/HG_1d.csv", "ZS": "data/raw/ZS_1d.csv", "AUDUSD": "data/raw/AUDUSD_1d.csv",
       "EURUSD": "data/raw/EURUSD_PERIOD_D1.csv", "GBPUSD": "data/raw/GBPUSD_PERIOD_D1.csv"}
data = {}
for k, p in SRC.items():
    df = engine.load_mt5_csv(p) if "PERIOD" in p else engine.load_csv(p)
    df.index = df.index.normalize(); data[k] = df[~df.index.duplicated()].close
w = im.weekly(data)
w = w[w.index >= "2000-09-01"]
r = w.diff()
rows = []
print("#################### лид-лаг: corr(x_t, y_{t+k}), недельные доходности; z vs циркулярный сдвиг x")
PAIRS = [("ZN", "DXY"), ("ZN", "EURUSD"), ("DXY", "ES"), ("DXY", "GC"), ("DXY", "CL"), ("DXY", "AUDUSD"), ("ES", "DXY"), ("GC", "DXY"), ("CL", "DXY"), ("ZN", "ES")]
for x, y in PAIRS:
    z, n = im.lag_xcorr(r[x], r[y], K=8, B=B)
    c0 = z[0]["real"]
    lead = {k: v for k, v in z.items() if k > 0}
    kmax = max(lead, key=lambda k: abs(lead[k]["z"]))
    lagm = {k: v for k, v in z.items() if k < 0}
    kmin = max(lagm, key=lambda k: abs(lagm[k]["z"]))
    sig_lead = sum(abs(v["z"]) >= 2.5 for k, v in lead.items())
    print(f"  {x:6s}→{y:7s} n={n:4d} corr0={c0:+.3f} (z{z[0]['z']:+.1f}) | лид x→y: сильнейший лаг k={kmax} corr={lead[kmax]['real']:+.3f} z{lead[kmax]['z']:+.1f}, лагов с |z|≥2.5: {sig_lead}/8 | обратно k={kmin} corr={lagm[kmin]['real']:+.3f} z{lagm[kmin]['z']:+.1f}")
    for k, v in z.items():
        rows.append({"test": "lag", "pair": f"{x}->{y}", "lag": k, **v})
print("\n#################### консолидация ZN (4-нед. диапазон/ATR, нижняя vs верхняя терциль) → диапазоны следующих 4 недель (отношение)")
for k, v in im.cons_zn(w, B=B).items():
    if "null_mean" in v:
        print(f"  {k:14s} real={v['real']:.3f} | null={v['null_mean']:.3f}±{v['null_sd']:.3f} z{v['z']:+.1f}")
    else:
        print(f"  {k:14s} синхронно: {v['real']:.3f}")
    rows.append({"test": "cons", "pair": k, **v})
print("\n#################### tandem ZN+DXY (8 нед. сонаправлены vs противонаправлены) → диапазон следующих 13 недель / ATR (отношение)")
for k, v in im.tandem(w, B=B).items():
    print(f"  {k:14s} real={v['real']:.3f} | null={v['null_mean']:.3f}±{v['null_sd']:.3f} z{v['z']:+.1f}")
    rows.append({"test": "tandem", "pair": k, **v})
print("\n#################### согласие классов (ES, сырьё, −DXY, −ZN за 13 нед.; |счёт| ≥ 3) → форвард 13 нед. пары в сторону risk-on")
for k, v in im.align(w, B=B).items():
    if "null_mean" in v:
        print(f"  {k:16s} ret={v['real']:+.3f} sd-ед. P>0={v['p_pos']:.2f} n={v['n']} | null={v['null_mean']:+.3f}±{v['null_sd']:.3f} z{v['z']:+.1f}")
    else:
        print(f"  {k:16s} corr(счёт, форвард)={v['real']:+.3f}")
    rows.append({"test": "align", "pair": k, **v})
print("\n#################### синхронные corr недельных доходностей (risk-on/off)")
print(r[["DXY", "ZN", "ES", "GC", "CL", "EURUSD", "AUDUSD"]].corr().round(2).to_string())
pd.DataFrame(rows).to_csv("out/x08.csv", index=False)
print("\nзаписано out/x08.csv")
