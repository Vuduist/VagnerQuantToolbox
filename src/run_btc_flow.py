"""Предсказательная сила потока агрессора BTC на 1/5/15/60 мин (экономно по памяти, float32).
Регрессия будущей доходности (в σ минуты) на: доходность минуты, дисбаланс агрессора, объём, их произведения; по годам.
Сортировки: «поглощение» (цена против потока) и «согласие» (цена и поток в одну сторону)."""
import sys
import numpy as np
import pandas as pd
import btc_flow as bf

d = bf.load()
cols = ["r", "ofi", "v", "f1", "f5", "f15", "f60"]
d = d[cols].astype("float32")
vol = d.r.rolling(1440, min_periods=500).std()
d["rz"] = (d.r / vol).astype("float32")
o_m = d.ofi.rolling(1440, min_periods=500).mean(); o_s = d.ofi.rolling(1440, min_periods=500).std()
d["oz"] = ((d.ofi - o_m) / o_s).astype("float32")
lv = np.log(d.v.where(d.v > 0))
d["vz"] = (lv - lv.rolling(1440, min_periods=500).mean()).astype("float32")
d["vol"] = vol.astype("float32")
d = d.dropna(subset=["rz", "oz", "vz", "f60", "vol"])
print(f"минут {len(d)}; corr(дисбаланс, доходность той же минуты) {np.corrcoef(d.ofi, d.r)[0, 1]:+.3f}", file=sys.stderr)
for k in (1, 5, 15, 60):
    y = (d[f"f{k}"] / d.vol / np.sqrt(k)).to_numpy("float64")
    X = np.column_stack([np.ones(len(d)), d.rz, d.oz, d.vz, d.rz * d.vz, d.oz * d.vz]).astype("float64")
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var()) * np.sqrt(k)
    print(f"{k:2d} мин: r {b[1]:+.4f} (t {b[1] / se[1]:+.1f}) | дисбаланс {b[2]:+.4f} (t {b[2] / se[2]:+.1f}) | объём {b[3]:+.4f} (t {b[3] / se[3]:+.1f}) | "
          f"r×объём {b[4]:+.4f} (t {b[4] / se[4]:+.1f}) | дисб.×объём {b[5]:+.4f} (t {b[5] / se[5]:+.1f})", file=sys.stderr, flush=True)
    del X, res
cases = {"поглощение: цена↓ при покупках": ((d.rz < -1) & (d.oz > 1), 1), "поглощение: цена↑ при продажах": ((d.rz > 1) & (d.oz < -1), -1),
         "согласие вверх": ((d.rz > 1) & (d.oz > 1), 1), "согласие вниз": ((d.rz < -1) & (d.oz < -1), -1)}
yrs = d.index.year
for nm, (m, sg) in cases.items():
    s = f"{nm:31s} n={int(m.sum()):6d}"
    for k in (5, 15, 60):
        rr = sg * d.loc[m, f"f{k}"] * 1e4
        s += f" | {k:2d} мин {rr.mean():+5.2f} бп (t {rr.mean() / rr.std() * np.sqrt(m.sum() / max(1, k / 5)):+.1f})"
    y60 = (sg * d.loc[m, "f60"]).groupby(yrs[m]).mean()
    s += f" | лет+ (60 мин) {(y60 > 0).sum()}/{len(y60)}"
    print(s, file=sys.stderr, flush=True)
