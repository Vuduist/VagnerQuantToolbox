"""X01: COT — позиция commercials как сигнал.

Сигналы (недельные, отчёт за вторник известен в пятницу → цена с ближайшего дня ≥ пятницы):
  idx    индекс commercials в трейлинг-окне W недель: (net − min)/(max − min); экстремумы ≥ 0.9 → ICT
         bullish (хеджеры «умные деньги» максимально длинные), ≤ 0.1 → bearish; первая неделя серии.
  chg4   знак изменения comm_net за 4 недели («hedging program»): рост → bullish.
  rho    Spearman между idx и форвард-доходностью (все недели).
Исход: форвард лог-доходность рынка за h недель в единицах 52-недельной sd недельных доходностей,
со знаком ожидания ICT; P(>0). Для JPY-фьючерса доходность = −USDJPY.
Нулл: циркулярный сдвиг ряда COT относительно цены на ≥ 52 недель (B раз).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def weekly_frame(cot_m: pd.DataFrame, price: pd.Series, invert=False) -> pd.DataFrame:
    """cot_m: строки одного рынка (date, comm_net, nc_net); price: дневной close."""
    p = np.log(price.astype(float))
    if invert:
        p = -p
    p = p[~p.index.duplicated()].sort_index()
    c = cot_m.set_index("date").sort_index()
    rel = c.index + pd.Timedelta(days=3)                       # пятница релиза
    pos = p.index.searchsorted(rel)
    ok = pos < len(p)
    f = pd.DataFrame({"date": c.index[ok], "rel_idx": pos[ok], "comm_net": c.comm_net.to_numpy()[ok],
                      "nc_net": c.nc_net.to_numpy()[ok]}).set_index("date")
    f["p0"] = p.to_numpy()[f.rel_idx]
    # недельная sd доходностей (по релизным датам)
    r1 = f.p0.diff()
    f["sd52"] = r1.rolling(52).std().shift(1)
    return f, p


def metrics(f: pd.DataFrame, p: pd.Series, W=52, hs=(4, 8, 13, 26), hi=0.9, lo=0.1, gap=4):
    f = f.copy()
    n = len(f)
    mn = f.comm_net.rolling(W).min(); mx = f.comm_net.rolling(W).max()
    f["idx"] = (f.comm_net - mn) / (mx - mn)
    f["chg4"] = np.sign(f.comm_net.diff(4))
    pv = p.to_numpy()
    out = {}
    for h in hs:
        # форвард: цена через h релизных недель (по индексу релизных дат)
        fwd = f.p0.shift(-h) - f.p0
        f[f"fwd{h}"] = fwd / f.sd52
    valid = f.idx.notna() & f.sd52.notna()

    def first_in_run(mask):
        idx = np.flatnonzero(mask.to_numpy())
        keep, last = [], -10**9
        for i in idx:
            if i - last > gap:
                keep.append(i)
            last = i
        return np.array(keep, int)

    bull = first_in_run(valid & (f.idx >= hi)); bear = first_in_run(valid & (f.idx <= lo))
    out["n_bull"] = float(len(bull)); out["n_bear"] = float(len(bear))
    for h in hs:
        col = f[f"fwd{h}"].to_numpy()
        rb = col[bull]; rs = -col[bear]
        r = np.concatenate([rb, rs]); r = r[np.isfinite(r)]
        out[f"ext_ret{h}"] = float(r.mean()) if len(r) else np.nan
        out[f"ext_p{h}"] = float((r > 0).mean()) if len(r) else np.nan
        rb = rb[np.isfinite(rb)]; rs = rs[np.isfinite(rs)]
        out[f"bull_ret{h}"] = float(rb.mean()) if len(rb) else np.nan
        out[f"bear_ret{h}"] = float(rs.mean()) if len(rs) else np.nan
        # hedging program: знак 4-нед. изменения
        m = valid & f.chg4.ne(0) & np.isfinite(col)
        r = (f.chg4 * f[f"fwd{h}"])[m]
        out[f"chg_ret{h}"] = float(r.mean()) if len(r) else np.nan
        out[f"chg_p{h}"] = float((r > 0).mean()) if len(r) else np.nan
        # корреляция индекса с форвардом
        m = valid & np.isfinite(col)
        if m.sum() > 50:
            out[f"rho{h}"] = float(spearmanr(f.idx[m], f[f"fwd{h}"][m]).statistic)
        # крупные спекулянты (зеркало): индекс nc_net, экстремумы → против ICT-commercials
    return out


def cot_test(cot_m, price, invert=False, W=52, B=300, seed=0, **kw):
    f, p = weekly_frame(cot_m, price, invert)
    real = metrics(f, p, W=W, **kw)
    rng = np.random.default_rng(seed)
    n = len(f)
    boots = []
    for _ in range(B):
        s = int(rng.integers(52, n - 52))
        g = f.copy()
        g["comm_net"] = np.roll(f.comm_net.to_numpy(), s)
        g["nc_net"] = np.roll(f.nc_net.to_numpy(), s)
        boots.append(metrics(g, p, W=W, **kw))
    z = {}
    for k, v in real.items():
        vals = np.array([b.get(k, np.nan) for b in boots], float); vals = vals[np.isfinite(vals)]
        if len(vals) < 10 or not np.isfinite(v):
            continue
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z, n
