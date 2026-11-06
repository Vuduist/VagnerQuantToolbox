"""X12: дифференциал ставок ЦБ → направление пары на 3–6 месяцев.

Месячная сетка (конец месяца). Пара base/quote: diff = r_base − r_quote (п.п.).
Сигналы (ICT: «деньги ищут доходность», пара идёт в сторону валюты с более высокой/растущей ставкой):
  lvl   знак дифференциала → форвард-доходность пары за h мес. в сторону высокой ставки;
  chg3  знак изменения дифференциала за 3 мес. → форвард в сторону роста дифференциала;
  rho   Spearman(Δdiff за 3 мес., форвард h мес.) и Spearman(уровень diff, форвард).
Доходность в единицах 12-месячной sd месячных доходностей. Нулл: циркулярный сдвиг ряда ставок
относительно цены (≥ 12 мес., B раз).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def monthly_frame(price: pd.Series, r_base: pd.Series, r_quote: pd.Series) -> pd.DataFrame:
    p = np.log(price.astype(float)).resample("ME").last()
    rb = r_base.resample("ME").last(); rq = r_quote.resample("ME").last()
    f = pd.concat({"p": p, "rb": rb, "rq": rq}, axis=1).dropna()
    f["diff"] = f.rb - f.rq
    f["sd12"] = f.p.diff().rolling(12).std().shift(1)
    return f


def metrics(f: pd.DataFrame, hs=(3, 6), gap=3):
    f = f.copy()
    f["dchg3"] = f["diff"].diff(3)
    out = {}
    for h in hs:
        f[f"fwd{h}"] = (f.p.shift(-h) - f.p) / f.sd12
    valid = f.sd12.notna()
    for h in hs:
        m = valid & f[f"fwd{h}"].notna()
        # уровень дифференциала
        s = np.sign(f["diff"])[m]; r = (s * f[f"fwd{h}"])[m & s.ne(0)]
        out[f"lvl_ret{h}"] = float(r.mean()); out[f"lvl_p{h}"] = float((r > 0).mean()); out[f"lvl_n"] = float(len(r))
        # изменение за 3 мес. (первое в серии, gap)
        s3 = np.sign(f.dchg3)
        idx = np.flatnonzero((m & s3.ne(0) & s3.notna()).to_numpy())
        keep, last, lastsign = [], -10**9, 0
        for i in idx:
            sg = s3.iloc[i]
            if sg != lastsign or i - last > gap:
                keep.append(i)
            last, lastsign = i, sg
        keep = np.array(keep, int)
        r = (s3.to_numpy()[keep] * f[f"fwd{h}"].to_numpy()[keep]); r = r[np.isfinite(r)]
        out[f"chg_ret{h}"] = float(r.mean()) if len(r) else np.nan; out[f"chg_p{h}"] = float((r > 0).mean()) if len(r) else np.nan
        out[f"chg_n{h}"] = float(len(r))
        mm = m & f.dchg3.notna()
        if mm.sum() > 40:
            out[f"rho_chg{h}"] = float(spearmanr(f.dchg3[mm], f[f"fwd{h}"][mm]).statistic)
            out[f"rho_lvl{h}"] = float(spearmanr(f["diff"][mm], f[f"fwd{h}"][mm]).statistic)
    # синхронная связь (что известно постфактум): corr(Δdiff за 3 мес., доходность за те же 3 мес.)
    same = (f.p - f.p.shift(3)) / f.sd12
    mm = valid & f.dchg3.notna() & same.notna()
    out["rho_same3"] = float(spearmanr(f.dchg3[mm], same[mm]).statistic) if mm.sum() > 40 else np.nan
    return out


def rates_test(price, r_base, r_quote, B=500, seed=0, **kw):
    f = monthly_frame(price, r_base, r_quote)
    real = metrics(f, **kw)
    rng = np.random.default_rng(seed); n = len(f); boots = []
    for _ in range(B):
        g = f.copy(); s = int(rng.integers(12, n - 12))
        g["rb"] = np.roll(f.rb.to_numpy(), s); g["rq"] = np.roll(f.rq.to_numpy(), s); g["diff"] = g.rb - g.rq
        boots.append(metrics(g, **kw))
    z = {}
    for k, v in real.items():
        vals = np.array([b.get(k, np.nan) for b in boots], float); vals = vals[np.isfinite(vals)]
        if len(vals) < 10 or not np.isfinite(v):
            continue
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z, n
