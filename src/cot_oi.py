"""X02: open interest (COT legacy, Open Interest All, недельный) как сигнал.

Правила ICT (сводка X02): падение OI ≥ 10–15% при снижении цены (к поддержке/discount) → рост;
рост OI ≥ 10–15% при росте цены (к premium) → падение; в тренде рост OI = продолжение, падение OI =
остановка/разворот. Всё это = ожидание sign(Δцена_k) · sign(ΔOI_k) для форвард-доходности.
Также: OI ниже своей сезонной (по неделе года) средней при снижении цены — бычий; «пики OI совпадают
с ценовыми хаями».
Событие: |ΔOI за k недель| ≥ thr, первая неделя серии (gap 4). Исход: форвард лог-доходность за h
недель в 52-недельных sd, со знаком ожидания ICT; P(>0). Нулл: циркулярный сдвиг ряда OI
относительно цены (≥ 52 нед., B раз); цена и её изменение — реальные.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def frame(cot_m: pd.DataFrame, price: pd.Series, invert=False):
    p = np.log(price.astype(float))
    if invert:
        p = -p
    p = p[~p.index.duplicated()].sort_index()
    c = cot_m.set_index("date").sort_index()
    rel = c.index + pd.Timedelta(days=3)
    pos = p.index.searchsorted(rel); ok = pos < len(p)
    f = pd.DataFrame({"date": c.index[ok], "oi": c.oi.to_numpy(float)[ok]}).set_index("date")
    f["p0"] = p.to_numpy()[pos[ok]]
    f["sd52"] = f.p0.diff().rolling(52).std().shift(1)
    # сезонная средняя OI по неделе года (все годы), отклонение
    woy = f.index.isocalendar().week.to_numpy()
    f["woy"] = woy
    return f


def metrics(f: pd.DataFrame, k=4, thr=0.15, hs=(4, 8, 13), gap=4):
    f = f.copy()
    f["doi"] = f.oi / f.oi.shift(k) - 1
    f["dp"] = np.sign(f.p0 - f.p0.shift(k))
    seas = f.groupby("woy").oi.transform("mean")
    f["oi_dev"] = f.oi / seas - 1
    for h in hs:
        f[f"fwd{h}"] = (f.p0.shift(-h) - f.p0) / f.sd52
    valid = f.doi.notna() & f.sd52.notna() & f.dp.ne(0)
    out = {}

    def first_in_run(mask):
        idx = np.flatnonzero(mask.to_numpy()); keep, last = [], -10**9
        for i in idx:
            if i - last > gap:
                keep.append(i)
            last = i
        return np.array(keep, int)

    sets = {"drop": valid & (f.doi <= -thr), "rise": valid & (f.doi >= thr),
            "drop_pdown": valid & (f.doi <= -thr) & (f.dp < 0), "rise_pup": valid & (f.doi >= thr) & (f.dp > 0),
            "drop_pup": valid & (f.doi <= -thr) & (f.dp > 0), "rise_pdown": valid & (f.doi >= thr) & (f.dp < 0),
            "dev_low_pdown": valid & (f.oi_dev <= -0.10) & (f.dp < 0), "dev_high_pup": valid & (f.oi_dev >= 0.10) & (f.dp > 0)}
    for name, mask in sets.items():
        idx = first_in_run(mask)
        out[f"{name}_n"] = float(len(idx))
        if len(idx) < 10:
            continue
        sgn_oi = np.sign(f.doi.to_numpy()[idx]) if "dev" not in name else np.sign(f.oi_dev.to_numpy()[idx])
        sgn = f.dp.to_numpy()[idx] * sgn_oi            # ожидание ICT
        for h in hs:
            r = sgn * f[f"fwd{h}"].to_numpy()[idx]; r = r[np.isfinite(r)]
            out[f"{name}_ret{h}"] = float(r.mean()); out[f"{name}_p{h}"] = float((r > 0).mean())
    # ICT-знак по всем неделям (без порога): corr(sign(dp)*doi, fwd)
    m = valid & f.fwd13.notna()
    if m.sum() > 100:
        out["rho13_all"] = float(spearmanr((f.dp * f.doi)[m], f.fwd13[m]).statistic)
        out["rho4_all"] = float(spearmanr((f.dp * f.doi)[m], f.fwd4[m]).statistic)
    # пики OI и ценовые хаи: позиция цены в 52-нед. диапазоне, когда OI на 52-нед. максимуме
    hi52 = f.oi.rolling(52).max(); pmax = f.p0.rolling(52).max(); pmin = f.p0.rolling(52).min()
    pos = (f.p0 - pmin) / (pmax - pmin)
    at_max = (f.oi >= hi52) & pos.notna()
    out["oi_max_n"] = float(at_max.sum())
    out["oi_max_price_pos"] = float(pos[at_max].mean()) if at_max.sum() > 10 else np.nan
    return out


def oi_test(cot_m, price, invert=False, B=300, seed=0, **kw):
    f = frame(cot_m, price, invert)
    real = metrics(f, **kw)
    rng = np.random.default_rng(seed); n = len(f); boots = []
    for _ in range(B):
        g = f.copy(); s = int(rng.integers(52, n - 52))
        g["oi"] = np.roll(f.oi.to_numpy(), s)
        boots.append(metrics(g, **kw))
    z = {}
    for k, v in real.items():
        vals = np.array([b.get(k, np.nan) for b in boots], float); vals = vals[np.isfinite(vals)]
        if len(vals) < 10 or not np.isfinite(v):
            continue
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z, n
