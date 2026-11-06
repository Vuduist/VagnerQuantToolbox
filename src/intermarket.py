"""X08: интермаркет — лид-лаг, risk-on/off, консолидация Treasury, tandem ZN+DXY, согласие классов.

Недельная сетка (пятница). Лог-доходности.
  lag_xcorr  corr(x_t, y_{t+k}), k = −8..8 недель; нулл — циркулярный сдвиг x на ≥ 26 недель (B раз):
             z по каждому лагу; лид x → y = значимые k > 0.
  cons_zn    4-недельный диапазон ZN / ATR в нижней терцили → диапазоны EURUSD/ES/DXY в следующие
             4 недели (относительно своего ATR) ниже, чем после верхней терцили? (ICT: консолидация
             Treasury подавляет остальных). Нулл — сдвиг ряда ZN.
  tandem     sign(ΔZN за 8 нед.) == sign(ΔDXY за 8 нед.) → диапазон EURUSD/GBPUSD следующих 13 недель /
             ATR ниже, чем при противонаправленности? Нулл — сдвиг ZN.
  align      согласие классов за 13 недель: risk-on = ES↑, сырьё↑ (равновзвешенно CL/GC/HG/ZS),
             DXY↓, ZN↓; счёт −4..+4; |счёт| ≥ 3 → форвард 13 нед. EURUSD/AUDUSD в сторону risk-on
             vs нулл (сдвиг всех классов относительно пары одним смещением).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def weekly(data: dict[str, pd.Series]) -> pd.DataFrame:
    w = pd.concat({k: np.log(v.astype(float)).resample("W-FRI").last() for k, v in data.items()}, axis=1)
    return w


def lag_xcorr(rx: pd.Series, ry: pd.Series, K=8, B=500, seed=0):
    d = pd.concat({"x": rx, "y": ry}, axis=1).dropna()
    x, y = d.x.to_numpy(), d.y.to_numpy(); n = len(x)
    def xc(x, y):
        out = {}
        for k in range(-K, K + 1):
            if k >= 0:
                a, b = x[:n - k], y[k:]
            else:
                a, b = x[-k:], y[:n + k]
            out[k] = np.corrcoef(a, b)[0, 1]
        return out
    real = xc(x, y)
    rng = np.random.default_rng(seed)
    boots = [xc(np.roll(x, int(rng.integers(26, n - 26))), y) for _ in range(B)]
    z = {}
    for k in real:
        vals = np.array([b[k] for b in boots]); sd = vals.std(ddof=1)
        z[k] = {"real": real[k], "null_sd": sd, "z": (real[k] - vals.mean()) / sd if sd > 0 else np.nan}
    return z, n


def _range_rel(logp: pd.Series, w=4):
    r = logp.diff().abs()
    atr = r.rolling(13).mean().shift(1)
    rng = (logp.rolling(w).max() - logp.rolling(w).min())
    return rng / (atr * np.sqrt(w))


def cons_zn(w: pd.DataFrame, targets=("EURUSD", "ES", "DXY"), B=500, seed=0):
    zr = _range_rel(w["ZN"], 4)
    q1, q2 = zr.quantile([1 / 3, 2 / 3])
    out = {}
    rng = np.random.default_rng(seed)
    n = len(w)
    for t in targets:
        tr = _range_rel(w[t], 4).shift(-4)       # диапазон следующих 4 недель
        def stat(zr):
            m = zr.notna() & tr.notna()
            lo = tr[m & (zr <= q1)].mean(); hi = tr[m & (zr >= q2)].mean()
            return lo / hi
        real = stat(zr)
        boots = np.array([stat(pd.Series(np.roll(zr.to_numpy(), int(rng.integers(26, n - 26))), index=zr.index)) for _ in range(B)])
        boots = boots[np.isfinite(boots)]
        out[f"cons_{t}"] = {"real": real, "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                            "z": (real - boots.mean()) / boots.std(ddof=1)}
        # синхронно (та же неделя)
        tr0 = _range_rel(w[t], 4)
        m = zr.notna() & tr0.notna()
        out[f"cons0_{t}"] = {"real": tr0[m & (zr <= q1)].mean() / tr0[m & (zr >= q2)].mean()}
    return out


def tandem(w: pd.DataFrame, targets=("EURUSD", "GBPUSD"), B=500, seed=0):
    dz = np.sign(w["ZN"].diff(8)); dd = np.sign(w["DXY"].diff(8))
    out = {}
    rng = np.random.default_rng(seed); n = len(w)
    for t in targets:
        fr = _range_rel(w[t], 13).shift(-13)
        def stat(dz):
            same = (dz == dd) & dz.ne(0) & fr.notna(); opp = (dz == -dd) & dz.ne(0) & fr.notna()
            return fr[same].mean() / fr[opp].mean()
        real = stat(dz)
        boots = np.array([stat(pd.Series(np.roll(dz.to_numpy(), int(rng.integers(26, n - 26))), index=dz.index)) for _ in range(B)])
        boots = boots[np.isfinite(boots)]
        out[f"tandem_{t}"] = {"real": real, "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                              "z": (real - boots.mean()) / boots.std(ddof=1)}
    return out


def align(w: pd.DataFrame, targets=("EURUSD", "AUDUSD"), B=500, seed=0, h=13):
    comm = w[["CL", "GC", "HG", "ZS"]].diff().mean(axis=1).cumsum()
    classes = pd.concat({"ES": w["ES"].diff(13), "COMM": comm.diff(13), "DXY": -w["DXY"].diff(13), "ZN": -w["ZN"].diff(13)}, axis=1)
    score = np.sign(classes).sum(axis=1)
    out = {}
    rng = np.random.default_rng(seed); n = len(w)
    for t in targets:
        sd = w[t].diff().rolling(52).std().shift(1)
        fwd = (w[t].shift(-h) - w[t]) / (sd * np.sqrt(h))
        def stat(score):
            m = (score.abs() >= 3) & fwd.notna()
            r = np.sign(score[m]) * fwd[m]
            return r.mean(), (r > 0).mean(), m.sum()
        real = stat(score)
        boots = np.array([stat(pd.Series(np.roll(score.to_numpy(), int(rng.integers(26, n - 26))), index=score.index))[0] for _ in range(B)])
        boots = boots[np.isfinite(boots)]
        out[f"align_{t}"] = {"real": real[0], "p_pos": real[1], "n": int(real[2]), "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                             "z": (real[0] - boots.mean()) / boots.std(ddof=1)}
        # без порога: corr(score, fwd)
        m = score.notna() & fwd.notna()
        out[f"align_corr_{t}"] = {"real": float(np.corrcoef(score[m], fwd[m])[0, 1])}
    return out
