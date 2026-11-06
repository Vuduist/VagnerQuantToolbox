"""S03: сжатие диапазона -> расширение (inside day/week, NR4/NR7, малый диапазон -> большой день).

Условия на D1 (и на неделях): inside bar, NR4, NR7, range < 0.6·ADR20, контроль: range > 1.5·ADR20.
Исход: диапазон следующего бара / ADR20 (ADR известен на момент условия), P(>1), P(>1.5),
максимум за 2 следующих бара. Сравнение: условно vs безусловно (ICT говорит «больше»),
и реал vs бутстреп (объясняется ли кластеризацией волы).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rel_ranges(df: pd.DataFrame, n_adr=20):
    h, l = df.high.to_numpy(), df.low.to_numpy()
    rng = h - l
    adr = pd.Series(rng).rolling(n_adr).mean().shift(1).to_numpy()
    rel = rng / adr
    rel_next = np.roll(rng, -1) / adr                            # следующий бар / ADR текущего
    rel_next2 = np.maximum(np.roll(rng, -1), np.roll(rng, -2)) / adr
    rel_next[-1] = np.nan
    rel_next2[-2:] = np.nan
    inside = np.zeros(len(h), bool)
    inside[1:] = (h[1:] < h[:-1]) & (l[1:] > l[:-1])
    nr4 = pd.Series(rng).rolling(4).min().to_numpy() == rng
    nr7 = pd.Series(rng).rolling(7).min().to_numpy() == rng
    ok = np.isfinite(rel) & np.isfinite(rel_next) & np.isfinite(rel_next2)
    return dict(rel=rel, rel_next=rel_next, rel_next2=rel_next2, inside=inside,
                nr4=nr4, nr7=nr7, small=rel < 0.6, big=rel > 1.5, ok=ok)


def expansion_stats(df: pd.DataFrame) -> dict:
    r = _rel_ranges(df)
    ok = r["ok"]
    out = {"n_bars": float(ok.sum())}
    base = r["rel_next"][ok]
    out["all_next_mean"] = float(base.mean())
    out["all_p_next_gt1"] = float((base > 1).mean())
    out["all_p_next_gt15"] = float((base > 1.5).mean())
    for name in ("inside", "nr4", "nr7", "small", "big"):
        m = r[name] & ok
        out[f"n_{name}"] = float(m.sum())
        if m.sum() < 30:
            continue
        nx, nx2 = r["rel_next"][m], r["rel_next2"][m]
        out[f"{name}_next_mean"] = float(nx.mean())
        out[f"{name}_p_next_gt1"] = float((nx > 1).mean())
        out[f"{name}_p_next_gt15"] = float((nx > 1.5).mean())
        out[f"{name}_next2_mean"] = float(nx2.mean())
        out[f"{name}_p_next2_gt15"] = float((nx2 > 1.5).mean())
        out[f"{name}_ratio_vs_all"] = float(nx.mean() / base.mean())
    return out


def weekly(df_d1: pd.DataFrame) -> pd.DataFrame:
    d = df_d1.copy()
    d.index = d.index.normalize()
    w = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return w
