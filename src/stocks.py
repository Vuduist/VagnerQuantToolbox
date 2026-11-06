"""X09: акции — higher low акции при lower low индекса (index SMT на акциях) и близость к 52-нед. хаю.

Данные: Dow 30 текущего состава (survivorship!) и ^DJI, Yahoo, цены скорректированы на сплиты/дивиденды
(adj/close). Событие: индекс делает новый L-дневный лоу (первый день серии, gap 10). Для каждой акции:
HL — за последние 5 дней не делала нового L-дневного лоу; LL — делала. Исход: форвард лог-доходность
акции минус индекса за h дней; метрика — средняя по событиям разность HL − LL и доля событий, где
HL > LL. Подвыборки: сезонные (событие в янв–фев или сен–окт → окна покупок ICT фев–май / окт–янв)
и остальные. Второй сигнал: близость к 52-нед. хаю (≤ 5% vs ≥ 15% ниже) → относительный форвард.
Зеркально для хаев индекса (bear). Нулл: перестановка меток HL/LL (или близости) внутри события.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_stock(path):
    df = pd.read_csv(path, header=None)
    cols = ["t", "open", "high", "low", "close", "volume"] + (["adj"] if df.shape[1] == 7 else [])
    df.columns = cols
    df["dt"] = pd.to_datetime(df.t, unit="ms").dt.normalize()
    df = df.drop_duplicates("dt").set_index("dt").sort_index()
    if "adj" in df:
        f = df.adj / df.close
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * f
    return df[["open", "high", "low", "close"]].astype(float)


def events_index(idx: pd.DataFrame, L=20, gap=10):
    newlow = idx.low < idx.low.rolling(L).min().shift(1)
    newhigh = idx.high > idx.high.rolling(L).max().shift(1)
    def first(mask):
        pos = np.flatnonzero(mask.to_numpy()); keep, last = [], -10**9
        for i in pos:
            if i - last > gap:
                keep.append(i)
            last = i
        return np.array(keep, int)
    return first(newlow), first(newhigh)


def smt_stocks(stocks: dict[str, pd.DataFrame], idx: pd.DataFrame, L=20, hs=(20, 60), B=1000, seed=0, lookback=5):
    dates = idx.index
    lp_idx = np.log(idx.close)
    ev_lo, ev_hi = events_index(idx, L)
    # матрицы по акциям (выровнены по датам индекса)
    names = list(stocks)
    lows = pd.concat({n: stocks[n].low.reindex(dates) for n in names}, axis=1)
    highs = pd.concat({n: stocks[n].high.reindex(dates) for n in names}, axis=1)
    closes = pd.concat({n: stocks[n].close.reindex(dates) for n in names}, axis=1)
    lp = np.log(closes)
    flag_lo = (lows < lows.rolling(L).min().shift(1)).rolling(lookback).max() > 0     # делала новый лоу за lookback дней
    flag_hi = (highs > highs.rolling(L).max().shift(1)).rolling(lookback).max() > 0
    near_hi = closes / highs.rolling(252).max() - 1                                   # близость к 52w хаю (≤0)
    rng = np.random.default_rng(seed)
    out = {}
    months = dates.month
    for kind, ev, flag, sign in (("bull", ev_lo, flag_lo, +1), ("bear", ev_hi, flag_hi, -1)):
        ev = ev[(ev > 260) & (ev < len(dates) - max(hs) - 1)]
        out[f"{kind}_n_events"] = float(len(ev))
        seas = np.isin(months[ev], [1, 2, 9, 10]) if kind == "bull" else np.isin(months[ev], [4, 5, 7, 8])
        for h in hs:
            rel = sign * ((lp.shift(-h) - lp).sub(lp_idx.shift(-h) - lp_idx, axis=0))   # акция − индекс, в сторону ICT
            R = rel.to_numpy()[ev]                 # события × акции
            F = (~flag.to_numpy()[ev]) & np.isfinite(R)   # «лидер» = НЕ делала нового экстремума
            G = flag.to_numpy()[ev] & np.isfinite(R)
            def stat(F, G, sel):
                d = []
                for i in np.flatnonzero(sel):
                    if F[i].sum() >= 2 and G[i].sum() >= 2:
                        d.append(np.nanmean(R[i][F[i]]) - np.nanmean(R[i][G[i]]))
                d = np.array(d)
                return (d.mean(), (d > 0).mean(), len(d)) if len(d) else (np.nan, np.nan, 0)
            for sub, sel in (("all", np.ones(len(ev), bool)), ("seas", seas), ("other", ~seas)):
                real = stat(F, G, sel)
                boots = []
                for _ in range(B):
                    Fp = np.zeros_like(F); Gp = np.zeros_like(G)
                    for i in np.flatnonzero(sel):
                        members = np.flatnonzero(F[i] | G[i]); k = F[i].sum()
                        perm = rng.permutation(members); Fp[i, perm[:k]] = True; Gp[i, perm[k:]] = True
                    boots.append(stat(Fp, Gp, sel)[0])
                boots = np.array(boots); boots = boots[np.isfinite(boots)]
                key = f"{kind}_{sub}_h{h}"
                out[key] = {"real": real[0], "p_pos": real[1], "n": real[2], "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                            "z": (real[0] - boots.mean()) / boots.std(ddof=1) if boots.std(ddof=1) > 0 else np.nan,
                            "lead": float(np.nanmean(np.where(F, R, np.nan)[sel])), "lag": float(np.nanmean(np.where(G, R, np.nan)[sel]))}
            # близость к 52w хаю (только bull)
            if kind == "bull":
                NH = near_hi.to_numpy()[ev]
                Fn = (NH >= -0.05) & np.isfinite(R); Gn = (NH <= -0.15) & np.isfinite(R)
                real = stat(Fn, Gn, np.ones(len(ev), bool))
                boots = []
                for _ in range(B):
                    Fp = np.zeros_like(Fn); Gp = np.zeros_like(Gn)
                    for i in range(len(ev)):
                        members = np.flatnonzero(Fn[i] | Gn[i]); k = Fn[i].sum()
                        perm = rng.permutation(members); Fp[i, perm[:k]] = True; Gp[i, perm[k:]] = True
                    boots.append(stat(Fp, Gp, np.ones(len(ev), bool))[0])
                boots = np.array(boots); boots = boots[np.isfinite(boots)]
                out[f"near52_h{h}"] = {"real": real[0], "p_pos": real[1], "n": real[2], "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                                      "z": (real[0] - boots.mean()) / boots.std(ddof=1) if boots.std(ddof=1) > 0 else np.nan}
    return out
