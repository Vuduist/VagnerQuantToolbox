"""Каскады агрессивных сделок на BTCUSDT perpetual (Binance aggTrades) — продолжение или откат?

День → посекундный ряд: last price, чистый агрессивный объём (покупки агрессора − продажи, BTC), число сделок.
Всплеск: |чистый объём за 10 с| в верхних 0.1% дня (относительно скользящих 6 ч) И |движение цены за 10 с| ≥ 3σ(10 с).
Направление всплеска = знак чистого объёма. После конца всплеска: доходность через 1, 5, 30, 60, 300 с в сторону всплеска.
Разметка «похоже на ликвидации»: в 5-минутном интервале всплеска открытый интерес упал (metrics) — закрытие позиций."""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

HZ = (1, 5, 30, 60, 300)


def day_seconds(path):
    z = zipfile.ZipFile(path)
    name = z.namelist()[0]
    with z.open(name) as f:
        head = f.read(64).decode()
    has_header = head.startswith("agg_trade_id")
    cols = ["id", "price", "qty", "f", "l", "t", "maker"]
    d = pd.read_csv(z.open(name), header=0 if has_header else None, names=cols, usecols=[1, 2, 5, 6],
                    dtype={"price": "float64", "qty": "float64"})
    d["maker"] = d.maker.astype(str).str.lower().eq("true")
    t = d.t.to_numpy()
    if t.max() > 10**14:
        t = t // 1000
    sec = (t // 1000).astype(np.int64)
    sv = np.where(d.maker.to_numpy(), -1.0, 1.0) * d.qty.to_numpy()       # maker=True → агрессор продавец
    g = pd.DataFrame({"sec": sec, "p": d.price.to_numpy(), "sv": sv, "n": 1})
    s = g.groupby("sec").agg(p=("p", "last"), sv=("sv", "sum"), n=("n", "sum"))
    full = np.arange(s.index.min(), s.index.max() + 1)
    s = s.reindex(full)
    s["p"] = s.p.ffill(); s[["sv", "n"]] = s[["sv", "n"]].fillna(0)
    return s


def bursts(s, q=0.999, k_sig=3.0, win=10, roll=6 * 3600, gap=60):
    lp = np.log(s.p.to_numpy())
    sv10 = pd.Series(s.sv.to_numpy()).rolling(win).sum().to_numpy()
    dp10 = np.r_[np.full(win, np.nan), lp[win:] - lp[:-win]]
    thr_v = pd.Series(np.abs(sv10)).rolling(roll, min_periods=3600).quantile(q).shift(1).to_numpy()
    sig = pd.Series(dp10).rolling(roll, min_periods=3600).std().shift(1).to_numpy()
    cand = (np.abs(sv10) >= thr_v) & (np.abs(dp10) >= k_sig * sig) & (np.sign(sv10) == np.sign(dp10))
    idx = np.flatnonzero(cand)
    ev, last = [], -10**9
    for i in idx:
        if i - last > gap:
            ev.append(i)
        last = i
    rows = []
    n = len(lp)
    for i in ev:
        # конец всплеска — первая секунда после i, где 10-секундный объём перестал быть экстремальным (не дальше 60 с)
        j = i
        while j + 1 < n and j - i < 60 and abs(sv10[j + 1]) >= thr_v[i] * 0.5:
            j += 1
        sd = np.sign(sv10[i])
        r = {"t": int(s.index[i]), "side": sd, "move_bp": sd * (lp[j] - lp[max(0, i - win)]) * 1e4, "sv": abs(sv10[i]),
             "len_s": j - i, "sig10_bp": sig[i] * 1e4}
        for h in HZ:
            r[f"f{h}"] = sd * (lp[min(n - 1, j + h)] - lp[j]) * 1e4
        rows.append(r)
    return pd.DataFrame(rows)


def oi_change(metrics_zip):
    m = pd.read_csv(metrics_zip)
    m["ts"] = (pd.to_datetime(m.create_time) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)   # pandas 3: не через astype
    m = m.drop_duplicates("ts").sort_values("ts")
    m["doi"] = m.sum_open_interest.pct_change()
    return m.set_index("ts").doi
