"""Трендовость на дневках (time-series momentum). Сигнал — знак доходности за k месяцев (21k дней), позиция на следующие 21 день,
вол-скейлинг по σ за 60 дней (цель 10% годовых на инструмент), издержки 3 бп на смену позиции. Пул по инструментам, половины."""
import sys
import numpy as np
import pandas as pd
import engine

IDX = ["SPX", "IXIC", "N225", "FTSE", "STOXX50", "ASX200", "TSX", "NZ50", "SMI"]; FX = ["USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
FUT = ["ES", "NQ", "ZB", "ZN", "GC", "SI", "HG", "CL", "NG", "ZC", "ZS", "ZW", "KC", "SB", "CT", "CC", "LE", "HE"]
def series(name):
    d = engine.load_csv(f"data/raw/{name}_1d.csv").close; d.index = d.index.normalize(); d = d[~d.index.duplicated()]
    return np.log(d[d.index.weekday < 5]).loc["2000":]
st = lambda x: f"{x.mean()*252*100:+5.1f}%/год (Sharpe {x.mean()/x.std()*np.sqrt(252):+.2f}, n {len(x)})"
for grp, names in (("индексы", IDX), ("FX", FX), ("фьючерсы (роллы!)", FUT)):
    print(f"\n== {grp}", file=sys.stderr)
    for k in (1, 3, 6, 12):
        pool = []
        for nm in names:
            try: lc = series(nm)
            except Exception: continue
            r = lc.diff(); sig = np.sign(lc - lc.shift(21 * k)); vol = r.rolling(60).std() * np.sqrt(252)
            pos = (sig * (0.10 / vol)).clip(-3, 3).shift(1)              # позиция на завтра известна сегодня
            pos_m = pos.resample("21D").first().reindex(pos.index, method="ffill")   # пересмотр раз в 21 день
            strat = pos_m * r - (pos_m.diff().abs() * 3e-4)
            pool.append(strat.dropna())
        P = pd.concat(pool, axis=1).mean(axis=1).dropna()
        print(f"  k={k:2d} мес: всё {st(P)} | 2000–12 {st(P[:'2012'])} | 2013–26 {st(P['2013':])} | 2017–26 {st(P['2017':])}", file=sys.stderr)
