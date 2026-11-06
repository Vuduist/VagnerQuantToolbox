"""Паттерны из run_patterns_by_state внутри bias старшего таймфрейма (20-дневный тренд; дневная структура за 10 дней)."""
import sys
import numpy as np
import pandas as pd
import run_patterns_by_state as R

rng = np.random.default_rng(6)
st = lambda x: f"{x.mean()*1e4:+6.1f}({x.mean()/x.std()*np.sqrt(len(x)):+.1f},{len(x)})" if len(x) > 30 else "   n/a   "
for name, path, start in (("NQ", "data/raw/NQH_M1.csv", "2010-11-20"), ("EURUSD", "data/raw/EURH_M1.csv", "2009-06-01")):
    b = R.load5(path, start); b = b[b.index.weekday < 5]
    day = b.index.normalize(); g = b.groupby(day); dh, dl, dc = g.high.max(), g.low.min(), g.close.last()
    trend = np.sign(dc - dc.shift(20)).shift(1)                                        # известен на начало дня
    hh = (dh > dh.shift(5)) & (dl > dl.shift(5)); ll = (dh < dh.shift(5)) & (dl < dl.shift(5))
    struct = pd.Series(np.where(hh, 1, np.where(ll, -1, 0)), index=dh.index).shift(1)
    E = R.events(b); d = E.index.normalize()
    E["trend"] = trend.reindex(d).values; E["struct"] = struct.reindex(d).values
    print(f"\n==== {name}", file=sys.stderr)
    for S in ("trend", "struct"):
        for pat, gg in E.groupby("pat"):
            al = gg.r[gg.dir * gg[S] > 0]; ag = gg.r[gg.dir * gg[S] < 0]; neu = gg.r[gg[S] == 0]
            lab = np.where(gg.dir * gg[S] > 0, "al", np.where(gg.dir * gg[S] < 0, "ag", "ne")); x = gg.r.values; yrs = gg.index.year.values
            def maxt(L): return max((abs(x[L == c].mean()) / x[L == c].std() * np.sqrt((L == c).sum()) if (L == c).sum() > 30 else 0) for c in np.unique(L))
            obs = maxt(lab); null = []
            for _ in range(300):
                L2 = lab.copy()
                for y in np.unique(yrs):
                    mk = yrs == y; L2[mk] = rng.permutation(lab[mk])
                null.append(maxt(L2))
            h = lambda z: f"[{st(z[z.index < '2018-01-01'])}/{st(z[z.index >= '2018-01-01'])}]"
            print(f"  {S:6s} {pat:8s}: по bias {st(al)} {h(al)} | против {st(ag)} {h(ag)} | нейтр. {st(neu)} | perm p {np.mean(np.array(null) >= obs):.2f}", file=sys.stderr)
