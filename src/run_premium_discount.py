"""Премиум/дисконт как состояние: положение цены события в диапазоне предыдущего дня (нижняя треть — дисконт, верхняя — премиум).
Бычьи паттерны в дисконте против премиума, медвежьи зеркально. Переиспользует события ранее (run_patterns_by_state)."""
import sys
import numpy as np
import pandas as pd
import run_patterns_by_state as R

rng = np.random.default_rng(5)
st = lambda x: f"{x.mean()*1e4:+6.1f}({x.mean()/x.std()*np.sqrt(len(x)):+.1f},{len(x)})" if len(x) > 30 else "   n/a   "
for name, path, start in (("NQ", "data/raw/NQH_M1.csv", "2010-11-20"), ("EURUSD", "data/raw/EURH_M1.csv", "2009-06-01")):
    b = R.load5(path, start); b = b[b.index.weekday < 5]
    # события с направлением: пересобираем, сохраняя знак
    o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close")); idx = b.index; day = idx.normalize()
    g = b.groupby(day); pdh = g.high.max().shift(1); pdl = g.low.min().shift(1)
    E = R.events(b)
    # знак события восстановим по знаку доходности нельзя; пересчитаем направление для P1/P2/P4 через повтор логики упрощённо:
    # P3/P4 направление = знак (r / |r|) не годится. Поэтому считаем положение в диапазоне и смотрим ТОЛЬКО «в сторону состояния»:
    # для события в дисконте ожидаем, что бычьи сделки лучше медвежьих. Нужен знак сделки → добавим его в events через монкипатч:
    pos = b.index.get_indexer(E.index); d = E.index.normalize()
    rngp = (b.close.iloc[pos].values - pdl.reindex(d).values) / (pdh.reindex(d).values - pdl.reindex(d).values)
    E["pos"] = rngp; z = pd.cut(E.pos, [-9, 1/3, 2/3, 99], labels=["discount", "mid", "premium"]).astype(object)
    # выровненность с направлением сделки: лонг в дисконте / шорт в премиуме = aligned
    E["zone"] = np.where(z.isna(), "nan", np.where(z == "mid", "mid", np.where(((E.dir > 0) & (z == "discount")) | ((E.dir < 0) & (z == "premium")), "aligned", "against")))
    print(f"\n==== {name}", file=sys.stderr)
    for pat, gg in E.groupby("pat"):
        parts = []
        for z in ("aligned", "mid", "against"):
            x = gg.r[gg.zone == z]; a = x[x.index < "2018-01-01"]; b_ = x[x.index >= "2018-01-01"]
            parts.append(f"{z} {st(x)} [{st(a)}/{st(b_)}]")
        lab = gg.zone.astype(str).values; ok = lab != "nan"; x = gg.r.values[ok]; lab = lab[ok]; yrs = gg.index.year.values[ok]
        def maxt(L): return max((abs(x[L == c].mean()) / x[L == c].std() * np.sqrt((L == c).sum()) if (L == c).sum() > 30 else 0) for c in np.unique(L))
        obs = maxt(lab); null = []
        for _ in range(300):
            L2 = lab.copy()
            for y in np.unique(yrs):
                mk = yrs == y; L2[mk] = rng.permutation(lab[mk])
            null.append(maxt(L2))
        print(f"  {pat}: " + " | ".join(parts) + f"   | perm p {np.mean(np.array(null) >= obs):.2f}", file=sys.stderr)
