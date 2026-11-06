"""COT — крайние позиции non-commercials (перцентиль nc_net за 26/52 недели) контрарно на 4 недели вперёд.
Вход по закрытию понедельника после публикации (позиции вторника публикуются в пятницу)."""
import sys
import numpy as np
import pandas as pd
import engine

C = pd.read_csv("data/raw/cot_legacy.csv", parse_dates=["date"])
st = lambda x: f"{x.mean()*100:+.2f}% (t {x.mean()/x.std()*np.sqrt(len(x)):+.1f}, n {len(x)})"
rows = []
for m, g in C.groupby("market"):
    try:
        D = engine.load_csv(f"data/raw/{m}_1d.csv").close; D.index = D.index.normalize(); D = D[~D.index.duplicated()]
    except Exception:
        continue
    g = g.sort_values("date").set_index("date")
    for W in (26, 52):
        pct = g.nc_net.rolling(W).rank(pct=True)
        for d, p in pct.dropna().items():
            entry = D[D.index >= d + pd.Timedelta(days=6)]            # понедельник после пятничной публикации
            if len(entry) < 21: continue
            e0 = entry.index[0]; r = np.log(entry.iloc[20] / entry.iloc[0])
            rows.append({"market": m, "W": W, "date": e0, "pct": p, "r4w": r})
R = pd.DataFrame(rows)
print("рынков:", R.market.nunique(), sorted(R.market.unique()), file=sys.stderr)
for W in (26, 52):
    X = R[R.W == W]
    hi, lo, mid = X[X.pct >= 0.9], X[X.pct <= 0.1], X[(X.pct > 0.3) & (X.pct < 0.7)]
    contr = pd.concat([-hi.r4w, lo.r4w])
    print(f"W={W}: спекулянты ≥90% → 4 нед. {st(hi.r4w)}; ≤10% → {st(lo.r4w)}; контрарно вместе {st(contr)}; середина {st(mid.r4w)}", file=sys.stderr)
    for lab, a, b in (("2000–2012", None, "2012-12-31"), ("2013–2026", "2013-01-01", None)):
        h = hi[(hi.date >= (a or "1900")) & (hi.date <= (b or "2100"))]; l = lo[(lo.date >= (a or "1900")) & (lo.date <= (b or "2100"))]
        print(f"   {lab}: контрарно {st(pd.concat([-h.r4w, l.r4w]))}", file=sys.stderr)
    # по рынкам
    print("   по рынкам (контрарно, t): " + " ".join(f"{m}:{(lambda x: x.mean()/x.std()*np.sqrt(len(x)))(pd.concat([-X[(X.market==m)&(X.pct>=0.9)].r4w, X[(X.market==m)&(X.pct<=0.1)].r4w])):+.1f}" for m in sorted(X.market.unique())), file=sys.stderr)
