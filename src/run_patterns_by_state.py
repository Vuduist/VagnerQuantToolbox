"""Закрытые паттерны ICT внутри состояний (S_vol, S_day, S_sess) на 5-минутках NQ/EURUSD, время NY."""
import sys
import numpy as np
import pandas as pd
import engine

rng = np.random.default_rng(3)
st = lambda x: f"{x.mean()*1e4:+6.1f}({x.mean()/x.std()*np.sqrt(len(x)):+.1f},{len(x)})" if len(x) > 30 else "   n/a   "

def load5(path, start):
    df = engine.load_csv(path); df.index = df.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    df = df.loc[start:]
    b = df.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return b

def states(b):
    day = b.index.normalize(); d = pd.DataFrame({"h": b.high, "l": b.low, "c": b.close}, index=b.index)
    drng = (d.h.groupby(day).max() - d.l.groupby(day).min()) / d.c.groupby(day).last()
    med = drng.rolling(20).median().shift(1); ratio = (drng.shift(1) / med)            # вчерашний диапазон к медиане 20 (известно утром)
    svol = pd.cut(ratio.rank(pct=True), [0, 1/3, 2/3, 1], labels=["vol_low", "vol_mid", "vol_high"], include_lowest=True)
    # S_day: эффективность 9:30→11:00
    m = b[(b.index.hour * 60 + b.index.minute >= 570) & (b.index.hour * 60 + b.index.minute < 660)]
    g = m.groupby(m.index.normalize()); eff = ((g.close.last() - g.open.first()).abs() / (g.high.max() - g.low.min())).replace(np.inf, np.nan)
    sday = pd.cut(eff.rank(pct=True), [0, 1/3, 2/3, 1], labels=["day_range", "day_mid", "day_trend"], include_lowest=True)
    return svol, sday

def sess(t):
    h = t.hour
    return "L_2-8" if 2 <= h < 8 else ("NY_8-12" if 8 <= h < 12 else ("PM_12-16" if 12 <= h < 16 else "other"))

def events(b):
    o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close")); idx = b.index; n = len(b)
    mins = idx.hour * 60 + idx.minute; day = idx.normalize(); rngm = pd.Series(h - l).rolling(200).mean().shift(1).to_numpy()
    lc = np.log(c); ev = []
    # P1: FVG — бычий: l[t] > h[t-2] с импульсом бара t-1; первое касание (l ≤ верх гэпа) в 48 барах; вход по close касания, 6 баров
    for t in range(2, n - 60):
        if not np.isfinite(rngm[t]) or (h[t - 1] - l[t - 1]) < 2 * rngm[t]: continue
        if l[t] > h[t - 2]:
            top, bot = l[t], h[t - 2]; sgn = 1
        elif h[t] < l[t - 2]:
            top, bot = h[t], l[t - 2]; sgn = -1
        else: continue
        for k in range(t + 1, min(t + 49, n - 7)):
            if day[k] != day[t]: break
            touched = (l[k] <= top) if sgn > 0 else (h[k] >= bot)
            if touched:
                broke = (c[k] < bot) if sgn > 0 else (c[k] > top)
                if not broke: ev.append(("P1_fvg", idx[k], sgn * (lc[k + 6] - lc[k]), sgn))
                break
    # P2: ORB 9:30–10:00, первое закрытие вне до 11:30, горизонт 12:00; P3: KZ 2–5 → 11:00; P4: Judas к 2:00 → 12:00
    for dd, g in b.groupby(day):
        mm = g.index.hour * 60 + g.index.minute; gi = g.index
        orb = g[(mm >= 570) & (mm < 600)]
        if len(orb) >= 5:
            hi, lo = orb.high.max(), orb.low.min(); after = g[(mm >= 600) & (mm < 690)]; end = g[mm == 715]
            if len(end):
                for t, row in after.iterrows():
                    if row.close > hi: ev.append(("P2_orb", t, np.log(end.close.iloc[0] / row.close), 1)); break
                    if row.close < lo: ev.append(("P2_orb", t, -np.log(end.close.iloc[0] / row.close), -1)); break
        kz = g[(mm >= 120) & (mm < 300)]; at5 = g[mm == 300]; at11 = g[mm == 660]
        if len(kz) >= 20 and len(at5) and len(at11):
            mid = (kz.high.max() + kz.low.min()) / 2; sgn = 1 if at5.close.iloc[0] > mid else -1
            ev.append(("P3_kz", at5.index[0], sgn * np.log(at11.close.iloc[0] / at5.close.iloc[0]), sgn))
        op = g[mm == 0]; at2 = g[mm == 120]; at12 = g[mm == 720]
        if len(op) and len(at2) and len(at12):
            sgn = 1 if at2.close.iloc[0] < op.open.iloc[0] else -1          # ниже открытия → Judas вниз → лонг
            ev.append(("P4_judas", at2.index[0], sgn * np.log(at12.close.iloc[0] / at2.close.iloc[0]), sgn))
    E = pd.DataFrame(ev, columns=["pat", "t", "r", "dir"]).set_index("t").sort_index()
    return E

def placebo(b, E, svol, sday, n_per=3):
    """случайные бары того же часа и состояния: доходность вперёд с горизонтом паттерна (P1 6 баров; P2/P3/P4 — по времени)."""
    lc = np.log(b.close); out = []
    hz = {"P1_fvg": 6, "P2_orb": 24, "P3_kz": 72, "P4_judas": 120}
    for pat, g in E.groupby("pat"):
        H = hz[pat]; pos = b.index.get_indexer(g.index)
        for p in pos:
            for _ in range(n_per):
                q = p + rng.integers(-2000, 2000)
                if 0 < q < len(b) - H - 1 and b.index[q].hour == b.index[p].hour:
                    out.append((pat, b.index[q], (np.sign(rng.standard_normal()) * (lc.iloc[q + H] - lc.iloc[q]))))
    return pd.DataFrame(out, columns=["pat", "t", "r"]).set_index("t")

def report(name, E, P, svol, sday):
    E = E.copy(); d = E.index.normalize(); E["svol"] = svol.reindex(d).values; E["sday"] = sday.reindex(d).values; E["sess"] = [sess(t) for t in E.index]
    E.loc[E.index.hour < 11, "sday"] = np.nan                       # S_day только для событий после 11:00
    half = E.index < "2018-01-01"
    print(f"\n==== {name}: события {E.groupby('pat').size().to_dict()}", file=sys.stderr)
    for pat, g in E.groupby("pat"):
        print(f"  {pat}: безусловно {st(g.r)} | плацебо {st(P[P.pat == pat].r)}", file=sys.stderr)
        for S in ("svol", "sday", "sess"):
            labs = g[S].astype(object).where(g[S].notna(), None)
            cats = sorted({c for c in labs if c is not None})
            if not cats: continue
            parts = []
            for c in cats:
                m = (labs == c).values; a = g.r[m & (g.index < "2018-01-01")]; b_ = g.r[m & (g.index >= "2018-01-01")]
                pl = P[(P.pat == pat)]; plm = pl.r if S == "sess" else pl.r
                parts.append(f"{c} {st(g.r[m])} [{st(a)}/{st(b_)}]")
            ok = labs.notna().values; x = g.r.values[ok]; lab = labs[ok].astype(str).values; yrs = g.index.year.values[ok]
            def maxt(L):
                return max((abs(x[L == c].mean()) / x[L == c].std() * np.sqrt((L == c).sum()) if (L == c).sum() > 30 else 0) for c in np.unique(L))
            obs = maxt(lab); null = []
            for _ in range(300):
                L2 = lab.copy()
                for y in np.unique(yrs):
                    mk = yrs == y; L2[mk] = rng.permutation(lab[mk])
                null.append(maxt(L2))
            print("    " + S + ": " + " | ".join(parts) + f"   | perm p {np.mean(np.array(null) >= obs):.2f}", file=sys.stderr)

if __name__ == "__main__":
    for name, path, start in (("NQ", "data/raw/NQH_M1.csv", "2010-11-20"), ("EURUSD", "data/raw/EURH_M1.csv", "2009-06-01")):
      b = load5(path, start); b = b[b.index.weekday < 5]
      svol, sday = states(b); E = events(b); P = placebo(b, E, svol, sday)
      report(name, E, P, svol, sday)
