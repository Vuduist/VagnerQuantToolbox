"""Ликвидность как свойство пути. Уровни PDH/PDL и азиатский диапазон (0:00–8:00 NY); при первом касании — доля проколов ≥ δ
и доля возвратов внутрь за 12 баров; плацебо — уровни, сдвинутые на случайные ±0.3…0.7 дневного ATR. 5-минутки, время NY."""
import sys
import numpy as np
import pandas as pd
import engine

rng = np.random.default_rng(4)
def load5(path, start):
    df = engine.load_csv(path); df.index = df.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    b = df.loc[start:].resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return b[b.index.weekday < 5]

def study(name, b):
    day = b.index.normalize(); days = sorted(set(day))
    g = b.groupby(day); pdh = g.high.max().shift(1); pdl = g.low.min().shift(1)
    datr = (g.high.max() - g.low.min()).rolling(20).mean().shift(1)
    asia = b[(b.index.hour < 8)]; ga = asia.groupby(asia.index.normalize()); ah, al = ga.high.max(), ga.low.min()
    rows = []
    for d in days[21:]:
        x = b[day == d]
        if len(x) < 100 or not np.isfinite(datr.get(d, np.nan)): continue
        A = datr[d]; h, l, c = x.high.values, x.low.values, x.close.values; t = x.index
        levels = [("PDH", pdh[d], +1, 0), ("PDL", pdl[d], -1, 0)]
        if d in ah.index: levels += [("AsiaH", ah[d], +1, 96), ("AsiaL", al[d], -1, 96)]   # после 8:00 (96-й бар)
        for nm, L, side, start in levels:
            if not np.isfinite(L): continue
            for tag, Lv in (("liq", L), ("placebo", L + side * A * rng.uniform(0.3, 0.7))):
                hit = None
                for i in range(start, len(x)):
                    if (h[i] >= Lv) if side > 0 else (l[i] <= Lv):
                        hit = i; break
                if hit is None or hit + 12 >= len(x): continue
                seg_h, seg_l, seg_c = h[hit:hit + 13], l[hit:hit + 13], c[hit:hit + 13]
                over = (seg_h.max() - Lv) / A if side > 0 else (Lv - seg_l.min()) / A      # максимальный прокол за 12 баров
                back = (seg_c[1:] < Lv).any() if side > 0 else (seg_c[1:] > Lv).any()      # вернулась внутрь
                # после прокола ≥ 0.1 ATR: ход за следующие 12 баров от первого закрытия после прокола
                rows.append({"day": d, "lvl": nm, "tag": tag, "over": over, "back": back, "hour": t[hit].hour, "c_after": (seg_c[-1] - seg_c[0]) / A * (-side)})
    R = pd.DataFrame(rows)
    print(f"\n== {name}: касаний {len(R)//2} уровней", file=sys.stderr)
    for lvl, gg in R.groupby("lvl"):
        liq, pl = gg[gg.tag == "liq"], gg[gg.tag == "placebo"]
        parts = []
        for dlt in (0.1, 0.25, 0.5):
            a, b_ = (liq.over >= dlt).mean(), (pl.over >= dlt).mean()
            parts.append(f"прокол≥{dlt}: {a:.2f} vs {b_:.2f}")
        rb = lambda z, dlt: z.back[z.over >= dlt].mean() if (z.over >= dlt).sum() > 30 else np.nan
        print(f"  {lvl:6s} n {len(liq)}: " + "; ".join(parts) + f" | возврат внутрь после прокола ≥0.25: {rb(liq, .25):.2f} vs {rb(pl, .25):.2f}"
              f" | ход против уровня 12 баров (ATR): {liq.c_after.mean():+.3f} (t {liq.c_after.mean()/liq.c_after.std()*np.sqrt(len(liq)):+.1f}) vs {pl.c_after.mean():+.3f}", file=sys.stderr)
        for lab, s in (("до 2018", liq.day < pd.Timestamp("2018-01-01")), ("с 2018", liq.day >= pd.Timestamp("2018-01-01"))):
            z = liq[s]; zp = pl[pl.day.isin(z.day)]
            print(f"     {lab}: прокол≥0.25 {(z.over>=.25).mean():.2f} vs {(zp.over>=.25).mean():.2f}; возврат {rb(z,.25):.2f} vs {rb(zp,.25):.2f}; ход {z.c_after.mean():+.3f} vs {zp.c_after.mean():+.3f}", file=sys.stderr)
    return R

for name, path, start in (("EURUSD", "data/raw/EURH_M1.csv", "2009-06-01"), ("NQ", "data/raw/NQH_M1.csv", "2010-11-20")):
    R = study(name, load5(path, start)); R.to_csv(f"out/liquidity_path_{name}.csv", index=False)
