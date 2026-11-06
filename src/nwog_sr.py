"""W04 (под-вариант): NWOG / Sunday gap как недельный динамический S/R.

Зона недели: [min, max] между close последнего бара прошлой недели и open первого бара новой
(ширина ≥ min_w·ATR100). Неделя = ISO-неделя для (t + 1 день), чтобы воскресный бар попал в неделю.
Касание: цена, находясь на стороне s (close выше hi или ниже lo) и отойдя от ближнего края на
≥ away·ATR, возвращается к краю (low ≤ hi при s=+1 / high ≥ lo при s=−1). Исход в следующие N баров:
  bounce  — ход от края в сторону s на ≥ k·ATR раньше, чем close за дальним краем зоны;
  through — close за дальним краем раньше;  neither — ни то, ни другое.
Признаки касания: side (+1 поддержка / −1 сопротивление), filled (до касания цена уже
закрывалась по обе стороны зоны), first (первое касание недели).
Нуллы: (1) та же зона, сдвинутая по цене на случайные ±[0.5, 3] ATR (R повторов) — «случайный
уровень той же ширины в той же неделе»; (3) зона той же ширины вокруг close случайного бара
прошлой недели («любой прошлый уровень»); (2) блочный бутстреп OHLC, полный конвейер.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine
from swings import atr


def week_ids(idx):
    sh = (idx.tz_localize(None) + pd.Timedelta(days=1)).isocalendar()
    return (sh.year * 100 + sh.week).to_numpy()


def zones(o, h, l, c, wk, min_w=0.2):
    a = atr(h, l, c, 100)
    first = np.r_[True, wk[1:] != wk[:-1]]
    out = []
    starts = np.flatnonzero(first)
    for j, i in enumerate(starts):
        if i == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        end = starts[j + 1] if j + 1 < len(starts) else len(c)
        lo, hi = min(c[i - 1], o[i]), max(c[i - 1], o[i])
        if (hi - lo) < min_w * a[i]:
            continue
        out.append((i, end, lo, hi, a[i]))
    return out


def scan(o, h, l, c, zone_list, offsets=None, away=1.0, k=1.0, N=24):
    """offsets: массив сдвигов зоны в ATR (нулл 1) или None."""
    recs = []
    for zi, (i, end, lo, hi, a) in enumerate(zone_list):
        if offsets is not None:
            lo, hi = lo + offsets[zi] * a, hi + offsets[zi] * a
        side, far = 0, False
        seen_above = seen_below = False
        for t in range(i, end):
            if c[t] > hi:
                seen_above = True
            if c[t] < lo:
                seen_below = True
            # обновление стороны
            if side == 0:
                if c[t] > hi:
                    side = 1; far = False
                elif c[t] < lo:
                    side = -1; far = False
                continue
            edge = hi if side > 0 else lo
            dist = (c[t] - edge) * side
            if dist >= away * a:
                far = True
            touched = (l[t] <= hi) if side > 0 else (h[t] >= lo)
            if far and touched:
                # исход
                res = "neither"
                far_edge = lo if side > 0 else hi
                for u in range(t, min(t + N, end)):
                    if (c[u] - far_edge) * side < 0:
                        res = "through"; break
                    if (h[u] - edge) * side >= k * a if side > 0 else (edge - l[u]) >= k * a:
                        res = "bounce"; break
                recs.append({"week": zi, "side": side, "filled": seen_above and seen_below,
                             "res": res, "t": t})
                # после касания состояние сбрасываем: ждём новой «дали» или смены стороны
                far = False
                if (c[t] - far_edge) * side < 0:
                    side = -side if (c[t] > hi or c[t] < lo) else 0
                continue
            # смена стороны без касания (гэп сквозь зону)
            if c[t] < lo and side > 0:
                side = -1; far = False
            elif c[t] > hi and side < 0:
                side = 1; far = False
    return pd.DataFrame(recs)


def summarize(recs: pd.DataFrame, n_weeks: int, prefix=""):
    out = {f"{prefix}n_weeks": float(n_weeks), f"{prefix}n_touch": float(len(recs))}
    if len(recs) == 0:
        return out
    out[f"{prefix}touch_per_week"] = len(recs) / n_weeks
    def rates(d, tag):
        if len(d) < 20:
            return
        out[f"{prefix}{tag}_n"] = float(len(d))
        out[f"{prefix}{tag}_p_bounce"] = float((d.res == "bounce").mean())
        out[f"{prefix}{tag}_p_through"] = float((d.res == "through").mean())
        dd = d[d.res != "neither"]
        out[f"{prefix}{tag}_p_bounce_dec"] = float((dd.res == "bounce").mean()) if len(dd) else np.nan
    rates(recs, "all")
    rates(recs[recs.filled], "filled")
    rates(recs[~recs.filled], "unfilled")
    rates(recs[recs.side > 0], "support")
    rates(recs[recs.side < 0], "resist")
    first = recs.groupby("week").head(1)
    rates(first, "first")
    return out


def nwog_test(df_et: pd.DataFrame, R=100, B=30, block=500, seed=0, **kw):
    o, h, l, c = (df_et[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    wk = week_ids(df_et.index)
    rng = np.random.default_rng(seed)

    def full(o, h, l, c):
        zl = zones(o, h, l, c, wk)
        return summarize(scan(o, h, l, c, zl, **kw), len(zl)), zl

    real, zl = full(o, h, l, c)
    n1 = []
    for _ in range(R):
        off = rng.uniform(0.5, 3.0, size=len(zl)) * rng.choice([-1, 1], size=len(zl))
        n1.append(summarize(scan(o, h, l, c, zl, offsets=off, **kw), len(zl)))
    # нулл 3: зона той же ширины вокруг close случайного бара прошлой недели (случайный прошлый уровень)
    n3 = []
    for _ in range(R):
        zl3 = []
        for (i, end, lo, hi, a) in zl:
            cr = c[max(0, i - 1 - int(rng.integers(1, 120)))]
            w = hi - lo
            zl3.append((i, end, cr - w / 2, cr + w / 2, a))
        n3.append(summarize(scan(o, h, l, c, zl3, **kw), len(zl3)))
    n2 = []
    for b in range(B):
        so, sh, sl, sc = engine.block_bootstrap_ohlc(o, h, l, c, block=block, seed=seed * 100 + b)
        n2.append(full(so, sh, sl, sc)[0])
    z = {}
    for kkey, v in real.items():
        row = {"real": v}
        for name, pool in (("n1", n1), ("n3", n3), ("n2", n2)):
            vals = np.array([s.get(kkey, np.nan) for s in pool], float); vals = vals[np.isfinite(vals)]
            if len(vals) >= 5:
                sd = vals.std(ddof=1)
                row[name + "_mean"] = vals.mean(); row[name + "_sd"] = sd
                row[name + "_z"] = (v - vals.mean()) / sd if sd > 0 else np.nan
        z[kkey] = row
    return real, z
