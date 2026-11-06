"""S12: трендовые линии — «третье касание диагонали по двум свинг-лоу — ловушка».

Утверждения ICT (Trendline Phantoms): линия по двум последовательным свинг-лоу (higher low)
на третьем касании не держит — пробой чаще отскока («монетка или хуже»), цена идёт к стопам
за точкой 2 (лоу L2), а не к хаю между касаниями 2 и 3; хай между 2 и 3 работает как bearish OB.

Разметка без заглядывания: свинги — каузальная машина состояний (swings.causal_swings, k*ATR),
линия L1→L2 считается существующей с бара подтверждения L2. Требование «линия не режет цену»:
все лоу между L1 и L2 ≥ линии − δ (δ = tol·ATR). Третье касание: первый бар t3 после подтверждения
L2 с low ≤ line(t3) + δ, при условии что до него не было закрытия ниже line − δ («пробой до
касания» считается отдельно). Горизонт N = clip(2·(i2−i1), 20, 400) баров.

Исходы после t3 (первое из двух событий):
  break  — close < line − δ;   bounce — high ≥ H23 (хай между L2 и t3)        → p_break_first
  p2     — low ≤ p2 (лоу точки 2); h23 — high ≥ H23                            → p_p2_first
  fade_ret10 / fade_retN — (c[t3] − c[t3+k]) / ATR (положительно = в сторону фейда ICT)
  h23_reject — после достижения H23: возврат к линии раньше, чем high ≥ H23 + 0.5·ATR
Зеркально для нисходящих линий по двум свинг-хаям (через инверсию цен).

Нуллы:
  1) случайная диагональ: тот же наклон (в единицах ATR), та же длительность i2−i1, та же
     задержка подтверждения; якорь — лоу случайного бара; требование «цена над линией» то же.
     Убирает привязку к двум свинг-лоу, сохраняя геометрию касания сверху.
  2) блочный бутстреп OHLC (engine.block_bootstrap_ohlc), полный конвейер.
"""

from __future__ import annotations

import numpy as np

import engine
from swings import atr, causal_swings

METRICS = ("p_touch", "p_break_first", "p_break_vs_1atr", "p_neither", "p_p2_first", "p_p2_neither",
           "fade_ret10", "fade_retN", "p_h23_reject", "p_break_bar")


def _touch(h, l, c, i1, p1, slope, start, i2, delta, N, a2):
    """Сканирует третье касание линии и исходы. Возвращает dict или None/'prebreak'."""
    n = len(c)
    hi_run = -np.inf
    for t in range(i2 + 1, min(start, n)):
        hi_run = max(hi_run, h[t])
    for t in range(start, min(start + N, n)):
        lv = p1 + slope * (t - i1)
        if l[t] <= lv + delta:
            t3 = t
            break
        if c[t] < lv - delta:
            return "prebreak"
        hi_run = max(hi_run, h[t])
    else:
        return None
    H23 = max(hi_run, h[t3])
    p2 = p1 + slope * (i2 - i1)
    end = min(t3 + N, n - 1)
    if end <= t3 + 1:
        return None
    first_break = first_bounce = first_p2 = first_b1 = None
    for t in range(t3, end + 1):
        lv = p1 + slope * (t - i1)
        if first_break is None and c[t] < lv - delta:
            first_break = t
        if t > t3:
            if first_bounce is None and h[t] >= H23:
                first_bounce = t
            if first_p2 is None and l[t] <= p2:
                first_p2 = t
            if first_b1 is None and h[t] >= lv + a2:
                first_b1 = t
        if first_break is not None and first_bounce is not None and first_p2 is not None and first_b1 is not None:
            break
    out = {"t3": t3, "H23": H23, "p2": p2, "N": N,
           "break_bar": first_break == t3}
    if first_break is None and first_bounce is None:
        out["bb"] = np.nan
    else:
        out["bb"] = 1.0 if (first_bounce is None or (first_break is not None and first_break <= first_bounce)) else 0.0
    if first_p2 is None and first_bounce is None:
        out["p2f"] = np.nan
    else:
        out["p2f"] = 1.0 if (first_bounce is None or (first_p2 is not None and first_p2 <= first_bounce)) else 0.0
    out["neither"] = float(first_break is None and first_bounce is None)
    if first_break is None and first_b1 is None:
        out["b1"] = np.nan
    else:
        out["b1"] = 1.0 if (first_b1 is None or (first_break is not None and first_break <= first_b1)) else 0.0
    out["p2neither"] = float(first_p2 is None and first_bounce is None)
    # возврат к H23 и отбой от него
    out["h23rej"] = np.nan
    if first_bounce is not None:
        tb = first_bounce
        rej = np.nan
        for t in range(tb + 1, min(tb + N, n)):
            lv = p1 + slope * (t - i1)
            if h[t] >= H23 + 0.5 * a2:                       # H23 + 0.5 ATR
                rej = 0.0; break
            if l[t] <= lv:
                rej = 1.0; break
        out["h23rej"] = rej
    return out


def find_lines(h, l, c, k_swing=3.0, tol=0.1, min_gap=5, max_gap=300):
    """Линии по двум последовательным свинг-лоу с higher low, не режущие цену."""
    sw_idx, sw_px, sw_dir, sw_conf = causal_swings(h, l, c, k_swing)
    a = atr(h, l, c)
    lows = [(sw_idx[j], sw_px[j], sw_conf[j]) for j in range(len(sw_idx)) if sw_dir[j] < 0]
    lines = []
    for j in range(1, len(lows)):
        i1, p1, _ = lows[j - 1]
        i2, p2, conf2 = lows[j]
        gap = i2 - i1
        if gap < min_gap or gap > max_gap or p2 <= p1 or not np.isfinite(a[i2]):
            continue
        slope = (p2 - p1) / gap
        delta = tol * a[i2]
        seg = l[i1 + 1:i2] - (p1 + slope * np.arange(1, gap))
        if seg.size and seg.min() < -delta:
            continue
        lines.append((i1, p1, i2, conf2, slope, delta, a[i2]))
    return lines


def _horizon(gap):
    return int(np.clip(2 * gap, 20, 400))


def events_for_lines(o, h, l, c, lines, c_real, side, bpd):
    """Исходы для набора линий (реальных или псевдо). Возвращает список записей."""
    n = len(c)
    recs, n_lines, n_pre = [], 0, 0
    a = atr(h, l, c)
    for (i1, p1, i2, conf2, slope, delta, a2) in lines:
        n_lines += 1
        N = _horizon(i2 - i1)
        r = _touch(h, l, c, i1, p1, slope, conf2 + 1, i2, delta, N, a2)
        if r == "prebreak":
            n_pre += 1
            continue
        if r is None:
            continue
        t3 = r["t3"]
        at = a[t3] if np.isfinite(a[t3]) else a2
        k10 = min(t3 + 10, n - 1)
        kN = min(t3 + N, n - 1)
        lb = t3 - 20 * bpd
        ret20 = c_real[t3] / c_real[lb] - 1 if lb >= 0 else np.nan
        recs.append({"t3": t3, "bb": r["bb"], "neither": r["neither"], "p2f": r["p2f"],
                     "p2neither": r["p2neither"], "h23rej": r["h23rej"],
                     "break_bar": float(r["break_bar"]), "b1": r["b1"],
                     "fr10": (c[t3] - c[k10]) / at, "frN": (c[t3] - c[kN]) / at,
                     "aligned": float(side * ret20 < 0) if np.isfinite(ret20) else np.nan})
    return recs, n_lines, n_pre


def summarize(recs, n_lines, n_pre, prefix=""):
    out = {prefix + "n_lines": float(n_lines), prefix + "n_touch": float(len(recs))}
    if n_lines:
        out[prefix + "p_touch"] = len(recs) / n_lines
        out[prefix + "p_prebreak"] = n_pre / n_lines
    if not recs:
        return out
    def m(key):
        v = np.array([r[key] for r in recs], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else np.nan
    out[prefix + "p_break_first"] = m("bb")
    out[prefix + "p_neither"] = m("neither")
    out[prefix + "p_p2_first"] = m("p2f")
    out[prefix + "p_p2_neither"] = m("p2neither")
    out[prefix + "fade_ret10"] = m("fr10")
    out[prefix + "fade_retN"] = m("frN")
    out[prefix + "p_h23_reject"] = m("h23rej")
    out[prefix + "p_break_bar"] = m("break_bar")
    out[prefix + "p_break_vs_1atr"] = m("b1")
    return out


def _both_sides(o, h, l, c, bpd, k_swing, tol):
    """Восходящие линии (фейд = шорт) на реальных ценах + нисходящие через инверсию."""
    res = []
    for side, (oo, hh, ll, cc) in ((1, (o, h, l, c)), (-1, (-o, -l, -h, -c))):
        lines = find_lines(hh, ll, cc, k_swing, tol)
        recs, nl, npre = events_for_lines(oo, hh, ll, cc, lines, c, side, bpd)
        res.append((side, lines, recs, nl, npre))
    return res


def pseudo_lines(h, l, c, lines, rng, tol=0.1, tries=30):
    """Нулл 1: та же геометрия (наклон в ATR, длительность, задержка), случайный якорь."""
    n = len(c)
    a = atr(h, l, c)
    out = []
    for (i1, p1, i2, conf2, slope, delta, a2) in lines:
        gap, lag = i2 - i1, conf2 - i2
        s_atr = slope / a2
        N = _horizon(gap)
        for _ in range(tries):
            j1 = int(rng.integers(100, n - gap - lag - N - 2))
            j2 = j1 + gap
            if not np.isfinite(a[j2]):
                continue
            sl = s_atr * a[j2]
            q1 = l[j1]
            d = tol * a[j2]
            seg = l[j1 + 1:j2] - (q1 + sl * np.arange(1, gap))
            if seg.size and seg.min() < -d:
                continue
            out.append((j1, q1, j2, j2 + lag, sl, d, a[j2]))
            break
    return out


def s12_stats(df, bpd, k_swing=3.0, tol=0.1, B=50, R=20, block=500, seed=0):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng = np.random.default_rng(seed)

    def full(o, h, l, c, c_real):
        real = {}
        sides = _both_sides(o, h, l, c, bpd, k_swing, tol)
        allrecs, nl, npre = [], 0, 0
        for side, lines, recs, n_l, n_p in sides:
            allrecs += recs; nl += n_l; npre += n_p
            real.update(summarize(recs, n_l, n_p, "up_" if side > 0 else "dn_"))
        real.update(summarize(allrecs, nl, npre, ""))
        al = [r for r in allrecs if r["aligned"] == 1.0]
        real.update(summarize(al, nl, npre, "al_"))
        return real, sides

    real, sides = full(o, h, l, c, c)
    # нулл 1: случайные диагонали, R повторов
    n1 = []
    for _ in range(R):
        allrecs, nl, npre = [], 0, 0
        for side, lines, _, _, _ in sides:
            hh, ll, cc, oo = (h, l, c, o) if side > 0 else (-l, -h, -c, -o)
            pl = pseudo_lines(hh, ll, cc, lines, rng, tol)
            recs, n_l, n_p = events_for_lines(oo, hh, ll, cc, pl, c, side, bpd)
            allrecs += recs; nl += n_l; npre += n_p
        s = summarize(allrecs, nl, npre, "")
        al = [r for r in allrecs if r["aligned"] == 1.0]
        s.update(summarize(al, nl, npre, "al_"))
        n1.append(s)
    # нулл 2: бутстреп
    n2 = []
    for b in range(B):
        so, sh, sl, sc = engine.block_bootstrap_ohlc(o, h, l, c, block=block, seed=seed * 1000 + b)
        s, _ = full(so, sh, sl, sc, sc)
        n2.append(s)
    z = {}
    for k, v in real.items():
        row = {"real": v}
        for name, pool in (("n1", n1), ("n2", n2)):
            vals = np.array([s.get(k, np.nan) for s in pool], float)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 5:
                sd = vals.std(ddof=1)
                row[name + "_mean"] = vals.mean(); row[name + "_sd"] = sd
                row[name + "_z"] = (v - vals.mean()) / sd if sd > 0 else np.nan
        z[k] = row
    return real, z
