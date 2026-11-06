"""Пять популярных ICT-моделей на минутках (каузально, сделка = лимит/маркет + стоп + цель + тайм-стоп).

Торговый день: 18:00 ET (вчера) → 17:00 ET. Окна заданы по часам Нью-Йорка.
FVG: бычий — high[i-2] < low[i] (зона [high[i-2], low[i]]), медвежий — low[i-2] > high[i].
Вход лимитом на ближней границе FVG (первое касание при откате). Исполнение консервативно:
лимит засчитан только при касании; на баре входа стоп проверяется первым; стоп и цель на одном
баре = стоп; цель до исполнения = отмена; выход по close в 16:00 ET, если ничего не задето.

Издержки (аргумент cost_bp, несмотря на имя, — в ПУНКТАХ цены за круг): два тика фьючерса с комиссией
(NQ/ES 0.5 пт), EURUSD 0.8 пипса, золото 0.30 $. R = валовый R − издержки/риск.

Модели:
  sb        Silver Bullet: первый FVG (M1), закрывшийся в окне 03–04 / 10–11 / 14–15 ET, у которого есть
            draw on liquidity (хай/лоу дня до FVG или PDH/PDL) не ближе framework-минимума (10 хэндлов ES
            ≈ 0.2% цены, FX 15 пипсов ≈ 0.15%); стоп за свечой 1 FVG; цель — эта ликвидность; вход до конца окна.
  sb5       то же, но цель фиксированная «5 хэндлов» (половина framework-минимума).
  m2022     Модель 2022: в окне 08:30–11:00 ET (FX/золото: London 02:00–05:00) снятие overnight high/low или
            PDH/PDL → MSS (закрытие за последним фрактальным свингом против снятия) → FVG в displacement-ноге →
            лимит на FVG, стоп за экстремумом снятия, цель — противоположная ликвидность (≥1R), вход до 12:00.
  venom     Venom: коробка 08:00–09:30 ET; после 09:30 до 11:00 снятие её хая/лоя → MSS → FVG → лимит,
            стоп за экстремумом снятия, цель — противоположная граница коробки (≥1R), вход до 12:00.
  son       Модель сына (30 s → M1, секундных данных нет): 09:30–12:00 ET, M5-бар снимает последний
            фрактальный свинг M5 и закрывается против снятия → первый M1 FVG в течение 30 мин →
            лимит, стоп 0.06% цены (≈11 пунктов NQ при 18000), цель 1.5R; DOL-фильтр: PDH/PDL дальше цели.
  ts        Turtle Soup (ICT, интрадей): 02:00–11:00 ET M5-бар выносит PDH/PDL, в течение 30 мин M5 закрывается
            обратно → вход по close, стоп за экстремумом выноса, цель 2R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine


def tmin(hh, mm=0):
    """Минуты от 18:00 ET (начало торгового дня)."""
    return ((hh * 60 + mm) - 18 * 60) % 1440


T1600 = tmin(16)


def load_days(path):
    df = engine.load_csv(path)
    et = df.index.tz_localize("UTC").tz_convert("America/New_York")
    clock = et.hour * 60 + et.minute
    tm = (np.asarray(clock) - 18 * 60) % 1440
    tday = (et + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    td = np.asarray(tday)
    starts = np.flatnonzero(np.r_[True, td[1:] != td[:-1]])
    ends = np.r_[starts[1:], len(td)]
    days = []
    for s, e in zip(starts, ends):
        if e - s < 300:
            continue
        days.append({"date": pd.Timestamp(td[s]), "tm": tm[s:e], "o": o[s:e], "h": h[s:e], "l": l[s:e], "c": c[s:e]})
    for i in range(1, len(days)):
        days[i]["pdh"] = days[i - 1]["h"].max(); days[i]["pdl"] = days[i - 1]["l"].min()
    return days[1:]


def idx_at(tm, t):
    """Первый бар с tm >= t."""
    return int(np.searchsorted(tm, t))


def simulate(d, side, entry, stop, target, j0, j_fill_end, o, h, l, c, tm, cost_bp, market=False):
    """R сделки или None (нет исполнения)."""
    n = len(c)
    j_end = min(idx_at(tm, T1600), n) - 1
    if j0 > j_end:
        return None
    risk = abs(entry - stop)
    if risk <= 0 or not np.isfinite(risk):
        return None
    jf = None
    if market:
        jf = j0
    else:
        for j in range(j0, min(j_fill_end, j_end) + 1):
            if side > 0:
                if l[j] <= entry:
                    jf = j; break
                if h[j] >= target:
                    return None
            else:
                if h[j] >= entry:
                    jf = j; break
                if l[j] <= target:
                    return None
    if jf is None:
        return None
    exit_px = c[j_end]
    for j in range(jf, j_end + 1):
        hit_s = l[j] <= stop if side > 0 else h[j] >= stop
        if hit_s:
            exit_px = stop; break
        if j > jf or market:
            hit_t = h[j] >= target if side > 0 else l[j] <= target
            if hit_t:
                exit_px = target; break
    rg = side * (exit_px - entry) / risk
    return {"date": d["date"], "side": side, "entry": entry, "stop": stop, "target": target,
            "risk_pct": risk / entry * 100, "R_gross": rg, "R": rg - cost_bp / risk}


def fractals(h, l):
    n = len(h)
    fh = np.zeros(n, bool); fl = np.zeros(n, bool)
    if n >= 5:
        fh[2:-2] = (h[2:-2] > h[1:-3]) & (h[2:-2] > h[:-4]) & (h[2:-2] >= h[3:-1]) & (h[2:-2] >= h[4:])
        fl[2:-2] = (l[2:-2] < l[1:-3]) & (l[2:-2] < l[:-4]) & (l[2:-2] <= l[3:-1]) & (l[2:-2] <= l[4:])
    return fh, fl


def last_true(mask, upto, lo=0):
    idx = np.flatnonzero(mask[lo:upto + 1])
    return lo + idx[-1] if len(idx) else None


# ---------------------------------------------------------------- Silver Bullet

def untaken_swings(d, i):
    """Неснятые к бару i фрактальные хаи/лои M5 этого торгового дня (buy-/sell-side ликвидность)."""
    if "_m5" not in d:
        b = m5(d); fh, fl = fractals(b["h"], b["l"]); d["_m5"] = (b, fh, fl)
    b, fh, fl = d["_m5"]
    kmax = int(np.searchsorted(b["en"], i - 1, side="right")) - 3      # фрактал подтверждён до бара i
    hs, ls = [], []
    for k in range(2, max(kmax, 2)):
        if fh[k] and d["h"][b["en"][k] + 1:i].max(initial=-np.inf) < b["h"][k]:
            hs.append(b["h"][k])
        if fl[k] and d["l"][b["en"][k] + 1:i].min(initial=np.inf) > b["l"][k]:
            ls.append(b["l"][k])
    return hs, ls




SB_WINDOWS = ((3, 4), (10, 11), (14, 15))


def model_sb(d, cost_bp, min_pct, fixed_pct=None):
    o, h, l, c, tm = d["o"], d["h"], d["l"], d["c"], d["tm"]
    out = []
    for a, b in SB_WINDOWS:
        w0, w1 = idx_at(tm, tmin(a)), idx_at(tm, tmin(b)) - 1
        if w1 - w0 < 10:
            continue
        for i in range(max(w0, 2), w1 + 1):
            side = 0
            if h[i - 2] < l[i]:
                side, entry, stop = 1, l[i], l[i - 2]
            elif l[i - 2] > h[i]:
                side, entry, stop = -1, h[i], h[i - 2]
            if side == 0:
                continue
            if fixed_pct is not None:
                target = entry * (1 + side * fixed_pct / 100)
                if side * (target - entry) <= 0:
                    continue
            else:
                hod, lod = h[:i - 1].max(), l[:i - 1].min()
                sw_h, sw_l = untaken_swings(d, i)
                if side > 0:
                    cands = [x for x in (hod, d["pdh"], *sw_h) if x >= entry * (1 + min_pct / 100)]
                    if not cands:
                        continue
                    target = min(cands)
                else:
                    cands = [x for x in (lod, d["pdl"], *sw_l) if x <= entry * (1 - min_pct / 100)]
                    if not cands:
                        continue
                    target = max(cands)
            r = simulate(d, side, entry, stop, target, i + 1, w1, o, h, l, c, tm, cost_bp)
            if r is not None:
                r["window"] = a
                out.append(r)
            break                        # один сетап на окно: первый FVG с DOL
    return out


# ---------------------------------------------------------------- снятие → MSS → FVG (2022, Venom)

def sweep_mss_fvg(d, hi_lvl, lo_lvl, w0, w1, fill_end_t, cost_bp, target_fn, min_rr=1.0):
    o, h, l, c, tm = d["o"], d["h"], d["l"], d["c"], d["tm"]
    fh, fl = fractals(h, l)
    n = len(c)
    j_fill_end = min(idx_at(tm, fill_end_t), n) - 1
    for j in range(w0, min(w1, n - 1) + 1):
        up, dn = h[j] > hi_lvl, l[j] < lo_lvl
        if up == dn:
            continue
        side = -1 if up else 1           # сняли хай → ищем шорт
        ext = j
        ref = last_true(fl if side < 0 else fh, ext - 2)
        mss = None
        for k in range(j + 1, min(j_fill_end, n - 1) + 1):
            if side < 0 and h[k] > h[ext]:
                ext = k; ref = last_true(fl, ext - 2); continue
            if side > 0 and l[k] < l[ext]:
                ext = k; ref = last_true(fh, ext - 2); continue
            if ref is None:
                continue
            if (side < 0 and c[k] < l[ref]) or (side > 0 and c[k] > h[ref]):
                mss = k; break
        if mss is None:
            return None
        fvg = None
        for i in range(min(mss + 2, n - 1), ext + 1, -1):
            if side < 0 and l[i - 2] > h[i] and i - 2 >= ext:
                fvg = i; break
            if side > 0 and h[i - 2] < l[i] and i - 2 >= ext:
                fvg = i; break
        if fvg is None:
            return None
        entry = h[fvg] if side < 0 else l[fvg]
        stop = h[ext] if side < 0 else l[ext]
        target = target_fn(side, entry)
        if target is None or side * (target - entry) < min_rr * abs(entry - stop):
            return None
        # решение принимается по закрытию бара min(mss+2, n-1): лимит живёт только после него
        decision = min(mss + 2, n - 1)
        return simulate(d, side, entry, stop, target, max(fvg, decision) + 1, j_fill_end, o, h, l, c, tm, cost_bp)
    return None


def model_2022(d, cost_bp, london=False):
    tm, h, l = d["tm"], d["h"], d["l"]
    if london:
        a0, a1, w0, w1 = tmin(19), tmin(0), tmin(2), tmin(5)       # Asian range → London killzone
    else:
        a0, a1, w0, w1 = tmin(18), tmin(8, 30), tmin(8, 30), tmin(11)
    i0, i1 = idx_at(tm, a0), idx_at(tm, a1)
    if i1 - i0 < 30:
        return None
    on_h, on_l = h[i0:i1].max(), l[i0:i1].min()
    hi_lvl, lo_lvl = on_h, on_l      # ближайшая неснятая ликвидность: PDH за overnight-хаем снимается позже

    def tgt(side, entry):
        lv = [x for x in ((on_l, d["pdl"]) if side < 0 else (on_h, d["pdh"])) if side * (x - entry) > 0]
        return (max(lv) if side < 0 else min(lv)) if lv else None
    return sweep_mss_fvg(d, hi_lvl, lo_lvl, idx_at(tm, w0), idx_at(tm, w1) - 1, tmin(12) if not london else tmin(8),
                         d["cost_bp"], tgt)


def model_venom(d, cost_bp):
    tm, h, l = d["tm"], d["h"], d["l"]
    b0, b1 = idx_at(tm, tmin(8)), idx_at(tm, tmin(9, 30))
    if b1 - b0 < 60:
        return None
    bh, bl = h[b0:b1].max(), l[b0:b1].min()

    def tgt(side, entry):
        return bl if side < 0 else bh
    return sweep_mss_fvg(d, bh, bl, b1, idx_at(tm, tmin(11)) - 1, tmin(12), cost_bp, tgt)


# ---------------------------------------------------------------- M5-хелпер

def m5(d):
    g = d["tm"] // 5
    st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    en = np.r_[st[1:], len(g)] - 1
    return {"st": st, "en": en, "tm": d["tm"][st] - d["tm"][st] % 5,
            "o": d["o"][st], "c": d["c"][en],
            "h": np.maximum.reduceat(d["h"], st), "l": np.minimum.reduceat(d["l"], st)}


def model_son(d, cost_bp, stop_pct=0.06, rr=1.5):
    o, h, l, c, tm = d["o"], d["h"], d["l"], d["c"], d["tm"]
    b = m5(d)
    fh, fl = fractals(b["h"], b["l"])
    k0, k1 = np.searchsorted(b["tm"], tmin(9, 30)), np.searchsorted(b["tm"], tmin(12))
    for k in range(max(k0, 5), min(k1, len(b["c"]))):
        side = 0
        rl, rh = last_true(fl, k - 3), last_true(fh, k - 3)
        if rl is not None and b["l"][k] < b["l"][rl] and b["c"][k] > b["o"][k]:
            side = 1
        elif rh is not None and b["h"][k] > b["h"][rh] and b["c"][k] < b["o"][k]:
            side = -1
        if side == 0:
            continue
        j0 = b["en"][k] + 1
        jl = min(idx_at(tm, b["tm"][k] + 5 + 30), len(c)) - 1
        for i in range(j0 + 2, jl + 1):
            if side > 0 and h[i - 2] < l[i]:
                entry = l[i]
            elif side < 0 and l[i - 2] > h[i]:
                entry = h[i]
            else:
                continue
            stop = entry * (1 - side * stop_pct / 100)
            target = entry + side * rr * abs(entry - stop)
            if (side > 0 and d["pdh"] <= target) or (side < 0 and d["pdl"] >= target):
                break
            return simulate(d, side, entry, stop, target, i + 1, jl, o, h, l, c, tm, cost_bp)
        # сетап без FVG/DOL — ищем следующий M5-сигнал
    return None


def model_ts(d, cost_bp, rr=2.0):
    o, h, l, c, tm = d["o"], d["h"], d["l"], d["c"], d["tm"]
    b = m5(d)
    k0, k1 = np.searchsorted(b["tm"], tmin(2)), np.searchsorted(b["tm"], tmin(11))
    for k in range(k0, min(k1, len(b["c"]))):
        up, dn = b["h"][k] > d["pdh"], b["l"][k] < d["pdl"]
        if up == dn:
            continue
        side = -1 if up else 1
        lvl = d["pdh"] if up else d["pdl"]
        ext = b["h"][k] if up else b["l"][k]
        for m in range(k, min(k + 6, len(b["c"]))):
            ext = max(ext, b["h"][m]) if up else min(ext, b["l"][m])
            if (up and b["c"][m] < lvl) or (dn and b["c"][m] > lvl):
                entry = b["c"][m]
                target = entry + side * rr * abs(entry - ext)
                return simulate(d, side, entry, ext, target, b["en"][m] + 1, b["en"][m] + 1,
                                o, h, l, c, tm, cost_bp, market=True)
        return None
    return None


def run_market(path, cost_bp, fx=False):
    days = load_days(path)
    min_pct = 0.15 if fx else 0.2
    five = 0.075 if fx else 0.1
    res = {k: [] for k in ("sb", "sb5", "m2022", "venom", "son", "ts")}
    for d in days:
        d["cost_bp"] = cost_bp
        res["sb"] += model_sb(d, cost_bp, min_pct)
        res["sb5"] += model_sb(d, cost_bp, min_pct, fixed_pct=five)
        for name, fn in (("m2022", lambda x: model_2022(x, cost_bp, london=fx)), ("venom", lambda x: model_venom(x, cost_bp)),
                         ("son", lambda x: model_son(x, cost_bp)), ("ts", lambda x: model_ts(x, cost_bp))):
            r = fn(d)
            if r is not None:
                res[name].append(r)
    return {k: pd.DataFrame(v) for k, v in res.items()}, len(days)
