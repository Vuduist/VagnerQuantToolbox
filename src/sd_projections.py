"""T10: SD-проекции дневного диапазона (CBDR 14–20 ET, Asian range 20–00, float 15–00,
floor pivots) → хай/лоу дня. Бары в ET (H1 или M5).

День = 00:00–16:00 ET (true day). Направление дня — знак close(16:00) − open(00:00) (hindsight,
как в утверждениях ICT «в бычий день»). Для up-дня меряется глубина лоу дня под нижней границей
опорного диапазона в единицах его ширины: d = (L_ref − low_day)/w; для down-дня зеркально.
Метрики: P(d ≤ 2) («1–2 SD»), P(d ≤ 3), P(d > 3) («редкость»), P(лоу внутри/над диапазоном),
P(d «на целом SD»: |d − round(d)| ≤ 0.1), медиана d; то же для терцилей ширины опорного
диапазона относительно ADR20 («narrow» = нижняя треть, «wide» = верхняя). Пивоты: число уровней лестницы (PP, M, R/S по OHLC прошлого дня), пройденных
диапазоном дня от open в сторону дня; попадание экстремума в ±5% шага от уровня.

Нуллы: (1) окно той же длительности со случайным стартом внутри предыдущих 24 часов (B раз);
(2) для пивотов — та же лестница со случайным сдвигом в пределах одного шага.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REFS = {"cbdr": (14, 20), "asian": (20, 24), "float": (15, 24)}   # часы ET предыдущего дня


def day_frames(df_et: pd.DataFrame, bar_minutes: int):
    """Матрицы день × слот для полных суток (00:00–24:00 ET) + индекс дней."""
    d = df_et.copy()
    idx = d.index.tz_localize(None)
    d["day"] = idx.normalize()
    d["slot"] = (idx.hour * 60 + idx.minute) // bar_minutes
    n_slots = 24 * 60 // bar_minutes
    d = d[idx.dayofweek < 5]
    d = d[~d.duplicated(["day", "slot"])]
    cnt = d.groupby("day").slot.nunique()
    full = cnt[cnt >= n_slots * 0.9].index
    d = d[d.day.isin(full)]
    piv = {k: d.pivot(index="day", columns="slot", values=k).reindex(columns=range(n_slots)).to_numpy()
           for k in ("open", "high", "low", "close")}
    days = piv["open"].shape[0]
    return piv, np.asarray(full), n_slots


def _rng(piv, s0, s1):
    """Диапазон слотов [s0, s1) для каждого дня (nan-safe)."""
    hi = np.nanmax(piv["high"][:, s0:s1], axis=1)
    lo = np.nanmin(piv["low"][:, s0:s1], axis=1)
    return hi, lo


def sd_stats(df_et: pd.DataFrame, bar_minutes=60, B=100, seed=0) -> tuple[dict, dict]:
    piv, days, n_slots = day_frames(df_et, bar_minutes)
    spb = 60 // bar_minutes                       # слотов в часе
    n = len(days)
    # день t использует предыдущий полный день t-1 (соседний по индексу; требуем разницу ≤ 4 дней)
    gap_ok = np.r_[False, (np.diff(days) <= np.timedelta64(4, "D"))]
    o00 = piv["open"][:, 0]
    c16 = piv["close"][:, 16 * spb - 1]
    hi_day, lo_day = _rng(piv, 0, 16 * spb)
    hi_lon, lo_lon = _rng(piv, 2 * spb, 5 * spb)
    up = c16 > o00
    # ADR20 прошлых дней (полные сутки)
    hi24, lo24 = _rng(piv, 0, n_slots)
    adr = pd.Series(hi24 - lo24).rolling(20).mean().shift(1).to_numpy()
    rng = np.random.default_rng(seed)

    def depth(ref_hi, ref_lo):
        """d для дня t по опорному диапазону дня t-1 (массивы длины n, первый элемент nan)."""
        w = ref_hi - ref_lo
        rh, rl = np.r_[np.nan, ref_hi[:-1]], np.r_[np.nan, ref_lo[:-1]]
        w = np.r_[np.nan, w[:-1]]
        d = np.where(up, (rl - lo_day) / w, (hi_day - rh) / w)
        d_lon = np.where(up, (rl - lo_lon) / w, (hi_lon - rh) / w)
        return d, d_lon, w

    def metrics(d, d_lon, w, prefix):
        ok = gap_ok & np.isfinite(d) & (w > 0) & np.isfinite(adr)
        out = {f"{prefix}_n": float(ok.sum())}
        rel = np.where(ok, w / adr, np.nan)
        q1, q2 = np.nanquantile(rel, [1 / 3, 2 / 3]) if ok.sum() > 50 else (np.nan, np.nan)
        for sub, m in (("all", ok), ("narrow", ok & (rel <= q1)), ("wide", ok & (rel >= q2))):
            if m.sum() < 50:
                continue
            x = d[m]
            out[f"{prefix}_{sub}_n"] = float(m.sum())
            out[f"{prefix}_{sub}_median_d"] = float(np.median(x))
            out[f"{prefix}_{sub}_p_inside"] = float((x < 0).mean())
            out[f"{prefix}_{sub}_p_0_2"] = float(((x >= 0) & (x <= 2)).mean())
            out[f"{prefix}_{sub}_p_0_3"] = float(((x >= 0) & (x <= 3)).mean())
            out[f"{prefix}_{sub}_p_gt3"] = float((x > 3).mean())
            xi = x[x >= 0.5]
            out[f"{prefix}_{sub}_p_on_integer"] = float((np.abs(xi - np.round(xi)) <= 0.1).mean()) if len(xi) > 30 else np.nan
            xl = d_lon[m]
            out[f"{prefix}_{sub}_p_london_0_2"] = float(((xl >= 0) & (xl <= 2)).mean())
        return out

    real, boots = {}, []
    for name, (h0, h1) in REFS.items():
        rh, rl = _rng(piv, h0 * spb, h1 * spb)
        d, dl, w = depth(rh, rl)
        real.update(metrics(d, dl, w, name))
    # --- пивоты по OHLC предыдущих суток
    def pivot_metrics(offset_frac, prefix="pivot"):
        pp = (hi24 + lo24 + piv["close"][:, n_slots - 1]) / 3
        r1 = 2 * pp - lo24; s1 = 2 * pp - hi24
        r2 = pp + (hi24 - lo24); s2 = pp - (hi24 - lo24)
        r3 = r1 + (hi24 - lo24); s3 = s1 - (hi24 - lo24)
        lv = np.stack([s3, (s3 + s2) / 2, s2, (s2 + s1) / 2, s1, (s1 + pp) / 2, pp,
                       (pp + r1) / 2, r1, (r1 + r2) / 2, r2, (r2 + r3) / 2, r3], axis=1)
        lv = np.r_[np.full((1, 13), np.nan), lv[:-1]]              # уровни дня t из суток t-1
        step = np.nanmedian(np.diff(lv, axis=1), axis=1)
        lv = lv + (offset_frac * step)[:, None]
        ok = gap_ok & np.isfinite(lv).all(axis=1) & (step > 0)
        cnt = np.where(up, ((lv > o00[:, None]) & (lv <= hi_day[:, None])).sum(1),
                       ((lv < o00[:, None]) & (lv >= lo_day[:, None])).sum(1))
        ext = np.where(up, hi_day, lo_day)
        dist = np.nanmin(np.abs(lv - ext[:, None]), axis=1) / step
        out = {f"{prefix}_n": float(ok.sum()), f"{prefix}_mean_count": float(cnt[ok].mean()),
               f"{prefix}_p_count_eq4": float((cnt[ok] == 4).mean()),
               f"{prefix}_p_count_3_5": float(((cnt[ok] >= 3) & (cnt[ok] <= 5)).mean()),
               f"{prefix}_p_ext_within_5pct": float((dist[ok] <= 0.05).mean()),
               f"{prefix}_p_ext_within_10pct": float((dist[ok] <= 0.10).mean())}
        return out
    real.update(pivot_metrics(np.zeros(n)))
    # --- нуллы
    real_w = {}
    for name, (h0, h1) in REFS.items():
        rh, rl = _rng(piv, h0 * spb, h1 * spb)
        real_w[name] = rh - rl
    for b in range(B):
        bo = {}
        for name, (h0, h1) in REFS.items():
            dur = (h1 - h0) * spb
            starts = rng.integers(0, n_slots - dur + 1, size=n)
            rh = np.array([np.nanmax(piv["high"][i, s:s + dur]) for i, s in enumerate(starts)])
            rl = np.array([np.nanmin(piv["low"][i, s:s + dur]) for i, s in enumerate(starts)])
            d, dl, w = depth(rh, rl)
            bo.update(metrics(d, dl, w, name))
            # нулл 2: та же ширина, что у реального окна, но положение случайного окна
            mid = (rh + rl) / 2
            d2, dl2, w2 = depth(mid + real_w[name] / 2, mid - real_w[name] / 2)
            bo.update(metrics(d2, dl2, w2, name + "POS"))
        bo.update(pivot_metrics(rng.random(n)))
        boots.append(bo)
    for name in list(REFS):
        for k in list(real):
            if k.startswith(name + "_"):
                real[name + "POS" + k[len(name):]] = real[k]
    z = {}
    for k, v in real.items():
        if k.endswith("_n"):
            continue
        vals = np.array([bb.get(k, np.nan) for bb in boots], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z
