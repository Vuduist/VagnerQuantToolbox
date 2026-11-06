"""T05 AM/PM и T04 lunch на 5-минутных слотах (true day 00:00–16:00 ET).

T05: PM (13:00–16:00) продолжает AM (9:30–12:00)? measured move: |PM| ≈ |AM|, диапазон дня
     ≈ 2×AM; PM-reversal после сильного AM.
T04: lunch (12:00–13:00) — консолидация; «lunch снимает стопы AM-позиций (пробивает AM low
     при AM вверх), затем PM продолжает AM».
Нулл: перестановка слотов внутри дня с профилем волы (killzones.shuffled).
"""

from __future__ import annotations

import numpy as np

import killzones as kz
from macros import rebuild_full, _slots

AM = _slots(9, 30, 12, 0)
LUNCH = _slots(12, 0, 13, 0)
PM = _slots(13, 0, 16, 0)


def am_pm_stats(comp, open0) -> dict:
    o, h, l, c = rebuild_full(comp, open0)
    out = {"n_days": float(len(c))}
    am_net = c[:, AM[-1]] - o[:, AM[0]]
    am_rng = h[:, AM].max(1) - l[:, AM].min(1)
    pm_net = c[:, PM[-1]] - o[:, PM[0]]
    pm_rng = h[:, PM].max(1) - l[:, PM].min(1)
    day_rng = h[:, AM[0]:].max(1) - l[:, AM[0]:].min(1)
    ok = (am_net != 0) & (pm_net != 0)
    out["p_pm_same_sign_as_am"] = float((np.sign(am_net[ok]) == np.sign(pm_net[ok])).mean())
    out["corr_am_pm_net"] = float(np.corrcoef(am_net, pm_net)[0, 1])
    strong = np.abs(am_net) / np.maximum(am_rng, 1e-12) > 0.6
    out["n_strong_am"] = float(strong.sum())
    out["p_pm_same_sign_given_strong_am"] = float((np.sign(am_net[strong]) == np.sign(pm_net[strong])).mean())
    out["pm_over_am_absnet_median"] = float(np.median(np.abs(pm_net) / np.maximum(np.abs(am_net), 1e-12)))
    out["pm_over_am_rng_median"] = float(np.median(pm_rng / np.maximum(am_rng, 1e-12)))
    out["day_over_am_rng_median"] = float(np.median(day_rng / np.maximum(am_rng, 1e-12)))
    out["p_day_rng_ge_1.8am"] = float((day_rng >= 1.8 * am_rng).mean())
    # PM по сравнению с продолжением AM-тренда в сторону AM: средняя PM-доходность × sign(AM), в долях AM range
    out["pm_net_in_am_dir_over_amrng"] = float(np.mean(np.sign(am_net) * pm_net / np.maximum(am_rng, 1e-12)))
    # --- T04 lunch
    l_rng = h[:, LUNCH].max(1) - l[:, LUNCH].min(1)
    out["lunch_rng_share"] = float(np.mean(l_rng / np.maximum(day_rng, 1e-12)))
    AM_LAST = _slots(11, 0, 12, 0)
    am_hi, am_lo = h[:, AM_LAST].max(1), l[:, AM_LAST].min(1)   # стопы AM-позиций: экстремумы последнего часа AM
    up = am_net > 0
    sweep_against = np.where(up, l[:, LUNCH].min(1) < am_lo, h[:, LUNCH].max(1) > am_hi)
    sweep_with = np.where(up, h[:, LUNCH].max(1) > am_hi, l[:, LUNCH].min(1) < am_lo)
    out["p_lunch_sweeps_am_against"] = float(sweep_against.mean())
    out["p_lunch_sweeps_am_with"] = float(sweep_with.mean())
    # после lunch-свипа против AM: PM продолжает AM?
    pm_dir = np.sign(am_net) * (c[:, PM[-1]] - c[:, LUNCH[-1]])
    m = sweep_against & ok
    out["n_lunch_sweep_against"] = float(m.sum())
    out["p_pm_cont_am_after_lunch_sweep"] = float((pm_dir[m] > 0).mean()) if m.sum() > 20 else np.nan
    m2 = (~sweep_against) & ok
    out["p_pm_cont_am_no_lunch_sweep"] = float((pm_dir[m2] > 0).mean()) if m2.sum() > 20 else np.nan
    # lunch 11:50–12:10 run: пробой high/low 11:00–11:50
    run_w = _slots(11, 50, 12, 10)
    prior = _slots(11, 0, 11, 50)
    brk = (h[:, run_w].max(1) > h[:, prior].max(1)) | (l[:, run_w].min(1) < l[:, prior].min(1))
    out["p_lunchrun_breaks_prior50"] = float(brk.mean())
    return out


def am_pm_test(df_et, B=200, seed=0):
    comp, open0, hours, days = kz.day_matrix(df_et, 5)
    slots = np.arange(comp.shape[2])
    real = am_pm_stats(comp, open0)
    rng = np.random.default_rng(seed)
    boots = [am_pm_stats(kz.shuffled(comp, slots, rng), open0) for _ in range(B)]
    z = {}
    for k, v in real.items():
        if k.startswith("n_"):
            continue
        vals = np.array([b[k] for b in boots], float)
        vals = vals[np.isfinite(vals)]
        sd = vals.std(ddof=1) if len(vals) > 5 else np.nan
        z[k] = {"real": v, "null_mean": vals.mean() if len(vals) else np.nan, "null_sd": sd,
                "z": (v - vals.mean()) / sd if sd and sd > 0 else np.nan}
    return real, z
