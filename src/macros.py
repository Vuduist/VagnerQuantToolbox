"""T02 macros и T09 NY open на 5-минутных слотах (true day 00:00–16:00 ET, 192 слота).

T02: в окнах 9:50–10:10, 10:50–11:10, 13:25–13:35, 15:15–15:45 «run к ликвидности»:
     диапазон, |нетто|, эффективность (|нетто|/путь), P(пробой high/low предыдущего часа)
     — против соседних 20-минуток и против нулла перестановки слотов (профиль волы сохранён).
T09: первый ход 9:30–9:45 против остатка дня; 8:30–9:30 против 9:30–16:00; 8:20 как
     локальный экстремум часа 7:50–8:50.
"""

from __future__ import annotations

import numpy as np

import killzones as kz

SLOT = 5


def _slots(h0, m0, h1, m1):
    return np.arange((h0 * 60 + m0) // SLOT, (h1 * 60 + m1) // SLOT)


WINDOWS = {
    "macro_0950_1010": _slots(9, 50, 10, 10),
    "pre_0930_0950": _slots(9, 30, 9, 50),
    "post_1010_1030": _slots(10, 10, 10, 30),
    "macro_1050_1110": _slots(10, 50, 11, 10),
    "pre_1030_1050": _slots(10, 30, 10, 50),
    "post_1110_1130": _slots(11, 10, 11, 30),
    "macro_1325_1345": _slots(13, 25, 13, 45),
    "pre_1305_1325": _slots(13, 5, 13, 25),
    "macro_1515_1545": _slots(15, 15, 15, 45),
    "pre_1445_1515": _slots(14, 45, 15, 15),
    "post_1545_1600": _slots(15, 45, 16, 0),
}


def rebuild_full(comp, open0):
    gap, oh, ol, oc = comp
    logc = np.log(open0)[:, None] + np.cumsum(gap + oc, axis=1)
    logo = logc - oc
    return np.exp(logo), np.exp(logo + oh), np.exp(logo + ol), np.exp(logc)


def macro_stats(comp, open0) -> dict:
    o, h, l, c = rebuild_full(comp, open0)
    day_path = np.abs(np.diff(c, axis=1)).sum(axis=1) + np.abs(c[:, 0] - o[:, 0])
    day_rng = h.max(axis=1) - l.min(axis=1)
    out = {"n_days": float(len(c))}
    for name, sl in WINDOWS.items():
        ws = sl[0]
        net = c[:, sl[-1]] - o[:, ws]
        rng_w = h[:, sl].max(axis=1) - l[:, sl].min(axis=1)
        path = np.abs(np.diff(c[:, sl], axis=1)).sum(axis=1) + np.abs(c[:, ws] - o[:, ws])
        er = np.where(path > 0, np.abs(net) / np.maximum(path, 1e-12), np.nan)
        prior = slice(max(ws - 12, 0), ws)
        brk = (h[:, sl].max(axis=1) > h[:, prior].max(axis=1)) | (l[:, sl].min(axis=1) < l[:, prior].min(axis=1))
        out[f"{name}_rng_share"] = float(np.mean(rng_w / day_rng))
        out[f"{name}_absnet_share"] = float(np.mean(np.abs(net) / day_rng))
        out[f"{name}_er"] = float(np.nanmean(er))
        out[f"{name}_p_break1h"] = float(brk.mean())
    # T09
    s0930, s0945, s0830, s1600 = _slots(9, 30, 9, 35)[0], _slots(9, 45, 9, 50)[0], _slots(8, 30, 8, 35)[0], c.shape[1] - 1
    first = c[:, s0945 - 1] - o[:, s0930]          # 9:30–9:45
    rest = c[:, s1600] - c[:, s0945 - 1]            # 9:45–16:00
    pre = c[:, s0930 - 1] - c[:, s0830 - 1]         # 8:30–9:30
    day = c[:, s1600] - o[:, s0930]                 # 9:30–16:00
    ok = (first != 0) & (rest != 0)
    out["t09_p_first15_opposite_rest"] = float((np.sign(first[ok]) != np.sign(rest[ok])).mean())
    ok2 = (pre != 0) & (day != 0)
    out["t09_p_pre0830_opposite_day"] = float((np.sign(pre[ok2]) != np.sign(day[ok2])).mean())
    # 8:20 как экстремум часа 7:50–8:50
    hour = _slots(7, 50, 8, 50)
    s820 = _slots(8, 15, 8, 25)
    hi_slot = hour[h[:, hour].argmax(axis=1)]
    lo_slot = hour[l[:, hour].argmin(axis=1)]
    out["t09_p_820_is_hour_extreme"] = float((np.isin(hi_slot, s820) | np.isin(lo_slot, s820)).mean())
    out["t09_p_820_expected"] = float(2 * len(s820) / len(hour) - (len(s820) / len(hour)) ** 2)
    # пробой pre-9:30 short-term high/low (8:30–9:30) в течение 9:30–11:00
    pre_hi = h[:, _slots(8, 30, 9, 30)].max(axis=1)
    pre_lo = l[:, _slots(8, 30, 9, 30)].min(axis=1)
    morn = _slots(9, 30, 11, 0)
    out["t09_p_break_pre930_by_11"] = float(((h[:, morn].max(axis=1) > pre_hi) | (l[:, morn].min(axis=1) < pre_lo)).mean())
    return out


def macro_test(df_et, B=200, seed=0):
    comp, open0, hours, days = kz.day_matrix(df_et, SLOT)
    slots = np.arange(comp.shape[2])
    real = macro_stats(comp, open0)
    rng = np.random.default_rng(seed)
    boots = [macro_stats(kz.shuffled(comp, slots, rng), open0) for _ in range(B)]
    z = {}
    for k, v in real.items():
        if k.startswith("n_"):
            continue
        vals = np.array([b[k] for b in boots])
        sd = vals.std(ddof=1)
        z[k] = {"real": v, "null_mean": vals.mean(), "null_sd": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan}
    return real, z
