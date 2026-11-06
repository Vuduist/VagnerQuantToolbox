"""Прямой исторический бэктест: событие -> сделка -> нетто-PnL с издержками.

Без нулевых моделей. Вопрос: что РЕАЛЬНО зарабатывало на истории,
и держится ли это по годам/ТФ/рынкам.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine


# ---------------------------------------------------------------- симуляция

def sim_rr(h, l, c, idx, d, entry, stop, target, K, cost_bp, limit=False):
    """Вход по entry, первое касание target/stop в течение K баров.
    limit=True: сделка существует только если цена коснулась entry,
    исход считается после касания (консервативно: одновременное
    касание entry и stop на одном баре = стоп)."""
    n = len(c)
    ok = (idx >= 0) & (idx + K < n)
    idx = idx[ok]
    if len(idx) == 0:
        return None
    d, entry, stop, target = d[ok], entry[ok], stop[ok], target[ok]
    steps = np.arange(1, K + 1)
    pos = idx[:, None] + steps[None, :]
    lows_f, highs_f = l[pos], h[pos]

    risk = np.abs(entry - stop)
    valid = (risk > 0) & np.isfinite(risk)
    if valid.sum() == 0:
        return None
    idx, d, entry, stop, target, risk = (idx[valid], d[valid], entry[valid],
                                         stop[valid], target[valid],
                                         risk[valid])
    lows_f, highs_f = lows_f[valid], highs_f[valid]

    hit_tgt = np.where(d[:, None] > 0, highs_f >= target[:, None],
                       lows_f <= target[:, None])
    hit_stp = np.where(d[:, None] > 0, lows_f <= stop[:, None],
                       highs_f >= stop[:, None])

    def first(mask):
        any_ = mask.any(axis=1)
        return np.where(any_, mask.argmax(axis=1), K + 1), any_

    if limit:
        touched = np.where(d[:, None] > 0, lows_f <= entry[:, None],
                           highs_f >= entry[:, None])
        t_fill, has_fill = first(touched)
        bar_ids = np.arange(1, K + 1)[None, :]
        after = bar_ids > t_fill[:, None]
        # стоп допускаем и в баре входа (консервативно), цель — после
        t_tgt, has_tgt = first(hit_tgt & after)
        t_stp, has_stp = first(hit_stp & (bar_ids >= t_fill[:, None]))
        active = has_fill
        win = active & has_tgt & (t_tgt < t_stp)
        loss = active & has_stp & (t_stp <= t_tgt)
    else:
        active = np.ones(len(idx), dtype=bool)
        t_tgt, has_tgt = first(hit_tgt)
        t_stp, has_stp = first(hit_stp)
        win = has_tgt & (t_tgt < t_stp)
        loss = has_stp & (t_stp <= t_tgt)
    # ни то, ни другое за K баров: выход по close[t+K]
    timeout = active & ~(win | loss)
    exit_px = np.where(win, target,
                       np.where(loss, stop, c[np.minimum(idx + K, n - 1)]))
    gross_bp = d * (exit_px - entry) / entry * 1e4
    net_bp = gross_bp - cost_bp
    rr = np.abs(target - entry) / risk
    pnl_R = np.where(win, rr, np.where(loss, -1.0,
                     d * (c[np.minimum(idx + K, n - 1)] - entry) / risk))
    pnl_R -= cost_bp * 1e-4 * entry / risk
    t_exit = np.where(win, t_tgt, np.where(loss, t_stp, K))
    return {"n": int(active.sum()), "net_bp": net_bp[active],
            "pnl_R": pnl_R[active], "win": win[active],
            "loss": loss[active], "timeout": timeout[active],
            "idx": idx[active], "t_exit": t_exit[active]}


def sim_time(c, idx, d, h_, cost_bp):
    """Вход по close, выход через h_ баров."""
    n = len(c)
    ok = idx + h_ < n
    idx = idx[ok]
    if len(idx) == 0:
        return None
    d = d[ok]
    gross_bp = d * (c[idx + h_] - c[idx]) / c[idx] * 1e4
    net_bp = gross_bp - cost_bp
    return {"n": len(idx), "net_bp": net_bp, "idx": idx}


# ---------------------------------------------------------------- стратегии

def gen_strategies(o, h, l, c):
    """Список (имя, idx, dir, entry, stop, target, K) или time-exit."""
    S = []
    ev, d = displacement_events = engine.displacement_events(o, h, l, c)
    idx = np.where(ev)[0]
    rng = h - l

    # 1. следование за импульсом, стоп за его экстремум, RR 1.5
    e = c[idx]; s = np.where(d[idx] > 0, l[idx], h[idx])
    t = e + d[idx] * 1.5 * np.abs(e - s)
    S.append(("disp_follow_RR1.5", "rr", idx, d[idx], e, s, t, 16))
    # 2. фейд импульса
    S.append(("disp_fade_RR1.5", "rr", idx, -d[idx], e,
              np.where(d[idx] > 0, h[idx], l[idx]),
              e - d[idx] * 1.5 * np.abs(e - s), 16))

    # 3. sweep_reject: вход по закрытию, стоп за фитиль, RR 1.5
    ru, rd = engine.sweep_reject_events(o, h, l, c)
    idx = np.where(ru | rd)[0]
    d_ = np.where(rd, 1.0, -1.0)[idx]
    e = c[idx]; s = np.where(d_ > 0, l[idx], h[idx])
    t = e + d_ * 1.5 * np.abs(e - s)
    S.append(("sweep_reject_RR1.5", "rr", idx, d_, e, s, t, 16))

    # 4. sweep_hold: следование пробою, стоп — другой край бара
    hold_up, fail_up, hold_dn, fail_dn = engine.sweep_events(h, l, c)
    idx = np.where(hold_up | hold_dn)[0]
    d_ = np.where(hold_up, 1.0, -1.0)[idx]
    e = c[idx]; s = np.where(d_ > 0, l[idx], h[idx])
    t = e + d_ * 1.5 * np.abs(e - s)
    S.append(("sweep_hold_RR1.5", "rr", idx, d_, e, s, t, 16))

    # 5. momentum/fade по времени: знак доходности за k баров, холд h_
    ret_k = np.sign(np.concatenate([np.full(16, np.nan),
                                    c[16:] / c[:-16] - 1]))
    allbars = np.where(np.isfinite(ret_k))[0]
    for hh in (4, 16):
        S.append((f"mom16_h{hh}", "time", allbars, ret_k[allbars], hh))
        S.append((f"fade16_h{hh}", "time", allbars, -ret_k[allbars], hh))

    # 6. вход на 50% импульса (FVG-скелет), цель — экстремум импульса
    idx = np.where(ev)[0]
    d_ = d[idx]
    e = np.where(d_ > 0, c[idx] - 0.5 * rng[idx], c[idx] + 0.5 * rng[idx])
    s = np.where(d_ > 0, l[idx], h[idx])
    t = np.where(d_ > 0, h[idx], l[idx])
    S.append(("fvg50_extreme", "rr_limit", idx, d_, e, s, t, 24))
    return S


def run_market(df: pd.DataFrame, cost_bp: float, name: str):
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    years = df.index.year.to_numpy()
    rows = []
    for spec in gen_strategies(o, h, l, c):
        nm, kind = spec[0], spec[1]
        if kind == "time":
            _, _, idx, d_, hh = spec
            r = sim_time(c, idx, d_, hh, cost_bp)
        else:
            _, _, idx, d_, e, s, t, K = spec
            r = sim_rr(h, l, c, idx, d_, e, s, t, K, cost_bp,
                       limit=(kind == "rr_limit"))
        if r is None or r["n"] < 30:
            continue
        row = {"market": name, "strategy": nm, "n": r["n"],
               "exp_bp": float(np.mean(r["net_bp"]))}
        if "pnl_R" in r:
            row["exp_R"] = float(np.mean(r["pnl_R"]))
            row["win_rate"] = float(np.mean(r["win"]))
        for y in np.unique(years):
            m = years[r["idx"]] == y
            if m.sum() >= 10:
                row[f"y{y}"] = float(np.mean(r["net_bp"][m]))
        rows.append(row)
    return rows
