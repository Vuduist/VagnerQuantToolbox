"""Каузальная разметка структурных экстремумов (машина состояний).

Экстремум подтверждается только после разворота цены на k*ATR от него.
Никакого заглядывания вперёд: в момент t известны лишь свинги,
подтверждённые до t. Демонстрация величины lookahead-иллюзии.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def causal_swings(h, l, c, k=3.0):
    """Машина состояний: вверх/вниз, разворот = откат на k*ATR от экстремума.

    Возвращает массивы: idx экстремума, цена, знак (+1 хай / -1 лоу),
    idx подтверждения. Все решения на баре t используют данные <= t.
    """
    n = len(c)
    a = atr(h, l, c)
    sw_idx, sw_px, sw_dir, sw_conf = [], [], [], []
    trend = 0                      # 0 старт, +1 вверх, -1 вниз
    hi, hi_i = h[0], 0
    lo, lo_i = l[0], 0
    for t in range(1, n):
        if trend >= 0:
            if h[t] > hi:
                hi, hi_i = h[t], t
            thr = k * a[t] if np.isfinite(a[t]) else np.inf
            if hi - c[t] >= thr:           # разворот вниз: хай подтверждён
                sw_idx.append(hi_i); sw_px.append(hi)
                sw_dir.append(1); sw_conf.append(t)
                trend = -1
                lo, lo_i = l[t], t
        if trend <= 0:
            if l[t] < lo:
                lo, lo_i = l[t], t
            thr = k * a[t] if np.isfinite(a[t]) else np.inf
            if c[t] - lo >= thr:           # разворот вверх: лоу подтверждён
                sw_idx.append(lo_i); sw_px.append(lo)
                sw_dir.append(-1); sw_conf.append(t)
                trend = 1
                hi, hi_i = h[t], t
    return (np.array(sw_idx), np.array(sw_px),
            np.array(sw_dir), np.array(sw_conf))


def sweep_of_structure(h, l, c, k=3.0, horizon=48):
    """Что происходит, когда цена возвращается к ПОДТВЕРЖДЁННОМУ
    структурному экстремуму и снимает его.

    Для каждого бара t известны только свинги с conf < t.
    Событие: high[t] > последний подтверждённый свинг-хай (и наоборот).
    Замер: знаковая доходность за horizon баров в сторону снятия
    и против него."""
    sw_idx, sw_px, sw_dir, sw_conf = causal_swings(h, l, c, k)
    n = len(c)
    # очередь подтверждённых уровней
    order = np.argsort(sw_conf)
    conf_s, px_s, dir_s = sw_conf[order], sw_px[order], sw_dir[order]

    last_hi = np.full(n, np.nan)   # последний подтверждённый хай на момент t
    last_lo = np.full(n, np.nan)
    j = 0
    cur_hi, cur_lo = np.nan, np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            if dir_s[j] > 0:
                cur_hi = px_s[j]
            else:
                cur_lo = px_s[j]
            j += 1
        last_hi[t], last_lo[t] = cur_hi, cur_lo

    ev_up = (h > last_hi) & np.isfinite(last_hi)
    ev_dn = (l < last_lo) & np.isfinite(last_lo)
    out = {}
    for name, mask, sgn in (("sweep_up", ev_up, 1.0),
                            ("sweep_dn", ev_dn, -1.0)):
        idx = np.where(mask)[0]
        idx = idx[idx + horizon < n]
        if len(idx) < 30:
            continue
        fwd = (c[idx + horizon] - c[idx]) / c[idx]
        out[name] = {
            "n": len(idx),
            "p_same": float(np.mean(fwd * sgn > 0)),   # продолжение за уровень
            "p_against": float(np.mean(fwd * sgn < 0)),  # отскок обратно
            "mean_bp": float(np.mean(fwd * sgn) * 1e4),
        }
    return out


def lookahead_illusion(h, l, c, k=3.0, horizon=48):
    """Демонстрация проблемы измерений: те же свинги, но считаем,
    что экстремум 'был известен' в момент его формирования
    (задним числом), и меряем 'реакцию' от него."""
    sw_idx, sw_px, sw_dir, sw_conf = causal_swings(h, l, c, k)
    ok = sw_idx + horizon < len(c)
    i, px, d, cf = sw_idx[ok], sw_px[ok], sw_dir[ok], sw_conf[ok]
    # 'Реакция' от экстремума (нелегальная: мы знаем его заранее)
    fwd_illegal = (c[i + horizon] - c[i]) / c[i] * (-d)   # против свинга
    # Честная реакция: от момента подтверждения
    ok2 = cf + horizon < len(c)
    fwd_legal = (c[cf[ok2] + horizon] - c[cf[ok2]]) / c[cf[ok2]] * (-d[ok2])
    return {
        "n": int(ok.sum()),
        "illegal_p_revert": float(np.mean(fwd_illegal > 0)),
        "illegal_mean_bp": float(np.mean(fwd_illegal) * 1e4),
        "legal_p_revert": float(np.mean(fwd_legal > 0)),
        "legal_mean_bp": float(np.mean(fwd_legal) * 1e4),
        "avg_conf_lag_bars": float(np.mean(cf - i)),
    }
