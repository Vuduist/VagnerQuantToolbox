"""Конструкции ВЫХОДА на случайном входе.

Все выходы получают идентичные (idx, dir) — парное сравнение чистого
вклада выхода. PnL нетто в бп от цены входа (close[t]).
Каждая функция возвращает (net_bp: np.ndarray, idx_used: np.ndarray).
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


def _fwd(h, l, c, idx, K):
    """Окна вперёд на K баров; возвращает (ok, idx2, hf, lf, cf)."""
    n = len(c)
    ok = idx + K < n
    idx2 = idx[ok]
    steps = np.arange(1, K + 1)
    pos = idx2[:, None] + steps[None, :]
    return ok, idx2, h[pos], l[pos], c[pos]


def _first(mask, K):
    any_ = mask.any(axis=1)
    return np.where(any_, mask.argmax(axis=1), K + 1), any_


def _bp(d, exit_px, entry_px, cost_bp):
    return d * (exit_px - entry_px) / entry_px * 1e4 - cost_bp


def exit_time(h, l, c, idx, d, K, cost_bp, **kw):
    """Выход по времени через K баров."""
    ok, idx2, _, _, cf = _fwd(h, l, c, idx, K)
    return _bp(d[ok], cf[:, -1], c[idx2], cost_bp), idx2


def exit_atr_rr(h, l, c, idx, d, K, cost_bp, atr_arr, atr_k=1.5, rr=2.0, **_):
    """Стоп atr_k*ATR, цель rr*риск; первое касание; таймаут K."""
    ok, idx2, hf, lf, cf = _fwd(h, l, c, idx, K)
    d2 = d[ok]
    e = c[idx2]
    a = atr_arr[idx2]
    stop = e - d2 * atr_k * a
    tgt = e + d2 * rr * atr_k * a
    hit_t = np.where(d2[:, None] > 0, hf >= tgt[:, None], lf <= tgt[:, None])
    hit_s = np.where(d2[:, None] > 0, lf <= stop[:, None], hf >= stop[:, None])
    t_t, has_t = _first(hit_t, K)
    t_s, has_s = _first(hit_s, K)
    win = has_t & (t_t < t_s)
    loss = has_s & (t_s <= t_t)
    exit_px = np.where(win, tgt, np.where(loss, stop, cf[:, -1]))
    return _bp(d2, exit_px, e, cost_bp), idx2


def exit_chandelier(h, l, c, idx, d, K, cost_bp, atr_arr, atr_k=3.0, **_):
    """Люстра: трейлинг-стоп на atr_k*ATR от бегущего экстремума."""
    ok, idx2, hf, lf, cf = _fwd(h, l, c, idx, K)
    d2 = d[ok]
    e = c[idx2]
    a = atr_arr[idx2]
    run_hi = np.maximum.accumulate(hf, axis=1)
    run_lo = np.minimum.accumulate(lf, axis=1)
    stop_line = np.where(d2[:, None] > 0,
                         run_hi - atr_k * a[:, None],
                         run_lo + atr_k * a[:, None])
    stopped = np.where(d2[:, None] > 0, lf <= stop_line, hf >= stop_line)
    t_s, has_s = _first(stopped, K)
    rows = np.arange(len(idx2))
    exit_px = np.where(has_s,
                       stop_line[rows, np.clip(t_s, 0, K - 1)],
                       cf[:, -1])
    return _bp(d2, exit_px, e, cost_bp), idx2


def exit_swing(h, l, c, idx, d, K, cost_bp, atr_arr, sw_conf, sw_dir,
               atr_k=2.0, **_):
    """Выход при подтверждении противоположного структурного свинга;
    защитный стоп atr_k*ATR; таймаут K баров."""
    ok, idx2, hf, lf, cf = _fwd(h, l, c, idx, K)
    d2 = d[ok]
    e = c[idx2]
    a = atr_arr[idx2]
    stop = e - d2 * atr_k * a
    hit_s = np.where(d2[:, None] > 0, lf <= stop[:, None], hf >= stop[:, None])
    t_s, has_s = _first(hit_s, K)

    n = len(c)
    pos_in_sw = np.searchsorted(sw_conf, idx2, side="right")
    t_struct = np.full(len(idx2), K + 1, dtype=int)
    for j in range(len(idx2)):
        t0 = idx2[j]
        s = pos_in_sw[j]
        while s < len(sw_conf):
            if sw_dir[s] == -d2[j]:
                rel = int(sw_conf[s] - t0)
                if 0 < rel <= K:
                    t_struct[j] = rel
                break
            s += 1

    struct = t_struct < np.where(has_s, t_s, K + 1)
    exit_px = np.where(struct,
                       c[np.minimum(idx2 + t_struct, n - 1)],
                       np.where(has_s, stop, cf[:, -1]))
    return _bp(d2, exit_px, e, cost_bp), idx2
