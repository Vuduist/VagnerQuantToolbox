"""Батарея популярных субъективных моделей с чистыми каузальными определениями.

Каждая функция возвращает (idx, dir): индексы баров сигнала и направление.
Никакого заглядывания вперёд — только данные <= t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sig_donchian(h, l, c, n=20):
    """Трендовый пробой канала Дончиана (Черепахи): закрытие выше
    максимума хаёв за n баров -> лонг; ниже минимума лоёв -> шорт."""
    hh = pd.Series(h).rolling(n).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(n).min().shift(1).to_numpy()
    up = np.where(c > hh)[0]
    dn = np.where(c < ll)[0]
    idx = np.sort(np.concatenate([up, dn]))
    d = np.where(np.isin(idx, up), 1.0, -1.0)
    return idx, d


def sig_rsi(c, n=2, lo=10.0, hi=90.0):
    """Коннорс RSI(2) mean-reversion: RSI<lo -> лонг, RSI>hi -> шорт."""
    diff = np.diff(c, prepend=np.nan)
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    au = pd.Series(up).rolling(n).mean().to_numpy()
    ad = pd.Series(dn).rolling(n).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + au / ad)
    long_i = np.where(rsi < lo)[0]
    short_i = np.where(rsi > hi)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def sig_ma_pullback(o, h, l, c, fast=20, slow=50):
    """Тренд + откат к быстрой МА: цена выше SMA(slow), бар коснулся
    SMA(fast) и закрылся выше неё -> лонг. Зеркально для шорта."""
    sf = pd.Series(c).rolling(fast).mean().to_numpy()
    ss = pd.Series(c).rolling(slow).mean().to_numpy()
    long_ok = (c > ss) & (l <= sf) & (c > sf) & (c > o)
    short_ok = (c < ss) & (h >= sf) & (c < sf) & (c < o)
    long_i, short_i = np.where(long_ok)[0], np.where(short_ok)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def sig_inside_breakout(h, l, c):
    n = len(c)
    inside = np.zeros(n, dtype=bool)
    inside[2:] = (h[1:-1] < h[:-2]) & (l[1:-1] > l[:-2])
    long_ok = np.zeros(n, dtype=bool)
    short_ok = np.zeros(n, dtype=bool)
    long_ok[2:] = inside[1:-1] & (h[2:] > h[1:-1]) & (c[2:] > h[1:-1])
    short_ok[2:] = inside[1:-1] & (l[2:] < l[1:-1]) & (c[2:] < l[1:-1])
    long_i, short_i = np.where(long_ok)[0], np.where(short_ok)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def sig_engulfing(o, h, l, c, k=1.5):
    n = len(c)
    body = np.abs(c - o)
    med = pd.Series(body).rolling(100).median().shift(1).to_numpy()
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull[1:] = ((c[1:] > o[1:]) & (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])
                & (body[1:] >= k * med[1:]))
    bear[1:] = ((c[1:] < o[1:]) & (c[1:] <= o[:-1]) & (o[1:] >= c[:-1])
                & (body[1:] >= k * med[1:]))
    long_i, short_i = np.where(bull)[0], np.where(bear)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def sig_pdhl_sweep(dt, h, l, c):
    """Judas swing: бар проколол high/low ПРОШЛОГО ДНЯ и закрылся обратно
    по другую сторону уровня -> сигнал против прокола."""
    day = dt.normalize()
    ph = pd.Series(h).groupby(day).max()
    pl = pd.Series(l).groupby(day).min()
    prev_h = ph.shift(1).reindex(day).to_numpy()
    prev_l = pl.shift(1).reindex(day).to_numpy()
    short_ok = (h > prev_h) & (c < prev_h)
    long_ok = (l < prev_l) & (c > prev_l)
    long_i, short_i = np.where(long_ok)[0], np.where(short_ok)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def sig_gap_fade(dt, o, h, l, c, min_gap_atr=0.3):
    """Индексная классика: гэп открытия дня против предыдущего закрытия
    -> фейд в сторону закрытия гэпа."""
    day = dt.normalize()
    first_bar = np.concatenate([[True], np.asarray(day[1:] != day[:-1])])
    pc = pd.Series(c).groupby(day).last()
    prev_close = pc.shift(1).reindex(day).to_numpy()
    a = atr100(h, l, c)
    gap = (o - prev_close) / np.where(a > 0, a, np.nan)
    long_ok = first_bar & (gap < -min_gap_atr)   # гэп вниз -> лонг
    short_ok = first_bar & (gap > min_gap_atr)
    long_i, short_i = np.where(long_ok)[0], np.where(short_ok)[0]
    idx = np.sort(np.concatenate([long_i, short_i]))
    d = np.where(np.isin(idx, long_i), 1.0, -1.0)
    return idx, d


def atr100(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()
