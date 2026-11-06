"""Ядро количественного исследования.

Философия: никаких графических паттернов. Событие = статистическое
свойство баров (импульс, обновление экстремума, серия). Альфа =
отличие условной статистики после события от той же статистики на
синтетических рядах, сохраняющих распределение доходностей,
кластеризацию волатильности и внутрибарную структуру (block bootstrap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_mt5_csv(path: str) -> pd.DataFrame:
    """CSV из MetaTrader 5 (UTF-16 LE/BE).
    Поддерживает оба формата первой колонки:
    'YYYY.MM.DD HH:MM' или unix-time в миллисекундах."""
    raw = open(path, "rb").read()
    for enc in ("utf-16", "utf-16-be", "utf-16-le", "utf-8"):
        try:
            text = raw.decode(enc)
            float(text.splitlines()[0].split(",")[1])
            break
        except Exception:
            continue
    from io import StringIO
    ncol = len(text.splitlines()[0].split(","))
    names = ["t", "open", "high", "low", "close", "volume", "spread"][:ncol]
    df = pd.read_csv(StringIO(text), header=None, names=names,
                     usecols=range(min(ncol, 6)))
    try:
        tnum = pd.to_numeric(df["t"])
        df["dt"] = pd.to_datetime(tnum.astype(np.int64), unit="ms")
    except (ValueError, TypeError):
        df["dt"] = pd.to_datetime(df["t"], format="%Y.%m.%d %H:%M")
    df = df.drop(columns=["t"])
    df = df.drop_duplicates("dt").sort_values("dt").set_index("dt")
    df["volume"] = df["volume"].clip(lower=1.0)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ---------------------------------------------------------------- данные

def load_csv(path: str) -> pd.DataFrame:
    """Загрузка CSV формата: open_time(ms),open,high,low,close[,volume]."""
    df = pd.read_csv(
        path,
        header=None,
        names=["open_time", "open", "high", "low", "close", "volume"],
        usecols=[0, 1, 2, 3, 4, 5],
    )
    # Binance: старые файлы в миллисекундах, новые — в микросекундах.
    # Определяем построчно по порядку величины.
    ts = df["open_time"].to_numpy(dtype=np.int64)
    # единицы по порядку величины: s / ms / us / ns -> миллисекунды
    factor = np.where(ts < 10**11, 10**-3,
             np.where(ts < 10**13, 1.0,
             np.where(ts < 10**16, 10**3, 10**6)))
    df["dt"] = pd.to_datetime((ts / factor).astype(np.int64), unit="ms")
    df = df.drop_duplicates("dt").sort_values("dt")
    df = df.set_index("dt").drop(columns=["open_time"])
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out[out["volume"] > 0]


# ---------------------------------------------------------------- события

def displacement_events(o, h, l, c, vol_window=200, k=2.0, body_min=0.6):
    """Импульс: диапазон бара > k * средний диапазон, тело доминирует.

    Никакого разрыва теней (FVG) не требуется — событие определяется
    чисто статистически.
    """
    rng = h - l
    rng_mean = pd.Series(rng).rolling(vol_window).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = rng / rng_mean
        body_ratio = np.abs(c - o) / rng
    direction = np.sign(c - o)
    ev = (norm >= k) & (body_ratio >= body_min) & (direction != 0)
    ev[~np.isfinite(norm)] = False
    return ev, direction


def sweep_events(h, l, c, lookback=50):
    """Обновление экстремума: бар делает новый N-барный хай/лоу.

    Дальше смотрим только факт удержания (закрытие за уровнем) —
    никаких теней/пин-баров.
    """
    hh = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    new_high = h > hh
    new_low = l < ll
    hold_up = new_high & (c > hh)      # пробил и удержал
    fail_up = new_high & (c <= hh)     # пробил и не удержал
    hold_dn = new_low & (c < ll)
    fail_dn = new_low & (c >= ll)
    return hold_up, fail_up, hold_dn, fail_dn


def run_events(c):
    """Серии однонаправленных закрытий: длина текущей серии."""
    sign = np.sign(np.diff(c, prepend=np.nan))
    streak = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if sign[i] == sign[i - 1] and sign[i] != 0:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 1 if sign[i] != 0 else 0
    return streak, sign


# ------------------------------------------------------- условная статистика

def forward_stats(c, ev_idx, direction, horizons=(1, 2, 4, 8, 16, 32)):
    """Знаковая доходность вперёд после события (в сторону импульса)."""
    out = {}
    n = len(c)
    for h in horizons:
        tgt = ev_idx + h
        ok = tgt < n
        idx, tg = ev_idx[ok], tgt[ok]
        if len(idx) == 0:
            continue
        ret = (c[tg] - c[idx]) / c[idx] * direction[idx]
        out[h] = {
            "n": len(ret),
            "mean": float(np.mean(ret)),
            "p_cont": float(np.mean(ret > 0)),  # вероятность продолжения
        }
    return out


def revisit_stats(h, l, c, ev_idx, direction, K=24):
    """Насколько глубоко цена возвращается в диапазон импульсного бара
    в течение K баров. depth=1.0 — полный возврат через весь бар."""
    n = len(c)
    ok = ev_idx + K < n
    idx = ev_idx[ok]
    if len(idx) == 0:
        return None
    steps = np.arange(1, K + 1)
    pos = idx[:, None] + steps[None, :]
    lows_f, highs_f = l[pos], h[pos]
    rng = (h[idx] - l[idx])
    rng[rng == 0] = np.nan
    depth = np.where(
        direction[idx, None] > 0,
        (c[idx, None] - lows_f) / rng[:, None],   # бычий бар: низ после
        (highs_f - c[idx, None]) / rng[:, None],  # медвежий: хай после
    )
    max_depth = np.nanmax(depth, axis=1)
    return {
        "n": len(idx),
        "median_depth": float(np.nanmedian(max_depth)),
        "p_depth_50": float(np.nanmean(max_depth >= 0.5)),
        "p_depth_100": float(np.nanmean(max_depth >= 1.0)),
    }


def trade_stats(h, l, c, ev_idx, direction, K=24):
    """Скелет сделки без графического подтверждения:
    вход на возврате к 50% диапазона импульсного бара,
    стоп — за его экстремум, цель — обновление экстремума импульса.
    Результат в единицах риска (R)."""
    n = len(c)
    ok = ev_idx + K < n
    idx = ev_idx[ok]
    if len(idx) == 0:
        return None
    steps = np.arange(1, K + 1)
    pos = idx[:, None] + steps[None, :]
    lows_f, highs_f = l[pos], h[pos]
    d = direction[idx]
    rng = h[idx] - l[idx]
    entry = np.where(d > 0, c[idx] - 0.5 * rng, c[idx] + 0.5 * rng)
    target = np.where(d > 0, h[idx], l[idx])
    stop = np.where(d > 0, l[idx], h[idx])

    risk = np.abs(entry - stop)
    reward = np.abs(target - entry)
    valid = risk > 0
    if valid.sum() == 0:
        return None
    lows_f, highs_f = lows_f[valid], highs_f[valid]
    entry, target, stop, risk, reward, d = (
        entry[valid], target[valid], stop[valid],
        risk[valid], reward[valid], d[valid],
    )

    touched = np.where(d[:, None] > 0,
                       lows_f <= entry[:, None],
                       highs_f >= entry[:, None])
    hit_tgt = np.where(d[:, None] > 0,
                       highs_f >= target[:, None],
                       lows_f <= target[:, None])
    hit_stp = np.where(d[:, None] > 0,
                       lows_f <= stop[:, None],
                       highs_f >= stop[:, None])

    def first_time(mask):
        any_hit = mask.any(axis=1)
        t = np.where(any_hit, mask.argmax(axis=1), K + 1)
        return t, any_hit

    t_entry, got_fill = first_time(touched)
    t_tgt, _ = first_time(hit_tgt)
    t_stp, _ = first_time(hit_stp)

    win = got_fill & (t_tgt < t_stp) & (t_entry <= t_tgt)
    loss = got_fill & (t_stp < t_tgt) & (t_entry <= t_stp)
    rr = reward / risk
    pnl_R = np.where(win, rr, np.where(loss, -1.0, 0.0))
    return {
        "n": int(valid.sum()),
        "fill_rate": float(got_fill.mean()),
        "win_rate": float(win[got_fill].mean()) if got_fill.any() else np.nan,
        "avg_rr": float(rr.mean()),
        "expectancy_R": float(pnl_R.sum() / got_fill.sum())
        if got_fill.any() else np.nan,
    }


# -------------------------------------------------- графические паттерны

def fvg_events(o, h, l, c):
    """Истинный Fair Value Gap (3 свечи): разрыв между тенями соседей
    среднего бара. Бычий FVG на баре t: low[t+1] > high[t-1]."""
    bull = np.zeros(len(c), dtype=bool)
    bear = np.zeros(len(c), dtype=bool)
    bull[1:-1] = l[2:] > h[:-2]
    bear[1:-1] = h[2:] < l[:-2]
    return bull, bear


def fvg_fill_stats(h, l, c, ev_idx, direction, K=24):
    """Заполнение FVG в течение K баров: касание края, 50% (CE),
    полное закрытие разрыва."""
    n = len(c)
    ok = (ev_idx >= 1) & (ev_idx + 1 < n) & (ev_idx + K < n)
    idx = ev_idx[ok]
    if len(idx) == 0:
        return None
    steps = np.arange(1, K + 1)
    pos = idx[:, None] + steps[None, :]
    lows_f, highs_f = l[pos], h[pos]
    d = direction[idx]
    # зона разрыва: для бычьего [high[t-1], low[t+1]]
    gap_lo = np.where(d > 0, h[idx - 1], h[idx + 1])
    gap_hi = np.where(d > 0, l[idx + 1], l[idx - 1])
    mid = 0.5 * (gap_lo + gap_hi)
    depth = np.where(
        d[:, None] > 0,
        (gap_hi[:, None] - lows_f),          # бычий: низ уходит вниз в зону
        (highs_f - gap_lo[:, None]),         # медвежий: хай уходит вверх
    )
    width = (gap_hi - gap_lo)[:, None]
    width[width == 0] = np.nan
    frac = depth / width                     # 1.0 = полное закрытие
    maxf = np.nanmax(frac, axis=1)
    return {
        "n": len(idx),
        "p_touch": float(np.nanmean(maxf > 0)),
        "p_fill_50": float(np.nanmean(maxf >= 0.5)),
        "p_fill_100": float(np.nanmean(maxf >= 1.0)),
    }


def sweep_reject_events(o, h, l, c, lookback=50, wick_min=0.4):
    """Графический свип: бар проколол N-барный экстремум, закрылся обратно
    внутрь диапазона и имеет фитиль отклонения >= wick_min диапазона."""
    hh = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    rng = h - l
    rng[rng == 0] = np.nan
    up_wick = (h - np.maximum(o, c)) / rng
    dn_wick = (np.minimum(o, c) - l) / rng
    rej_up = (h > hh) & (c <= hh) & (up_wick >= wick_min)
    rej_dn = (l < ll) & (c >= ll) & (dn_wick >= wick_min)
    return rej_up, rej_dn


def combo_events(o, h, l, c, lookback=50, window=6, k_disp=2.0,
                 body_min=0.6):
    """Комплекс: свип экстремума, затем в течение window баров
    противоположный дисплейсмент, образующий FVG (MSS по-ICT)."""
    hh = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    swept_up = h > hh      # сняли ликвидность сверху -> ждём медвежий импульс
    swept_dn = l < ll
    disp, d = displacement_events(o, h, l, c, k=k_disp, body_min=body_min)
    fvg_bull, fvg_bear = fvg_events(o, h, l, c)
    n = len(c)
    bear_combo = np.zeros(n, dtype=bool)   # свип хая -> медвежий disp+FVG
    bull_combo = np.zeros(n, dtype=bool)
    sw_up_idx = np.where(swept_up)[0]
    sw_dn_idx = np.where(swept_dn)[0]
    bear_disp = np.where(disp & (d < 0) & fvg_bear)[0]
    bull_disp = np.where(disp & (d > 0) & fvg_bull)[0]
    for i in bear_disp:
        lo = max(0, i - window)
        if sw_up_idx.size and np.any((sw_up_idx >= lo) & (sw_up_idx < i)):
            bear_combo[i] = True
    for i in bull_disp:
        lo = max(0, i - window)
        if sw_dn_idx.size and np.any((sw_dn_idx >= lo) & (sw_dn_idx < i)):
            bull_combo[i] = True
    combo = bear_combo | bull_combo
    cdir = np.where(bull_combo, 1.0, -1.0)
    return combo, cdir


# ---------------------------------------------------------------- нулевая модель

def block_bootstrap_ohlc(o, h, l, c, block=50, seed=None):
    """Синтетический ряд: блочный бутстреп лог-компонентов свечей.
    Сохраняет распределение доходностей, кластеризацию волатильности,
    внутрибарную форму и гэпы."""
    rng = np.random.default_rng(seed)
    n = len(c)
    r_gap = np.log(o[1:] / c[:-1])
    r_oh = np.log(h[1:] / o[1:])
    r_ol = np.log(l[1:] / o[1:])
    r_oc = np.log(c[1:] / o[1:])
    m = n - 1
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m - block, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
    so = np.empty(n); sh = np.empty(n); sl = np.empty(n); sc = np.empty(n)
    so[0], sh[0], sl[0], sc[0] = o[0], h[0], l[0], c[0]
    # close[t] = close[t-1] * exp(gap[t] + oc[t]) -> кумулятивно
    sc[1:] = sc[0] * np.exp(np.cumsum(r_gap[idx] + r_oc[idx]))
    so[1:] = sc[:-1] * np.exp(r_gap[idx])
    sh[1:] = so[1:] * np.exp(r_oh[idx])
    sl[1:] = so[1:] * np.exp(r_ol[idx])
    return so, sh, sl, sc


# ---------------------------------------------------------------- прогон

def run_single(o, h, l, c, horizons=(1, 2, 4, 8, 16, 32), K=24,
               k_disp=2.0):
    """Полный набор статистик для одного ряда."""
    ev, direction = displacement_events(o, h, l, c, k=k_disp)
    ev_idx = np.where(ev)[0]
    res = {"n_bars": len(c), "n_disp": len(ev_idx)}
    res["disp_fwd"] = forward_stats(c, ev_idx, direction, horizons)
    res["disp_revisit"] = revisit_stats(h, l, c, ev_idx, direction, K)
    res["disp_trade"] = trade_stats(h, l, c, ev_idx, direction, K)

    hold_up, fail_up, hold_dn, fail_dn = sweep_events(h, l, c)
    sdir = np.where(hold_up | fail_up, 1.0, -1.0)  # по всей длине ряда
    for name, mask in (
        ("hold", hold_up | hold_dn),
        ("fail", fail_up | fail_dn),
    ):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            res[f"sweep_{name}"] = None
            continue
        res[f"sweep_{name}"] = forward_stats(c, idx, sdir, horizons)

    streak, sign = run_events(c)
    run_res = {}
    for min_len in (2, 3, 4):
        mask = (streak >= min_len) & (sign != 0)
        idx = np.where(mask)[0]
        idx = idx[idx + 1 < len(c)]
        if len(idx) == 0:
            continue
        nxt = (c[idx + 1] - c[idx]) * sign[idx]
        run_res[min_len] = {
            "n": len(idx),
            "p_cont": float(np.mean(nxt > 0)),
        }
    res["runs"] = run_res

    # ---------------- графические паттерны ----------------
    fvg_bull, fvg_bear = fvg_events(o, h, l, c)
    fvg_any = fvg_bull | fvg_bear
    fdir = np.where(fvg_bull, 1.0, -1.0)
    res["fvg"] = {"n": int(fvg_any.sum())}
    res["fvg_fill"] = fvg_fill_stats(h, l, c, np.where(fvg_any)[0], fdir, K)

    # дисплейсмент С видимым разрывом против чисто статистического
    disp_fvg = ev & fvg_any
    res["disp_fvg"] = {"n": int(disp_fvg.sum())}
    res["disp_fvg_trade"] = trade_stats(h, l, c, np.where(disp_fvg)[0],
                                        direction, K)
    res["disp_fvg_fwd"] = forward_stats(c, np.where(disp_fvg)[0],
                                        direction, horizons)

    rej_up, rej_dn = sweep_reject_events(o, h, l, c)
    rej_any = rej_up | rej_dn
    rdir = np.where(rej_dn, 1.0, -1.0)   # отклонение вверх -> шорт-контекст
    res["sweep_reject"] = {"n": int(rej_any.sum())}
    res["sweep_reject_fwd"] = forward_stats(c, np.where(rej_any)[0],
                                            rdir, horizons)

    combo, cdir = combo_events(o, h, l, c)
    res["combo"] = {"n": int(combo.sum())}
    res["combo_fwd"] = forward_stats(c, np.where(combo)[0], cdir, horizons)
    res["combo_trade"] = trade_stats(h, l, c, np.where(combo)[0], cdir, K)
    return res


def run_with_null(df: pd.DataFrame, B=100, block=50, seed=7, **kw):
    """Статистики на реале + распределение тех же статистик на синтетике."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    real = run_single(o, h, l, c, **kw)

    boots = []
    for b in range(B):
        so, sh, sl, sc = block_bootstrap_ohlc(o, h, l, c, block=block,
                                              seed=seed + b)
        boots.append(run_single(so, sh, sl, sc, **kw))
    return real, boots
