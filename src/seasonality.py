"""X04: сезонность по календарю (месяц года, окна ICT) на D1.

Нуллы: (1) месяцы — перестановка 12 месячных доходностей внутри каждого года
(сохраняет годовой дрейф и режим, рушит календарь); (2) окна ICT — то же окно по длине,
но со случайным стартом внутри года (B раз), z относительно распределения.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (имя, рынки, (мес, день) старт, (мес, день) конец, ожидаемый знак по ICT)
WINDOWS = [
    ("EUR: лоу Jun–Jul → лонг лето (Jul1–Sep30)", ("EURUSD",), (7, 1), (9, 30), +1),
    ("EUR: вниз 2-я пол. марта→апрель (Mar15–Apr30)", ("EURUSD",), (3, 15), (4, 30), -1),
    ("GBP: лоу март → хай май (Mar1–May31)", ("GBPUSD",), (3, 1), (5, 31), +1),
    ("Gold: лоу Jul–Aug → рост к концу года (Aug1–Dec31)", ("XAUUSD", "GC"), (8, 1), (12, 31), +1),
    ("Gold: bullish Dec–mid-Feb (Dec1–Feb15)", ("XAUUSD", "GC"), (12, 1), (2, 15), +1),
    ("Gold: слабость перед лоу (Jun1–Aug15)", ("XAUUSD", "GC"), (6, 1), (8, 15), -1),
    ("Idx: снижение посл. нед. апреля–май (Apr24–May31)", ("NQ", "ES"), (4, 24), (5, 31), -1),
    ("Idx: sell May→June (May1–Jun30)", ("NQ", "ES"), (5, 1), (6, 30), -1),
    ("Idx: sell May–mid-Sep (May1–Sep15)", ("NQ", "ES"), (5, 1), (9, 15), -1),
    ("Idx: buy Feb–May (Feb1–May1)", ("NQ", "ES"), (2, 1), (5, 1), +1),
    ("Idx: buy mid-Sep–Dec (Sep15–Dec31)", ("NQ", "ES"), (9, 15), (12, 31), +1),
    ("Idx: лонг осень→февраль (Oct1–Feb28)", ("NQ", "ES"), (10, 1), (2, 28), +1),
    ("Idx: 3-я неделя мая buy (May15–May31)", ("NQ", "ES"), (5, 15), (5, 31), +1),
    ("Idx: 3-я неделя июня buy (Jun15–Jun30)", ("NQ", "ES"), (6, 15), (6, 30), +1),
    ("Idx: sell mid-Jun→early Jul (Jun15–Jul10)", ("NQ", "ES"), (6, 15), (7, 10), -1),
    ("Idx: sell 1-я нед. января→февраль (Jan1–Jan31)", ("NQ", "ES"), (1, 1), (1, 31), -1),
    ("Idx: buy посл. нед. ноября→1-я дек. (Nov22–Dec7)", ("NQ", "ES"), (11, 22), (12, 7), +1),
    ("All: август — слабый/вялый месяц (Aug1–Aug31, |ER|)", ("*",), (8, 1), (8, 31), 0),
]


def _ts(year, month, day):
    """Timestamp с защитой от 29 февраля в невисокосном году."""
    if month == 2 and day == 29:
        day = 28
    return pd.Timestamp(year=year, month=month, day=day)


def window_return(c: pd.Series, start, end):
    """Доходность окна по годам: close последнего дня < старт -> close последнего дня <= конец."""
    years = sorted(set(c.index.year))
    out = {}
    for y in years:
        y_end = y + 1 if (end[0], end[1]) < (start[0], start[1]) else y
        t0 = _ts(y, start[0], start[1])
        t1 = _ts(y_end, end[0], end[1])
        a = c[c.index < t0]
        b = c[c.index <= t1]
        if len(a) == 0 or len(b) == 0 or b.index[-1] < t0 or (t1 - b.index[-1]).days > 7 or (t0 - a.index[-1]).days > 7:
            continue
        out[y] = np.log(b.iloc[-1] / a.iloc[-1])
    return pd.Series(out)


def efficiency(c: pd.Series, start, end):
    """Kaufman ER окна по годам: |net| / сумма |дневных|."""
    out = {}
    for y in sorted(set(c.index.year)):
        t0 = _ts(y, start[0], start[1])
        t1 = _ts(y, end[0], end[1])
        seg = c[(c.index >= t0) & (c.index <= t1)]
        if len(seg) < 10:
            continue
        r = np.log(seg).diff().dropna()
        out[y] = abs(r.sum()) / r.abs().sum() if r.abs().sum() > 0 else np.nan
    return pd.Series(out)


def window_test(c: pd.Series, start, end, sign, B=1000, seed=0):
    """Реал vs случайные окна той же длины (старт равномерно по году)."""
    rng = np.random.default_rng(seed)
    length = (pd.Timestamp(year=2001 if (end[0], end[1]) < (start[0], start[1]) else 2000, month=end[0], day=end[1])
              - pd.Timestamp(year=2000, month=start[0], day=start[1])).days
    if sign == 0:
        real = efficiency(c, start, end)
        stat = float(real.mean())
        boots = []
        for _ in range(B):
            s0 = pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=int(rng.integers(0, 365 - length)))
            s1 = s0 + pd.Timedelta(days=length)
            boots.append(efficiency(c, (s0.month, s0.day), (s1.month, s1.day)).mean())
        boots = np.array(boots)
        return {"n_years": len(real), "mean": stat, "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                "z": (stat - boots.mean()) / boots.std(ddof=1), "t": np.nan, "p_pos": np.nan, "h1": np.nan, "h2": np.nan}
    real = window_return(c, start, end) * sign
    stat = float(real.mean())
    t = stat / (real.std(ddof=1) / np.sqrt(len(real))) if len(real) > 3 else np.nan
    half = len(real) // 2
    boots = []
    for _ in range(B):
        s0 = pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=int(rng.integers(0, 365)))
        s1 = s0 + pd.Timedelta(days=length)
        boots.append((window_return(c, (s0.month, s0.day), (s1.month, s1.day)) * sign).mean())
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    return {"n_years": len(real), "mean": stat, "t": t, "p_pos": float((real > 0).mean()),
            "h1": float(real.iloc[:half].mean()), "h2": float(real.iloc[half:].mean()),
            "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
            "z": (stat - boots.mean()) / boots.std(ddof=1)}


def monthly_test(c: pd.Series, B=2000, seed=0):
    """Месячные доходности: по месяцам mean/t/доля плюс; нулл — перестановка месяцев в году."""
    m = np.log(c.resample("ME").last()).diff().dropna()
    tab = pd.DataFrame({"r": m, "y": m.index.year, "mo": m.index.month})
    piv = tab.pivot_table(index="y", columns="mo", values="r")
    piv = piv.dropna(how="any")                        # полные годы
    X = piv.to_numpy()
    real_mean = X.mean(axis=0)
    real_t = real_mean / (X.std(axis=0, ddof=1) / np.sqrt(len(X)))
    rng = np.random.default_rng(seed)
    null_means = np.empty((B, 12))
    null_maxt = np.empty(B)
    for b in range(B):
        P = np.array([row[rng.permutation(12)] for row in X])
        mu = P.mean(axis=0)
        null_means[b] = mu
        null_maxt[b] = np.max(np.abs(mu / (P.std(axis=0, ddof=1) / np.sqrt(len(X)))))
    z = (real_mean - null_means.mean(axis=0)) / null_means.std(axis=0, ddof=1)
    return {"n_years": len(X), "months": piv.columns.to_list(), "mean_pct": real_mean * 100,
            "t": real_t, "p_pos": (X > 0).mean(axis=0), "z": z,
            "max_abs_t": float(np.max(np.abs(real_t))),
            "p_maxt": float((null_maxt >= np.max(np.abs(real_t))).mean())}
