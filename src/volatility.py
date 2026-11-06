"""Предсказуемость волатильности: HAR-модель, walk-forward OOS.

RV_t+1 ~ a + b1*RV_t + b2*mean(RV_{t-4..t}) + b3*mean(RV_{t-21..t})
Для интрадей-данных RV дня = сумма кв. лог-доходностей за день;
для дневных — прокси Паркинсона (ln H/L)^2 / (4 ln 2).

Оценка: out-of-sample R^2 против наивного прогноза (RV_t) + по годам.
Нулл: блочный бутстреп самого ряда RV (разрушает длинную память).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine


def daily_rv(df: pd.DataFrame) -> pd.Series:
    """Дневная реализованная волатильность."""
    if (df.index.hour == 0).all():
        # дневные данные -> Паркинсон
        rv = (np.log(df["high"] / df["low"]) ** 2) / (4 * np.log(2))
        return pd.Series(rv, index=df.index)
    r = np.log(df["close"]).diff()
    rv = (r ** 2).groupby(df.index.normalize()).sum()
    rv = rv[rv > 0]
    return rv


def har_design(rv: np.ndarray):
    n = len(rv)
    d = rv[:-1]                                   # RV_t
    w = pd.Series(rv).rolling(5).mean().to_numpy()[:-1]
    m = pd.Series(rv).rolling(22).mean().to_numpy()[:-1]
    y = rv[1:]                                     # RV_t+1
    X = np.column_stack([np.ones(n - 1), d, w, m])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[ok], y[ok]


def har_oos(rv_series: pd.Series, train0=500, refit=21):
    """Walk-forward: первые train0 дней обучение, дальше OOS,
    переобучение раз в refit дней."""
    rv = np.log(rv_series.to_numpy())   # лог-РВ для стабильности
    X, y = har_design(rv)
    start = train0
    preds, actual = [], []
    for t0 in range(start, len(y), refit):
        t1 = min(t0 + refit, len(y))
        beta, *_ = np.linalg.lstsq(X[:t0], y[:t0], rcond=None)
        preds.extend(X[t0:t1] @ beta)
        actual.extend(y[t0:t1])
    preds, actual = np.array(preds), np.array(actual)
    naive = rv[1:][start - 1:][:len(actual)]  # прогноз "завтра = сегодня"
    # R^2 против наивного и против среднего
    sse = np.sum((actual - preds) ** 2)
    sst_naive = np.sum((actual - naive) ** 2)
    sst_mean = np.sum((actual - actual.mean()) ** 2)
    return {
        "n_oos": len(actual),
        "R2_vs_mean": 1 - sse / sst_mean,
        "R2_vs_naive": 1 - sse / sst_naive,
        "corr": float(np.corrcoef(preds, actual)[0, 1]),
        "idx": rv_series.index[1:][start:][:len(actual)],
        "actual": actual, "preds": preds,
    }


def rv_bootstrap(rv: pd.Series, block=30, seed=0):
    """Блочный бутстреп ряда RV: сохраняет распределение, убивает память."""
    v = rv.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(v)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=nb)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return pd.Series(v[idx], index=rv.index)
