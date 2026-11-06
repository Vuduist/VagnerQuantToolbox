"""X03: сырьё — сезонные окна ICT (лекция Ideal Seasonal Tendencies, R2NjnYozhdw) и отбор лидера
сектора по failure swing (Relative Strength Analysis, ygtvWKyOzHQ / u5S6Zt1ZIpA).

Сезонность: seasonality.monthly_test (перестановка месяцев в году) и window_test (случайное окно
той же длины) на непрерывных фьючерсах Yahoo 2000–2026.
Лидер сектора: в день t рынок m «лидер», если за последние 5 дней он не делал нового L-дневного лоу,
а хотя бы один сосед по сектору — делал (failure swing на фоне lower low соседей). Исход: разность
форвард-доходностей за h дней (лидер − среднее по соседям с новым лоу), каждая нормирована на
60-дневную sd дневных доходностей × √h. Зеркально для хаев (медвежий лидер = не сделал higher high).
Нулл: случайная перестановка ролей внутри события (кто лидер, кто сосед) — B раз.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = {"grains": ["ZW", "ZS", "ZC"], "metals": ["GC", "SI", "HG"], "energy": ["CL", "NG"],
           "softs": ["KC", "CT", "SB", "CC"], "livestock": ["LE", "HE"]}

# (рынок, метка, старт (м, д), конец (м, д), знак ICT)
WINDOWS = [
    ("ZC", "кукуруза: хай май–июнь → лоу сен–окт (Jun1–Sep30)", (6, 1), (9, 30), -1),
    ("ZW", "пшеница: вниз в 1-й половине года (Jan1–Jun30)", (1, 1), (6, 30), -1),
    ("ZW", "пшеница: лоу июнь–август → покупки (Aug1–Dec31)", (8, 1), (12, 31), +1),
    ("CC", "какао: лоу конец мая → хай конец авг–сен (Jun1–Sep15)", (6, 1), (9, 15), +1),
    ("KC", "кофе: лоу середина июня–июль → рост (Jul1–Oct31)", (7, 1), (10, 31), +1),
    ("CT", "хлопок: распродажа от хая апр–май (May1–Jun30)", (5, 1), (6, 30), -1),
    ("CL", "нефть: бычий уклон с начала года до хая сен–окт (Jan1–Sep15)", (1, 1), (9, 15), +1),
    ("CL", "нефть: бычий сер. февраля – май/июнь (Feb15–May31)", (2, 15), (5, 31), +1),
    ("HG", "медь: покупки в июне (Jun1–Aug31)", (6, 1), (8, 31), +1),
    ("HG", "медь: продажи в сентябре (Sep1–Oct31)", (9, 1), (10, 31), -1),
    ("HG", "медь: продажи в апреле (Apr1–May31)", (4, 1), (5, 31), -1),
    ("HG", "медь: покупки в ноябре (Nov1–Dec31)", (11, 1), (12, 31), +1),
    ("HE", "свинина: покупки фев–апр → хай май–июль (Feb1–May31)", (2, 1), (5, 31), +1),
    ("HE", "свинина: хай май–июль → лоу сен–окт (Jul1–Sep30)", (7, 1), (9, 30), -1),
    ("LE", "скот: хай фев–март → лоу июнь (Mar1–Jun15)", (3, 1), (6, 15), -1),
    ("LE", "скот: markup 2-я половина года (Jun15–Dec31)", (6, 15), (12, 31), +1),
    ("SI", "серебро: лоу июль–август → покупки (Aug1–Dec31)", (8, 1), (12, 31), +1),
    ("SI", "серебро: слабость в феврале (Feb1–Mar15)", (2, 1), (3, 15), -1),
    ("ZS", "соя: рост с февраля к хаю июнь–июль (Feb1–Jun15)", (2, 1), (6, 15), +1),
    ("ZS", "соя: от хая июнь–июль к лоу сен–окт (Jul1–Sep30)", (7, 1), (9, 30), -1),
    ("ZB", "30Y: лоу май–июнь → рост (Jun1–Oct31)", (6, 1), (10, 31), +1),
]


def leader_test(data: dict[str, pd.Series], sector: list[str], L=20, hs=(20, 60), lookback=5, gap=10, B=500, seed=0):
    """data: close-серии по рынкам (выровнены по дате внутри). Векторизовано."""
    px = pd.concat({m: data[m] for m in sector}, axis=1).dropna()
    lp = np.log(px.to_numpy(float))
    n, M = lp.shape
    sd = pd.DataFrame(lp).diff().rolling(60).std().shift(1).to_numpy()
    fwd = {h: (np.r_[lp[h:], np.full((h, M), np.nan)] - lp) / (sd * np.sqrt(h)) for h in hs}
    low_new = np.zeros((n, M), bool); high_new = np.zeros((n, M), bool)
    for j in range(M):
        s = px.iloc[:, j]
        low_new[:, j] = (s < s.rolling(L).min().shift(1)).rolling(lookback).max().fillna(0).astype(bool).to_numpy()
        high_new[:, j] = (s > s.rolling(L).max().shift(1)).rolling(lookback).max().fillna(0).astype(bool).to_numpy()
    rng = np.random.default_rng(seed)
    out = {}
    for kind, F, sign in (("bull", low_new, +1), ("bear", high_new, -1)):
        ev_t, ev_lead, ev_lag = [], [], []
        last = np.full(M, -10**9)
        for t in range(L + 60, n - max(hs) - 1):
            row = F[t]
            if not row.any() or row.all():
                continue
            for m in np.flatnonzero(~row):
                if t - last[m] <= gap:
                    continue
                last[m] = t
                ev_t.append(t); ev_lead.append(m); ev_lag.append(row.copy())
        out[f"{kind}_n"] = float(len(ev_t))
        if len(ev_t) < 20:
            continue
        ev_t = np.array(ev_t); ev_lead = np.array(ev_lead); ev_lag = np.array(ev_lag)
        for h in hs:
            R = sign * fwd[h][ev_t]
            r_lead = R[np.arange(len(ev_t)), ev_lead]
            r_lag = np.nanmean(np.where(ev_lag, R, np.nan), axis=1)
            d = r_lead - r_lag; ok = np.isfinite(d); d = d[ok]
            real = d.mean(); p_pos = (d > 0).mean()
            members = ev_lag.copy(); members[np.arange(len(ev_t)), ev_lead] = True
            boots = []
            for _ in range(B):
                u = rng.random((len(ev_t), M)) * members
                rl = u.argmax(axis=1)
                others = members.copy(); others[np.arange(len(ev_t)), rl] = False
                r_l = R[np.arange(len(ev_t)), rl]
                r_o = np.nanmean(np.where(others, R, np.nan), axis=1)
                boots.append(np.nanmean((r_l - r_o)[ok]))
            boots = np.array(boots)
            out[f"{kind}_d{h}"] = {"real": real, "p_pos": p_pos, "null_mean": boots.mean(), "null_sd": boots.std(ddof=1),
                                   "z": (real - boots.mean()) / boots.std(ddof=1) if boots.std(ddof=1) > 0 else np.nan,
                                   "lead": float(np.nanmean(r_lead[ok])), "lag": float(np.nanmean(r_lag[ok]))}
    return out
