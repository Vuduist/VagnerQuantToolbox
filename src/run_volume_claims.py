"""14 открытых утверждений ICT про ОБЪЁМ на данных с реальным объёмом.

Данные: ES/NQ 1h и 1d (Yahoo, фьючерсы, реальный объём CME), BTCUSDT 1m Binance 2020-01…2025-09
(агрегируется в 5m по месяцам). ZB-данных нет — утверждения про ZB проверяются на ES/NQ как замене.
FX-объёма нет — утверждения из FX-лекций проверяются на ES/NQ и BTC.

Метод: каждое утверждение → одно правило с конкретными окнами NY; метрика; нулл = перестановка
метки объёма внутри геометрического события (проверяет, добавляет ли объём что-то к геометрии) или
плацебо-окна (те же правила для других часов/дней); n, эффект, t; знак в двух половинах выборки.
Издержки: ES/NQ 0.25 пт спред (~0.4 бп), BTC тейкер 5 бп за сторону (10 бп круг).

Запуск: PYTHONPATH=src.venv/bin/python src/run_volume_claims.py
"""
from __future__ import annotations

import glob
import os
import sys
import time

import warnings

import numpy as np
import pandas as pd

import engine

warnings.filterwarnings("ignore", category=RuntimeWarning)

NY = "America/New_York"
RNG = np.random.default_rng(7)
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def hdr(title):
    say("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


# ---------------------------------------------------------------- данные

def load_fut_1h(sym):
    df = engine.load_csv(f"data/raw/{sym}_1h.csv")
    df = df[df.index.minute == 0]                      # редкие бары :30 / обрывки — вон
    df.loc[df.volume <= 1, "volume"] = np.nan          # первый бар Globex-дня у Yahoo = мусор (volume=1)
    return to_ny(df)


def to_ny(df):
    loc = df.index.tz_localize("UTC").tz_convert(NY)
    d = df.copy()
    d.index = loc
    naive = loc.tz_localize(None)
    d["date"] = naive.normalize()
    d["clock"] = naive.hour * 60 + naive.minute
    d["wd"] = naive.weekday
    return d


def load_btc_5m():
    files = sorted(glob.glob("data/raw/BTCUSDT_1m/BTCUSDT-1m-*.csv"))
    parts = []
    for f in files:                                    # по месяцам: память < 50 МБ
        m = engine.load_csv(f)
        parts.append(engine.resample(m, "5min"))
    b = pd.concat(parts)
    b = b[~b.index.duplicated()].sort_index()
    return to_ny(b)


def rel_volume(d, window=40):
    """Объём / медиана объёма того же слота (clock) за предыдущие `window` дней — причинная нормировка профиля дня."""
    g = d.groupby("clock")["volume"]
    med = g.transform(lambda s: s.shift(1).rolling(window, min_periods=10).median())
    return d["volume"] / med


# ---------------------------------------------------------------- статистика

def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return len(x), np.nan, np.nan
    return len(x), x.mean(), x.mean() / x.std(ddof=1) * np.sqrt(len(x))


def perm_diff(y, lab, n_perm=2000):
    """Разница средних (lab=True) − (lab=False); t Уэлча; p и z против перестановки метки."""
    y = np.asarray(y, float); lab = np.asarray(lab, bool)
    ok = np.isfinite(y); y, lab = y[ok], lab[ok]
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 < 8 or n0 < 8:
        return dict(n1=n1, n0=n0, m1=np.nan, m0=np.nan, diff=np.nan, t=np.nan, p=np.nan, z=np.nan)
    m1, m0 = y[lab].mean(), y[~lab].mean()
    obs = m1 - m0
    se = np.sqrt(y[lab].var(ddof=1) / n1 + y[~lab].var(ddof=1) / n0)
    tot = y.sum(); null = np.empty(n_perm)
    for i in range(n_perm):
        s = y[RNG.permutation(len(y))[:n1]].sum()
        null[i] = s / n1 - (tot - s) / n0
    p = float((np.abs(null) >= abs(obs)).mean())
    z = (obs - null.mean()) / null.std()
    return dict(n1=n1, n0=n0, m1=m1, m0=m0, diff=obs, t=obs / se if se > 0 else np.nan, p=p, z=z)


def halves_sign(dates, y, lab):
    """Знак разницы (lab − ¬lab) в первой и второй половине по времени."""
    dates = pd.DatetimeIndex(dates); y = np.asarray(y, float); lab = np.asarray(lab, bool)
    cut = dates.sort_values()[len(dates) // 2]
    out = []
    for m in (dates < cut, dates >= cut):
        a, b = y[m & lab], y[m & ~lab]
        sg = np.sign(np.nanmean(a) - np.nanmean(b)) if len(a) > 3 and len(b) > 3 else 0
        out.append(0 if not np.isfinite(sg) else sg)
    return "/".join({1.0: "+", -1.0: "−", 0: "0"}[s] for s in out)


def fmt_pd(r, unit="бп", mult=1e4):
    return (f"n={r['n1']}/{r['n0']}  лёгкий={r['m1'] * mult:+.2f} тяжёлый={r['m0'] * mult:+.2f} {unit}  "
            f"разн={r['diff'] * mult:+.2f}  t={r['t']:+.2f}  perm p={r['p']:.3f} z={r['z']:+.2f}")


# ---------------------------------------------------------------- 1. профиль объёма по часам NY

def hour_profile(d, name, sess_hours=None):
    """Средняя доля объёма дня по часам NY (по дням Пн–Пт с ≥ 18 барами), первая/вторая половина."""
    x = d[d.wd < 5].dropna(subset=["volume"])
    cnt = x.groupby("date").volume.count()
    x = x[x.date.isin(cnt[cnt >= 18].index)]
    tot = x.groupby("date").volume.transform("sum")
    x = x.assign(share=x.volume / tot, hour=x.clock // 60)
    prof = x.groupby("hour").share.mean() * 100
    cut = x.date.sort_values().iloc[len(x) // 2]
    p1 = x[x.date < cut].groupby("hour").share.mean() * 100
    p2 = x[x.date >= cut].groupby("hour").share.mean() * 100
    tab = pd.DataFrame({"доля,%": prof, "1-я половина": p1, "2-я половина": p2}).round(2)
    say(f"\n{name}: {x.date.nunique()} дней {x.date.min().date()}…{x.date.max().date()}; доля объёма дня по часам NY (бар HH = HH:00–HH:59)")
    say(tab.T.to_string())
    top = prof.sort_values(ascending=False)
    say("топ-3 часа:", ", ".join(f"{h:02d}:00 ({v:.1f}%)" for h, v in top.head(3).items()))
    return prof


def window_share(prof, lo, hi):
    return prof[(prof.index >= lo) & (prof.index < hi)].sum()


# ---------------------------------------------------------------- 2. opening range + объём (утв. 2, 10)

def or_break_test(d, name, or_lo, or_hi, ev_lo, ev_hi, horizons, bar_label):
    """Первый пробой high/low opening range в окне события. Метка «лёгкий» = объём бара пробоя < среднего
    объёма бара OR (сырой) и (вариант) относительный объём бара < относительного объёма OR.
    Исход: лог-доходность в сторону пробоя за N баров (ICT: лёгкий → разворот, т.е. меньше/отрицательно),
    и P(закрытие вернулось внутрь OR через N баров)."""
    d = d[d.wd < 5]
    rv = rel_volume(d)
    c = d.close.to_numpy(); h = d.high.to_numpy(); l = d.low.to_numpy(); v = d.volume.to_numpy(); rvv = rv.to_numpy()
    clock = d.clock.to_numpy(); dates = d.date.to_numpy()
    rows = []
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], len(d)]
    N = max(horizons)
    for s, e in zip(starts, ends):
        ck = clock[s:e]
        o_m = (ck >= or_lo) & (ck < or_hi)
        if o_m.sum() < (or_hi - or_lo) // (ck[1] - ck[0] if e - s > 1 else 60) * 0.8:
            continue
        orh, orl = h[s:e][o_m].max(), l[s:e][o_m].min()
        orv = np.nanmean(v[s:e][o_m]); orrv = np.nanmean(rvv[s:e][o_m])
        if not np.isfinite(orv) or not np.isfinite(orrv):
            continue
        done = {+1: False, -1: False}
        for i in range(s, e):
            if clock[i] < ev_lo or clock[i] >= ev_hi:
                continue
            if i + N >= e:
                break
            for dirn, brk in ((+1, h[i] > orh), (-1, l[i] < orl)):
                if brk and not done[dirn]:
                    done[dirn] = True
                    if not np.isfinite(v[i]):
                        continue
                    row = dict(date=dates[i], dirn=dirn, light_raw=v[i] < orv, light_rel=rvv[i] < orrv)
                    for n in horizons:
                        row[f"r{n}"] = dirn * np.log(c[i + n] / c[i])
                        row[f"back{n}"] = float(orl <= c[i + n] <= orh)
                    rows.append(row)
    ev = pd.DataFrame(rows)
    say(f"\n{name}: OR {or_lo // 60:02d}:{or_lo % 60:02d}–{or_hi // 60:02d}:{or_hi % 60:02d} NY, пробой в {ev_lo // 60:02d}:{ev_lo % 60:02d}–{ev_hi // 60:02d}:{ev_hi % 60:02d}; "
        f"событий {len(ev)} (вверх {int((ev.dirn > 0).sum())}, вниз {int((ev.dirn < 0).sum())}); доля «лёгких» raw {ev.light_raw.mean():.2f}, rel {ev.light_rel.mean():.2f}")
    for dirn, lab in ((+1, "пробой OR-high вверх (утв. 2)"), (-1, "пробой OR-low вниз (утв. 10)"), (0, "оба направления")):
        sub = ev if dirn == 0 else ev[ev.dirn == dirn]
        say(f"  -- {lab}: n={len(sub)}")
        for n in horizons:
            k, m, t = tstat(sub[f"r{n}"])
            say(f"     безусловно: продолжение за {n} {bar_label}: {m * 1e4:+.2f} бп (t={t:+.2f}); P(вернулись в OR)={sub[f'back{n}'].mean():.2f}")
            for key, nm in (("light_raw", "сырой объём"), ("light_rel", "относит. объём")):
                r = perm_diff(sub[f"r{n}"], sub[key])
                say(f"     {nm:15s} продолжение {n}{bar_label}: {fmt_pd(r)}  половины {halves_sign(sub.date, sub[f'r{n}'], sub[key])}")
                rb = perm_diff(sub[f"back{n}"], sub[key])
                say(f"     {'':15s} P(возврат в OR) лёгкий={rb['m1']:.2f} тяжёлый={rb['m0']:.2f} разн={rb['diff']:+.3f} t={rb['t']:+.2f} perm p={rb['p']:.3f}")
    return ev


# ---------------------------------------------------------------- 3. ретест лоу дня на меньшем объёме (утв. 11)

def retest_low_test(d, name, sess_lo, sess_hi, min_gap, eps_frac, horizons, bar_label):
    """Лоу сессии L (бар iL, объём vL). Ретест: первый бар j ≥ iL+min_gap с low ≤ L + eps и close > L.
    Метка «лёгкий»: v_j < vL. Исход: P(close < L в следующие N баров) и лог-доходность за N баров (ICT: лёгкий → держит и растёт)."""
    d = d[d.wd < 5]
    c = d.close.to_numpy(); h = d.high.to_numpy(); l = d.low.to_numpy(); v = d.volume.to_numpy()
    clock = d.clock.to_numpy(); dates = d.date.to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]]); ends = np.r_[starts[1:], len(d)]
    N = max(horizons); rows = []
    for s, e in zip(starts, ends):
        idx = [i for i in range(s, e) if sess_lo <= clock[i] < sess_hi]
        if len(idx) < 6:
            continue
        L, iL, hi_so_far = np.inf, None, -np.inf
        for i in idx:
            hi_so_far = max(hi_so_far, h[i])
            if l[i] < L:
                L, iL = l[i], i
                continue
            if iL is None or i < iL + min_gap or i + N >= e:
                continue
            eps = eps_frac * (hi_so_far - L)
            if l[i] <= L + eps and c[i] > L and np.isfinite(v[i]) and np.isfinite(v[iL]):
                row = dict(date=dates[i], light=v[i] < v[iL])
                for n in horizons:
                    row[f"brk{n}"] = float(c[i + 1:i + 1 + n].min() < L)
                    row[f"r{n}"] = np.log(c[i + n] / c[i])
                rows.append(row)
                L = np.inf; iL = None                 # один ретест на уровень; следующий лоу — новый уровень
    ev = pd.DataFrame(rows)
    say(f"\n{name}: сессия {sess_lo // 60:02d}:{sess_lo % 60:02d}–{sess_hi // 60:02d}:00 NY, ретест = low ≤ L + {eps_frac:.2f}×(диапазон сессии), close > L; событий {len(ev)}, лёгких {ev.light.mean():.2f}")
    for n in horizons:
        k, m, t = tstat(ev[f"r{n}"])
        say(f"  безусловно: доходность {n} {bar_label}: {m * 1e4:+.2f} бп (t={t:+.2f}); P(пробой L)={ev[f'brk{n}'].mean():.2f}")
        r = perm_diff(ev[f"r{n}"], ev.light)
        say(f"  доходность {n}{bar_label}: {fmt_pd(r)}  половины {halves_sign(ev.date, ev[f'r{n}'], ev.light)}")
        rb = perm_diff(ev[f"brk{n}"], ev.light)
        say(f"  P(пробой L за {n}): лёгкий={rb['m1']:.3f} тяжёлый={rb['m0']:.3f} разн={rb['diff']:+.3f} t={rb['t']:+.2f} perm p={rb['p']:.3f} половины {halves_sign(ev.date, ev[f'brk{n}'], ev.light)}")
    return ev


# ---------------------------------------------------------------- 4. новый хай дня на падающем объёме (утв. 14)

def new_high_div_test(d, name, sess_lo, sess_hi, first_skip, min_gap, horizons, bar_label, or_lo, or_hi):
    """Бар k делает новый хай сессии (после first_skip минут), предыдущий хай-бар k0 ≥ min_gap баров раньше.
    Дивергенция: v_k < v_k0. Исход: лог-доходность за N баров; P(close вернулся к середине OR за N баров)."""
    d = d[d.wd < 5]
    c = d.close.to_numpy(); h = d.high.to_numpy(); l = d.low.to_numpy(); v = d.volume.to_numpy()
    clock = d.clock.to_numpy(); dates = d.date.to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]]); ends = np.r_[starts[1:], len(d)]
    N = max(horizons); rows = []
    for s, e in zip(starts, ends):
        idx = [i for i in range(s, e) if sess_lo <= clock[i] < sess_hi]
        if len(idx) < 6:
            continue
        om = [i for i in idx if or_lo <= clock[i] < or_hi]
        if not om:
            continue
        ormid = (max(h[om]) + min(l[om])) / 2
        H, iH = -np.inf, None
        for i in idx:
            if h[i] > H:
                prev = iH
                H, iH = h[i], i
                if prev is None or clock[i] < sess_lo + first_skip or i - prev < min_gap or i + N >= e:
                    continue
                if not (np.isfinite(v[i]) and np.isfinite(v[prev])):
                    continue
                row = dict(date=dates[i], light=v[i] < v[prev])
                for n in horizons:
                    row[f"r{n}"] = np.log(c[i + n] / c[i])
                    row[f"or{n}"] = float(c[i + 1:i + 1 + n].min() <= ormid)
                rows.append(row)
    ev = pd.DataFrame(rows)
    say(f"\n{name}: сессия {sess_lo // 60:02d}:{sess_lo % 60:02d}–{sess_hi // 60:02d}:00 NY, новый хай после {(sess_lo + first_skip) // 60:02d}:{(sess_lo + first_skip) % 60:02d}, разрыв ≥ {min_gap} баров; событий {len(ev)}, с дивергенцией {ev.light.mean():.2f}")
    for n in horizons:
        k, m, t = tstat(ev[f"r{n}"])
        say(f"  безусловно: доходность {n} {bar_label}: {m * 1e4:+.2f} бп (t={t:+.2f}); P(возврат к mid OR)={ev[f'or{n}'].mean():.2f}")
        r = perm_diff(ev[f"r{n}"], ev.light)
        say(f"  доходность {n}{bar_label} (дивергенция − нет): {fmt_pd(r)}  половины {halves_sign(ev.date, ev[f'r{n}'], ev.light)}")
        rb = perm_diff(ev[f"or{n}"], ev.light)
        say(f"  P(возврат к mid OR за {n}): див={rb['m1']:.3f} нет={rb['m0']:.3f} разн={rb['diff']:+.3f} t={rb['t']:+.2f} perm p={rb['p']:.3f}")
    return ev


# ---------------------------------------------------------------- 5. mitigation: возврат в объёмную свечу (утв. 1)

def mitigation_test(d, name, look, horizons, bar_label, top_q=0.9, low_q=0.5):
    """Свеча B: тело ≥ 50% диапазона, относительный объём ≥ q90 (объёмная) либо ≤ q50 (плацебо-геометрия).
    Против хода: для бычьей B — в следующие `look` баров close < B.low; затем первый бар с high ≥ B.low (касание) = событие T.
    Исход после T: доходность за N баров в сторону хода (вниз для бычьей B) и расширение = сумма диапазонов N баров после / N до."""
    rv = rel_volume(d).to_numpy()
    o = d.open.to_numpy(); c = d.close.to_numpy(); h = d.high.to_numpy(); l = d.low.to_numpy()
    dates = d.date.to_numpy(); n = len(d)
    rng = h - l
    # диапазон бара / медиана диапазона того же слота за 40 дней (причинно): «расширение» без профиля дня
    rr = (d.assign(volume=pd.Series(rng, index=d.index)).pipe(rel_volume)).to_numpy()
    rr = np.where(rr > 0, rr, np.nan)
    body = np.abs(c - o) / np.where(rng > 0, rng, np.nan)
    q_hi = np.nanquantile(rv, top_q); q_lo = np.nanquantile(rv, low_q)
    cand = np.flatnonzero((body >= 0.5) & np.isfinite(rv) & ((rv >= q_hi) | (rv <= q_lo)))
    N = max(horizons); rows = []
    for b in cand:
        if b + 2 * look + N >= n:
            break
        sgn = 1 if c[b] > o[b] else -1                 # бычья свеча: ход против = вниз
        # 1) ход против: close за дальним краем свечи
        if sgn > 0:
            away = np.flatnonzero(c[b + 1:b + 1 + look] < l[b])
        else:
            away = np.flatnonzero(c[b + 1:b + 1 + look] > h[b])
        if len(away) == 0:
            continue
        a = b + 1 + away[0]
        # 2) возврат: касание диапазона свечи
        if sgn > 0:
            back = np.flatnonzero(h[a + 1:a + 1 + look] >= l[b])
        else:
            back = np.flatnonzero(l[a + 1:a + 1 + look] <= h[b])
        if len(back) == 0:
            continue
        t = a + 1 + back[0]
        if t + N >= n or t - N < 0:
            continue
        row = dict(date=dates[t], heavy=rv[b] >= q_hi)
        for k in horizons:
            row[f"r{k}"] = -sgn * np.log(c[t + k] / c[t])   # + = ускорение в сторону хода (против свечи)
            a_, b_ = rng[t + 1:t + 1 + k].sum(), rng[t + 1 - k:t + 1].sum()
            row[f"exp{k}"] = np.log(a_ / b_) if a_ > 0 and b_ > 0 else np.nan
            row[f"rr{k}"] = np.nanmean(np.log(rr[t + 1:t + 1 + k]))  # относительно нормы слота
            row[f"rd{k}"] = row[f"rr{k}"] - np.nanmean(np.log(rr[t + 1 - k:t + 1]))  # после − до, оба относительно нормы слота
        rows.append(row)
    ev = pd.DataFrame(rows)
    say(f"\n{name}: кандидатов {len(cand)}, событий касания {len(ev)} (объёмных {int(ev.heavy.sum())}, плацебо-малообъёмных {int((~ev.heavy).sum())}), look={look} баров")
    for k in horizons:
        for key, nm in ((f"r{k}", "ускорение в сторону хода, бп"), (f"exp{k}", "лог-расширение (после/до)"), (f"rr{k}", "лог-диапазон / норма слота"), (f"rd{k}", "норм. диапазон: после − до")):
            mult = 1e4 if key[0] == "r" and key[1].isdigit() else 1.0
            kk, m, t = tstat(ev[key])
            r = perm_diff(ev[key], ev.heavy)
            say(f"  {nm} за {k}{bar_label}: все {m * mult:+.3f} (t={t:+.2f}); объёмные={r['m1'] * mult:+.3f} плацебо={r['m0'] * mult:+.3f} "
                f"разн={r['diff'] * mult:+.3f} t={r['t']:+.2f} perm p={r['p']:.3f} половины {halves_sign(ev.date, ev[key], ev.heavy)}")
    return ev


# ---------------------------------------------------------------- 6. lunch hour и «свежий» AM-хай (утв. 13)

def lunch_test_btc(d):
    """BTC 5m. AM = 09:30–W; RH = последний подтверждённый пивот-хай (2 бара с каждой стороны) до W; DH = макс. хай AM.
    Дни с RH < DH. В окне W…W+60: P(хай > RH), P(хай > DH), P(RH < хай ≤ DH). W=12:00 (lunch) vs плацебо 11:00, 13:00, 14:00."""
    d = d[d.wd < 5]
    h = d.high.to_numpy(); clock = d.clock.to_numpy(); dates = d.date.to_numpy()
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]]); ends = np.r_[starts[1:], len(d)]
    out = {}
    for W in (11 * 60, 12 * 60, 13 * 60, 14 * 60):
        rows = []
        for s, e in zip(starts, ends):
            am = [i for i in range(s, e) if 570 <= clock[i] < W]
            win = [i for i in range(s, e) if W <= clock[i] < W + 60]
            if len(am) < 12 or len(win) < 10:
                continue
            DH = h[am].max()
            piv = [i for i in am[2:-2] if h[i] > max(h[i - 2], h[i - 1], h[i + 1], h[i + 2])]
            if not piv:
                continue
            RH = h[piv[-1]]
            if RH >= DH:
                continue
            wh = h[win].max()
            rows.append((wh > RH, wh > DH, (wh > RH) & (wh <= DH), (DH - RH) / DH))
        a = np.array(rows, float)
        out[W] = a
        say(f"  окно {W // 60:02d}:00–{W // 60 + 1:02d}:00: n={len(a)} дней, P(хай > свежий RH)={a[:, 0].mean():.3f}, P(хай > DH дня)={a[:, 1].mean():.3f}, "
            f"P(RH < хай ≤ DH)={a[:, 2].mean():.3f}, медиана (DH−RH)={np.median(a[:, 3]) * 1e4:.0f} бп")
    return out


def lunch_test_fut(d, name):
    d = d[d.wd < 5]
    piv = d.pivot_table(index="date", columns="clock", values="high")
    rows = []
    for W in (11, 12, 13, 14):
        hw = piv[W * 60]; prev = piv[(W - 1) * 60]
        am = piv[[c for c in piv.columns if 540 <= c < W * 60]].max(axis=1)
        m = (prev < am) & hw.notna() & prev.notna()
        rows.append((f"{W:02d}:00", int(m.sum()), (hw[m] > prev[m]).mean(), (hw[m] > am[m]).mean(), ((hw[m] > prev[m]) & (hw[m] <= am[m])).mean()))
    t = pd.DataFrame(rows, columns=["окно", "n", "P(> хай пред. часа)", "P(> хай AM)", "P(между)"]).set_index("окно").round(3)
    say(f"  {name} (1h; «свежий хай» = хай предыдущего часа, дни где он ниже хая 09:00–W):\n" + t.to_string())


# ---------------------------------------------------------------- 7. праздники (утв. 7, 9)

def holiday_tests(sym):
    import holidays
    d = engine.load_csv(f"data/raw/{sym}_1d.csv")
    d = d[d.volume > 1]
    d.index = d.index.tz_localize("UTC").tz_convert(NY).tz_localize(None).normalize()
    d = d[~d.index.duplicated()]
    d = d[d.index.weekday < 5]
    hol = holidays.NYSE(years=range(d.index[0].year, d.index[-1].year + 1))
    hol = pd.DatetimeIndex(sorted(pd.Timestamp(x) for x in hol))
    hol = hol[(hol.weekday < 5) & (hol >= d.index[0]) & (hol <= d.index[-1])]
    hol = hol[~hol.isin(d.index)]                     # только дни, когда биржа реально закрыта в данных
    d["vrel"] = d.volume / d.volume.shift(1).rolling(20).median()
    d["ret"] = np.log(d.close / d.close.shift(1))
    d["er"] = (d.close - d.open).abs() / (d.high - d.low).where(d.high > d.low)
    d["rng"] = np.log(d.high / d.low)
    d["rngrel"] = d.rng / d.rng.shift(1).rolling(20).median()
    d["week"] = d.index.to_period("W-SUN")
    hol_weeks = set(hol.to_period("W-SUN"))
    post_weeks = set((hol + pd.Timedelta(days=7)).to_period("W-SUN"))
    # предпраздничный день = последний торговый день до праздника (разрыв ≤ 4 дн)
    pos = d.index.searchsorted(hol) - 1
    pre = d.index[pos]
    ok = (hol - pre).days <= 4
    pre, hol = pre[ok], hol[ok]
    names = [holidays.NYSE(years=[x.year]).get(x.date()) for x in hol]
    pre_df = d.loc[pre].assign(hol=names, hdate=hol)
    base = d[~d.index.isin(pre)].dropna(subset=["vrel"])
    say(f"\n{sym} дневки {d.index[0].date()}…{d.index[-1].date()}, {len(d)} дней; праздников (закрыто) {len(hol)}")
    k, m, t = tstat(pre_df.vrel); k2, m2, t2 = tstat(base.vrel)
    r = perm_diff(np.r_[pre_df.vrel, base.vrel], np.r_[np.ones(len(pre_df)), np.zeros(len(base))], 1000)
    say(f"  предпраздничный день: объём/медиана20 = {m:.3f} (n={k}) против {m2:.3f} у остальных; разн t={r['t']:+.2f} perm p={r['p']:.3f}")
    mem = pre_df[pre_df.hol.str.contains("Memorial")]
    k, m, t = tstat(mem.vrel)
    say(f"  пятница перед Memorial Day: n={k}, объём/медиана20 = {m:.3f}, лет с объёмом ниже медианы {(mem.vrel < 1).sum()}/{k}; "
        f"|ret| {mem.ret.abs().mean() * 1e4:.0f} бп vs {base.ret.abs().mean() * 1e4:.0f} бп; диапазон/медиана20 {mem.rngrel.mean():.2f}")
    k, m, t = tstat(pre_df.ret)
    say(f"  направление предпраздничного дня: {m * 1e4:+.1f} бп (t={t:+.2f}); следующий торговый день после праздника: "
        f"{d.ret.shift(-1).loc[pre].mean() * 1e4:+.1f} бп (t={tstat(d.ret.shift(-1).loc[pre])[2]:+.2f})")
    # неделя после праздничной недели: первые 3 торговых дня
    g = d.groupby("week")
    d["rank"] = g.cumcount()
    wk_vol = g.volume.mean()
    base2 = d.week.map(lambda w: wk_vol.get(w - 2, np.nan))      # средний объём недели за 2 недели до (= неделя перед праздничной)
    d["v2w"] = d.volume / base2
    first3 = d[d["rank"] < 3].copy()
    first3["kind"] = np.where(first3.week.isin(post_weeks), "после праздничной", np.where(first3.week.isin(hol_weeks), "праздничная", "обычная"))
    tab = first3.groupby("kind").agg(n=("ret", "size"), vrel=("vrel", "mean"), v2w=("v2w", "mean"), er=("er", "mean"), absret_bp=("ret", lambda x: x.abs().mean() * 1e4), rngrel=("rngrel", "mean")).round(3)
    say("  первые 3 торговых дня недели по типу недели (vrel = объём/медиана20, v2w = объём/средний объём недели за 2 нед. до, er = |c−o|/(h−l), rngrel = диапазон/медиана20):\n" + tab.to_string())
    a, b = first3[first3.kind == "после праздничной"], first3[first3.kind == "обычная"]
    for col, nm in (("vrel", "объём/медиана20"), ("v2w", "объём/неделя −2"), ("er", "ER"), ("rngrel", "диапазон")):
        r = perm_diff(np.r_[a[col], b[col]], np.r_[np.ones(len(a)), np.zeros(len(b))], 1000)
        say(f"    {nm}: после праздничной − обычная = {r['diff']:+.3f}, t={r['t']:+.2f}, perm p={r['p']:.3f}, половины {halves_sign(np.r_[a.index, b.index], np.r_[a[col], b[col]], np.r_[np.ones(len(a)), np.zeros(len(b))].astype(bool))}")
    # дни 1–5 после праздничной недели (восстановление объёма)
    for k_ in range(1, 6):
        sub = d[(d["rank"] == k_ - 1) & d.week.isin(post_weeks)]
        say(f"    день {k_} недели после праздничной: n={len(sub)}, объём/медиана20 {sub.vrel.mean():.3f}, ER {sub.er.mean():.3f}, ret {sub.ret.mean() * 1e4:+.1f} бп (t={tstat(sub.ret)[2]:+.2f})")
    return pre_df


def preholiday_intraday(d, name):
    """1h: предпраздничные дни в выборке 1h — доля объёма AM (09–12) и |ход| AM/PM против обычных дней."""
    import holidays
    x = d[(d.wd < 5)].dropna(subset=["volume"])
    days = pd.DatetimeIndex(sorted(x.date.unique()))
    hol = pd.DatetimeIndex(sorted(pd.Timestamp(h) for h in holidays.NYSE(years=range(days[0].year, days[-1].year + 1))))
    hol = hol[(hol >= days[0]) & (hol <= days[-1]) & ~hol.isin(days)]
    pos = days.searchsorted(hol) - 1
    pre = set(days[pos][(hol - days[pos]).days <= 4])
    am = x[(x.clock >= 540) & (x.clock < 720)].groupby("date").agg(v=("volume", "sum"), o=("open", "first"), c=("close", "last"))
    pm = x[(x.clock >= 720) & (x.clock < 960)].groupby("date").agg(v=("volume", "sum"), o=("open", "first"), c=("close", "last"))
    tot = x[(x.clock >= 540) & (x.clock < 960)].groupby("date").volume.sum()
    t = pd.DataFrame({"tot": tot, "am_share": am.v / tot, "am_move": np.log(am.c / am.o).abs() * 1e4, "pm_move": np.log(pm.c / pm.o).abs() * 1e4}).dropna()
    t["vrel"] = t.tot / t.tot.shift(1).rolling(20).median()
    t["pre"] = t.index.isin(pre)
    g = t.groupby("pre").agg(n=("tot", "size"), vrel=("vrel", "mean"), am_share=("am_share", "mean"), am_move=("am_move", "mean"), pm_move=("pm_move", "mean")).round(3)
    say(f"  {name} 1h: предпраздничные дни vs остальные (vrel = объём RTH/медиана20, am_share = доля 09–12, ходы |o→c| в бп):\n" + g.to_string())


# ---------------------------------------------------------------- 8. 10:00–11:00 profit taking (утв. 6)

def profit_taking_test(d, name):
    """corr(ret 10–11, ret 03–10) и средний |ret| по часам; плацебо — те же корреляции для других часов."""
    x = d[d.wd < 5]
    piv_o = x.pivot_table(index="date", columns="clock", values="open")
    piv_c = x.pivot_table(index="date", columns="clock", values="close")
    r = np.log(piv_c / piv_o)
    pre = np.log(piv_c[9 * 60] / piv_o[3 * 60])
    say(f"\n{name}: ход 03:00–10:00 vs следующий час (n≈{int(pre.notna().sum())}):")
    for H in (10, 11, 12, 13, 14, 15):
        y = r[H * 60]; m = pre.notna() & y.notna()
        beta = np.polyfit(pre[m], y[m], 1)[0]
        corr = np.corrcoef(pre[m], y[m])[0, 1]
        k = int(m.sum())
        idx = pre[m].index; cut = idx[len(idx) // 2]
        c1 = np.corrcoef(pre[m][idx < cut], y[m][idx < cut])[0, 1]; c2 = np.corrcoef(pre[m][idx >= cut], y[m][idx >= cut])[0, 1]
        say(f"  час {H:02d}: corr={corr:+.3f} (t={corr * np.sqrt(k - 2) / np.sqrt(1 - corr ** 2):+.2f}; половины {c1:+.3f}/{c2:+.3f}), beta={beta:+.3f}, |ret| часа {y[m].abs().mean() * 1e4:.1f} бп")
    say("  средний |ход| по часам, бп:", " ".join(f"{h:02d}:{r[h * 60].abs().mean() * 1e4:.0f}" for h in range(0, 24) if h * 60 in r.columns))


# ================================================================ main

def main():
    pd.set_option("display.width", 200)
    say("Загрузка данных …")
    ES = load_fut_1h("ES"); NQ = load_fut_1h("NQ")
    BTC = load_btc_5m()
    BTC1h = to_ny(engine.resample(BTC.drop(columns=["date", "clock", "wd"]).tz_convert("UTC").tz_localize(None), "1h"))
    say(f"ES 1h {len(ES)} баров, NQ 1h {len(NQ)}, BTC 5m {len(BTC)} баров {BTC.index[0].date()}…{BTC.index[-1].date()}; {time.time() - T0:.0f} с")

    hdr("A. Профиль объёма по часам NY (утв. 3, 4, 5, 6a). ZB-данных нет: 8:00–9:30 ZB проверяется на ES/NQ как замене; "
        "1h-бары не делят 09:00–09:30 и 09:30–10:00")
    profs = {}
    for name, d in (("ES", ES), ("NQ", NQ), ("BTC", BTC1h)):
        profs[name] = hour_profile(d, name)
    for name, p in profs.items():
        say(f"{name}: доля 08:00–10:00 = {window_share(p, 8, 10):.1f}%, 09:00–10:00 = {window_share(p, 9, 10):.1f}%, 10:00–11:00 = {window_share(p, 10, 11):.1f}%, "
            f"03:00–10:00 = {window_share(p, 3, 10):.1f}%, 10:00–16:00 = {window_share(p, 10, 16):.1f}%, 15:00–16:00 = {window_share(p, 15, 16):.1f}%")
    # BTC: профиль по получасам 08:00–11:00 (единственный ряд с 30-мин разрешением и реальным объёмом)
    b = BTC[(BTC.wd < 5)].copy()
    b["slot"] = (b.clock // 30) * 30
    tot = b.groupby("date").volume.transform("sum")
    sh = (b.volume / tot).groupby(b.slot).mean() * 100
    say("BTC 5m→30m: доля объёма дня по получасам 07:30–11:30 NY: " + " ".join(f"{s // 60:02d}:{s % 60:02d}={v:.2f}%" for s, v in sh.items() if 450 <= s < 690))
    say("BTC: топ-3 получаса дня: " + ", ".join(f"{s // 60:02d}:{s % 60:02d} ({v:.2f}%)" for s, v in sh.sort_values(ascending=False).head(3).items()))

    hdr("B. 10:00–11:00 NY = profit taking / London close → цели после NY open меньше (утв. 6b)")
    for name, d in (("ES", ES), ("NQ", NQ), ("BTC", BTC1h)):
        profit_taking_test(d, name)

    hdr("C. Пробой opening range на лёгком объёме = stop run → разворот (утв. 2 вверх, утв. 10 вниз)")
    say("BTC 5m: OR 09:30–10:00, пробой 10:00–12:00, горизонты 6/12 баров = 30/60 мин; издержки 10 бп круг")
    or_break_test(BTC, "BTC 5m", 570, 600, 600, 720, (6, 12), "×5m")
    say("\nES/NQ 1h: OR = бар 09:00 (09:00–10:00, содержит cash open), пробой барами 10:00/11:00, горизонты 1/3 ч; издержки ~0.4 бп")
    or_break_test(ES, "ES 1h", 540, 600, 600, 720, (1, 3), "ч")
    or_break_test(NQ, "NQ 1h", 540, 600, 600, 720, (1, 3), "ч")

    hdr("D. Ретест лоу дня на меньшем объёме не пробивает → рост (утв. 11)")
    retest_low_test(BTC, "BTC 5m", 570, 960, 3, 0.10, (6, 12), "×5m")
    retest_low_test(ES, "ES 1h", 540, 960, 2, 0.10, (1, 3), "ч")
    retest_low_test(NQ, "NQ 1h", 540, 960, 2, 0.10, (1, 3), "ч")

    hdr("E. Новый хай дня на падающем объёме = слабость → возврат к OR / консолидация (утв. 14)")
    new_high_div_test(BTC, "BTC 5m", 570, 960, 30, 3, (6, 12), "×5m", 570, 600)
    new_high_div_test(ES, "ES 1h", 540, 960, 60, 2, (1, 3), "ч", 540, 600)
    new_high_div_test(NQ, "NQ 1h", 540, 960, 60, 2, (1, 3), "ч", 540, 600)

    hdr("F. Mitigation: возврат в диапазон объёмной свечи против хода → ускорение (утв. 1). Плацебо = та же геометрия на малообъёмных свечах")
    mitigation_test(BTC, "BTC 5m", 36, (6, 12), "×5m")
    mitigation_test(ES, "ES 1h", 12, (1, 3), "ч")
    mitigation_test(NQ, "NQ 1h", 12, (1, 3), "ч")

    hdr("G. Lunch 12:00–13:00 целит buy stops над самым свежим AM-хаем, а не над хаем дня (утв. 13). Плацебо = окна 11, 13, 14")
    say("BTC 5m:")
    lunch_test_btc(BTC)
    lunch_test_fut(ES, "ES"); lunch_test_fut(NQ, "NQ")

    hdr("H. Праздники: лёгкий объём перед праздником, sloppy price action после праздничной недели (утв. 7, 9)")
    for sym in ("ES", "NQ"):
        holiday_tests(sym)
    say("")
    preholiday_intraday(ES, "ES"); preholiday_intraday(NQ, "NQ")
    for sym, d in (("ES", ES), ("NQ", NQ)):
        for day in ("2025-05-26", "2026-05-25"):
            m = d[(d.date == pd.Timestamp(day))].dropna(subset=["volume"])
            mon = d[(d.wd == 0)].dropna(subset=["volume"]).groupby("date").volume.sum()
            say(f"  {sym} Memorial Day {day}: баров {len(m)}, объём {m.volume.sum():.0f} = {m.volume.sum() / mon.median():.2f} медианного понедельника")

    hdr("I. Непроверяемые (нарратив / управление позицией, а не рыночный объём): утв. 8 (partial exits к weekly/monthly целям), "
        "утв. 12 (partial exits выше середины диапазона)")
    say("Это правила управления размером позиции («объём» = объём позиции), не свойства рынка: ожидаемая доходность частичных выходов "
        "равна ожидаемой доходности полного удержания при нулевой направленной альфе (все предыдущие тесты выходов проекта — exits.py). Не проверяется на данных об объёме торгов.")
    say(f"\nГотово за {time.time() - T0:.0f} с")


if __name__ == "__main__":
    main()
