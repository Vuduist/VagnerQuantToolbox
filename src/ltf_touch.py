"""LTF-микроструктура в момент касания свежей FVG-зоны.

Зоны строятся на HTF (15m), первое касание находится, поведение вокруг
касания измеряется на LTF (1m). Группы: отскок против полного пробоя.
Вопрос: отличается ли LTF-картина у отскоков от пробоев?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import engine


def build_touch_events(df_htf, df_ltf, K=96, M=48, W_BEFORE=20, W_AFTER=60):
    """Зоны на HTF -> момент первого касания -> фичи LTF-окна."""
    o, h, l, c = (df_htf[x].to_numpy() for x in ("open", "high", "low", "close"))
    ol, hl, ll, cl = (df_ltf[x].to_numpy() for x in ("open", "high", "low", "close"))
    t_ltf = df_ltf.index.to_numpy()
    t_htf = df_htf.index.to_numpy()
    n = len(c)

    # LTF ATR для нормировки
    a1 = pd.Series(hl - ll).rolling(100).mean().to_numpy()

    rows = []
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:
            d, bottom, top = 1, h[t - 1], l[t + 1]
        elif h[t + 1] < l[t - 1]:
            d, bottom, top = -1, h[t + 1], l[t - 1]
        else:
            continue
        born = t + 1
        start = born + 1
        end = min(start + K, n)
        touch_mask = (l[start:end] <= top) if d > 0 else (h[start:end] >= bottom)
        if not touch_mask.any():
            continue
        t_touch = start + int(np.argmax(touch_mask))

        # исход: отскок vs полное заполнение (окно M баров HTF)
        s2, e2 = t_touch, min(t_touch + M, n)
        if d > 0:
            fill = np.argmax(l[s2:e2] <= bottom) if (l[s2:e2] <= bottom).any() else M + 1
            bounce = np.argmax(c[s2:e2] > top) if (c[s2:e2] > top).any() else M + 1
        else:
            fill = np.argmax(h[s2:e2] >= top) if (h[s2:e2] >= top).any() else M + 1
            bounce = np.argmax(c[s2:e2] < bottom) if (c[s2:e2] < bottom).any() else M + 1
        if fill > M and bounce > M:
            continue
        bounced = bounce < fill

        # LTF-окно вокруг касания
        t0 = t_htf[t_touch]
        i0 = int(np.searchsorted(t_ltf, t0))
        lo_i, hi_i = i0 - W_BEFORE, i0 + W_AFTER
        if lo_i < 100 or hi_i >= len(cl):
            continue
        wh, wl, wc, wo = hl[lo_i:hi_i], ll[lo_i:hi_i], cl[lo_i:hi_i], ol[lo_i:hi_i]
        a = a1[i0]
        if not np.isfinite(a) or a <= 0:
            continue

        rng_l = wh - wl
        rng_full = hl - ll
        med_rng = np.nanmedian(rng_full[max(0, i0 - 200):i0])
        if med_rng <= 0 or not np.isfinite(med_rng):
            continue
        body = np.abs(wc - wo)
        disp = (rng_l > 2 * med_rng) & (body / np.where(rng_l > 0, rng_l, 1) > 0.6)
        dirn = np.sign(wc - wo)

        n_disp_with = int((disp & (dirn == d)).sum())     # LTF-импульсы по ходу зоны
        n_disp_against = int((disp & (dirn == -d)).sum()) # против

        # микро-свип: после касания LTF делает новый лой и закрывается обратно
        after_lo = wl[W_BEFORE:W_BEFORE + 10]
        micro_sweep = bool((after_lo.min() < wl[W_BEFORE - 5:W_BEFORE].min()) and
                           (wc[W_BEFORE + 9] * d > wo[W_BEFORE + 9] * d))

        approach = d * (cl[i0] - cl[i0 - W_BEFORE]) / a   # скорость захода
        touch_depth = (top - wl[W_BEFORE:W_BEFORE + 3].min()) / max(top - bottom, 1e-12) if d > 0 else \
                      (hl[W_BEFORE:W_BEFORE + 3].max() - bottom) / max(top - bottom, 1e-12)
        vol_ratio = float(np.mean(rng_l[W_BEFORE:]) / med_rng)

        rows.append({
            "bounced": bounced, "approach_atr": float(approach),
            "touch_depth": float(touch_depth),
            "n_disp_with": n_disp_with, "n_disp_against": n_disp_against,
            "micro_sweep": micro_sweep, "vol_ratio": vol_ratio,
        })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame, name):
    b, f = df[df.bounced], df[~df.bounced]
    print(f"\n=== {name}: отскоков {len(b)}, пробоев {len(f)} ===")
    print(f"{'фича':16s} {'отскок (мед)':>13s} {'пробой (мед)':>13s} {'d':>6s}")
    for col in ("approach_atr", "touch_depth", "n_disp_with",
                "n_disp_against", "vol_ratio"):
        mb, mf = b[col].median(), f[col].median()
        sd = np.sqrt((b[col].var() + f[col].var()) / 2)
        d_ = (b[col].mean() - f[col].mean()) / sd if sd > 0 else 0
        print(f"{col:16s} {mb:13.2f} {mf:13.2f} {d_:+6.2f}")
    for col in ("micro_sweep",):
        pb, pf = b[col].mean(), f[col].mean()
        print(f"{col:16s} {pb:13.3f} {pf:13.3f} (p)")
