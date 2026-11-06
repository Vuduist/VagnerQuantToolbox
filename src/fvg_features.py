"""Объединённые фичи FVG-зоны: LTF-микроструктура + контекст + геометрия.

Для классификатора «хороший FVG»: label = отскок после первого касания
(до полного заполнения). Все фичи известны на момент касания (каузально).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod

FEATURES = ["approach_atr", "touch_depth", "n_disp_with", "n_disp_against",
            "vol_ratio", "micro_sweep", "zone_width_atr", "disp_strength",
            "er20", "dist_level_atr"]


def _atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def _er(c, W=20):
    net = np.abs(c[W:] - c[:-W])
    path = pd.Series(np.abs(np.diff(c))).rolling(W).sum().to_numpy()[W - 1:]
    return np.concatenate([np.full(W, np.nan), net / np.where(path > 0, path, np.nan)])


def build(df_htf, df_ltf, K=96, M=48, WB=20, WA=60):
    o, h, l, c = (df_htf[x].to_numpy() for x in ("open", "high", "low", "close"))
    ol, hl, ll, cl = (df_ltf[x].to_numpy() for x in ("open", "high", "low", "close"))
    t_ltf = df_ltf.index.to_numpy()
    t_htf = df_htf.index.to_numpy()
    n = len(c)

    a_htf = _atr(h, l, c)
    er = _er(c)
    a1 = pd.Series(hl - ll).rolling(100).mean().to_numpy()
    rng_full = hl - ll

    _, px_all, _, conf_all = swmod.causal_swings(h, l, c, k=3.0)
    order = np.argsort(conf_all)
    conf_s, px_s = conf_all[order], px_all[order]
    last_px = np.full(n, np.nan)
    j = 0
    cur = np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            cur = px_s[j]
            j += 1
        last_px[t] = cur

    rows = []
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:
            d, bottom, top = 1, h[t - 1], l[t + 1]
        elif h[t + 1] < l[t - 1]:
            d, bottom, top = -1, h[t + 1], l[t - 1]
        else:
            continue
        born = t + 1
        if not (np.isfinite(a_htf[born]) and a_htf[born] > 0
                and np.isfinite(er[born])):
            continue
        start = born + 1
        end = min(start + K, n)
        tm = (l[start:end] <= top) if d > 0 else (h[start:end] >= bottom)
        if not tm.any():
            continue
        t_touch = start + int(np.argmax(tm))

        s2, e2 = t_touch, min(t_touch + M, n)
        if d > 0:
            fm, bm = l[s2:e2] <= bottom, c[s2:e2] > top
        else:
            fm, bm = h[s2:e2] >= top, c[s2:e2] < bottom
        fb = np.argmax(fm) if fm.any() else M + 1
        bb = np.argmax(bm) if bm.any() else M + 1
        if fb > M and bb > M:
            continue
        bounced = bool(bb < fb)

        i0 = int(np.searchsorted(t_ltf, t_htf[t_touch]))
        lo_i, hi_i = i0 - WB, i0 + WA
        if lo_i < 100 or hi_i >= len(cl):
            continue
        wh, wl, wc, wo = hl[lo_i:hi_i], ll[lo_i:hi_i], cl[lo_i:hi_i], ol[lo_i:hi_i]
        a = a1[i0]
        med_rng = np.nanmedian(rng_full[max(0, i0 - 200):i0])
        if not np.isfinite(a) or a <= 0 or not np.isfinite(med_rng) or med_rng <= 0:
            continue

        rng_l = wh - wl
        body = np.abs(wc - wo)
        disp = (rng_l > 2 * med_rng) & (body / np.where(rng_l > 0, rng_l, 1) > 0.6)
        dirn = np.sign(wc - wo)
        after_lo = wl[WB:WB + 10]
        micro_sweep = float((after_lo.min() < wl[WB - 5:WB].min()) and
                            (wc[WB + 9] * d > wo[WB + 9] * d))

        rows.append({
            "t": t_htf[t_touch], "bounced": bounced,
            "approach_atr": float(d * (cl[i0] - cl[i0 - WB]) / a),
            "touch_depth": float(((top - wl[WB:WB + 3].min()) if d > 0
                                  else (hl[WB:WB + 3].max() - bottom))
                                 / max(top - bottom, 1e-12)),
            "n_disp_with": float((disp & (dirn == d)).sum()),
            "n_disp_against": float((disp & (dirn == -d)).sum()),
            "vol_ratio": float(np.mean(rng_l[WB:]) / med_rng),
            "micro_sweep": micro_sweep,
            "zone_width_atr": float((top - bottom) / a_htf[born]),
            "disp_strength": float((h[t] - l[t]) / a_htf[t]),
            "er20": float(er[born]),
            "dist_level_atr": float(abs(c[born] - last_px[born]) / a_htf[born])
                              if np.isfinite(last_px[born]) else np.nan,
        })
    return pd.DataFrame(rows).dropna()
