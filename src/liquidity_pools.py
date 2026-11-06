"""Equal highs/lows (пулы ликвидности) — последний примитив ICT.

Каузально: два ПОДТВЕРЖДЁННЫХ структурных хая в пределах 0.25*ATR друг
от друга, разнесённых >=10 баров -> пул известен после подтверждения
второго. Событие: снятие пула (прокол уровня).
Контрольная группа: снятие ОДИНОЧНЫХ хаёв (соседние хаи далеки).
Гипотеза ICT: снятие пула ведёт себя иначе (ускорение/реверс).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod


def _atr(h, l, c, n=100):
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def pool_sweep_events(o, h, l, c, tol_atr=0.75, min_gap=6, max_gap=300, K=24):
    """Пул = два подтверждённых односторонних свинга в пределах tol*ATR,
    уровень первого НЕ снимался между его подтверждением и подтверждением
    второго (чистый двойной экстремум). Событие: первое снятие уровня
    max(px1,px2) после подтверждения второго. Контроль: одиночные свинги
    (без партнёра в tol)."""
    n = len(c)
    a = _atr(h, l, c)
    sw_idx, sw_px, sw_dir, sw_conf = swmod.causal_swings(h, l, c, k=3.0)

    out = {"pool_hi": [], "single_hi": [], "pool_lo": [], "single_lo": []}
    for side, kp, ks in ((1, "pool_hi", "single_hi"), (-1, "pool_lo", "single_lo")):
        sel = [(i, px, cf) for i, px, sd, cf in zip(sw_idx, sw_px, sw_dir, sw_conf)
               if sd == side]
        sel.sort(key=lambda x: x[2])
        used_partners = set()
        for j, (idx2, px2, cf2) in enumerate(sel):
            # партнёр: предыдущий свинг той же стороны в пределах tol
            partner = None
            for idx1, px1, cf1 in reversed(sel[:j]):
                if not (min_gap <= idx2 - idx1 <= max_gap):
                    continue
                if not np.isfinite(a[cf2]) or a[cf2] <= 0:
                    continue
                if abs(px1 - px2) > tol_atr * a[cf2]:
                    continue
                # чистота: уровень не снимался между conf1 и conf2
                if side > 0:
                    clean = not (h[cf1:cf2 + 1] > max(px1, px2)).any()
                else:
                    clean = not (l[cf1:cf2 + 1] < min(px1, px2)).any()
                if clean:
                    partner = idx1
                    break
            # событие: первое снятие после подтверждения второго
            lvl = max(px1, px2) if (partner and side > 0) else \
                  (min(px1, px2) if partner else (px2 if side > 0 else px2))
            for t in range(cf2 + 1, n):
                hit = h[t] > lvl if side > 0 else l[t] < lvl
                if hit:
                    if t + K < n:
                        out[kp if partner else ks].append(t)
                    break
    return out


def fwd_stats(c, idx, d, horizons=(4, 12, 24)):
    n = len(c)
    res = {}
    for H in horizons:
        ok = np.array([i + H < n for i in idx])
        i2 = np.array(idx)[ok]
        if len(i2) < 30:
            continue
        r = d * (c[i2 + H] - c[i2]) / c[i2]
        res[H] = {"n": len(i2), "p_same": float(np.mean(r > 0)),
                  "mean_bp": float(np.mean(r) * 1e4)}
    return res
