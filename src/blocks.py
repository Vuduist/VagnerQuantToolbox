"""ICT-блоки как зоны с жизненным циклом: OB, Rejection, Breaker, Mitigation.

Каузальные определения:
- Order Block: последняя противоположная свеча перед дисплейсментом
  (range>2*средний, тело>=60%). Зона = её [low, high]. Рождается на
  закрытии дисплейсмента.
- Rejection Block: свеча с фитилём >=55% диапазона и диапазоном
  >=1.5*среднего. Зона = фитиль. Медвежья: верхний фитиль [max(o,c), h].
- Breaker: бычий OB, пробитый закрытием ниже своего низа -> инвертированная
  медвежья зона; событие = возврат в неё снизу, замер реакции.
- Mitigation: после подтверждённого свинга (causal_swings) — свеча-
  источник хода в экстремум (последняя противоположная перед финальной
  ногой); возврат в неё -> реакция против хода.
Исходы как у FVG: касание свежей зоны -> отскок vs проход.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import swings as swmod
from fvg_deep2 import _atr, _outcome


def _disp(h, l, c, o):
    rng = h - l
    mean = pd.Series(rng).rolling(200).mean().shift(1).to_numpy()
    body = np.abs(c - o) / np.where(rng > 0, rng, np.nan)
    ev = (rng > 2 * mean) & (body >= 0.6)
    return ev, np.sign(c - o)


def ob_zones(o, h, l, c, K=96, M=48):
    """Order Block -> исход первого касания."""
    n = len(c)
    disp, d = _disp(h, l, c, o)
    res = {"n": 0, "touch": 0, "bounce": 0}
    for t in np.where(disp)[0]:
        dd = d[t]
        if dd == 0:
            continue
        src = None
        for j in range(t - 1, max(t - 6, 0), -1):   # последняя противоположная
            if dd > 0 and c[j] < o[j]:
                src = j
                break
            if dd < 0 and c[j] > o[j]:
                src = j
                break
        if src is None:
            continue
        bottom, top = (l[src], h[src])
        bnc = _outcome(l, h, c, t, bottom, top, int(dd), K, M)
        if bnc is None:
            continue
        res["n"] += 1
        res["touch"] += 1
        res["bounce"] += int(bnc)
    return res


def rejection_zones(o, h, l, c, K=96, M=48):
    """Rejection Block: длинный фитиль -> возврат в зону фитиля."""
    n = len(c)
    rng = h - l
    mean = pd.Series(rng).rolling(200).mean().shift(1).to_numpy()
    up_w = (h - np.maximum(o, c)) / np.where(rng > 0, rng, np.nan)
    dn_w = (np.minimum(o, c) - l) / np.where(rng > 0, rng, np.nan)
    res = {"n": 0, "touch": 0, "bounce": 0}
    for t in range(1, n - 2):
        if not np.isfinite(mean[t]) or rng[t] < 1.5 * mean[t]:
            continue
        if up_w[t] >= 0.55:          # медвежья зона (верхний фитиль)
            bottom, top, d = max(o[t], c[t]), h[t], -1
        elif dn_w[t] >= 0.55:        # бычья (нижний фитиль)
            bottom, top, d = l[t], min(o[t], c[t]), 1
        else:
            continue
        bnc = _outcome(l, h, c, t, bottom, top, d, K, M)
        if bnc is None:
            continue
        res["n"] += 1
        res["touch"] += 1
        res["bounce"] += int(bnc)
    return res


def breaker_events(o, h, l, c, M=48, K2=192):
    """Breaker: бычий OB пробит закрытием -> возврат снизу -> реакция."""
    n = len(c)
    disp, d = _disp(h, l, c, o)
    res = {"n": 0, "reject": 0}
    for t in np.where(disp)[0]:
        if d[t] <= 0:
            continue
        src = None
        for j in range(t - 1, max(t - 6, 0), -1):
            if c[j] < o[j]:
                src = j
                break
        if src is None:
            continue
        ob_lo = l[src]
        # ждём пробоя закрытием ниже низа OB
        viol = None
        for k in range(t + 1, min(t + K2, n)):
            if c[k] < ob_lo:
                viol = k
                break
        if viol is None:
            continue
        # возврат снизу в зону OB [ob_lo, h[src]]
        for k in range(viol + 1, min(viol + K2, n)):
            if h[k] >= ob_lo:
                e = min(k + M, n)
                cont_up = (c[k:e] > h[src]).any()      # проход насквозь
                rej_dn = (c[k:e] < ob_lo).any()        # отбой вниз
                if cont_up or rej_dn:
                    res["n"] += 1
                    res["reject"] += int(rej_dn and not cont_up
                                         if rej_dn != cont_up else rej_dn)
                break
    return res


def mitigation_events(o, h, l, c, M=48, K2=192):
    """Mitigation: после подтверждённого медвежьего свинга (хай) —
    возврат в последнюю бычью свечу перед ним -> реакция вниз."""
    n = len(c)
    sw_idx, sw_px, sw_dir, sw_conf = swmod.causal_swings(h, l, c, k=3.0)
    res = {"n": 0, "reject": 0}
    for si, px, sd, cf in zip(sw_idx, sw_px, sw_dir, sw_conf):
        if sd != 1:            # только подтверждённые хаи (медвежий контекст)
            continue
        src = None
        for j in range(si, max(si - 8, 0), -1):
            if c[j] > o[j]:
                src = j
                break
        if src is None:
            continue
        bottom, top = o[src], c[src]      # тело свечи-источника
        for k in range(cf, min(cf + K2, n)):
            if l[k] <= top and h[k] >= bottom:   # возврат в тело
                e = min(k + M, n)
                rej = (c[k:e] < bottom).any()
                thru = (c[k:e] > px).any()
                if rej or thru:
                    res["n"] += 1
                    res["reject"] += int(rej and (not thru or
                        np.argmax(c[k:e] < bottom) < np.argmax(c[k:e] > px)))
                break
    return res
