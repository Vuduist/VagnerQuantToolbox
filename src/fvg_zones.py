"""FVG как зоны с жизненным циклом (не статический паттерн).

Зона бычья на баре t: [high[t-1], low[t+1]], известна с закрытия t+1.
Жизненный цикл с рождения:
  fresh -> touched (low <= верх зоны) -> half (<= 50%) -> filled (<= низ)
Ключевые метрики по СВЕЖИМ зонам (ещё не тронутым):
  - время до первого касания;
  - после первого касания: отскок (закрытие обратно выше верха зоны до
    полного заполнения) против пробоя (полное заполнение).
"""

from __future__ import annotations

import numpy as np


def fvg_zones_lifecycle(o, h, l, c, K=96, M=48):
    n = len(c)
    bull_lo = np.where(np.arange(n) >= 1, h[np.maximum(np.arange(n) - 1, 0)], np.nan)  # низ зоны бычьей = high[t-1]
    zones = []
    for t in range(1, n - 2):
        if l[t + 1] > h[t - 1]:   # бычий FVG
            zones.append((t + 1, h[t - 1], l[t + 1], 1))   # born, bottom, top
        elif h[t + 1] < l[t - 1]:  # медвежий
            zones.append((t + 1, h[t + 1], l[t - 1], -1))  # bottom, top
    res = {"n": len(zones), "touched": 0, "t_first": [], "bounce": 0,
           "filled_no_bounce": 0, "stale": 0}
    for born, bottom, top, d in zones:
        start = born + 1
        end = min(start + K, n)
        if start >= n:
            continue
        path_lo = l[start:end] if d > 0 else None
        path_hi = h[start:end] if d < 0 else None
        touch_mask = (path_lo <= top) if d > 0 else (path_hi >= bottom)
        if not touch_mask.any():
            res["stale"] += 1
            continue
        res["touched"] += 1
        t_touch = int(np.argmax(touch_mask))
        res["t_first"].append(t_touch)
        # после касания: отскок против полного заполнения, окно M
        s2 = start + t_touch
        e2 = min(s2 + M, n)
        if d > 0:
            fill_m = l[s2:e2] <= bottom
            bounce_m = c[s2:e2] > top
        else:
            fill_m = h[s2:e2] >= top
            bounce_m = c[s2:e2] < bottom
        fb = np.argmax(fill_m) if fill_m.any() else M + 1
        bb = np.argmax(bounce_m) if bounce_m.any() else M + 1
        if bb < fb:
            res["bounce"] += 1
        elif fb <= bb and fill_m.any():
            res["filled_no_bounce"] += 1
    t = res.pop("t_first")
    res["t_first_median"] = float(np.median(t)) if t else np.nan
    res["p_bounce_given_touch"] = (res["bounce"] / res["touched"]
                                   if res["touched"] else np.nan)
    res["p_touch"] = res["touched"] / res["n"] if res["n"] else np.nan
    return res
