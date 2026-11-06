"""B01 Propulsion block и F01 Volume imbalance — зоны с жизненным циклом по конвейеру blocks.py.

Propulsion (бычий): есть бычий OB (последняя down-свеча перед дисплейсментом вверх); позже
цена вернулась в OB и down-свеча ЗАКРЫЛАСЬ внутри OB (это propulsion-свеча); зона = её
[low, high]; событие = возврат в propulsion-зону -> отскок vs проход (_outcome).
Контроль: обычный OB-ретест того же OB.

Volume imbalance (бычий): тела соседних свечей не перекрываются (open[t] > close[t-1]), но
фитили перекрываются (low[t] <= high[t-1]); зона = [close[t-1], open[t]], |зона| ≥ 0.1·ATR.
Событие = возврат в зону -> отскок vs проход. Контроль: 3-свечной FVG (ранее: 0.371).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blocks import _disp
from fvg_deep2 import _atr, _outcome


def propulsion_stats(df: pd.DataFrame, K=96, M=48, K2=192) -> dict:
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    disp, d = _disp(h, l, c, o)
    res = {"n_ob": 0.0, "n_prop": 0.0, "ob_bounce": 0.0, "prop_bounce": 0.0, "n_ob_touch": 0.0, "n_prop_touch": 0.0}
    for t in np.flatnonzero(disp):
        dd = d[t]
        if dd == 0:
            continue
        src = None
        for j in range(t - 1, max(t - 6, 0), -1):
            if (dd > 0 and c[j] < o[j]) or (dd < 0 and c[j] > o[j]):
                src = j
                break
        if src is None:
            continue
        bottom, top = l[src], h[src]
        res["n_ob"] += 1
        # обычный OB-ретест
        bnc = _outcome(l, h, c, t, bottom, top, int(dd), K, M)
        if bnc is not None:
            res["n_ob_touch"] += 1
            res["ob_bounce"] += int(bnc)
        # propulsion: противоположная свеча, закрывшаяся внутри OB после ухода цены
        away = False
        for k in range(t + 1, min(t + K2, n)):
            if dd > 0:
                if c[k] > top + (top - bottom):      # цена ушла выше на высоту блока
                    away = True
                if away and c[k] < o[k] and bottom <= c[k] <= top:
                    pb, pt = l[k], h[k]
                    lv = np.flatnonzero(c[k + 1:min(k + K2, n)] > pt)      # сначала уход выше зоны
                    pbnc = _outcome(l, h, c, k + 1 + lv[0], pb, pt, 1, K, M) if len(lv) else None
                    res["n_prop"] += 1
                    if pbnc is not None:
                        res["n_prop_touch"] += 1
                        res["prop_bounce"] += int(pbnc)
                    break
                if c[k] < bottom:
                    break
            else:
                if c[k] < bottom - (top - bottom):
                    away = True
                if away and c[k] > o[k] and bottom <= c[k] <= top:
                    pb, pt = l[k], h[k]
                    lv = np.flatnonzero(c[k + 1:min(k + K2, n)] < pb)
                    pbnc = _outcome(l, h, c, k + 1 + lv[0], pb, pt, -1, K, M) if len(lv) else None
                    res["n_prop"] += 1
                    if pbnc is not None:
                        res["n_prop_touch"] += 1
                        res["prop_bounce"] += int(pbnc)
                    break
                if c[k] > top:
                    break
    res["p_bounce_ob"] = res["ob_bounce"] / res["n_ob_touch"] if res["n_ob_touch"] > 30 else np.nan
    res["p_bounce_prop"] = res["prop_bounce"] / res["n_prop_touch"] if res["n_prop_touch"] > 30 else np.nan
    return res


def volume_imbalance_stats(df: pd.DataFrame, K=96, M=48, min_atr=0.1) -> dict:
    """Отскок от свежей зоны по корзинам ширины (в ATR): VI vs 3-свечной FVG."""
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    a = _atr(h, l, c, 100)
    n = len(c)
    bins = ((0.1, 0.3), (0.3, 0.6), (0.6, 9.9))
    cnt = {f"{kind}_{lo}_{hi}": [0, 0] for kind in ("vi", "fvg") for lo, hi in bins}

    def add(kind, width, bnc):
        for lo, hi in bins:
            if lo <= width < hi:
                key = f"{kind}_{lo}_{hi}"
                cnt[key][0] += 1
                cnt[key][1] += int(bnc)

    for t in range(1, n - 2):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        if o[t] > c[t - 1] and l[t] <= h[t - 1] and (o[t] - c[t - 1]) >= min_atr * a[t]:
            # уход выше зоны, затем ретест
            lv = np.flatnonzero(c[t:min(t + K, n)] > o[t])
            bnc = _outcome(l, h, c, t + lv[0], c[t - 1], o[t], 1, K, M) if len(lv) else None
            if bnc is not None:
                add("vi", (o[t] - c[t - 1]) / a[t], bnc)
        elif o[t] < c[t - 1] and h[t] >= l[t - 1] and (c[t - 1] - o[t]) >= min_atr * a[t]:
            lv = np.flatnonzero(c[t:min(t + K, n)] < o[t])
            bnc = _outcome(l, h, c, t + lv[0], o[t], c[t - 1], -1, K, M) if len(lv) else None
            if bnc is not None:
                add("vi", (c[t - 1] - o[t]) / a[t], bnc)
        if l[t + 1] > h[t - 1] and (l[t + 1] - h[t - 1]) >= min_atr * a[t]:
            bnc = _outcome(l, h, c, t + 1, h[t - 1], l[t + 1], 1, K, M)
            if bnc is not None:
                add("fvg", (l[t + 1] - h[t - 1]) / a[t], bnc)
        elif h[t + 1] < l[t - 1] and (l[t - 1] - h[t + 1]) >= min_atr * a[t]:
            bnc = _outcome(l, h, c, t + 1, h[t + 1], l[t - 1], -1, K, M)
            if bnc is not None:
                add("fvg", (l[t - 1] - h[t + 1]) / a[t], bnc)
    res = {}
    for key, (nn, b) in cnt.items():
        res[f"n_{key}"] = float(nn)
        res[f"p_bounce_{key}"] = b / nn if nn > 30 else np.nan
    return res
