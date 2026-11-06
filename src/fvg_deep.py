"""Глубокие проблемные стороны FVG.

1) Старение: вероятность отскока по номеру касания (1-е, 2-е, 3+).
2) Инверсия: полностью заполненная зона -> возврат с другой стороны ->
   реакция как у уровня противоположной полярности.
3) Фитиль против тела: слом по касанию против слома по закрытию.
4) Премиум/дисконт: позиция зоны в диапазоне последних подтверждённых
   структурных экстремумов; HTF-направление (знак движения за 100 баров).
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


def deep_fvg(o, h, l, c, K=192, M=48):
    n = len(c)
    a = _atr(h, l, c)
    _, px_all, dir_all, conf_all = swmod.causal_swings(h, l, c, k=3.0)
    order = np.argsort(conf_all)
    conf_s, px_s, dir_s = conf_all[order], px_all[order], dir_all[order]
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    j = 0
    chi, clo = np.nan, np.nan
    for t in range(n):
        while j < len(conf_s) and conf_s[j] <= t:
            if dir_s[j] > 0:
                chi = px_s[j]
            else:
                clo = px_s[j]
            j += 1
        last_hi[t], last_lo[t] = chi, clo

    out = {"touch_num": [], "touch_bounce": [], "pd": [], "pd_bounce": [],
           "htf": [], "htf_bounce": [], "wick_vs_body": {"wick": [0, 0], "body": [0, 0]},
           "inv_n": 0, "inv_reject": 0}

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
        if start >= end or not np.isfinite(a[born]) or a[born] <= 0:
            continue

        # --- старение: последовательные касания до полного заполнения
        touches = 0
        filled = False
        i = start
        while i < end and not filled and touches < 3:
            in_zone = (l[i] <= top) if d > 0 else (h[i] >= bottom)
            if not in_zone:
                i += 1
                continue
            touches += 1
            s2, e2 = i, min(i + M, n)
            if d > 0:
                fm, bm = l[s2:e2] <= bottom, c[s2:e2] > top
            else:
                fm, bm = h[s2:e2] >= top, c[s2:e2] < bottom
            fb = np.argmax(fm) if fm.any() else M + 1
            bb = np.argmax(bm) if bm.any() else M + 1
            bounced = bb < fb
            out["touch_num"].append(min(touches, 3))
            out["touch_bounce"].append(bounced)
            if fb <= M:
                filled = True
                # --- фитиль против тела
                if d > 0:
                    wick_break = fm.any()
                    body_break = (c[s2:e2] <= bottom).any()
                else:
                    wick_break = fm.any()
                    body_break = (c[s2:e2] >= top).any()
                key = "body" if body_break else "wick"
                out["wick_vs_body"][key][0] += 1
                out["wick_vs_body"][key][1] += int(bounced)
                break
            i = s2 + max(int(bb), 1)
        if touches == 0:
            continue

        # --- премиум/дисконт (только первое касание)
        if np.isfinite(last_hi[born]) and np.isfinite(last_lo[born]) and \
           last_hi[born] > last_lo[born]:
            pos = (0.5 * (top + bottom) - last_lo[born]) / \
                  (last_hi[born] - last_lo[born])
            discount = pos < 0.5
            good = (d > 0 and discount) or (d < 0 and not discount)
            out["pd"].append(good)
            out["pd_bounce"].append(out["touch_bounce"][-1])

        # --- HTF-направление: знак движения за 100 баров
        if born > 100:
            htf_dir = np.sign(c[born] - c[born - 100])
            if htf_dir != 0:
                aligned = (htf_dir == d)
                out["htf"].append(aligned)
                out["htf_bounce"].append(out["touch_bounce"][-1])

        # --- инверсия: после полного заполнения возврат с другой стороны
        if filled:
            s3 = min(s2 + int(fb) + 1, end - 1)
            for k in range(s3, min(s3 + K, n)):
                rev_touch = (h[k] >= bottom) if d > 0 else (l[k] <= top)
                if rev_touch:
                    out["inv_n"] += 1
                    e4 = min(k + M, n)
                    if d > 0:
                        rej = (c[k:e4] < bottom).any()
                    else:
                        rej = (c[k:e4] > top).any()
                    out["inv_reject"] += int(rej)
                    break

    res = {}
    tn = np.array(out["touch_num"])
    tb = np.array(out["touch_bounce"])
    res["aging"] = {k: float(tb[tn == k].mean()) for k in (1, 2, 3)
                    if (tn == k).sum() > 30}
    res["aging_n"] = {k: int((tn == k).sum()) for k in (1, 2, 3)}
    pdg = np.array(out["pd"])
    pdb = np.array(out["pd_bounce"])
    res["pd_good"] = float(pdb[pdg].mean()) if pdg.any() else np.nan
    res["pd_bad"] = float(pdb[~pdg].mean()) if (~pdg).any() else np.nan
    htfa = np.array(out["htf"])
    htfb = np.array(out["htf_bounce"])
    res["htf_aligned"] = float(htfb[htfa].mean()) if htfa.any() else np.nan
    res["htf_against"] = float(htfb[~htfa].mean()) if (~htfa).any() else np.nan
    wb = out["wick_vs_body"]
    res["wick_bounce"] = wb["wick"][1] / wb["wick"][0] if wb["wick"][0] else np.nan
    res["body_bounce"] = wb["body"][1] / wb["body"][0] if wb["body"][0] else np.nan
    res["inv_n"] = out["inv_n"]
    res["inv_reject_rate"] = (out["inv_reject"] / out["inv_n"]
                              if out["inv_n"] else np.nan)
    return res
