"""Мейкерский фейд каскада. После конца всплеска (сек. j, цена P) — лимит против всплеска на P·(1 + side·δ)
(δ бп дальше по ходу всплеска), живёт 30 с; исполнение — только если последняя цена секунды прошла уровень строго.
Выход через 300 с от исполнения по рынку (тейкер). Комиссии: базовый уровень Binance (мейкер 2, тейкер 5 бп) и
продвинутый (мейкер 0, тейкер 3 бп). Отдельно — все всплески и сильные (движение ≥ 80-го перцентиля). out/micro_fade.csv"""
import os
import sys
import numpy as np
import pandas as pd
import micro_bursts as mb

FEES = {"базовый 2/5": (2.0, 5.0), "продвинутый 0/3": (0.0, 3.0)}
rows = []
for f in sorted(x for x in os.listdir("data/raw/binance_um/aggTrades") if x.endswith(".zip")):
    s = mb.day_seconds(os.path.join("data/raw/binance_um/aggTrades", f))
    b = mb.bursts(s)
    if not len(b):
        continue
    lp = np.log(s.p.to_numpy()); t0 = int(s.index[0]); n = len(lp)
    for _, e in b.iterrows():
        j = int(e.t - t0) + int(e.len_s)
        if j >= n - 400:
            continue
        side = e.side
        for dl in (0.0, 2.0, 5.0):
            lvl = lp[j] + side * dl * 1e-4
            w = lp[j + 1:j + 31]
            hit = np.flatnonzero(w > lvl) if side > 0 else np.flatnonzero(w < lvl)
            if not len(hit):
                rows.append({"day": f[18:28], "move": e.move_bp, "delta": dl, "filled": False}); continue
            k = j + 1 + hit[0]
            gross = -side * (lp[min(n - 1, k + 300)] - lvl) * 1e4     # фейд: против всплеска
            rows.append({"day": f[18:28], "move": e.move_bp, "delta": dl, "filled": True, "gross": gross})
R = pd.DataFrame(rows)
R.to_csv("out/micro_fade.csv", index=False)
strong = R.move >= R.move.quantile(0.8)
for nm, m in (("все всплески", R.move.notna()), ("сильные (≥80-й перцентиль)", strong)):
    for dl in (0.0, 2.0, 5.0):
        x = R[m & (R.delta == dl)]
        fl = x[x.filled]
        s = f"{nm:27s} δ={dl:.0f} бп: исполнено {len(fl)}/{len(x)} ({len(fl) / max(1, len(x)):.0%}), валово {fl.gross.mean():+6.2f} бп (t {fl.gross.mean() / fl.gross.std() * np.sqrt(len(fl)):+.1f})"
        for fn, (mk, tk) in FEES.items():
            s += f" | {fn}: {fl.gross.mean() - mk - tk:+6.2f}"
        print(s, file=sys.stderr)
