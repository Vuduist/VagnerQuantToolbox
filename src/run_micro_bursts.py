"""Каскады агрессивных сделок BTCUSDT perp по всем скачанным дням; разметка по изменению открытого интереса. out/micro_bursts.csv"""
import os
import sys
import numpy as np
import pandas as pd
import micro_bursts as mb

rows = []
files = sorted(f for f in os.listdir("data/raw/binance_um/aggTrades") if f.endswith(".zip"))
for f in files:
    day = f.split("aggTrades-")[1][:10]
    try:
        s = mb.day_seconds(os.path.join("data/raw/binance_um/aggTrades", f))
    except Exception as e:
        print(f, "ошибка", e, file=sys.stderr); continue
    b = mb.bursts(s)
    mz = f"data/raw/binance_um/metrics/BTCUSDT-metrics-{day}.zip"
    if len(b) and os.path.exists(mz):
        doi = mb.oi_change(mz)
        end = ((b.t // 300) + 1) * 300
        b["doi"] = doi.reindex(end).to_numpy()
    b["day"] = day
    rows.append(b)
B = pd.concat(rows, ignore_index=True)
B.to_csv("out/micro_bursts.csv", index=False)
print(f"дней {len(files)}, всплесков {len(B)}", file=sys.stderr)


def show(x, nm):
    s = f"{nm:40s} n={len(x):4d}"
    for h in mb.HZ:
        v = x[f"f{h}"]; s += f" | {h:3d}с {v.mean():+6.2f} бп (t {v.mean() / v.std() * np.sqrt(len(v)):+.1f})"
    print(s, file=sys.stderr)


show(B, "все всплески (в сторону всплеска)")
show(B[B.side > 0], "  покупки")
show(B[B.side < 0], "  продажи")
q = B.doi.quantile([0.2, 0.8])
show(B[B.doi <= q.iloc[0]], f"OI упал сильнее всего (≤ {q.iloc[0] * 100:.2f}%)")
show(B[B.doi < 0], "OI упал")
show(B[B.doi > 0], "OI вырос")
show(B[B.doi >= q.iloc[1]], f"OI вырос сильнее всего (≥ {q.iloc[1] * 100:.2f}%)")
mq = B.move_bp.quantile(0.8)
show(B[B.move_bp >= mq], f"самые сильные движения (≥ {mq:.0f} бп)")
show(B[(B.move_bp >= mq) & (B.doi < 0)], "  … и OI упал (похоже на ликвидации)")
