"""Прогон пяти ICT-моделей (ict_models.py) на минутках HistData → out/ict_models_trades.csv, out/ict_models.txt."""

import sys

import numpy as np
import pandas as pd

import ict_models as im

# издержки в пунктах цены за круг (2 тика фьючерса + комиссия; FX 0.8 пипса; золото 0.30 $)
MARKETS = [("NQ", "data/raw/NQH_M1.csv", 0.5, False), ("SPX", "data/raw/SPH_M1.csv", 0.5, False),
           ("EURUSD", "data/raw/EURH_M1.csv", 0.00008, True), ("XAUUSD", "data/raw/XAUH_M1.csv", 0.30, False)]


def main():
    allt = []
    lines = []
    for mk, path, cost, fx in MARKETS:
        try:
            res, nd = im.run_market(path, cost, fx=fx)
        except FileNotFoundError:
            print(mk, "нет файла", file=sys.stderr); continue
        for k, v in res.items():
            if len(v) == 0:
                continue
            v = v.assign(model=k, market=mk)
            allt.append(v)
            y = v.groupby(v.date.dt.year).R.sum()
            t = v.R.mean() / v.R.std() * np.sqrt(len(v)) if len(v) > 2 else np.nan
            line = (f"{k:6s} {mk:7s} дней {nd:5d} n={len(v):5d} win={(v.R > 0).mean():.2f} meanR={v.R.mean():+.3f} "
                    f"(gross {v.R_gross.mean():+.3f}) t={t:+.1f} risk%={v.risk_pct.median():.3f} "
                    f"лет+ {int((y > 0).sum())}/{len(y)}")
            print(line, file=sys.stderr); lines.append(line)
    out = pd.concat(allt, ignore_index=True)
    out.to_csv("out/ict_models_trades.csv", index=False)
    open("out/ict_models.txt", "w").write("\n".join(lines) + "\n")
    print("записано out/ict_models_trades.csv", len(out), file=sys.stderr)


if __name__ == "__main__":
    main()
