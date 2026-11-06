"""COT legacy (futures only) с CFTC: deacot<year>.zip → data/raw/cot_legacy.csv
(market, date, oi, nc_long, nc_short, comm_long, comm_short, comm_net, nc_net).

Запуск: PYTHONPATH=src.venv/bin/python src/fetch_cot.py 2000 2026
"""

import io
import sys
import time
import zipfile

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}
MARKETS = {
    "EUR": ("EURO FX - CHICAGO MERCANTILE EXCHANGE",),
    "GBP": ("BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE", "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE"),
    "JPY": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",),
    "GOLD": ("GOLD - COMMODITY EXCHANGE INC.",),
    "ES": ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE"),
}
COLS = {"Market and Exchange Names": "name", "As of Date in Form YYYY-MM-DD": "date",
        "Open Interest (All)": "oi", "Noncommercial Positions-Long (All)": "nc_long",
        "Noncommercial Positions-Short (All)": "nc_short", "Commercial Positions-Long (All)": "comm_long",
        "Commercial Positions-Short (All)": "comm_short"}


def fetch_year(year: int) -> pd.DataFrame:
    r = requests.get(f"https://www.cftc.gov/files/dea/history/deacot{year}.zip", headers=UA, timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), encoding="latin1", low_memory=False)
    df = df[list(COLS)].rename(columns=COLS)
    df["name"] = df["name"].str.strip()
    out = []
    for key, names in MARKETS.items():
        sub = df[df.name.isin(names)].copy()
        sub["market"] = key
        out.append(sub)
    return pd.concat(out)


if __name__ == "__main__":
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    frames = []
    for y in range(y0, y1 + 1):
        try:
            f = fetch_year(y); frames.append(f); print(y, len(f), "строк")
        except Exception as e:
            print(y, "ошибка:", e)
        time.sleep(0.5)
    cot = pd.concat(frames)
    cot["date"] = pd.to_datetime(cot["date"])
    for c in ("oi", "nc_long", "nc_short", "comm_long", "comm_short"):
        cot[c] = pd.to_numeric(cot[c], errors="coerce")
    cot["comm_net"] = cot.comm_long - cot.comm_short
    cot["nc_net"] = cot.nc_long - cot.nc_short
    cot = cot.sort_values(["market", "date"]).drop_duplicates(["market", "date"])
    cot[["market", "date", "name", "oi", "nc_long", "nc_short", "comm_long", "comm_short", "comm_net", "nc_net"]].to_csv(
        "data/raw/cot_legacy.csv", index=False)
    print(cot.groupby("market").date.agg(["min", "max", "count"]))
