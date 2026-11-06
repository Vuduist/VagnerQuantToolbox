"""Загрузка дневок/часовиков с Yahoo chart API в формат data/raw/<name>.csv
(open_time_ms, open, high, low, close, volume) — как NQ_1d.csv.

Запуск: PYTHONPATH=src.venv/bin/python src/fetch_yahoo.py DX-Y.NYB DXY_1d 1d max
"""

import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(symbol: str, interval: str = "1d", rng: str = "max") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": interval, "range": rng}, headers=UA, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"t": [int(x) * 1000 for x in ts], "open": q["open"], "high": q["high"],
                       "low": q["low"], "close": q["close"], "volume": q["volume"]})
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0)
    return df


if __name__ == "__main__":
    symbol, name = sys.argv[1], sys.argv[2]
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
    rng = sys.argv[4] if len(sys.argv) > 4 else "max"
    df = fetch(symbol, interval, rng)
    df.to_csv(f"data/raw/{name}.csv", header=False, index=False)
    print(name, len(df), pd.to_datetime(df.t.iloc[0], unit="ms"), "→", pd.to_datetime(df.t.iloc[-1], unit="ms"))
    time.sleep(1)
