"""Дневки Dow 30 (текущий состав, survivorship!) и ^DJI с Yahoo → data/raw/stocks/<TICKER>_1d.csv

Запуск: PYTHONPATH=src.venv/bin/python src/fetch_stocks.py
"""

import os
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}
DOW30 = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM",
         "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"]
os.makedirs("data/raw/stocks", exist_ok=True)


def fetch(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"interval": "1d", "period1": 946684800, "period2": int(time.time())}, headers=UA, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({"t": [int(x) * 1000 for x in res["timestamp"]], "open": q["open"], "high": q["high"], "low": q["low"],
                       "close": q["close"], "volume": q["volume"]})
    if adj is not None:
        df["adj"] = adj
    df = df.dropna(subset=["open", "high", "low", "close"]); df["volume"] = df["volume"].fillna(0)
    return df


if __name__ == "__main__":
    for sym in DOW30 + ["^DJI"]:
        name = sym.replace("^", "")
        for attempt in range(3):
            try:
                df = fetch(sym); df.to_csv(f"data/raw/stocks/{name}_1d.csv", header=False, index=False)
                print(name, len(df), pd.to_datetime(df.t.iloc[0], unit="ms").date(), "→", pd.to_datetime(df.t.iloc[-1], unit="ms").date()); break
            except Exception as e:
                print(name, "попытка", attempt, str(e)[:60]); time.sleep(3)
        time.sleep(0.6)
