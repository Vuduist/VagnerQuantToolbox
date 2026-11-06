"""Дневки Yahoo со скорректированным закрытием (дивиденды) → data/raw/<имя>_adj_1d.csv: date,close,adjclose."""
import json
import sys
import time
import urllib.request
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
for sym, name in (("TLT", "TLT"), ("IEF", "IEF"), ("SPY", "SPY")):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&period1=946684800&period2={int(time.time())}&events=div"   # range=max отдаёт месячные бары
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    d = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s").normalize(), "close": res["indicators"]["quote"][0]["close"],
                      "adjclose": res["indicators"]["adjclose"][0]["adjclose"]}).dropna()
    d.to_csv(f"data/raw/{name}_adj_1d.csv", index=False)
    print(sym, len(d), d.date.iloc[0].date(), "—", d.date.iloc[-1].date(), file=sys.stderr)
    time.sleep(2)
