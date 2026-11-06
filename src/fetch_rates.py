"""Ставки ЦБ из BIS WS_CBPOL (bulk zip, дневные) → data/raw/cb_rates.csv (daily, ffill):
USD, EUR (XM), GBP, JPY, AUD, NZD, CAD, CHF.
Источник: https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip (FRED таймаутит).
"""

import io
import sys
import zipfile

import pandas as pd

ZIP = sys.argv[1] if len(sys.argv) > 1 else "data/raw/bis_cbpol.zip"
AREAS = {"US": "USD", "XM": "EUR", "GB": "GBP", "JP": "JPY", "AU": "AUD", "NZ": "NZD", "CA": "CAD", "CH": "CHF"}

z = zipfile.ZipFile(ZIP)
df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), low_memory=False)
df = df.rename(columns={"FREQ:Frequency": "freq", "REF_AREA:Reference area": "area",
                        "TIME_PERIOD:Time period or range": "t", "OBS_VALUE:Observation Value": "v"})
df["code"] = df.area.str.split(":").str[0]
d = df[(df.freq.str.startswith("D")) & df.code.isin(AREAS)]
out = {}
for code, cur in AREAS.items():
    s = d[d.code == code].copy()
    s["t"] = pd.to_datetime(s.t); s = s.set_index("t").v.astype(float).sort_index()
    out[cur] = s[~s.index.duplicated()]
tab = pd.concat(out, axis=1).sort_index().ffill()
tab.to_csv("data/raw/cb_rates.csv")
print(tab.apply(lambda c: f"{c.first_valid_index().date()}→{c.last_valid_index().date()} ({c.iloc[-1]:.2f}%)").to_string())
print("дней:", len(tab))
