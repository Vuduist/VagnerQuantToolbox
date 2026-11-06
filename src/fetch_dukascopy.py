"""Загрузка минутных свечей Dukascopy (бесплатный datafeed) в data/raw/<NAME>_M1.csv.

Файл дня: https://datafeed.dukascopy.com/datafeed/<SYM>/<YYYY>/<MM-1>/<DD>/BID_candles_min_1.bi5
(LZMA, записи по 24 байта big-endian: смещение сек от 00:00 UTC, open, close, low, high (uint32 × 1/scale), volume float32).
Кэш сырых файлов: data/raw/duka/<SYM>/<YYYY-MM-DD>.bi5. Бары с нулевым объёмом выбрасываются (выходные/праздники).
Время в CSV — UTC, миллисекунды (engine.load_csv).

Запуск: PYTHONPATH=src.venv/bin/python -I src/fetch_dukascopy.py USATECHIDXUSD NQD 2013-01-01 2026-10-16 1000
"""

import datetime as dt
import lzma
import os
import struct
import sys
import time

import requests

UA = {"User-Agent": "Mozilla/5.0"}
URL = "https://datafeed.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def get_day(sym, day, cache_dir, pause=0.35):
    path = os.path.join(cache_dir, f"{day:%Y-%m-%d}.bi5")
    if os.path.exists(path):
        return open(path, "rb").read()
    url = URL.format(s=sym, y=day.year, m=day.month - 1, d=day.day)
    for k in range(8):
        try:
            r = requests.get(url, headers=UA, timeout=30)
        except requests.RequestException:
            time.sleep(5 * (k + 1)); continue
        if r.status_code == 200:
            open(path, "wb").write(r.content)
            time.sleep(pause)
            return r.content
        if r.status_code == 404:
            open(path, "wb").write(b"")
            return b""
        time.sleep(10 * (k + 1))          # 503 = троттлинг
    raise RuntimeError(f"{sym} {day} не скачан")


def decode(raw, day, scale):
    if not raw:
        return []
    b = lzma.decompress(raw)
    t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()) * 1000
    out = []
    for i in range(len(b) // 24):
        sec, o, c, lo, hi, v = struct.unpack(">IIIIIf", b[i * 24:(i + 1) * 24])
        if v > 0:
            out.append((t0 + sec * 1000, o / scale, hi / scale, lo / scale, c / scale, v))
    return out


def main():
    sym, name, d0, d1 = sys.argv[1:5]
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1000.0
    cache = os.path.join("data/raw/duka", sym)
    os.makedirs(cache, exist_ok=True)
    day = dt.date.fromisoformat(d0)
    end = dt.date.fromisoformat(d1)
    rows = 0
    with open(f"data/raw/{name}_M1.csv", "w") as f:
        while day <= end:
            if day.weekday() != 5:                       # суббота пустая
                for rec in decode(get_day(sym, day, cache), day, scale):
                    f.write("%d,%.5f,%.5f,%.5f,%.5f,%.4f\n" % rec)
                    rows += 1
            if day.day == 1:
                print(sym, day, rows, flush=True)
            day += dt.timedelta(days=1)
    print("записано", name, rows, flush=True)


if __name__ == "__main__":
    main()
