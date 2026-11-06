"""Загрузка минуток HistData.com (ASCII M1) в data/raw/<NAME>_M1.csv (t_ms UTC, o, h, l, c, vol).

HistData: годовые zip за прошлые годы, помесячные за текущий. Время в файлах — нью-йоркское С переходом на
летнее (проверено ранее: всплеск открытия cash в 09:30 и летом, и зимой). Загрузчик пишет (время + 5 ч), затем
src/fix_histdata_tz.py переводит в настоящий UTC. Объёма нет (0) → пишем 1.
Кэш zip: data/raw/histdata/<sym>/<period>.zip.

Запуск: PYTHONPATH=src.venv/bin/python -I src/fetch_histdata.py nsxusd NQH 2010 2026
"""

import io
import os
import re
import sys
import time
import zipfile

import requests

UA = {"User-Agent": "Mozilla/5.0"}
PAGE = "http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{s}/{p}"


def get_zip(sym, period, cache):
    path = os.path.join(cache, f"{period.replace('/', '-')}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return open(path, "rb").read()
    for k in range(5):
        try:
            s = requests.Session()
            page = s.get(PAGE.format(s=sym, p=period), headers=UA, timeout=60)
            form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', page.text))
            if "tk" not in form:
                return None
            r = s.post("http://www.histdata.com/get.php", data=form, timeout=300,
                       headers={**UA, "Referer": PAGE.format(s=sym, p=period)})
            break
        except requests.RequestException:
            time.sleep(20 * (k + 1))
    else:
        return None
    if r.status_code != 200 or len(r.content) < 1000 or r.content[:2] != b"PK":
        return None
    open(path, "wb").write(r.content)
    time.sleep(2)
    return r.content


def parse(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
    out = []
    import calendar
    for line in z.read(name).decode("ascii", "ignore").splitlines():
        p = line.strip().split(";")
        if len(p) < 5:
            continue
        d, t = p[0].split(" ")
        ts = calendar.timegm((int(d[:4]), int(d[4:6]), int(d[6:8]), int(t[:2]), int(t[2:4]), int(t[4:6]), 0, 0, 0))
        out.append(((ts + 5 * 3600) * 1000, p[1], p[2], p[3], p[4]))
    return out


def main():
    sym, name, y0, y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cache = os.path.join("data/raw/histdata", sym)
    os.makedirs(cache, exist_ok=True)
    rows = []
    for y in range(y0, y1 + 1):
        blob = get_zip(sym, str(y), cache)
        if blob is not None:
            got = parse(blob); rows += got
            print(sym, y, "год", len(got), flush=True)
            continue
        if y < time.gmtime().tm_year:                 # прошлый год без годового архива — данных нет
            continue
        for m in range(1, 13):                        # текущий год — помесячно
            blob = get_zip(sym, f"{y}/{m}", cache)
            if blob is None:
                continue
            got = parse(blob); rows += got
            print(sym, f"{y}-{m:02d}", len(got), flush=True)
    rows.sort()
    with open(f"data/raw/{name}_M1.csv", "w") as f:
        last = None
        for t, o, h, l, c in rows:
            if t != last:
                f.write(f"{t},{o},{h},{l},{c},1\n")
            last = t
    print("записано", name, len(rows), flush=True)


if __name__ == "__main__":
    main()
