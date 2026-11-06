"""Тики bid/ask HistData (ASCII, timeframe T) только за окна дней NFP.
Для каждого месяца с NFP (data/calendar_events.csv): скачать месячный zip, вырезать строки дней NFP с 08:00:00 до 09:35:00
(время файлов — нью-йоркское локальное, как у M1; проверено по скачку 08:30 NFP), записать в
data/raw/histdata_ticks/<sym>/<YYYY-MM-DD>.csv (local_time,bid,ask), zip удалить.
Запуск: PYTHONPATH=src.venv/bin/python -I src/fetch_histdata_ticks.py xauusd 2016 2026 [worker_id n_workers]"""
import io, os, re, sys, time, zipfile
import requests
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
PAGE = "http://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{s}/{y}/{m}"


def download(sym, y, m):
    for k in range(6):
        try:
            s = requests.Session()
            url = PAGE.format(s=sym, y=y, m=m)
            page = s.get(url, headers=UA, timeout=60)
            form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', page.text))
            if "tk" not in form:
                return None
            r = s.post("http://www.histdata.com/get.php", data=form, timeout=900, headers={**UA, "Referer": url})
            if r.status_code == 200 and r.content[:2] == b"PK":
                return r.content
        except requests.RequestException:
            pass
        time.sleep(30 * (k + 1))
    return None


def main():
    sym, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    wid, nw = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    cal = pd.read_csv("data/calendar_events.csv", parse_dates=["date"])
    nfp = cal[(cal.event == "NFP") & (cal.date.dt.year >= y0) & (cal.date.dt.year <= y1)].date.sort_values()
    months = sorted({(d.year, d.month) for d in nfp})
    out = os.path.join("data/raw/histdata_ticks", sym); os.makedirs(out, exist_ok=True)
    for i, (y, m) in enumerate(months):
        if i % nw != wid:
            continue
        days = [d for d in nfp if d.year == y and d.month == m]
        if all(os.path.exists(os.path.join(out, f"{d:%Y-%m-%d}.csv")) for d in days):
            continue
        t = time.time()
        blob = download(sym, y, m)
        if blob is None:
            print(sym, y, m, "нет архива", flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(blob))
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        keys = {f"{d:%Y%m%d}": d for d in days}
        buf = {k: [] for k in keys}
        with z.open(name) as f:
            for line in f:
                k = line[:8].decode()
                if k in buf and b"080000000" <= line[9:18] <= b"093500000":
                    t_, b, a = line.decode().strip().split(",")[:3]
                    buf[k].append(f"{t_[9:]},{b},{a}")
        for k, rows in buf.items():
            with open(os.path.join(out, f"{keys[k]:%Y-%m-%d}.csv"), "w") as g:
                g.write("t,bid,ask\n" + "\n".join(rows) + ("\n" if rows else ""))
        print(sym, y, m, "дней", len(days), "тиков", sum(len(v) for v in buf.values()), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
