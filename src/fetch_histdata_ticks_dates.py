"""Тики bid/ask HistData за произвольные даты и окно (местное время NY файлов HistData).
Запуск: PYTHONPATH=src.venv/bin/python -I src/fetch_histdata_ticks_dates.py eurusd dates.txt 090000000 130000000 <подпапка> [wid nw]
Выход: data/raw/histdata_ticks/<подпапка>/<sym>/<дата>.csv (t,bid,ask)."""
import io, os, sys, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # -I игнорирует PYTHONPATH
from fetch_histdata_ticks import download


def main():
    sym, dfile, t0, t1, sub = sys.argv[1:6]
    wid, nw = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (0, 1)
    days = sorted(open(dfile).read().split())
    out = os.path.join("data/raw/histdata_ticks", sub, sym); os.makedirs(out, exist_ok=True)
    months = sorted({d[:7] for d in days})
    for i, ym in enumerate(months):
        if i % nw != wid:
            continue
        md = [d for d in days if d.startswith(ym)]
        if all(os.path.exists(os.path.join(out, f"{d}.csv")) for d in md):
            continue
        t = time.time()
        blob = download(sym, int(ym[:4]), int(ym[5:7]))
        if blob is None:
            print(sym, ym, "нет архива", flush=True); continue
        z = zipfile.ZipFile(io.BytesIO(blob))
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        keys = {d.replace("-", ""): d for d in md}
        buf = {k: [] for k in keys}
        b0, b1 = t0.encode(), t1.encode()
        with z.open(name) as f:
            for line in f:
                k = line[:8].decode()
                if k in buf and b0 <= line[9:18] <= b1:
                    tt, bb, aa = line.decode().strip().split(",")[:3]
                    buf[k].append(f"{tt[9:]},{bb},{aa}")
        for k, rows in buf.items():
            with open(os.path.join(out, f"{keys[k]}.csv"), "w") as g:
                g.write("t,bid,ask\n" + "\n".join(rows) + ("\n" if rows else ""))
        print(sym, ym, "тиков", sum(len(v) for v in buf.values()), f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
