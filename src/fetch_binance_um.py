"""Загрузка архива Binance USDT-M futures (data.binance.vision): aggTrades / metrics / bookTicker по дням.
Файлы — data/raw/binance_um/<тип>/<SYMBOL>-<тип>-<YYYY-MM-DD>.zip (кэш). Запуск:
PYTHONPATH=src.venv/bin/python -I src/fetch_binance_um.py aggTrades BTCUSDT 2021-05-19 [2021-05-20...]"""
import os, sys, time
import requests

BASE = "https://data.binance.vision/data/futures/um/daily/{t}/{s}/{s}-{t}-{d}.zip"


def get(t, s, d):
    out = os.path.join("data/raw/binance_um", t); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{s}-{t}-{d}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    for k in range(5):
        try:
            r = requests.get(BASE.format(t=t, s=s, d=d), timeout=600, stream=True)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                with open(path + ".part", "wb") as f:
                    for ch in r.iter_content(1 << 20):
                        f.write(ch)
                os.replace(path + ".part", path)
                return path
        except requests.RequestException:
            pass
        time.sleep(10 * (k + 1))
    return None


if __name__ == "__main__":
    t, s = sys.argv[1], sys.argv[2]
    for d in sys.argv[3:]:
        p = get(t, s, d)
        print(t, s, d, "ok" if p else "нет", os.path.getsize(p) // 2**20 if p else "", "МБ", flush=True)
