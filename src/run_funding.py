"""Цена BTC вокруг выплат фандинга (00/08/16 UTC), Binance. Цена — spot BTCUSDT 1m (close), ставка — USDT-M perp.
Окна: pre = T−30 → T, post = T → T+30, post2 = T → T+120 мин. Сигнал: ставка этой выплаты (почти известна заранее как прогнозная;
осторожность — небольшое заглядывание) и ставка прошлой выплаты (строго каузально). Плацебо — те же окна в T+4 ч."""
import sys
import numpy as np
import pandas as pd
import engine

px = engine.load_csv("data/raw/BTCUSDT_1m.csv").close
F = pd.read_csv("data/raw/binance_um/BTCUSDT_funding.csv")
F["T"] = pd.to_datetime(F.calc_time, unit="ms").dt.round("min")
F = F.drop_duplicates("T").set_index("T").sort_index()
F["rate_prev"] = F.last_funding_rate.shift(1)
lp = np.log(px)


def at(ts):
    return lp.reindex(ts, method="ffill", tolerance=pd.Timedelta("3min")).to_numpy()


def frame(shift_h=0):
    T = F.index + pd.Timedelta(hours=shift_h)
    m = {k: at(T + pd.Timedelta(minutes=k)) for k in (-30, 0, 30, 120)}
    return pd.DataFrame({"pre": m[0] - m[-30], "post": m[30] - m[0], "post2": m[120] - m[0],
                         "rate": F.last_funding_rate.to_numpy(), "rate_prev": F.rate_prev.to_numpy()}, index=T).dropna()


def st(x):
    x = x.dropna() * 1e4
    return f"{x.mean():+6.2f} бп (t {x.mean() / x.std() * np.sqrt(len(x)):+.1f}, n {len(x)})"


for nm, sh in (("выплата", 0), ("плацебо +4 ч", 4)):
    D = frame(sh)
    print(f"\n=== {nm}: все {len(D)}; pre {st(D.pre)} post {st(D.post)} post2 {st(D.post2)}", file=sys.stderr)
    for sig in ("rate", "rate_prev"):
        q = D[sig].quantile([0.2, 0.8])
        hi = D[sig] >= q.iloc[1]; lo = D[sig] <= q.iloc[0]
        print(f"  {sig:9s} высокий фандинг (≥{q.iloc[1] * 1e4:.2f} бп): pre {st(D.pre[hi])} post {st(D.post[hi])} post2 {st(D.post2[hi])}", file=sys.stderr)
        print(f"  {sig:9s} низкий фандинг  (≤{q.iloc[0] * 1e4:.2f} бп): pre {st(D.pre[lo])} post {st(D.post[lo])} post2 {st(D.post2[lo])}", file=sys.stderr)
    D["yr"] = D.index.year
    if sh == 0:
        print("  по годам (pre, высокий rate_prev):", " ".join(f"{y}:{g.pre[g.rate_prev >= g.rate_prev.quantile(0.8)].mean() * 1e4:+.1f}" for y, g in D.groupby("yr")), file=sys.stderr)
