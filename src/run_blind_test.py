"""Настоящие отрезки против суррогатов. Генерирует (1) 40 PNG для слепого теста (out/blind/*.png + out/blind/key.json),
(2) признаки второго порядка для 3 × 3000 отрезков и классификатор, (3) проверку признаков подлинности против направления."""
import json
import sys
import numpy as np
import pandas as pd
import engine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
df = engine.load_csv("data/raw/NQH_M1.csv"); df.index = df.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
df = df.loc["2016":]; df = df[df.index.weekday < 5]
r1 = np.log(df.close).diff(); prof = r1.groupby(df.index.hour * 60 + df.index.minute).std().reindex(range(960)).interpolate().bfill().ffill().values
b5 = df.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
b5 = b5[(b5.index.weekday < 5) & (b5.index.hour < 16)]
days = [g for _, g in b5.groupby(b5.index.normalize()) if len(g) == 192]
print(f"реальных дней по 192 бара: {len(days)}", file=sys.stderr)
# пул 5-мин лог-доходностей и форм баров для бутстрепа (по часам, чтобы сохранить профиль)
def to_rel(g):
    o = g.open.values[0]; return np.log(g[["open", "high", "low", "close"]].values / o)

def gbm_day():
    """случайное блуждание по минутам с профилем волатильности → 5-мин OHLC (лог-цены относительно открытия)."""
    sub = 12                                                        # 12 шагов в минуте: бары из 60 точек, иначе закрытия слишком часто на экстремуме
    p = np.cumsum(rng.standard_normal(960 * sub) * np.repeat(prof, sub) / np.sqrt(sub)); p = p - p[0]
    P = p.reshape(192, 5 * sub); o = np.r_[0.0, P[:-1, -1]]
    return np.c_[o, np.maximum(P.max(axis=1), o), np.minimum(P.min(axis=1), o), P[:, -1]]

real_rel = [to_rel(g) for g in days]
def boot_day():
    """блочный бутстреп: 16 блоков по 12 баров из случайных реальных дней, тот же час дня (сохраняет профиль и кластеры внутри часа)."""
    out = []; level = 0.0
    for h in range(16):
        g = real_rel[rng.integers(len(real_rel))][h * 12:(h + 1) * 12]; base = g[0, 0]
        blk = g - base + level; out.append(blk); level = blk[-1, 3]
    return np.vstack(out)

def feats(X):
    o, h, l, c = X.T; r = np.diff(c); ar = np.abs(r)
    ac = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1] if np.std(x[:-k]) > 0 and np.std(x[k:]) > 0 else 0
    rng_ = h - l; body = np.abs(c - o)
    vr = np.var(c[4::4] - c[:-4:4]) / (4 * np.var(r)) if np.var(r) > 0 else 1
    signs = np.sign(r[r != 0]); runs = 1 + (signs[1:] != signs[:-1]).sum() if len(signs) > 1 else 1
    # R/S Хёрст (грубо)
    def hurst(x):
        n = len(x); 
        if n < 20: return 0.5
        y = np.cumsum(x - x.mean()); R = y.max() - y.min(); S = x.std()
        return np.log(R / S) / np.log(n) if S > 0 and R > 0 else 0.5
    return {"kurt": pd.Series(r).kurt(), "skew": pd.Series(r).skew(), "ac_abs1": ac(ar, 1), "ac_abs6": ac(ar, 6), "ac_r1": ac(r, 1), "ac_r2": ac(r, 2),
            "vr4": vr, "runs": runs / max(len(signs), 1), "hurst": hurst(r), "close_at_ext": np.mean((body > 0) & (np.minimum(h - np.maximum(o, c), np.minimum(o, c) - l) < 0.1 * rng_ + 1e-12)),
            "body_ratio": np.mean(body / (rng_ + 1e-12)), "range_cv": np.std(rng_) / (np.mean(rng_) + 1e-12), "max_abs_r": ar.max() / (ar.mean() + 1e-12),
            "zero_bars": np.mean(rng_ == 0)}

# (1) слепой тест: 40 отрезков
plt.rcParams.update({"figure.facecolor": "white"})
key = []; idx_real = rng.choice(len(days), 14, replace=False)
items = [("real", real_rel[i]) for i in idx_real] + [("gbm", gbm_day()) for _ in range(13)] + [("boot", boot_day()) for _ in range(13)]
rng.shuffle(items)
for k, (cls, X) in enumerate(items):
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110); o, h, l, c = X.T; x = np.arange(len(c))
    up = c >= o
    ax.vlines(x, l, h, color="#555", lw=0.7)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=0.8, color="#2a9d8f"); ax.bar(x[~up], (c - o)[~up], bottom=o[~up], width=0.8, color="#e76f51")
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(); fig.savefig(f"out/blind/chart_{k+1:02d}.png"); plt.close(fig); key.append({"n": k + 1, "cls": cls})
json.dump(key, open("out/blind/key.json", "w")); print("40 графиков записаны", file=sys.stderr)

# (2) классификатор
rows = []
for i in range(3000):
    rows.append({"cls": "real", **feats(real_rel[rng.integers(len(real_rel))]), "t": i})
    rows.append({"cls": "gbm", **feats(gbm_day()), "t": i}); rows.append({"cls": "boot", **feats(boot_day()), "t": i})
F = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
Xc = F.drop(columns=["cls", "t"]); yc = F.cls; tr = F.t < 2000; te = ~tr
clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0).fit(Xc[tr], yc[tr])
pred = clf.predict(Xc[te]); acc = (pred == yc[te]).mean()
print(f"классификатор 3 класса: точность {acc:.3f} (случай 0.333)", file=sys.stderr)
cm = pd.DataFrame(confusion_matrix(yc[te], pred, labels=["real", "gbm", "boot"]), index=["real", "gbm", "boot"], columns=["→real", "→gbm", "→boot"]); print(cm.to_string(), file=sys.stderr)
for a, b_ in (("real", "gbm"), ("real", "boot")):
    m = yc.isin([a, b_]); c2 = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05, random_state=0).fit(Xc[tr & m], yc[tr & m])
    print(f"  {a} против {b_}: точность {(c2.predict(Xc[te & m]) == yc[te & m]).mean():.3f}", file=sys.stderr)
from sklearn.inspection import permutation_importance
imp = permutation_importance(clf, Xc[te], yc[te], n_repeats=5, random_state=0).importances_mean
print("важность признаков:", dict(sorted(zip(Xc.columns, imp.round(3)), key=lambda z: -z[1])[:8]), file=sys.stderr)
print("средние по классам:", file=sys.stderr); print(F.groupby("cls")[["kurt", "ac_abs1", "ac_r1", "vr4", "hurst", "close_at_ext", "body_ratio", "range_cv"]].mean().round(3).to_string(), file=sys.stderr)
F.to_pickle("out/blind/features.pkl")
