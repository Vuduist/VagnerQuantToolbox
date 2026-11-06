"""Классификатор «хороший FVG»: реал против того же пайплайна на нулле.

Сплит по времени (70/30), логистическая регрессия и градиентный бустинг.
Метрика — AUC на тесте. Нулл: бутстреп 1m -> ресемпл HTF -> тот же код.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import engine
from fvg_features import build, FEATURES


def fit_auc(df):
    df = df.sort_values("t")
    X = df[FEATURES].to_numpy()
    y = df["bounced"].astype(int).to_numpy()
    cut = int(len(df) * 0.7)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    out = {}
    for name, model in (("logreg", make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.1))),
                        ("gbm", HistGradientBoostingClassifier(
                            max_iter=200, learning_rate=0.05,
                            max_depth=3, l2_regularization=5.0))):
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        out[name] = roc_auc_score(yte, p)
    return out


def build_from_1m(df1m, htf_rule, boot=False, seed=0):
    if boot:
        o, h, l, c = (df1m[x].to_numpy() for x in ("open", "high", "low", "close"))
        so, sh, sl, sc = engine.block_bootstrap_ohlc(o, h, l, c, block=500,
                                                     seed=seed)
        df1m = pd.DataFrame({"open": so, "high": sh, "low": sl, "close": sc,
                             "volume": df1m["volume"].to_numpy()},
                            index=df1m.index)
    htf = engine.resample(df1m, htf_rule)
    return build(htf, df1m)


def main():
    for name, path, fmt, rule in (
            ("BTC", "data/raw/BTCUSDT_1m.csv", "binance", "15min"),
            ("USTEC", "data/raw/USTEC_PERIOD_M1.csv", "mt5", "5min")):
        df1m = (engine.load_mt5_csv(path) if fmt == "mt5"
                else engine.load_csv(path))
        real_df = build_from_1m(df1m, rule)
        real = fit_auc(real_df)
        print(f"=== {name} (зон с исходом: {len(real_df)}) ===")
        print(f"  реал: logreg AUC={real['logreg']:.4f} | gbm AUC={real['gbm']:.4f}")
        nulls = []
        for b in range(8):
            ndf = build_from_1m(df1m, rule, boot=True, seed=500 + b)
            nulls.append(fit_auc(ndf))
        for m in ("logreg", "gbm"):
            bv = np.array([x[m] for x in nulls])
            z = (real[m] - bv.mean()) / bv.std() if bv.std() > 0 else np.nan
            print(f"  нулл {m}: AUC={bv.mean():.4f}±{bv.std():.4f} "
                  f"| z реала={z:+.2f}")


if __name__ == "__main__":
    main()
