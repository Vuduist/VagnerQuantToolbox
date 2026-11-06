"""Оркестратор исследования: один CSV -> таблицы эффектов по ТФ.

python run_study.py data/raw/BTCUSDT_1m.csv --boots 50 --out out/BTCUSDT
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import engine

TFS = {"1m": None, "5m": "5min", "15m": "15min", "1h": "1h",
       "4h": "4h", "1d": "1D", "1w": "1W"}
TF_ORDER = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]


def zscore(real_val, boot_vals):
    b = np.asarray(boot_vals, dtype=float)
    b = b[np.isfinite(b)]
    if len(b) < 10 or b.std() == 0:
        return np.nan
    return (real_val - b.mean()) / b.std()


def flatten(tf, real, boots):
    rows = []

    def add(group, metric, real_val, key_path):
        bv = []
        for b in boots:
            cur = b
            for k in key_path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, dict) and metric in cur:
                bv.append(cur[metric])
            elif isinstance(cur, (int, float, np.integer, np.floating)):
                bv.append(cur)
        rows.append({
            "tf": tf, "group": group, "metric": metric,
            "real": real_val, "z": zscore(real_val, bv),
            "null_mean": float(np.nanmean(bv)) if bv else np.nan,
        })

    rows.append({"tf": tf, "group": "meta", "metric": "n_bars",
                 "real": real["n_bars"], "z": np.nan, "null_mean": np.nan})
    add("disp", "count", real["n_disp"], ["n_disp"])

    for h_, st in real["disp_fwd"].items():
        for m in ("mean", "p_cont"):
            add(f"disp_fwd_{h_}", m, st[m], ["disp_fwd", h_])

    if real["disp_revisit"]:
        for m in ("median_depth", "p_depth_50", "p_depth_100"):
            add("disp_revisit", m, real["disp_revisit"][m], ["disp_revisit"])

    if real["disp_trade"]:
        for m in ("fill_rate", "win_rate", "expectancy_R"):
            add("disp_trade", m, real["disp_trade"][m], ["disp_trade"])

    for grp in ("sweep_hold", "sweep_fail"):
        if real.get(grp):
            for h_, st in real[grp].items():
                add(f"{grp}_{h_}", "p_cont", st["p_cont"], [grp, h_])

    for n_, st in real["runs"].items():
        add(f"runs_{n_}", "p_cont", st["p_cont"], ["runs", n_])

    # графические паттерны
    add("fvg", "count", real["fvg"]["n"], ["fvg", "n"])
    if real["fvg_fill"]:
        for m in ("p_touch", "p_fill_50", "p_fill_100"):
            add("fvg_fill", m, real["fvg_fill"][m], ["fvg_fill"])
    add("disp_fvg", "count", real["disp_fvg"]["n"], ["disp_fvg", "n"])
    if real["disp_fvg_trade"]:
        for m in ("fill_rate", "win_rate", "expectancy_R"):
            add("disp_fvg_trade", m, real["disp_fvg_trade"][m],
                ["disp_fvg_trade"])
    for h_, st in real["disp_fvg_fwd"].items():
        for m in ("mean", "p_cont"):
            add(f"disp_fvg_fwd_{h_}", m, st[m], ["disp_fvg_fwd", h_])
    add("sweep_reject", "count", real["sweep_reject"]["n"],
        ["sweep_reject", "n"])
    for h_, st in real["sweep_reject_fwd"].items():
        for m in ("mean", "p_cont"):
            add(f"sweep_reject_fwd_{h_}", m, st[m], ["sweep_reject_fwd", h_])
    add("combo", "count", real["combo"]["n"], ["combo", "n"])
    for h_, st in real["combo_fwd"].items():
        for m in ("mean", "p_cont"):
            add(f"combo_fwd_{h_}", m, st[m], ["combo_fwd", h_])
    if real["combo_trade"]:
        for m in ("fill_rate", "win_rate", "expectancy_R"):
            add("combo_trade", m, real["combo_trade"][m], ["combo_trade"])

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--boots", type=int, default=50)
    ap.add_argument("--out", default="out/study")
    ap.add_argument("--min-tf", default="1m")
    ap.add_argument("--min-bars", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.csv, "rb") as f:
        magic = f.read(4)
    is_utf16 = (magic[:2] in (b"\xff\xfe", b"\xfe\xff")
                or magic[0] == 0 or magic[1] == 0)
    if is_utf16:
        base = engine.load_mt5_csv(args.csv)
    else:
        base = engine.load_csv(args.csv)
    print(f"загружено {len(base)} баров: "
          f"{base.index[0]} .. {base.index[-1]}")

    all_rows = []
    started = False
    for tf in TF_ORDER:
        if tf == args.min_tf:
            started = True
        if not started:
            continue
        rule = None if tf == args.min_tf else TFS[tf]
        df = base if rule is None else engine.resample(base, rule)
        if len(df) < args.min_bars:
            print(f"{tf}: мало данных ({len(df)}), пропуск")
            continue
        print(f"{tf}: {len(df)} баров, real+{args.boots} синтетик...")
        real, boots = engine.run_with_null(df, B=args.boots)
        all_rows.extend(flatten(tf, real, boots))

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    sig = res[(res["z"].abs() >= 3)]
    print("\n=== Эффекты с |z| >= 3 (реал вне нулевой модели) ===")
    print(sig.to_string(index=False))
    print("\nПолная таблица ->", os.path.join(args.out, "summary.csv"))


if __name__ == "__main__":
    main()
