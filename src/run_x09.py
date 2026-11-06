"""Прогон X09 акции Dow 30. out/x09.csv"""

import glob
import os
import sys

import pandas as pd

import stocks as st

B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
idx = st.load_stock("data/raw/stocks/DJI_1d.csv")
stk = {os.path.basename(p)[:-7]: st.load_stock(p) for p in sorted(glob.glob("data/raw/stocks/*_1d.csv")) if "DJI" not in p}
print("акций:", len(stk), "дней индекса:", len(idx))
rows = []
for L in (20, 40):
    out = st.smt_stocks(stk, idx, L=L, B=B)
    print(f"\n===== L={L}: событий новый лоу индекса {int(out['bull_n_events'])}, новый хай {int(out['bear_n_events'])}")
    for k, r in out.items():
        if isinstance(r, dict):
            s = f"  {k:16s} HL−LL={r['real']*100:+.2f}% (P>0 {r['p_pos']:.2f}, n={r['n']})"
            if "lead" in r:
                s += f" [лидер {r['lead']*100:+.2f}%, отстающие {r['lag']*100:+.2f}%]"
            s += f" | null={r['null_mean']*100:+.2f}%±{r['null_sd']*100:.2f} z{r['z']:+.1f}"
            print(s)
            rows.append({"L": L, "metric": k, **r})
pd.DataFrame(rows).to_csv("out/x09.csv", index=False)
print("\nзаписано out/x09.csv")
