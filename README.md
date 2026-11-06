# VagnerQuantToolbox

Research tooling from a multi-year personal study of intraday and daily market effects (FX, index futures/CFDs, commodities, crypto).
The focus is **methodology**: causal event definitions, honest nulls (block bootstrap, shuffles, placebo windows, synthetic random walks
with the real intraday volatility profile), clustered errors for overlapping targets, pre-registered hypotheses, walk-forward and
split-sample confirmation.

This repository contains **scripts only**. No market data, no result tables, no live trading infrastructure and no conclusions are
published. Scripts expect data under `data/raw/` (see *Data layout*) and write their output to `out/`.

Инструментарий количественного исследования рыночных эффектов. В репозитории только код: без данных, без результатов и без
боевой инфраструктуры. Докстринги модулей на русском описывают, что именно считает каждый скрипт.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw out
PYTHONPATH=src python src/fetch_yahoo.py          # example: daily/hourly bars from Yahoo
PYTHONPATH=src python src/run_study.py data/raw/EURUSD_M1.csv
```

All scripts are run as `PYTHONPATH=src python src/<script>.py` from the repository root. Minute data loaders accept the
`open_time,open,high,low,close,volume` CSV layout (epoch in s/ms/us auto-detected) and MetaTrader 5 exports (UTF-16).

## Data layout

| Path | Content | Produced by |
|---|---|---|
| `data/raw/<NAME>_M1.csv` | minute bars, UTC | `fetch_histdata.py`, `fetch_dukascopy.py` |
| `data/raw/<NAME>_1d.csv`, `_1h.csv` | daily / hourly bars | `fetch_yahoo.py`, `fetch_yahoo_adj.py` |
| `data/raw/histdata_ticks/` | bid/ask ticks for selected dates | `fetch_histdata_ticks*.py` |
| `data/raw/binance_um/` | Binance futures aggTrades / bookTicker / metrics | `fetch_binance_um.py` |
| `data/raw/cot_legacy.csv`, `cb_rates.csv`, `stocks/` | CFTC COT, central-bank policy rates (BIS), Dow 30 dailies | `fetch_cot.py`, `fetch_rates.py`, `fetch_stocks.py` |
| `data/calendar_events.csv` | FOMC and NFP dates | `calendar_events.py` |

## Map of tools

### Core library
| Module | Purpose |
|---|---|
| `engine.py` | Loaders (generic CSV, MT5 UTF-16), resampling, statistical events on bars, block-bootstrap surrogates that keep the return distribution, volatility clustering and intrabar structure; conditional statistics real-vs-null. |
| `swings.py` | Causal structural swing detection (state machine, confirmation after a k·ATR reversal); shows the size of the look-ahead illusion of non-causal swings. |
| `backtest.py` | Event → trade → net PnL with costs; the common trade skeleton used by all scanners. |
| `exits.py` / `run_exits.py` | Exit constructions evaluated on random entries (paired test): isolates the contribution of the exit rule from the entry. |
| `models.py` / `run_models.py` | Battery of popular discretionary models with strict causal definitions; scanner across markets and timeframes. |
| `sessions.py` / `run_sessions.py` | Session layer: price behaviour by time of day (ET), opening-range and session-transition events. |
| `volatility.py` / `run_vol.py` | HAR model of realised volatility, walk-forward out-of-sample predictability. |
| `calendar_events.py` | FOMC dates from the Fed pages and NFP dates by rule with verification against the 08:30 ET spike. |
| `run_study.py` | Orchestrator: one CSV → tables of effects by timeframe. |
| `run_backtest.py` | Scanner: all markets × all timeframes × strategies → net expectancy by year. |

### Data acquisition
| Module | Purpose |
|---|---|
| `fetch_yahoo.py`, `fetch_yahoo_adj.py` | Daily/hourly bars and dividend-adjusted closes from the Yahoo chart API. |
| `fetch_histdata.py`, `fix_histdata_tz.py` | HistData.com minute bars; repair of the HistData time stamps (EST-without-DST convention → UTC). |
| `fetch_histdata_ticks.py`, `fetch_histdata_ticks_dates.py` | HistData bid/ask ticks for selected dates and intraday windows. |
| `fetch_dukascopy.py` | Dukascopy minute candles (free datafeed, bi5 → CSV). |
| `fetch_binance_um.py` | Binance USDT-M futures archive: aggTrades, metrics, bookTicker by day. |
| `fetch_cot.py`, `fetch_rates.py`, `fetch_stocks.py` | CFTC COT legacy, BIS central-bank policy rates, Dow 30 dailies (survivorship warning in the docstring). |
| `vtt_clean.py` | Utility: WebVTT subtitles → clean text for note-taking. |

### Structure, levels and zones (ICT-style concepts under causal definitions)
| Module | Purpose |
|---|---|
| `killzones.py`, `macros.py`, `am_pm.py` | Do killzones / macros / AM-PM sessions form the extreme of the day? Slot-based tests on the "true day". |
| `reflevels.py`, `ndog.py`, `nwog_sr.py`, `rounds.py` | Reference opens, day/week opening gaps as magnets and S/R, round numbers. |
| `range_expansion.py`, `geometry.py`, `sd_projections.py` | Range compression → expansion (NR4/NR7, inside bars), leg projections, consolidation duration, oscillators, standard-deviation projections of daily ranges. |
| `trendlines.py` | Causal trend lines on two swing lows and the "third touch" event. |
| `liquidity_pools.py`, `dealing_range.py` | Equal highs/lows as liquidity pools; dealing range between confirmed swings. |
| `fvg_zones.py`, `fvg_context.py`, `fvg_features.py`, `fvg_deep.py`, `fvg_deep2.py`, `ltf_touch.py` | Fair-value gaps as zones with a life cycle; birth context, geometry, lower-timeframe microstructure at the touch. |
| `blocks.py`, `zones2.py` | Order / rejection / breaker / mitigation / propulsion blocks and volume imbalances as zones through one pipeline. |
| `amd.py` | Accumulation–manipulation–distribution ("power of three") as a testable daily template. |
| `ict_checks.py`, `ict_models.py` | Day-of-week / TGIF checks; five popular intraday models as causal rules with stop, target and time stop. |
| `run_fvg_ml.py` | "Good FVG" classifier: real data against the same pipeline run on the null. |
| `run_patterns_by_state.py`, `run_premium_discount.py`, `run_htf_bias.py`, `run_liquidity_path.py` | Pattern events inside pre-defined market states (volatility, day type, session, higher-timeframe bias, premium/discount); liquidity as a property of the path. |
| `run_volume_claims.py` | Open claims about volume tested on instruments with real volume. |
| `run_batch4.py`, `run_t01_s03.py`, `run_t10.py`, `run_s12.py`, `run_w03_t02_t09_s06.py`, `run_w04_sr.py`, `run_ict_checks.py`, `run_ict_models.py` | Batch runners for the modules above. |

### Cross-market, positioning and fundamentals
| Module | Purpose |
|---|---|
| `cross.py`, `run_cross.py`, `smt_dxy.py`, `intermarket.py` | SMT divergences on confirmed structure, lead–lag, spread reversion, risk-on/off, DXY vs pairs. |
| `cot.py`, `cot_oi.py`, `run_authors_cot.py` | COT commercials / non-commercials and open interest as signals, including contrarian extremes. |
| `rates.py`, `news_days.py`, `seasonality.py`, `ipda.py`, `commod.py`, `stocks.py` | Policy-rate differentials, news days, calendar seasonality, quarterly shifts and look-back ranges, commodity seasonal windows, stock-vs-index divergence. |
| `run_authors_wyckoff.py` | Wyckoff-style volume/spread events on daily futures, ATR-normalised, with a permutation null. |
| `run_tsmom.py` | Time-series momentum on dailies with volatility scaling and costs. |
| `run_x01.py` … `run_x12.py` | Batch runners for the fundamentals block. |

### Portfolio and account simulation
| Module | Purpose |
|---|---|
| `run_portfolio.py`, `sim_account.py` | Donchian breakout basket, risk per trade, leverage under a drawdown constraint, CAGR / maxDD / Calmar. |
| `anatomy.py`, `run_anatomy.py` | "Anatomy of a move": invariants of price paths across markets and epochs. |

### Crypto microstructure
| Module | Purpose |
|---|---|
| `btc_flow.py`, `run_btc_flow.py` | Aggressor flow from 1-minute klines (taker volume, trade counts) and its predictive power at 1–60 minutes. |
| `micro_book.py` | Best-quote imbalance from bookTicker → mid move over 1/5/30 s. |
| `micro_bursts.py`, `run_micro_bursts.py`, `run_micro_fade.py` | Cascades of aggressive trades: continuation vs reversal, and a maker-side fade with realistic fee tiers. |
| `run_funding.py` | Price around perpetual funding payments. |

### Surrogates and the blind test
| Module | Purpose |
|---|---|
| `run_blind_test.py` | Real chart segments vs surrogates (geometric random walk at sub-minute steps, block bootstrap); generates a blind test and trains a classifier on texture features to show what is and is not distinguishable. |

## Method notes

- Every event is defined causally: all decisions at bar *t* use data up to *t*; swing confirmation is delayed on purpose.
- Nulls are matched to the question: shuffles for conditional statistics, block bootstrap for path properties, placebo days/hours for
  calendar effects, synthetic random walks with the real intraday volatility profile for session and "liquidity" claims.
- With overlapping targets the mean is taken over bars with errors clustered by day; "mean of daily means" produces artefacts.
- Hypotheses are written down before the run, with expected signs; split-sample and walk-forward confirmation; BH-FDR over families.
- Costs are part of every number: spread, commissions and slippage per leg.

## License

MIT, see `LICENSE`.
