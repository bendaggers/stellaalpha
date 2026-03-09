# Stella Alpha — Multi-Timeframe Pure ML Trading Pipeline

> **Extends V4 Pure ML** with Daily (D1) context, high R:R configurations,  
> statistical feature validation, and a loss analysis system that generates  
> actionable trade filters.

---

## What's New vs V4

| Feature | V4 | Stella Alpha |
|---|---|---|
| Timeframes | H4 only | H4 + D1 (multi-timeframe) |
| TP range | 30–80 pips | 50–150 pips |
| SL | 40 pips (fixed) | 30 pips (fixed) |
| Total configs | 24 | 66 |
| Min R:R | 0.6:1 | 1.67:1 |
| Tier system | None | Tier 0 Runners (≥2.5:1), Tier 1 Ideal (1.67–2.49:1) |
| Acceptance criteria | Fixed 55% precision | R:R-derived breakeven + 5% edge |
| Statistical pre-filter | No | Yes (Welch t-test + Cohen's d before RFE) |
| Loss analysis | No | Yes — per-trade recording + pattern detection |
| Filter recommendations | No | Yes — auto-generated, P&L-simulated |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Validate your data (recommended)

```bash
# Check H4 + D1 for quality issues and date range
python validate_leakage.py \
    --h4  data/EURUSD_H4_features.csv \
    --d1  data/EURUSD_D1_features.csv \
    --test-merge
```

### 3. Run the pipeline

```bash
# Full multi-timeframe run
python run_pure_ml_stella_alpha.py \
    --config  config/pure_ml_stella_alpha_settings.yaml \
    --input   data/EURUSD_H4_features.csv \
    --d1      data/EURUSD_D1_features.csv \
    --workers 8

# H4-only fallback (no D1 data)
python run_pure_ml_stella_alpha.py \
    --config config/pure_ml_stella_alpha_settings.yaml \
    --input  data/EURUSD_H4_features.csv
```

### 4. Analyse results

```bash
# Diagnostic report
python diagnose.py --db artifacts/pure_ml_stella_alpha.db

# Loss analysis
python analyze_losses.py --filters

# Filter recommendations
python recommend_filters.py --tp 100 --sl 30

# Compare with V4
python compare_versions.py \
    --v4     ../version-4/artifacts/pure_ml.db \
    --stella artifacts/pure_ml_stella_alpha.db
```

---

## Project Structure

```
stella-alpha/
│
├── run_pure_ml_stella_alpha.py   # Main pipeline entry point
├── diagnose.py                   # Diagnostic analyser (11 sections)
├── analyze_losses.py             # Loss analysis CLI
├── recommend_filters.py          # Filter recommendation CLI
├── compare_versions.py           # V4 vs Stella Alpha comparison
├── validate_leakage.py           # Data quality + leakage checker
│
├── config/
│   └── pure_ml_stella_alpha_settings.yaml
│
├── src/
│   ├── data_merger.py            # D1 safe merge (leakage prevention)
│   ├── tier_classification.py    # Tier 0 / Tier 1 logic
│   ├── checkpoint_db.py          # SQLite checkpoint + resume
│   ├── feature_engineering.py    # H4 + D1 derived + MTF features
│   ├── statistical_validation.py # Welch t-test + Cohen's d pre-filter
│   ├── pure_ml_labels.py         # TP/SL/Hold labelling
│   ├── training.py               # LightGBM training + calibration
│   ├── evaluation.py             # Metrics + R:R-aware acceptance
│   ├── experiment.py             # Parallel experiment runner
│   ├── trade_recorder.py         # Per-trade SQLite logger
│   ├── loss_analysis.py          # Win vs loss pattern analysis
│   └── filter_recommendations.py # Auto filter generation + ranking
│
├── artifacts/                    # Created at runtime
│   ├── pure_ml_stella_alpha.db   # Checkpoint database
│   ├── trades_stella_alpha.db    # Trade log (for loss analysis)
│   ├── models/                   # Trained models per config
│   ├── trading_config_stella_alpha.json
│   ├── metrics_stella_alpha.json
│   ├── features_stella_alpha.json
│   ├── loss_analysis_stella_alpha.json
│   └── filter_recommendations.json
│
└── tests/
    └── test_suite.py             # 39 unit + integration tests
```

---

## Pipeline Steps

The main pipeline runs 10 steps automatically:

```
Step 1   Load H4 CSV (and D1 if provided)
Step 2   Feature engineering
           ├── H4 derived features  (~104)
           ├── D1 derived features  (~32, new)
           └── MTF cross-TF features (~20, new)
Step 3   Pre-compute labels for all (TP, SL, Hold) combos
Step 4   Define walk-forward folds (5 expanding folds)
Step 5   Generate configuration space (66 configs)
Step 6   Run experiments in parallel
           ├── Statistical pre-filter (noise removal)
           ├── RFE feature selection per fold
           ├── LightGBM training + calibration
           ├── Threshold optimisation
           └── Trade recording (wins + losses)
Step 7   Select best configuration (R:R-aware EV ranking)
Step 8   Train final production model
Step 9   Save all artifacts
Step 10  Run loss analysis + filter recommendations
```

---

## Tier System

Stella Alpha introduces a **tier system** based on Risk:Reward ratio:

| Tier | Name | R:R | Required win rate | Notes |
|---|---|---|---|---|
| **Tier 0** | Runner | ≥ 2.5:1 | ≤ 28.6% | High flyers — large TP, very low win rate ok |
| **Tier 1** | Ideal  | 1.67–2.49:1 | 28.6–37.5% | Core target configs |

All configs in Stella Alpha have R:R ≥ 1.67:1 (TP ≥ 50 with SL = 30).

---

## Acceptance Criteria

Unlike V4's fixed 55% precision requirement, Stella Alpha uses **R:R-derived criteria**:

```
min_winrate   = SL / (TP + SL)
edge_required = min_winrate + 0.05          ← 5% edge above breakeven
passes if:
  precision_mean > edge_required
  ev_mean > 0
  precision_cv  < 0.30                      ← stability check
  avg_trades_per_fold ≥ 30                  ← volume check
```

**Example — TP=100, SL=30:**
```
min_winrate   = 30 / 130 = 23.1%
edge_required = 23.1% + 5% = 28.1%
→ Only need to WIN 29% of trades to pass (vs V4's fixed 55%)
```

---

## Data Leakage Prevention

This is the most critical correctness requirement. For each H4 candle:

```
D1 data used = last COMPLETED daily candle BEFORE the H4 date
```

**What this means in practice:**
- H4 candle at `2024-01-15 08:00` → uses D1 from `2024-01-14` (yesterday's close)
- H4 candle at `2024-01-16 20:00` → uses D1 from `2024-01-15` (yesterday's close)
- H4 candle at `2024-01-16 00:00` → uses D1 from `2024-01-15` (NOT same-day Jan 16)

The merge uses `pd.merge_asof(direction='backward')` on a helper column
`_d1_available_from = d1_timestamp + 1 day`, guaranteeing no same-day D1 data.

**Validate before running:**
```bash
python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --test-merge
```

---

## Feature Engineering

### H4 Derived Features (~104 features, carried from V4)
BB position, RSI momentum, ATR regime, volume analysis, candle patterns,
exhaustion score, session flags, swing detection, and more.

### D1 Derived Features (~32 features, new in Stella Alpha)
| Category | Examples |
|---|---|
| Momentum | `d1_rsi_slope_3`, `d1_rsi_slope_5`, `d1_price_momentum` |
| Mean Reversion | `d1_rsi_oversold`, `d1_bb_below_lower`, `d1_mean_rev_score` |
| Volatility | `d1_bb_squeeze`, `d1_atr_percentile`, `d1_atr_expanding` |
| Regime | `d1_is_trending_up`, `d1_is_trending_down`, `d1_adx` |
| Pattern | `d1_shooting_star`, `d1_bearish_divergence`, `d1_consecutive_bullish` |
| Z-scores | `d1_rsi_zscore`, `d1_volume_zscore` |

### MTF Cross-Timeframe Features (~20 features, new in Stella Alpha)
| Category | Examples |
|---|---|
| Alignment | `mtf_rsi_aligned`, `mtf_bb_aligned`, `mtf_trend_aligned` |
| Divergence | `mtf_rsi_divergence`, `mtf_trend_divergence` |
| Relative | `h4_position_in_d1_range`, `h4_vs_d1_rsi`, `mtf_confluence_score` |
| Context | `d1_supports_short`, `d1_opposes_short`, `mtf_strong_short_setup` |

**`mtf_confluence_score`** is the key composite feature — it weights RSI, BB, trend,
and bearish divergence alignment across both timeframes (0.0–1.0 scale).

---

## Loss Analysis System

After the pipeline completes, loss analysis automatically runs on the recorded trades.
You can also run it standalone:

```bash
python analyze_losses.py
python analyze_losses.py --config TP100_SL30_H48 --filters
python analyze_losses.py --list-configs     # show all configs in the DB
```

The analysis includes:
- **Feature comparison** — Welch's t-test + Cohen's d for every feature (wins vs losses)
- **Session analysis** — win rate by Asian / London / NY / Overlap
- **Confidence analysis** — win rate by probability bucket (0.50–0.55, ..., 0.80+)
- **D1 alignment analysis** — impact of `d1_supports_short` / `d1_opposes_short`
- **MTF confluence analysis** — high/medium/low confluence win rates
- **Optimal threshold** — grid-search for best confidence cutoff

### Sample output
```
SESSION ANALYSIS
  London Overlap:  67.3%  (312 trades)
  NY Session:      62.1%  (445 trades)
  London Session:  59.8%  (523 trades)
  Asian Session:   41.2%  (198 trades)  ⚠️ PROBLEMATIC (-21.1pp)

→ Avoid Asian session  ←  filter: h4_is_asian_session == 0
```

---

## Filter Recommendations

```bash
python recommend_filters.py --tp 100 --sl 30
python recommend_filters.py --tp 100 --sl 30 --combinations
```

Each filter is evaluated by **net pip improvement**:

```
pips_saved = losses_removed × SL_pips
pips_lost  = wins_removed   × TP_pips
net        = pips_saved − pips_lost

For TP=100, SL=30: need to remove 3.33 losses per missed win to break even
```

Filters are rated: `🟢 STRONG` (efficiency ≥ 3:1, net > 1000 pips),
`🟡 MODERATE` (efficiency ≥ 2:1, net > 300 pips), `🔴 NOT RECOMMENDED`.

---

## Configuration Reference

Key settings in `config/pure_ml_stella_alpha_settings.yaml`:

```yaml
config_space:
  tp_pips:          {min: 50, max: 150, step: 10}   # 11 values
  sl_pips:          {fixed: 30}                      # fixed SL
  max_holding_bars: {min: 12, max: 72, step: 12}     # 6 values
  # Total: 11 × 1 × 6 = 66 configs

acceptance_criteria:
  min_edge_above_breakeven: 0.05   # 5% above SL/(TP+SL)
  max_precision_cv:         0.30   # stability
  min_trades_per_fold:      30     # volume

loss_analysis:
  enabled: true
  db_path: artifacts/trades_stella_alpha.db

multi_timeframe:
  compute_d1_derived: true
  compute_cross_tf:   true
```

---

## CLI Reference

### `run_pure_ml_stella_alpha.py`
```
-c, --config   YAML config file (required)
-i, --input    H4 CSV input (required)
    --d1       D1 CSV input (optional, enables MTF features)
-w, --workers  Parallel workers (default: from config)
-o, --output   Override artifacts directory
-d, --debug    Verbose error output
```

### `diagnose.py`
```
-d, --db          Checkpoint database (default: artifacts/pure_ml_stella_alpha.db)
-t, --top         Top N configs to show (default: 10)
    --compare-v4  Path to V4 database for comparison
```

### `analyze_losses.py`
```
-t, --trades      Trades database (default: artifacts/trades_stella_alpha.db)
-c, --config      Specific config_id to analyse
    --all         All configs combined (default)
-o, --output      Save JSON report
    --top         Top N features to show (default: 20)
    --filters     Print quick filter hints
    --list-configs List all config_ids and exit
```

### `recommend_filters.py`
```
-t, --trades       Trades database
    --tp           Take-profit pips (auto-reads from trading_config if omitted)
    --sl           Stop-loss pips (default: 30)
-c, --config       Limit to a specific config_id
    --top-n        Number of single-filter recommendations (default: 20)
    --combinations Also test pairs/triples of top filters
    --top-singles  How many singles to combine (default: 5)
-o, --output       Save JSON report
```

### `compare_versions.py`
```
    --v4       Path to V4 checkpoint DB (required)
    --stella   Path to Stella Alpha DB (required)
-o, --output   Save JSON comparison report
-v, --verbose  Show top-10 configs from each version
```

### `validate_leakage.py`
```
    --h4         H4 CSV path
    --d1         D1 CSV path
    --merged     Pre-merged CSV path
    --test-merge Test-merge H4+D1 and check for leakage
    --strict     Warnings also cause non-zero exit
-o, --output     Save JSON report
```

---

## Running Tests

```bash
# All tests
python tests/test_suite.py

# Single group
python tests/test_suite.py TradeRecorderTests
python tests/test_suite.py LossAnalysisTests
python tests/test_suite.py FilterRecommendationTests
python tests/test_suite.py StatisticalValidationTests
python tests/test_suite.py EvaluationTests
python tests/test_suite.py IntegrationTests
python tests/test_suite.py LeakageValidationTests     # requires data_merger.py
```

**Test groups:**

| Group | Tests | What's covered |
|---|---|---|
| `LeakageValidationTests` | 5 | D1 merge correctness, leakage prevention |
| `TradeRecorderTests` | 7 | SQLite insert, schema, thread safety |
| `LossAnalysisTests` | 8 | Report structure, planted pattern detection |
| `FilterRecommendationTests` | 8 | Rule apply, P&L math, pattern surfacing |
| `StatisticalValidationTests` | 3 | Noise filtering, significant feature kept |
| `EvaluationTests` | 5 | R:R-aware breakeven formula, CV rejection |
| `IntegrationTests` | 3 | End-to-end module chain |

---

## Artifacts Reference

After a successful pipeline run:

| File | Contents |
|---|---|
| `pure_ml_stella_alpha.db` | Checkpoint DB — all 66 config results |
| `trades_stella_alpha.db` | Per-trade log — features + outcome for every trade |
| `trading_config_stella_alpha.json` | **Best config for deployment** |
| `metrics_stella_alpha.json` | Run summary, pass rates, tier/classification counts |
| `features_stella_alpha.json` | Consensus selected features |
| `loss_analysis_stella_alpha.json` | Loss pattern report |
| `filter_recommendations.json` | Ranked filter rules with P&L impact |
| `models/stella_<config_id>.pkl` | Trained model per passing config |
| `logs/training_stella_alpha_*.log` | Full execution log |

### `trading_config_stella_alpha.json` format
```json
{
  "version": "Stella Alpha",
  "best_config": {
    "config_id":        "TP100_SL30_H48",
    "tier":             0,
    "tier_name":        "TIER 0 - RUNNER",
    "tp_pips":          100,
    "sl_pips":          30,
    "max_holding_bars": 48,
    "threshold":        0.55,
    "classification":   "GOLD"
  },
  "performance": {
    "ev_mean":              15.2,
    "precision_mean":       0.38,
    "precision_cv":         0.10,
    "risk_reward_ratio":    3.33,
    "min_winrate_required": 0.231,
    "edge_above_breakeven": 0.149
  },
  "features": ["rsi_value", "bb_position", "d1_trend_direction", "mtf_confluence_score", "..."],
  "model_file": "models/stella_TP100_SL30_H48.pkl"
}
```

---

## D1 Data Requirements

Your D1 CSV must have the same structure as H4 but calculated on daily bars:

**Required columns:** `timestamp, open, high, low, close, volume`

**Recommended feature columns** (same names as H4 equivalents):  
`rsi_value, bb_position, atr_pct, trend_strength, lower_band, upper_band, middle_band,`  
`volume_ratio, candle_rejection, candle_body_pct, prev_candle_body_pct`

The pipeline will compute all D1 derived and MTF features from these base columns.
If D1 feature columns are missing, only the base OHLCV merge will be performed
and D1 derived / MTF features will be skipped gracefully.

**Timezone:** Use the same broker timezone as your H4 data. The daily candle is
considered complete at `00:00` in that timezone (i.e. the daily close from the
previous `00:00→00:00` window).

---

## Troubleshooting

**"No configurations passed acceptance criteria"**  
→ Check data quality with `validate_leakage.py`.  
→ Try relaxing `min_edge_above_breakeven: 0.03` in config.  
→ Run `diagnose.py` to see near-misses.

**"D1 merge failed"**  
→ Check D1 timestamps are daily frequency (`freq='D'`).  
→ Ensure D1 start date is at least 1 day before H4 start.  
→ Run `validate_leakage.py --h4 ... --d1 ... --test-merge` for details.

**"Loss analysis module not available"**  
→ Confirm `src/trade_recorder.py` and `src/loss_analysis.py` are in the `src/` folder.  
→ Check `loss_analysis.enabled: true` in your YAML config.

**Pipeline resumes unexpectedly from checkpoint**  
→ Delete `artifacts/pure_ml_stella_alpha.db` to start fresh.  
→ Or run with `--output` pointing to a new directory.

**High memory usage**  
→ Reduce `--workers` (each worker holds a copy of the full DataFrame).  
→ Reduce `max_features` in the RFE config section.

---

## How It Differs from the Signal Filter Model (V3)

| Aspect | V3 Signal Filter | Stella Alpha |
|---|---|---|
| Pre-filter | BB ≥ 0.80 AND RSI ≥ 70 | None (ML evaluates every candle) |
| Candles evaluated | ~5% of all candles | 100% |
| Timeframes | H4 only | H4 + D1 |
| TP/SL | Configurable, often equal R:R | High R:R only (≥1.67:1) |
| Win rate needed | ~50–60% | ~25–40% (breakeven-derived) |
| ML role | "Is this signal good?" | "Is this candle a good SHORT?" |

Stella Alpha is best used **alongside** V3, not instead of it. V3 takes fewer, 
higher-confidence trades; Stella Alpha hunts for high-R:R runner opportunities 
with a different entry philosophy.
