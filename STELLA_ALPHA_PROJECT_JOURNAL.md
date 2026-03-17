# STELLA ALPHA V3 - PROJECT JOURNAL

> **Purpose:** This document is a complete reference for AI assistants (Claude, GPT, etc.) to understand the full context of this ML trading project. Read this FIRST before making any changes.

> **Last Updated:** 2025-03-17

---

## 📍 QUICK STATUS

```
PROJECT:        Stella Alpha V3 - ML Trading System for EURUSD
STATUS:         ✅ READY FOR DEPLOYMENT (Paper Trade)
BEST RESULT:    +16,500 pips over 14 years (+1,179 pips/year)
NEXT STEP:      Export model to ONNX → Build MT5 EA → Paper trade
```

---

## 📁 PROJECT LOCATION

```
C:\Users\Ben Michael Oracion\AppData\Roaming\MetaQuotes\Terminal\
D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\Advisors\Solara\
Model Training\stellaalpha\v3\
```

---

## 🎯 FINAL WINNING CONFIGURATION

```
STRATEGY:       MTF Trend Aligned LONG
SIGNAL:         H4 uptrend + D1 uptrend aligned
ML MODEL:       LightGBM (25 features after RFE)
ML THRESHOLD:   0.45 (lower = more trades = more profit)

TRADE PARAMS:
├── Take Profit:  100 pips
├── Stop Loss:    50 pips
├── R:R Ratio:    2.0:1
└── Max Hold:     72 bars (12 days)

EXPECTED PERFORMANCE:
├── Win Rate:       38.4%
├── Avg Pips/Trade: +7.67
├── Trades/Year:    154
├── Annual Profit:  +1,179 pips
├── Profit Factor:  1.25
├── Sharpe Ratio:   1.67
└── Max Drawdown:   4,150 pips
```

---

## 📜 COMPLETE HISTORY

### Phase 0: Failed Attempts (V1-V4, Stella V1-V2)

| Version | Strategy | Result | Why Failed |
|---------|----------|--------|------------|
| V1-V2 | Various | ❌ Failed | Exploratory, no edge |
| V3 | BB Touch → ML Filter | ❌ Failed | Signal too weak |
| V4 | Pure ML (predict everything) | ❌ Failed | AUC ~0.52, no edge |
| Stella V1 | BB/RSI SHORT mean reversion | ❌ Failed | Only 30-pip moves predictable |
| Stella V2 | Multi-timeframe SHORT | ❌ Failed | Still couldn't predict large moves |

**Key Learning:** BB/RSI SHORT mean reversion strategy was fundamentally flawed. Could only predict 30-pip reversions, not 50+ pip moves needed for good R:R.

### Phase 1: Strategy Pivot (Critical Decision)

**Date:** Early in Stella V3 development

**Analysis Done:**
- Tested every signal type in the data
- Found MTF Trend Aligned has small but real edge (+11.6%, p<0.0001)
- Decided to PIVOT from SHORT mean reversion to LONG trend following

```
OLD STRATEGY (Failed):
├── Signal: BB Upper Touch + RSI Overbought
├── Direction: SHORT (mean reversion)
└── Problem: Fighting the trend

NEW STRATEGY (Success):
├── Signal: H4 uptrend + D1 uptrend
├── Direction: LONG (trend following)
└── Advantage: Trading WITH the trend
```

### Phase 2: AUC Optimization Attempts

| Test | Features | AUC | Notes |
|------|----------|-----|-------|
| Baseline | 215 | 0.52 | LightGBM starting point |
| Option A (TP/SL test) | 215 | **0.545** | TP=100/SL=50 best |
| Option B (every bar) | 215 | 0.519 | Worse - signal filtering helps |
| Expanded features | 362 | 0.537 | WORSE - more features = more noise |

**Key Learning:** 
- AUC ~0.54-0.55 is the ceiling for this data
- Adding more indicators (EMA, MACD, Stochastic, ADX, Ichimoku) did NOT help
- Model still preferred simple RSI/BB/D1 features
- Market is efficient - public indicators are already priced in

### Phase 3: Pivot to Profit Optimization (Option G)

**Decision:** Stop chasing AUC. Accept ~0.54 AUC and optimize for PROFIT instead.

**Option G Results:**

```
THRESHOLD ANALYSIS:
┌──────────┬────────┬────────────┬──────────┬────────┐
│ Threshold│ Trades │ Total Pips │ Avg Pips │ Sharpe │
├──────────┼────────┼────────────┼──────────┼────────┤
│ 0.45     │ 2,151  │ +16,500    │ +7.67    │ 1.67   │ ← BEST BY TOTAL
│ 0.50     │ 1,704  │ +13,650    │ +8.01    │ 1.74   │
│ 0.60     │ 892    │ +7,900     │ +8.86    │ 1.92   │
│ 0.65     │ 616    │ +5,500     │ +8.93    │ 1.93   │ ← BEST BY SHARPE
└──────────┴────────┴────────────┴──────────┴────────┘

INSIGHT: Lower threshold = More trades = More total profit
         (Even though per-trade stats are slightly worse)
```

**Trade Filter Analysis:**

```
SURPRISING FINDING: ALL filters REDUCED profits!

Baseline (no filters):     +16,500 pips
With session filters:      +10,400 pips (-37%)
With volatility filters:   +4,850 pips (-71%)
With ALL filters:          +750 pips (-95%)

CONCLUSION: The ML model is already filtering well.
            Adding manual filters removes GOOD trades.
            DO NOT add extra filters!
```

---

## 📊 KEY DATA FILES

```
v3/
├── data/
│   ├── EURUSD_H4_features.csv    # 41,023 rows, H4 candles + indicators
│   └── EURUSD_D1_features.csv    # 7,314 rows, D1 candles + indicators
│
├── combined_features.py          # 215 features (USE THIS - better performance)
├── expanded_features.py          # 362 features (DO NOT USE - worse performance)
│
├── option_g_profit_optimization.py  # Threshold + filter analysis
├── export_model_for_mt5.py          # Export to ONNX for MT5
│
└── models/                       # Output directory (after export)
    ├── stella_alpha_model.onnx   # For MT5
    ├── stella_alpha_model.pkl    # Python backup
    ├── stella_alpha_features.json
    ├── stella_alpha_scaler.json
    ├── stella_alpha_config.json
    └── stella_alpha_features.mqh # MQL5 helper
```

---

## 🔬 TECHNICAL DETAILS

### Feature Engineering

```python
# USE combined_features.py (215 features) - NOT expanded_features.py

Top Features Selected by RFE:
1. rsi_min_50          # RSI minimum over 50 bars
2. d1_time_since_last_touch
3. rsi_max_50
4. d1_bb_width_pct     # D1 volatility
5. d1_atr_pct          # D1 ATR
6. trend_strength
7. rsi_max_20
8. rsi_range_50
9. price_vs_kijun      # Ichimoku
10. close_std_50
```

### Model Configuration

```python
model = lgb.LGBMClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# RFE selects 25 features from 215
# StandardScaler normalizes features
# Threshold = 0.45 for trade decision
```

### Signal Logic

```python
# MTF Trend Aligned LONG Signal
h4_trend_up = (trend_strength > 0.3) OR (close > EMA21)
d1_trend_up = (d1_trend_strength > 0.3) OR (d1_close > d1_EMA21)

signal = h4_trend_up AND d1_trend_up  # Both timeframes agree

# D1 data uses PREVIOUS day (no lookahead bias)
```

---

## ⚠️ CRITICAL WARNINGS

### DO NOT:
1. ❌ Use expanded_features.py (362 features) - worse performance
2. ❌ Add session/day/volatility filters - reduces profit
3. ❌ Use threshold > 0.50 if you want max total pips
4. ❌ Expect AUC > 0.55 - this is the ceiling for this data
5. ❌ Go live without 1-3 months paper trading

### DO:
1. ✅ Use combined_features.py (215 features)
2. ✅ Use threshold 0.45 for maximum profit
3. ✅ Let ML model do all the filtering
4. ✅ Paper trade on demo account first
5. ✅ Expect ~38% win rate (this is FINE with 2:1 R:R)

---

## 🚀 NEXT STEPS

### Immediate (Current Session)
```
1. [ ] Install ONNX: pip install onnx onnxmltools onnxconverter-common
2. [ ] Run export: python export_model_for_mt5.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
3. [ ] Verify models/ folder created with all files
```

### Short Term (Next 1-2 Weeks)
```
4. [ ] Build MT5 EA that:
       - Calculates 25 features
       - Loads ONNX model
       - Runs inference
       - Executes trades at threshold >= 0.45
5. [ ] Deploy EA on MT5 demo account
6. [ ] Verify trades are being placed correctly
```

### Medium Term (Months 2-3)
```
7. [ ] Paper trade for 1-3 months
8. [ ] Compare live results to backtest:
       - Expected: ~13 trades/month
       - Expected: ~38% win rate
       - Expected: +98 pips/month average
9. [ ] Debug any discrepancies
```

### Long Term (Month 4+)
```
10. [ ] If paper trade matches backtest → Go live
11. [ ] Start with minimal position size (0.01 lots)
12. [ ] Scale up gradually as confidence grows
13. [ ] Monitor monthly, adjust if needed
```

---

## 📈 SUCCESS METRICS

### Paper Trade Success Criteria
```
After 3 months of paper trading, consider successful if:
├── Trades taken: 30-50 (matches expected 13/month)
├── Win rate: 30-45% (within range of 38%)
├── Total pips: Positive (even if less than backtest)
├── No major bugs or execution issues
└── Drawdown < 6,000 pips
```

### Live Trade Success Criteria
```
After 6 months of live trading:
├── Annual pips on track for +800 to +1,500
├── Max drawdown < 5,000 pips
├── Profit factor > 1.1
└── Consistent monthly results (no catastrophic months)
```

---

## 🤖 FOR AI ASSISTANTS

### If User Asks About This Project:
1. Read this journal FIRST
2. The strategy is FINALIZED - don't try to optimize AUC further
3. Focus on DEPLOYMENT (MT5 EA, paper trading)
4. Don't suggest adding more indicators - already tried, made it worse

### Common Questions:
- "Why is win rate only 38%?" → R:R is 2:1, so 38% is profitable
- "Can we improve AUC?" → No, 0.54-0.55 is the ceiling. Focus on profit.
- "Should we add more features?" → No, 362 features performed WORSE than 215
- "Why threshold 0.45?" → Maximizes total pips despite lower per-trade quality

### Key Files to Reference:
- `combined_features.py` - Feature engineering (USE THIS)
- `option_g_profit_optimization.py` - Threshold/filter analysis
- `export_model_for_mt5.py` - Model export script
- `STELLA_ALPHA_PROJECT_JOURNAL.md` - This file

---

## 📝 CHANGE LOG

| Date | Change | Result |
|------|--------|--------|
| 2025-03-17 | Pivoted from AUC optimization to profit optimization | +16,500 pips found |
| 2025-03-17 | Tested expanded features (362) | WORSE than 215 features |
| 2025-03-17 | Tested trade filters | ALL filters reduced profit |
| 2025-03-17 | Finalized config: TP=100, SL=50, Threshold=0.45 | Ready for deployment |
| 2025-03-17 | Created export script for MT5 | Pending execution |

---

*End of Journal*
