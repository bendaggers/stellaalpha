# STELLA ALPHA V2: INSTITUTIONAL-GRADE ML TRADING SYSTEM
## Complete Functional Specification

---

# DOCUMENT INFORMATION

| Field | Value |
|-------|-------|
| **Version** | 2.0.0 |
| **Status** | Ready for Implementation |
| **Created** | 2026-03-10 |
| **Purpose** | Complete specification for building Stella Alpha V2 from scratch |

---

# TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Specifications](#4-data-specifications)
5. [Phase 1: Alpha Research Module](#5-phase-1-alpha-research-module)
6. [Phase 2: Multi-Target Label Generation](#6-phase-2-multi-target-label-generation)
7. [Phase 3: Probability Model](#7-phase-3-probability-model)c
8. [Phase 4: Dynamic Strategy Engine](#8-phase-4-dynamic-strategy-engine)
9. [Phase 5: Integration & Validation](#9-phase-5-integration--validation)
10. [Configuration System](#10-configuration-system)
11. [File Structure](#11-file-structure) *(See companion document: STELLA_ALPHA_V2_FILE_ARCHITECTURE.md)*
12. [Command Line Interface](#12-command-line-interface)
13. [Output Artifacts](#13-output-artifacts)
14. [Testing Requirements](#14-testing-requirements)
15. [Implementation Checklist](#15-implementation-checklist)
16. [Appendices](#16-appendices)

---

# COMPANION DOCUMENTS

| Document | Description |
|----------|-------------|
| **STELLA_ALPHA_V2_SPECIFICATION.md** | This file - Complete functional specification |
| **STELLA_ALPHA_V2_FILE_ARCHITECTURE.md** | Detailed file structure with descriptions |

---

# 1. EXECUTIVE SUMMARY

## 1.1 What is Stella Alpha V2?

Stella Alpha V2 is an **institutional-grade** machine learning trading system for EURUSD forex trading. It represents a complete redesign from V1, shifting from retail-style direction prediction to probability-based decision making with dynamic risk management.

## 1.2 Core Philosophy

```
V1 APPROACH (Failed):                   V2 APPROACH (Institutional):
─────────────────────                   ─────────────────────────────
"Will price hit TP?"              →     "What's the probability distribution?"
Fixed TP/SL in pips               →     Dynamic TP/SL based on ATR + volatility
Binary WIN/LOSS labels            →     Multi-target probability + regression
Same strategy always              →     Adapt to volatility regime
Predict direction                 →     Predict outcome distributions
Hope for profit                   →     Calculate expected value mathematically
```

## 1.3 Key Innovations

1. **Alpha Research First** - Statistical analysis before building any model
2. **Multi-Target Labels** - Rich labels capturing full price path information
3. **Probability Predictions** - Calibrated probabilities for multiple targets
4. **Dynamic Exits** - ATR-based stops and EV-optimized take profits
5. **Kelly Position Sizing** - Scale position size by edge confidence

## 1.4 Expected Outcomes

```
TARGET METRICS:
──────────────────────────────────────────
Win Rate:           55-65% (not fixed TP/SL dependent)
Profit Factor:      1.5-2.0
Expected Value:     +5 to +15 pips per trade (dynamic)
Sharpe Ratio:       >1.5
Max Drawdown:       <15%
Calibration Error:  <5% (predicted prob matches actual)
```

---

# 2. BACKGROUND & MOTIVATION

## 2.1 What Failed in V1

Stella Alpha V1 attempted to predict large directional moves (50-150 pips) using:
- H4 Bollinger Band + RSI signals
- D1 timeframe alignment
- Binary classification (WIN/LOSS)
- Fixed pip-based TP/SL

**Result:** 0/66 configurations passed acceptance criteria.

## 2.2 Root Cause Analysis

```
FUNDAMENTAL PROBLEMS:
─────────────────────

1. WRONG QUESTION
   V1 asked: "Will price hit exactly 50 pips in 48 bars?"
   Reality: Markets don't care about arbitrary pip levels
   
2. FIXED EXITS DON'T WORK
   50 pips in low volatility = huge move (rare)
   50 pips in high volatility = small move (common)
   Same target, completely different probabilities
   
3. BINARY LABELS LOSE INFORMATION
   A trade that reached 45 pips then reversed = LOSS
   A trade that reached 5 pips then reversed = LOSS
   Both labeled identically, but completely different situations
   
4. NO PROBABILITY CALIBRATION
   Model says 60% confidence, but actual win rate is 45%
   Can't make rational decisions with uncalibrated probabilities
   
5. SAME STRATEGY EVERYWHERE
   Trending market needs different approach than ranging
   V1 used identical strategy regardless of regime
```

## 2.3 The Institutional Approach

Professional trading firms don't predict direction. They:

1. **Quantify Edge** - Statistical research to find WHERE alpha exists
2. **Estimate Distributions** - What are the possible outcomes and probabilities?
3. **Optimize Risk-Adjusted Returns** - Dynamic TP/SL that maximize EV
4. **Scale by Confidence** - Position size proportional to edge
5. **Monitor Regime** - Adapt strategy to current market conditions

---

# 3. SYSTEM ARCHITECTURE OVERVIEW

## 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STELLA ALPHA V2 PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

DATA LAYER
═══════════════════════════════════════════════════════════════════════════════
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐
│   H4 OHLCV   │    │   D1 OHLCV   │    │         DATA MERGER              │
│   25 Years   │───▶│   25 Years   │───▶│   (Leakage-safe D1 alignment)    │
│   ~41,000    │    │   ~6,500     │    │                                  │
└──────────────┘    └──────────────┘    └──────────────┬───────────────────┘
                                                       │
                                                       ▼
RESEARCH LAYER (PHASE 1)
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ALPHA RESEARCH                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Mean Reversion  │  │   Volatility    │  │   Session/MTF   │              │
│  │    Analysis     │  │  Regime Study   │  │    Analysis     │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│                   ┌─────────────────────┐                                    │
│                   │  RESEARCH REPORT    │                                    │
│                   │  (Where is alpha?)  │                                    │
│                   └─────────────────────┘                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
LABELING LAYER (PHASE 2)
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-TARGET LABEL GENERATOR                           │
│                                                                              │
│  For each candle, compute:                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Binary Targets:  │  │ Regression:      │  │ Quality Flags:   │           │
│  │ • reached_30_pip │  │ • mfe_pips       │  │ • safe_trade     │           │
│  │ • reached_50_pip │  │ • mae_pips       │  │ • clean_winner   │           │
│  │ • reached_80_pip │  │ • bars_to_target │  │ • fast_mover     │           │
│  │ • reached_100_pip│  │ • mfe_mae_ratio  │  │                  │           │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
MODEL LAYER (PHASE 3)
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PROBABILITY MODEL                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MULTI-OUTPUT PREDICTOR                           │    │
│  │                                                                     │    │
│  │  Inputs: Features (H4 + D1 + Cross-TF)                              │    │
│  │                                                                     │    │
│  │  Outputs:                                                           │    │
│  │    • P(reach 30 pips) = 0.72                                        │    │
│  │    • P(reach 50 pips) = 0.48                                        │    │
│  │    • P(reach 80 pips) = 0.29                                        │    │
│  │    • E[MFE] = 45 pips                                               │    │
│  │    • E[MAE] = 22 pips                                               │    │
│  │    • E[bars_to_target] = 15                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PROBABILITY CALIBRATOR                           │    │
│  │         (Platt Scaling / Isotonic Regression)                       │    │
│  │                                                                     │    │
│  │  Raw prediction: 0.65  →  Calibrated: 0.58                          │    │
│  │  (If model says 58%, it should win 58% of the time)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
STRATEGY LAYER (PHASE 4)
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DYNAMIC STRATEGY ENGINE                               │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  DYNAMIC SL     │  │  EV OPTIMIZER   │  │ POSITION SIZER  │              │
│  │                 │  │                 │  │                 │              │
│  │ SL = f(ATR,     │  │ Find TP that    │  │ Kelly fraction  │              │
│  │   regime,       │  │ maximizes:      │  │ based on edge   │              │
│  │   predicted_MAE)│  │ P(TP)×TP -      │  │ and variance    │              │
│  │                 │  │ P(SL)×SL        │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      TRADE DECISION                                 │    │
│  │                                                                     │    │
│  │  {                                                                  │    │
│  │    "action": "SHORT",                                               │    │
│  │    "entry": 1.0850,                                                 │    │
│  │    "stop_loss": 1.0895 (45 pips, dynamic),                          │    │
│  │    "take_profit": 1.0810 (40 pips, EV-optimized),                   │    │
│  │    "position_size": 0.09 (9% of capital, half-Kelly),               │    │
│  │    "expected_value": +12.5 pips,                                    │    │
│  │    "confidence": "HIGH"                                             │    │
│  │  }                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
VALIDATION LAYER (PHASE 5)
═══════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION & REPORTING                               │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ WALK-FORWARD    │  │ MONTE CARLO     │  │ REGIME          │              │
│  │ VALIDATION      │  │ SIMULATION      │  │ BREAKDOWN       │              │
│  │ (5 folds)       │  │ (1000 samples)  │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      FINAL REPORT                                   │    │
│  │  • Equity curve                                                     │    │
│  │  • Performance metrics (Sharpe, Sortino, Max DD)                    │    │
│  │  • Calibration analysis                                             │    │
│  │  • Regime performance breakdown                                     │    │
│  │  • Confidence intervals                                             │    │
│  │  • Deployment recommendation                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Data Flow Summary

```
Raw OHLCV Data
    │
    ▼
Merged H4+D1 (with leakage prevention)
    │
    ▼
Feature Engineering (~200 features)
    │
    ▼
Multi-Target Labels (for each candle)
    │
    ▼
Train/Calibration/Test Splits (walk-forward)
    │
    ├──▶ Train Probability Model
    │         │
    │         ▼
    ├──▶ Calibrate Probabilities
    │         │
    │         ▼
    └──▶ Dynamic Strategy (find optimal TP/SL per prediction)
              │
              ▼
         Trade Decisions + Performance Metrics
```

---

# 4. DATA SPECIFICATIONS

## 4.1 Input Data Requirements

### 4.1.1 H4 Data (Primary Timeframe)

```
FILE: data/EURUSD_H4_features.csv
ROWS: ~41,000 (approximately 25 years of 4-hour candles)
FREQUENCY: Every 4 hours (6 candles per day)

REQUIRED COLUMNS:
─────────────────────────────────────────────────────────────────────────────
Column              Type        Description                      Example
─────────────────────────────────────────────────────────────────────────────
timestamp           datetime    Candle open time                 2024.01.15 08:00:00
open                float       Opening price                    1.0850
high                float       Highest price                    1.0890
low                 float       Lowest price                     1.0840
close               float       Closing price                    1.0875
volume              int         Tick volume                      1250

OPTIONAL COLUMNS (Pre-computed by EA - preferred):
─────────────────────────────────────────────────────────────────────────────
lower_band          float       Bollinger lower (20,2)           1.0820
middle_band         float       Bollinger middle (SMA20)         1.0855
upper_band          float       Bollinger upper (20,2)           1.0890
bb_position         float       Position in BB (0-1 scale)       0.85
bb_width_pct        float       BB width as % of price           0.0065
rsi_value           float       RSI(14)                          72.5
atr_pct             float       ATR(14) as % of price            0.0035
volume_ratio        float       Volume / SMA(volume,20)          1.3
trend_strength      float       ADX-based trend measure          1.2
candle_body_pct     float       |close-open| / (high-low)        0.6
candle_rejection    float       Wick ratio (rejection signal)    0.4
─────────────────────────────────────────────────────────────────────────────
```

### 4.1.2 D1 Data (Higher Timeframe Context)

```
FILE: data/EURUSD_D1_features.csv
ROWS: ~6,500 (approximately 25 years of daily candles)
FREQUENCY: Once per day

REQUIRED COLUMNS:
─────────────────────────────────────────────────────────────────────────────
Column              Type        Description                      Example
─────────────────────────────────────────────────────────────────────────────
timestamp           datetime    Daily candle close time          2024.01.15 00:00:00
open                float       Daily open                       1.0820
high                float       Daily high                       1.0900
low                 float       Daily low                        1.0810
close               float       Daily close                      1.0878
volume              int         Daily tick volume                45000

OPTIONAL COLUMNS (Pre-computed):
─────────────────────────────────────────────────────────────────────────────
(Same indicators as H4, but calculated on D1 timeframe)
d1_bb_position, d1_rsi_value, d1_atr_pct, d1_trend_strength, etc.
─────────────────────────────────────────────────────────────────────────────
```

## 4.2 Data Quality Requirements

```
MANDATORY CHECKS:
═════════════════════════════════════════════════════════════════════════════

1. TIMESTAMP INTEGRITY
   □ Chronologically sorted (ascending)
   □ No duplicate timestamps
   □ Gaps acceptable (weekends, holidays) but flagged
   □ Format: "YYYY.MM.DD HH:MM:SS" or ISO 8601

2. PRICE INTEGRITY  
   □ high >= low (always)
   □ high >= max(open, close)
   □ low <= min(open, close)
   □ No zero or negative prices
   □ No extreme spikes (>10% in single candle) without review

3. VOLUME INTEGRITY
   □ Non-negative integers
   □ Zero volume acceptable (low liquidity periods)

4. CONTINUITY
   □ H4 data aligned to 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
   □ D1 data aligned to 00:00 UTC (NY close preferred)

5. INDICATOR VALIDITY (if pre-computed)
   □ BB position between 0 and 1 (with slight overshoot allowed)
   □ RSI between 0 and 100
   □ ATR > 0
   □ No NaN in critical columns (except warmup period)
```

## 4.3 Data Merge Specification (H4 + D1)

```
CRITICAL: Data Leakage Prevention
═════════════════════════════════════════════════════════════════════════════

PROBLEM:
D1 candle for January 15 closes at January 16 00:00 UTC.
Any H4 candle ON January 15 should NOT see January 15 D1 data.
Using same-day D1 data = FUTURE DATA = LEAKAGE.

RULE:
For H4 candle at timestamp T:
    → Use D1 candle from floor(T.date) - 1 day
    → This ensures only COMPLETED D1 data is used

EXAMPLE:
┌─────────────────────┬────────────────────┬─────────────────────┐
│ H4 Timestamp        │ H4 Date            │ D1 Data From        │
├─────────────────────┼────────────────────┼─────────────────────┤
│ 2024-01-15 04:00    │ 2024-01-15         │ 2024-01-14          │
│ 2024-01-15 08:00    │ 2024-01-15         │ 2024-01-14          │
│ 2024-01-15 12:00    │ 2024-01-15         │ 2024-01-14          │
│ 2024-01-15 16:00    │ 2024-01-15         │ 2024-01-14          │
│ 2024-01-15 20:00    │ 2024-01-15         │ 2024-01-14          │
│ 2024-01-16 00:00    │ 2024-01-16         │ 2024-01-15 ✓        │
│ 2024-01-16 04:00    │ 2024-01-16         │ 2024-01-15          │
└─────────────────────┴────────────────────┴─────────────────────┘

Note: H4 candle at exactly 00:00 on Jan 16 CAN use Jan 15 D1 data
      because the D1 candle has just closed.

IMPLEMENTATION:
```python
def merge_h4_d1_safe(df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> pd.DataFrame:
    """
    Merge D1 data into H4 data without data leakage.
    
    For each H4 candle, uses the most recent COMPLETED D1 candle,
    which means the D1 candle from the previous day.
    """
    df_h4 = df_h4.copy()
    df_d1 = df_d1.copy()
    
    # Parse timestamps
    df_h4['timestamp'] = pd.to_datetime(df_h4['timestamp'])
    df_d1['timestamp'] = pd.to_datetime(df_d1['timestamp'])
    
    # Get date of each H4 candle
    df_h4['_h4_date'] = df_h4['timestamp'].dt.normalize()
    
    # D1 data becomes available the NEXT day
    # So D1 dated Jan 15 is available from Jan 16 00:00
    df_d1['_d1_date'] = df_d1['timestamp'].dt.normalize()
    df_d1['_d1_available_from'] = df_d1['_d1_date'] + pd.Timedelta(days=1)
    
    # Rename D1 columns with 'd1_' prefix
    d1_rename = {col: f'd1_{col}' for col in df_d1.columns 
                 if col not in ['timestamp', '_d1_date', '_d1_available_from']}
    d1_rename['timestamp'] = 'd1_timestamp'
    df_d1 = df_d1.rename(columns=d1_rename)
    
    # Sort by time
    df_h4 = df_h4.sort_values('timestamp').reset_index(drop=True)
    df_d1 = df_d1.sort_values('_d1_available_from').reset_index(drop=True)
    
    # Merge using merge_asof - get most recent available D1
    df_merged = pd.merge_asof(
        df_h4,
        df_d1.drop(columns=['_d1_date']),
        left_on='_h4_date',
        right_on='_d1_available_from',
        direction='backward'
    )
    
    # Cleanup helper columns
    df_merged = df_merged.drop(columns=['_h4_date', '_d1_available_from'], errors='ignore')
    
    return df_merged
```

VALIDATION:
After merge, verify:
□ d1_timestamp.date < timestamp.date for all rows (except 00:00)
□ No NaN in d1_timestamp (except first ~50 H4 rows)
□ D1 columns properly prefixed with 'd1_'
```

## 4.4 Pip Value Configuration

```
EURUSD PIP VALUE:
═════════════════════════════════════════════════════════════════════════════
1 pip = 0.0001 (4th decimal place)
1 point = 0.00001 (5th decimal place, if available)

CONVERSIONS:
price_change_pips = (price_after - price_before) / 0.0001

EXAMPLE:
Entry: 1.0850
Exit:  1.0800
Change: 1.0800 - 1.0850 = -0.0050
Pips:   -0.0050 / 0.0001 = -50 pips (SHORT profit of 50 pips)
```

---

# 5. PHASE 1: ALPHA RESEARCH MODULE

## 5.1 Purpose

Before building ANY predictive model, analyze the historical data to find WHERE alpha (trading edge) actually exists. This prevents building models for non-existent patterns.

## 5.2 Research Questions

```
MEAN REVERSION:
□ At what BB position does reversal probability significantly increase?
□ At what RSI level does mean reversion become likely?
□ How far does price typically revert? (Distribution, not just average)
□ How long does reversion take? (Time to target analysis)

VOLATILITY:
□ Do signals perform better in high or low volatility?
□ What ATR percentile is optimal for trading?
□ How should stop loss scale with volatility?

SESSION/TIME:
□ Which trading session has best signal performance?
□ Are there day-of-week effects?
□ Are there hour-of-day effects?

MULTI-TIMEFRAME:
□ Does D1 trend alignment improve H4 signal performance?
□ By how much? (Quantify the edge)
□ Should we filter for D1 alignment?

ADVERSE EXCURSION:
□ What's the typical MAE (max adverse excursion) before a winning trade?
□ What stop loss level captures 90% of winners?
□ How does MAE vary by market conditions?
```

## 5.3 Statistical Framework

```
FOR EACH FINDING, COMPUTE:
═════════════════════════════════════════════════════════════════════════════

1. SAMPLE SIZE
   - Number of observations in each bucket/condition
   - Minimum: 100 for any statistical claim
   - Preferred: 500+ for high confidence

2. EFFECT SIZE (Cohen's d)
   d = (mean_group1 - mean_group2) / pooled_std
   
   Interpretation:
   - d < 0.2: Negligible
   - d = 0.2-0.5: Small
   - d = 0.5-0.8: Medium
   - d > 0.8: Large

3. STATISTICAL SIGNIFICANCE (p-value)
   - Use t-test for continuous comparisons
   - Use chi-square for categorical comparisons
   - Threshold: p < 0.05 for significance
   - p < 0.01 for high confidence

4. CONFIDENCE LEVEL
   - HIGH: p < 0.01 AND sample > 500 AND effect_size > 0.5
   - MEDIUM: p < 0.05 AND sample > 200
   - LOW: p < 0.10 OR sample < 200

5. ACTIONABLE INSIGHT
   - What should we DO based on this finding?
   - Specific parameter recommendations
```

## 5.4 Forward Return Analysis

```
FOR EACH H4 CANDLE, CALCULATE:
═════════════════════════════════════════════════════════════════════════════

Given: Entry at candle close price
Direction: SHORT (looking for price to drop)
Horizons: 6, 12, 24, 48, 72 bars (H4)

METRICS:
┌────────────────────────────────────────────────────────────────────────────┐
│ MFE (Max Favorable Excursion):                                             │
│   Maximum profit achieved during holding period                            │
│   For SHORT: entry_price - min(low[entry+1 : entry+horizon])               │
│                                                                            │
│ MAE (Max Adverse Excursion):                                               │
│   Maximum drawdown experienced during holding period                       │
│   For SHORT: max(high[entry+1 : entry+horizon]) - entry_price              │
│                                                                            │
│ Final Return:                                                              │
│   Profit/loss at horizon end                                               │
│   For SHORT: entry_price - close[entry+horizon]                            │
│                                                                            │
│ Time to Target:                                                            │
│   Bars until price reached X pips profit                                   │
│   For SHORT: first bar where entry_price - low >= X pips                   │
└────────────────────────────────────────────────────────────────────────────┘

CONVERT TO PIPS:
mfe_pips = mfe / 0.0001
mae_pips = mae / 0.0001
```

## 5.5 Mean Reversion Analysis

```python
def research_mean_reversion(df: pd.DataFrame) -> List[ResearchFinding]:
    """
    Analyze mean reversion patterns.
    
    Buckets to analyze:
    - BB Position: [0.90-0.95], [0.95-0.98], [0.98-1.00], [>1.00]
    - RSI: [65-70], [70-75], [75-80], [80-85], [85+]
    - Combined: BB >= 0.95 AND RSI >= 70
    
    For each bucket, calculate:
    - P(reach 30 pips) within 48 bars
    - P(reach 50 pips) within 48 bars
    - P(reach 80 pips) within 48 bars
    - Average MFE
    - Average MAE
    - MFE/MAE ratio
    - Edge vs baseline
    """
    
    # BUCKET ANALYSIS
    bb_buckets = [
        (0.90, 0.95, 'BB 90-95%'),
        (0.95, 0.98, 'BB 95-98%'),
        (0.98, 1.00, 'BB 98-100%'),
        (1.00, 1.10, 'BB >100%'),
    ]
    
    # Calculate baseline (all candles)
    baseline_prob_30 = (df['mfe_short_48_pips'] >= 30).mean()
    
    results = {}
    for low, high, label in bb_buckets:
        mask = (df['bb_position'] >= low) & (df['bb_position'] < high)
        subset = df[mask]
        
        if len(subset) >= 100:
            results[label] = {
                'count': len(subset),
                'prob_30_pip': (subset['mfe_short_48_pips'] >= 30).mean(),
                'prob_50_pip': (subset['mfe_short_48_pips'] >= 50).mean(),
                'prob_80_pip': (subset['mfe_short_48_pips'] >= 80).mean(),
                'avg_mfe': subset['mfe_short_48_pips'].mean(),
                'avg_mae': subset['mae_short_48_pips'].mean(),
                'mfe_mae_ratio': avg_mfe / avg_mae,
                'edge_vs_baseline': prob_30_pip - baseline_prob_30,
            }
    
    # Statistical test between best bucket and rest
    # ... (t-test, effect size calculation)
    
    return findings
```

## 5.6 Volatility Regime Analysis

```python
def research_volatility_regimes(df: pd.DataFrame) -> List[ResearchFinding]:
    """
    Analyze how volatility affects signal performance.
    
    Volatility measure: ATR percentile (rolling 100-bar rank)
    
    Regimes:
    - Low Vol: 0-25th percentile
    - Below Avg: 25-50th percentile  
    - Above Avg: 50-75th percentile
    - High Vol: 75-100th percentile
    
    For signals (BB >= 0.95 AND RSI >= 70), compare:
    - Win rate by regime
    - MFE/MAE ratio by regime
    - Optimal stop loss by regime
    """
    
    # Calculate ATR percentile
    df['atr_percentile'] = df['atr_pct'].rolling(100).rank(pct=True)
    
    vol_regimes = [
        (0.00, 0.25, 'Low Vol'),
        (0.25, 0.50, 'Below Avg'),
        (0.50, 0.75, 'Above Avg'),
        (0.75, 1.00, 'High Vol'),
    ]
    
    signal_mask = (df['bb_position'] >= 0.95) & (df['rsi_value'] >= 70)
    
    results = {}
    for low, high, label in vol_regimes:
        mask = signal_mask & (df['atr_percentile'] >= low) & (df['atr_percentile'] < high)
        subset = df[mask]
        
        if len(subset) >= 50:
            results[label] = {
                'count': len(subset),
                'prob_30_pip': (subset['mfe_short_48_pips'] >= 30).mean(),
                'avg_mfe': subset['mfe_short_48_pips'].mean(),
                'avg_mae': subset['mae_short_48_pips'].mean(),
                'mfe_mae_ratio': avg_mfe / avg_mae,
            }
    
    return findings
```

## 5.7 ATR-Based Stop Loss Analysis

```python
def research_atr_stop_loss(df: pd.DataFrame) -> List[ResearchFinding]:
    """
    Analyze what ATR multiple best captures winning trades.
    
    For signals, calculate:
    - What % of trades have MAE < 0.5x ATR?
    - What % of trades have MAE < 1.0x ATR?
    - What % of trades have MAE < 1.5x ATR?
    - What % of trades have MAE < 2.0x ATR?
    
    This tells us: If we set SL = 1.5x ATR, what % of winners survive?
    """
    
    signal_mask = (df['bb_position'] >= 0.95) & (df['rsi_value'] >= 70)
    signal_df = df[signal_mask].copy()
    
    # Convert ATR to pips
    signal_df['atr_pips'] = signal_df['atr_pct'] / 0.0001
    
    atr_multiples = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    coverage = {}
    
    for mult in atr_multiples:
        sl_level = signal_df['atr_pips'] * mult
        # What % of trades had MAE below this SL?
        coverage[f'{mult}x ATR'] = (signal_df['mae_short_48_pips'] <= sl_level).mean()
    
    return findings
```

## 5.8 Multi-Timeframe Analysis

```python
def research_mtf_effects(df: pd.DataFrame) -> List[ResearchFinding]:
    """
    Analyze D1 alignment impact on H4 signals.
    
    For H4 SHORT signals (BB >= 0.95, RSI >= 70), compare:
    - Performance when D1 is bearish (supports SHORT)
    - Performance when D1 is bullish (opposes SHORT)
    - Performance when D1 is neutral
    
    D1 trend determination:
    - Bearish: D1 close < D1 MA20 (price below daily moving average)
    - Bullish: D1 close > D1 MA20
    - Neutral: |D1 close - D1 MA20| < 0.1% of price
    """
    
    # Calculate D1 trend if not present
    if 'd1_trend_direction' not in df.columns:
        d1_ma20 = df['d1_close'].rolling(20).mean()
        df['d1_trend_direction'] = np.sign(df['d1_close'] - d1_ma20)
    
    signal_mask = (df['bb_position'] >= 0.95) & (df['rsi_value'] >= 70)
    
    # D1 bearish (supports our SHORT)
    bearish_mask = signal_mask & (df['d1_trend_direction'] < 0)
    # D1 bullish (opposes our SHORT)
    bullish_mask = signal_mask & (df['d1_trend_direction'] > 0)
    
    d1_impact = {
        'd1_bearish': {
            'count': bearish_mask.sum(),
            'prob_30_pip': (df[bearish_mask]['mfe_short_48_pips'] >= 30).mean(),
            'avg_mfe': df[bearish_mask]['mfe_short_48_pips'].mean(),
        },
        'd1_bullish': {
            'count': bullish_mask.sum(),
            'prob_30_pip': (df[bullish_mask]['mfe_short_48_pips'] >= 30).mean(),
            'avg_mfe': df[bullish_mask]['mfe_short_48_pips'].mean(),
        },
    }
    
    edge_from_alignment = d1_impact['d1_bearish']['prob_30_pip'] - d1_impact['d1_bullish']['prob_30_pip']
    
    return findings
```

## 5.9 Research Output Format

```json
{
  "generated_at": "2026-03-10T12:00:00",
  "data_range": "2000-01-01 to 2025-12-31",
  "total_candles": 41523,
  
  "findings": {
    "mean_reversion": [
      {
        "name": "BB Position Effect",
        "description": "BB position >= 95% shows 68% probability of 30-pip move vs 56% baseline",
        "sample_size": 2341,
        "effect_size": 0.45,
        "p_value": 0.0001,
        "is_significant": true,
        "confidence": "HIGH",
        "actionable_insight": "Filter for BB position >= 0.95 adds 12% edge",
        "raw_data": {
          "bb_buckets": {
            "BB 90-95%": {"count": 3200, "prob_30_pip": 0.58},
            "BB 95-98%": {"count": 1800, "prob_30_pip": 0.65},
            "BB 98-100%": {"count": 541, "prob_30_pip": 0.72},
            "BB >100%": {"count": 89, "prob_30_pip": 0.78}
          }
        }
      },
      {
        "name": "Combined BB + RSI Signal",
        "description": "BB >= 0.95 AND RSI >= 70: 72% prob of 30 pips, 48% prob of 50 pips",
        "sample_size": 1245,
        "effect_size": 0.52,
        "p_value": 0.0001,
        "is_significant": true,
        "confidence": "HIGH",
        "actionable_insight": "Use combined signal as primary entry criteria",
        "raw_data": {
          "prob_30": 0.72,
          "prob_50": 0.48,
          "prob_80": 0.29,
          "avg_mfe": 52.3,
          "avg_mae": 28.7,
          "mfe_mae_ratio": 1.82
        }
      }
    ],
    
    "volatility": [
      {
        "name": "Volatility Regime Impact",
        "description": "High volatility regime shows MFE/MAE = 1.95 vs Low volatility = 1.32",
        "actionable_insight": "Prefer Above Avg and High Vol regimes. Avoid Low Vol.",
        "raw_data": {
          "Low Vol": {"prob_30": 0.58, "mfe_mae_ratio": 1.32},
          "Below Avg": {"prob_30": 0.62, "mfe_mae_ratio": 1.45},
          "Above Avg": {"prob_30": 0.68, "mfe_mae_ratio": 1.72},
          "High Vol": {"prob_30": 0.71, "mfe_mae_ratio": 1.95}
        }
      },
      {
        "name": "ATR-Based Stop Loss",
        "description": "1.5x ATR stop captures 90% of winning trades",
        "actionable_insight": "Use SL = 1.5x ATR instead of fixed pips",
        "raw_data": {
          "0.5x ATR": {"coverage": 0.52},
          "1.0x ATR": {"coverage": 0.78},
          "1.5x ATR": {"coverage": 0.90},
          "2.0x ATR": {"coverage": 0.96}
        }
      }
    ],
    
    "mtf": [
      {
        "name": "D1 Trend Alignment",
        "description": "D1 bearish: 74% win rate vs D1 bullish: 52% win rate",
        "effect_size": 0.22,
        "is_significant": true,
        "confidence": "HIGH",
        "actionable_insight": "D1 alignment adds 22% edge - CRITICAL FILTER!",
        "raw_data": {
          "d1_bearish": {"count": 605, "prob_30": 0.74},
          "d1_bullish": {"count": 640, "prob_30": 0.52}
        }
      }
    ],
    
    "session": [
      {
        "name": "Session Performance",
        "description": "Overlap session: 73% win rate, Asian: 58% win rate",
        "actionable_insight": "Prefer London and Overlap sessions",
        "raw_data": {
          "Asian": {"count": 312, "prob_30": 0.58},
          "London": {"count": 458, "prob_30": 0.68},
          "NY": {"count": 321, "prob_30": 0.65},
          "Overlap": {"count": 154, "prob_30": 0.73}
        }
      }
    ]
  },
  
  "recommended_strategy": {
    "entry_filters": [
      {"filter": "BB position >= 0.95", "edge": 0.12},
      {"filter": "RSI >= 70", "edge": 0.08},
      {"filter": "D1 trend bearish", "edge": 0.22},
      {"filter": "ATR percentile >= 50", "edge": 0.10}
    ],
    "exit_strategy": {
      "stop_loss": {
        "method": "ATR-based",
        "multiplier": 1.5,
        "expected_coverage": 0.90
      },
      "take_profit": {
        "method": "EV-optimized",
        "typical_range": "30-60 pips depending on volatility"
      }
    },
    "expected_edge": {
      "combined_win_rate": 0.68,
      "avg_mfe": 52,
      "avg_mae": 28
    }
  }
}
```

## 5.10 Phase 1 Files to Create

```
FILES:
═════════════════════════════════════════════════════════════════════════════

src/research/
├── alpha_research.py       # Main research module
├── forward_returns.py      # MFE/MAE calculation
├── mean_reversion.py       # Mean reversion analysis
├── volatility_analysis.py  # Volatility regime analysis
├── session_analysis.py     # Session/time analysis
├── mtf_analysis.py         # Multi-timeframe analysis
└── research_report.py      # Report generation

run_alpha_research.py       # CLI entry point

OUTPUT:
artifacts/alpha_research_report.json
```

---

# 6. PHASE 2: MULTI-TARGET LABEL GENERATION

## 6.1 Purpose

Replace binary WIN/LOSS labels with rich multi-target labels that capture the full price path information. This enables probability prediction and dynamic exit optimization.

## 6.2 Label Types

```
BINARY TARGETS (Classification):
═════════════════════════════════════════════════════════════════════════════
For each candle, at each horizon (6, 12, 24, 48, 72 bars):

reached_30_pips     0/1     Did MFE reach 30 pips?
reached_50_pips     0/1     Did MFE reach 50 pips?
reached_80_pips     0/1     Did MFE reach 80 pips?
reached_100_pips    0/1     Did MFE reach 100 pips?
reached_150_pips    0/1     Did MFE reach 150 pips?


REGRESSION TARGETS (Continuous):
═════════════════════════════════════════════════════════════════════════════
mfe_pips            float   Max favorable excursion in pips
mae_pips            float   Max adverse excursion in pips
final_return_pips   float   Return at horizon end
bars_to_30_pips     int     Bars to reach 30 pips (NaN if never reached)
bars_to_50_pips     int     Bars to reach 50 pips (NaN if never reached)
mfe_mae_ratio       float   MFE / MAE (trade quality score)


QUALITY FLAGS (Derived):
═════════════════════════════════════════════════════════════════════════════
safe_30_pips        0/1     Reached 30 pips with MAE < 25 pips
safe_50_pips        0/1     Reached 50 pips with MAE < 35 pips
clean_winner        0/1     Smooth path, no deep drawdown
fast_30_pips        0/1     Reached 30 pips in < 12 bars
```

## 6.3 Label Calculation Logic

```python
def generate_multi_target_labels(
    df: pd.DataFrame,
    max_holding_bars: int = 48,
    pip_value: float = 0.0001
) -> pd.DataFrame:
    """
    Generate multi-target labels for each candle.
    
    Args:
        df: DataFrame with OHLC data
        max_holding_bars: Maximum holding period
        pip_value: Value of 1 pip (0.0001 for EURUSD)
    
    Returns:
        DataFrame with all label columns added
    """
    result = df.copy()
    n = len(df)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Initialize arrays
    mfe_pips = np.full(n, np.nan)
    mae_pips = np.full(n, np.nan)
    final_return_pips = np.full(n, np.nan)
    bars_to_30 = np.full(n, np.nan)
    bars_to_50 = np.full(n, np.nan)
    
    # For each candle (entry point)
    for i in range(n - max_holding_bars):
        entry_price = close[i]
        
        # Look forward over holding period
        future_highs = high[i+1 : i+1+max_holding_bars]
        future_lows = low[i+1 : i+1+max_holding_bars]
        future_closes = close[i+1 : i+1+max_holding_bars]
        
        # For SHORT direction:
        # MFE = entry - min(future_lows) (price dropped, we profit)
        # MAE = max(future_highs) - entry (price rose, we lose)
        
        mfe = entry_price - np.min(future_lows)
        mae = np.max(future_highs) - entry_price
        final_ret = entry_price - future_closes[-1]
        
        mfe_pips[i] = mfe / pip_value
        mae_pips[i] = mae / pip_value
        final_return_pips[i] = final_ret / pip_value
        
        # Time to reach targets
        for bar_idx in range(len(future_lows)):
            favorable_move = (entry_price - future_lows[bar_idx]) / pip_value
            
            if np.isnan(bars_to_30[i]) and favorable_move >= 30:
                bars_to_30[i] = bar_idx + 1
            if np.isnan(bars_to_50[i]) and favorable_move >= 50:
                bars_to_50[i] = bar_idx + 1
    
    # Add to dataframe
    result['mfe_pips'] = mfe_pips
    result['mae_pips'] = mae_pips
    result['final_return_pips'] = final_return_pips
    result['bars_to_30_pips'] = bars_to_30
    result['bars_to_50_pips'] = bars_to_50
    
    # Binary targets
    result['reached_30_pips'] = (mfe_pips >= 30).astype(int)
    result['reached_50_pips'] = (mfe_pips >= 50).astype(int)
    result['reached_80_pips'] = (mfe_pips >= 80).astype(int)
    result['reached_100_pips'] = (mfe_pips >= 100).astype(int)
    
    # Quality flags
    result['safe_30_pips'] = ((mfe_pips >= 30) & (mae_pips < 25)).astype(int)
    result['safe_50_pips'] = ((mfe_pips >= 50) & (mae_pips < 35)).astype(int)
    result['mfe_mae_ratio'] = mfe_pips / mae_pips.replace(0, np.nan)
    result['clean_winner'] = ((mfe_pips >= 30) & (result['mfe_mae_ratio'] > 1.5)).astype(int)
    result['fast_30_pips'] = (bars_to_30 <= 12).astype(int)
    
    return result
```

## 6.4 Multi-Horizon Labels

```python
def generate_labels_multiple_horizons(
    df: pd.DataFrame,
    horizons: List[int] = [12, 24, 48, 72]
) -> pd.DataFrame:
    """
    Generate labels for multiple holding periods.
    
    Creates columns like:
    - mfe_12_pips, mfe_24_pips, mfe_48_pips, mfe_72_pips
    - reached_30_pips_12, reached_30_pips_24, etc.
    """
    result = df.copy()
    
    for horizon in horizons:
        labels = generate_multi_target_labels(df, max_holding_bars=horizon)
        
        # Rename columns with horizon suffix
        for col in ['mfe_pips', 'mae_pips', 'reached_30_pips', 'reached_50_pips']:
            result[f'{col}_{horizon}'] = labels[col]
    
    return result
```

## 6.5 Signal Filtering

```python
def filter_to_signals(
    df: pd.DataFrame,
    bb_min: float = 0.95,
    rsi_min: float = 70,
    require_d1_alignment: bool = True
) -> pd.DataFrame:
    """
    Filter to only rows where we would consider trading.
    
    This reduces dataset to ~2-5% of candles (actual trading opportunities).
    """
    # Base signal condition
    mask = (df['bb_position'] >= bb_min) & (df['rsi_value'] >= rsi_min)
    
    # Optional D1 alignment
    if require_d1_alignment and 'd1_trend_direction' in df.columns:
        mask = mask & (df['d1_trend_direction'] < 0)  # D1 bearish for SHORT
    
    return df[mask].copy()
```

## 6.6 Label Statistics Summary

```python
def summarize_labels(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for labels.
    
    Useful for understanding data distribution before modeling.
    """
    return {
        'total_candles': len(df),
        'signal_candles': (df['bb_position'] >= 0.95).sum(),
        
        'prob_30_pips': df['reached_30_pips'].mean(),
        'prob_50_pips': df['reached_50_pips'].mean(),
        'prob_80_pips': df['reached_80_pips'].mean(),
        
        'avg_mfe': df['mfe_pips'].mean(),
        'avg_mae': df['mae_pips'].mean(),
        'avg_mfe_mae_ratio': df['mfe_mae_ratio'].mean(),
        
        'avg_bars_to_30': df['bars_to_30_pips'].mean(),
        'pct_safe_30': df['safe_30_pips'].mean(),
        'pct_clean_winner': df['clean_winner'].mean(),
    }
```

## 6.7 Phase 2 Files to Create

```
FILES:
═════════════════════════════════════════════════════════════════════════════

src/labels/
├── multi_target_labels.py  # Main label generation
├── forward_metrics.py      # MFE/MAE calculation
├── quality_flags.py        # Safe/clean/fast flags
└── label_statistics.py     # Summary statistics

run_generate_labels.py      # CLI entry point

OUTPUT:
data/EURUSD_H4_labeled.csv  # H4 data with all labels
artifacts/label_statistics.json
```

---

# 7. PHASE 3: PROBABILITY MODEL

## 7.1 Purpose

Train a model that predicts PROBABILITY DISTRIBUTIONS instead of binary outcomes. The model outputs calibrated probabilities for multiple targets.

## 7.2 Model Architecture

```
MULTI-OUTPUT MODEL:
═════════════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────────────────┐
                    │            INPUT FEATURES               │
                    │  (H4 + D1 + Cross-TF: ~100-200 features)│
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │         SHARED FEATURE LAYER            │
                    │     (Feature selection / embedding)      │
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │ CLASSIFIER HEAD 1 │ │ CLASSIFIER HEAD 2 │ │ REGRESSION HEAD   │
        │  P(reach 30 pips) │ │  P(reach 50 pips) │ │  E[MFE], E[MAE]   │
        └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                  │                     │                     │
                  ▼                     ▼                     ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │   CALIBRATOR 1    │ │   CALIBRATOR 2    │ │  (no calibration) │
        │ (Platt scaling)   │ │ (Platt scaling)   │ │                   │
        └─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                  │                     │                     │
                  └─────────────────────┼─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            MODEL OUTPUT                 │
                    │  {                                      │
                    │    "p_reach_30": 0.72,                  │
                    │    "p_reach_50": 0.48,                  │
                    │    "p_reach_80": 0.29,                  │
                    │    "expected_mfe": 45.2,                │
                    │    "expected_mae": 22.8                 │
                    │  }                                      │
                    └─────────────────────────────────────────┘
```

## 7.3 Implementation Options

```
OPTION A: Separate Models (Recommended for simplicity)
═════════════════════════════════════════════════════════════════════════════

Train separate model for each target:
- Model 1: LightGBM classifier for reached_30_pips
- Model 2: LightGBM classifier for reached_50_pips
- Model 3: LightGBM classifier for reached_80_pips
- Model 4: LightGBM regressor for mfe_pips
- Model 5: LightGBM regressor for mae_pips

Pros: Simple, proven, each model optimized for its target
Cons: No shared learning, more models to manage


OPTION B: Multi-Output Model
═════════════════════════════════════════════════════════════════════════════

Single model predicting all outputs:
- Use MultiOutputClassifier or custom architecture
- Share feature representations

Pros: Efficiency, shared learning
Cons: More complex, potential target interference


OPTION C: Neural Network Multi-Task Learning
═════════════════════════════════════════════════════════════════════════════

Neural network with shared layers and task-specific heads.

Pros: Best potential accuracy with enough data
Cons: Requires more data, more complex to tune


RECOMMENDATION: Start with Option A (separate models), upgrade later if needed.
```

## 7.4 Model Training Specification

```python
class ProbabilityModelTrainer:
    """
    Train probability models for multiple targets.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.calibrators = {}
        self.feature_columns = []
    
    def train(
        self,
        df_train: pd.DataFrame,
        df_calibration: pd.DataFrame,
        targets: List[str] = ['reached_30_pips', 'reached_50_pips', 'reached_80_pips']
    ):
        """
        Train models on training set, calibrate on calibration set.
        
        CRITICAL: Calibration MUST be done on held-out data, not training data.
        """
        # Get feature columns
        self.feature_columns = self._get_feature_columns(df_train)
        
        X_train = df_train[self.feature_columns]
        X_cal = df_calibration[self.feature_columns]
        
        for target in targets:
            y_train = df_train[target]
            y_cal = df_calibration[target]
            
            # Train classifier
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, y_train)
            self.models[target] = model
            
            # Calibrate probabilities
            raw_probs = model.predict_proba(X_cal)[:, 1]
            calibrator = CalibratedClassifierCV(
                model, 
                method='sigmoid',  # Platt scaling
                cv='prefit'
            )
            calibrator.fit(X_cal, y_cal)
            self.calibrators[target] = calibrator
    
    def predict_probabilities(self, X: pd.DataFrame) -> dict:
        """
        Get calibrated probability predictions for all targets.
        """
        predictions = {}
        
        for target, model in self.models.items():
            calibrator = self.calibrators[target]
            prob = calibrator.predict_proba(X[self.feature_columns])[:, 1]
            predictions[f'p_{target}'] = prob
        
        return predictions
```

## 7.5 Probability Calibration

```
WHAT IS CALIBRATION?
═════════════════════════════════════════════════════════════════════════════

Uncalibrated: Model says 70% probability, but actual win rate is 55%.
Calibrated: Model says 70% probability, actual win rate is ~70%.

WHY IT MATTERS:
- Position sizing relies on accurate probabilities
- EV calculations require true probabilities
- Decision thresholds need calibrated inputs


CALIBRATION METHODS:
═════════════════════════════════════════════════════════════════════════════

1. PLATT SCALING (Sigmoid)
   - Fits sigmoid: P_cal = 1 / (1 + exp(A * P_raw + B))
   - Works well for imbalanced data
   - Smooth, monotonic transformation
   
2. ISOTONIC REGRESSION
   - Non-parametric, piecewise constant
   - More flexible, can handle non-sigmoid relationships
   - Needs more calibration data

3. TEMPERATURE SCALING
   - P_cal = softmax(logits / T)
   - Simple, single parameter
   - Good for neural networks


EVALUATION METRICS:
═════════════════════════════════════════════════════════════════════════════

1. BRIER SCORE
   brier = mean((predicted_prob - actual_outcome)^2)
   Lower is better (0 = perfect)
   
2. EXPECTED CALIBRATION ERROR (ECE)
   Divide predictions into bins, compare predicted vs actual in each bin
   ECE = sum(|bin_accuracy - bin_confidence| * bin_count / total)
   Lower is better (0 = perfect)

3. CALIBRATION CURVE
   Plot predicted probability (x) vs actual frequency (y)
   Perfect calibration = diagonal line
```

```python
def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    Evaluate probability calibration.
    """
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() > 0:
            bin_confidence = y_prob[mask].mean()
            bin_accuracy = y_true[mask].mean()
            ece += abs(bin_accuracy - bin_confidence) * mask.sum() / len(y_true)
    
    # Calibration curve data
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    return {
        'brier_score': brier,
        'ece': ece,
        'calibration_curve': {
            'predicted': prob_pred.tolist(),
            'actual': prob_true.tolist()
        }
    }
```

## 7.6 Walk-Forward Training

```
WALK-FORWARD CROSS-VALIDATION:
═════════════════════════════════════════════════════════════════════════════

Purpose: Train on past, test on future (simulate real trading).

DATA SPLIT (25 years):
├── Fold 1: Train [2000-2014], Calibrate [2014-2016], Test [2016-2018]
├── Fold 2: Train [2000-2016], Calibrate [2016-2018], Test [2018-2020]
├── Fold 3: Train [2000-2018], Calibrate [2018-2020], Test [2020-2022]
├── Fold 4: Train [2000-2020], Calibrate [2020-2022], Test [2022-2024]
└── Fold 5: Train [2000-2022], Calibrate [2022-2024], Test [2024-2025]

CRITICAL RULES:
1. Training data ALWAYS before calibration data
2. Calibration data ALWAYS before test data
3. Never use future data for any calculation
4. Recalibrate for each fold (calibration can drift)
```

```python
def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    calibration_pct: float = 0.15,
    test_years: int = 2
) -> List[dict]:
    """
    Generate walk-forward splits.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    timestamps = pd.to_datetime(df['timestamp'])
    
    min_date = timestamps.min()
    max_date = timestamps.max()
    
    splits = []
    
    for fold in range(n_folds):
        # Calculate boundaries
        test_end = max_date - pd.Timedelta(days=(n_folds - fold - 1) * test_years * 365)
        test_start = test_end - pd.Timedelta(days=test_years * 365)
        
        # Calibration before test
        cal_duration = (test_start - min_date) * calibration_pct
        cal_start = test_start - cal_duration
        
        # Training before calibration
        train_start = min_date
        train_end = cal_start
        
        splits.append({
            'fold': fold + 1,
            'train_idx': (timestamps >= train_start) & (timestamps < train_end),
            'cal_idx': (timestamps >= cal_start) & (timestamps < test_start),
            'test_idx': (timestamps >= test_start) & (timestamps < test_end),
        })
    
    return splits
```

## 7.7 Phase 3 Files to Create

```
FILES:
═════════════════════════════════════════════════════════════════════════════

src/models/
├── probability_model.py    # Multi-target probability model
├── calibration.py          # Platt scaling, isotonic regression
├── walk_forward.py         # Walk-forward CV implementation
└── model_persistence.py    # Save/load models

src/evaluation/
├── probability_metrics.py  # Brier score, ECE
├── calibration_curves.py   # Visualization
└── model_comparison.py     # Compare model variants

run_train_model.py          # CLI entry point

OUTPUT:
artifacts/models/
├── prob_model_fold_1.pkl
├── prob_model_fold_2.pkl
├── ...
├── calibration_fold_1.pkl
└── calibration_metrics.json
```

---

# 8. PHASE 4: DYNAMIC STRATEGY ENGINE

## 8.1 Purpose

Use probability predictions to make optimal trading decisions: dynamic stop loss, EV-optimized take profit, and Kelly-based position sizing.

## 8.2 Dynamic Stop Loss

```
CORE PRINCIPLE:
═════════════════════════════════════════════════════════════════════════════

Fixed stop losses (30 pips, 50 pips) ignore market conditions.
Dynamic stops adapt to current volatility and expected price behavior.

DYNAMIC SL FORMULA:
SL = max(
    ATR_multiple × current_ATR,
    predicted_MAE × safety_factor,
    min_sl_pips
)

PARAMETERS:
- ATR_multiple: 1.5 (captures 90% of winning trades based on research)
- safety_factor: 1.2 (buffer above predicted MAE)
- min_sl_pips: 15 (never less than this)


EXAMPLE:
─────────────────────────────────────────────────────────────────────────────
Current ATR = 35 pips
Model predicted MAE = 28 pips
Volatility regime = HIGH

SL = max(
    1.5 × 35,      = 52.5 pips (ATR-based)
    28 × 1.2,      = 33.6 pips (MAE-based)
    15             = 15 pips (minimum)
)

Final SL = 52.5 pips (rounded to 53 pips)
```

```python
def calculate_dynamic_sl(
    current_atr_pips: float,
    predicted_mae: float,
    volatility_percentile: float,
    atr_multiple: float = 1.5,
    mae_safety_factor: float = 1.2,
    min_sl: float = 15,
    max_sl: float = 100
) -> float:
    """
    Calculate dynamic stop loss.
    """
    # Base calculations
    atr_based_sl = atr_multiple * current_atr_pips
    mae_based_sl = predicted_mae * mae_safety_factor
    
    # Adjust for volatility regime
    if volatility_percentile > 0.75:
        # High vol: use larger of ATR and MAE
        base_sl = max(atr_based_sl, mae_based_sl)
    elif volatility_percentile < 0.25:
        # Low vol: tighter stops
        base_sl = min(atr_based_sl, mae_based_sl) * 1.1
    else:
        # Normal: average
        base_sl = (atr_based_sl + mae_based_sl) / 2
    
    # Apply bounds
    final_sl = np.clip(base_sl, min_sl, max_sl)
    
    return round(final_sl, 1)
```

## 8.3 EV-Optimized Take Profit

```
CORE PRINCIPLE:
═════════════════════════════════════════════════════════════════════════════

For each possible TP level, calculate expected value:
EV(TP) = P(reach TP) × TP - P(stopped out) × SL

Where:
- P(reach TP) comes from model (e.g., P(reach 50 pips))
- P(stopped out) = 1 - P(reach TP)  # Simplified assumption
- TP and SL are in pips

Find TP that maximizes EV while meeting minimum requirements.


EXAMPLE:
─────────────────────────────────────────────────────────────────────────────
Given: SL = 40 pips (dynamic)

Model predictions:
- P(reach 30 pips) = 0.72
- P(reach 50 pips) = 0.48
- P(reach 80 pips) = 0.29
- P(reach 100 pips) = 0.18

EV Calculations:
TP=30: EV = 0.72 × 30 - 0.28 × 40 = 21.6 - 11.2 = +10.4 pips
TP=50: EV = 0.48 × 50 - 0.52 × 40 = 24.0 - 20.8 = +3.2 pips
TP=80: EV = 0.29 × 80 - 0.71 × 40 = 23.2 - 28.4 = -5.2 pips
TP=100: EV = 0.18 × 100 - 0.82 × 40 = 18.0 - 32.8 = -14.8 pips

OPTIMAL TP = 30 pips (highest EV = +10.4)

But wait - TP=30 means R:R = 30/40 = 0.75:1 (unfavorable)
TP=50 has R:R = 50/40 = 1.25:1 and still positive EV

DECISION: Choose TP=50 for better R:R while maintaining positive EV
```

```python
def optimize_take_profit(
    probabilities: dict,
    sl_pips: float,
    target_levels: List[int] = [30, 40, 50, 60, 80, 100],
    min_rr_ratio: float = 1.0,
    min_ev: float = 0.0
) -> dict:
    """
    Find optimal take profit level.
    
    Args:
        probabilities: Dict with P(reach X pips) for each level
        sl_pips: Dynamic stop loss in pips
        target_levels: TP levels to evaluate
        min_rr_ratio: Minimum risk:reward ratio
        min_ev: Minimum expected value
    
    Returns:
        Optimal TP with analysis
    """
    results = []
    
    for tp in target_levels:
        # Get probability (interpolate if needed)
        p_reach = get_probability_for_level(probabilities, tp)
        p_stop = 1 - p_reach
        
        # Calculate EV
        ev = p_reach * tp - p_stop * sl_pips
        
        # Calculate R:R ratio
        rr_ratio = tp / sl_pips
        
        # Check if meets criteria
        is_valid = (rr_ratio >= min_rr_ratio) and (ev >= min_ev)
        
        results.append({
            'tp_pips': tp,
            'probability': p_reach,
            'sl_pips': sl_pips,
            'rr_ratio': rr_ratio,
            'expected_value': ev,
            'is_valid': is_valid
        })
    
    # Filter valid options
    valid_options = [r for r in results if r['is_valid']]
    
    if not valid_options:
        return {'action': 'NO_TRADE', 'reason': 'No valid TP meets criteria'}
    
    # Choose best (highest EV among valid)
    best = max(valid_options, key=lambda x: x['expected_value'])
    
    return {
        'action': 'TRADE',
        'optimal_tp': best['tp_pips'],
        'probability': best['probability'],
        'expected_value': best['expected_value'],
        'rr_ratio': best['rr_ratio'],
        'all_options': results
    }
```

## 8.4 Kelly Criterion Position Sizing

```
KELLY CRITERION:
═════════════════════════════════════════════════════════════════════════════

Kelly fraction = (p × b - q) / b

Where:
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (TP / SL)

This tells you what % of capital to risk to maximize long-term growth.


EXAMPLE:
─────────────────────────────────────────────────────────────────────────────
P(win) = 0.60
TP = 50 pips, SL = 40 pips, b = 1.25

kelly = (0.60 × 1.25 - 0.40) / 1.25
      = (0.75 - 0.40) / 1.25
      = 0.35 / 1.25
      = 0.28 (28% of capital)

HALF-KELLY (recommended):
kelly_half = 0.28 / 2 = 0.14 (14% of capital)

Half-Kelly reduces volatility while sacrificing only ~25% of optimal growth.


POSITION SIZE CALCULATION:
─────────────────────────────────────────────────────────────────────────────
Account balance = $10,000
Half-Kelly fraction = 0.14
SL = 40 pips

Risk amount = $10,000 × 0.14 = $1,400
Position size = Risk amount / (SL pips × pip value)
             = $1,400 / (40 × $10)  # $10 per pip per lot
             = 3.5 lots
```

```python
def calculate_kelly_position(
    win_probability: float,
    tp_pips: float,
    sl_pips: float,
    account_balance: float,
    pip_value_per_lot: float = 10.0,
    kelly_fraction: float = 0.5,  # Half-Kelly
    max_risk_pct: float = 0.10    # Max 10% per trade
) -> dict:
    """
    Calculate position size using Kelly criterion.
    """
    # Win/loss ratio
    b = tp_pips / sl_pips
    
    # Kelly formula
    q = 1 - win_probability
    full_kelly = (win_probability * b - q) / b
    
    # Apply fraction (half-Kelly, quarter-Kelly, etc.)
    kelly = full_kelly * kelly_fraction
    
    # Cap at maximum risk
    risk_pct = min(kelly, max_risk_pct)
    risk_pct = max(risk_pct, 0)  # No negative
    
    # Calculate position size
    risk_amount = account_balance * risk_pct
    position_lots = risk_amount / (sl_pips * pip_value_per_lot)
    
    return {
        'full_kelly': full_kelly,
        'adjusted_kelly': kelly,
        'risk_pct': risk_pct,
        'risk_amount': risk_amount,
        'position_lots': round(position_lots, 2),
        'confidence': 'HIGH' if full_kelly > 0.20 else 'MEDIUM' if full_kelly > 0.10 else 'LOW'
    }
```

## 8.5 Trade Decision Engine

```python
class TradeDecisionEngine:
    """
    Combines all components to make trading decisions.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.probability_model = None
        
    def evaluate_signal(
        self,
        features: pd.Series,
        current_atr_pips: float,
        volatility_percentile: float,
        account_balance: float
    ) -> dict:
        """
        Evaluate a trading signal and generate trade decision.
        
        Returns complete trade specification or NO_TRADE.
        """
        # 1. Get probability predictions
        probabilities = self.probability_model.predict(features)
        
        # 2. Calculate dynamic SL
        sl_pips = calculate_dynamic_sl(
            current_atr_pips=current_atr_pips,
            predicted_mae=probabilities.get('expected_mae', 25),
            volatility_percentile=volatility_percentile
        )
        
        # 3. Find optimal TP
        tp_result = optimize_take_profit(
            probabilities=probabilities,
            sl_pips=sl_pips,
            min_rr_ratio=self.config.get('min_rr_ratio', 1.0),
            min_ev=self.config.get('min_ev', 0.0)
        )
        
        if tp_result['action'] == 'NO_TRADE':
            return {
                'action': 'NO_TRADE',
                'reason': tp_result['reason'],
                'probabilities': probabilities
            }
        
        # 4. Calculate position size
        position = calculate_kelly_position(
            win_probability=tp_result['probability'],
            tp_pips=tp_result['optimal_tp'],
            sl_pips=sl_pips,
            account_balance=account_balance
        )
        
        # 5. Final decision
        return {
            'action': 'TRADE',
            'direction': 'SHORT',  # Our strategy is SHORT on BB/RSI signals
            'entry_price': features.get('close'),
            'stop_loss_pips': sl_pips,
            'take_profit_pips': tp_result['optimal_tp'],
            'position_lots': position['position_lots'],
            'risk_pct': position['risk_pct'],
            'expected_value': tp_result['expected_value'],
            'win_probability': tp_result['probability'],
            'rr_ratio': tp_result['rr_ratio'],
            'confidence': position['confidence'],
            'probabilities': probabilities
        }
```

## 8.6 Phase 4 Files to Create

```
FILES:
═════════════════════════════════════════════════════════════════════════════

src/strategy/
├── dynamic_exits.py        # Dynamic SL calculation
├── ev_optimizer.py         # Take profit optimization
├── position_sizing.py      # Kelly criterion
├── trade_decision.py       # Decision engine
└── regime_detector.py      # Volatility regime detection

OUTPUT:
artifacts/strategy_config.json   # Optimized strategy parameters
```

---

# 9. PHASE 5: INTEGRATION & VALIDATION

## 9.1 Purpose

Integrate all components into a complete pipeline, run comprehensive validation, and generate production-ready outputs.

## 9.2 Complete Pipeline Flow

```python
def run_stella_v2_pipeline(config: dict) -> dict:
    """
    Main entry point for Stella Alpha V2.
    
    Steps:
    1. Load and merge data
    2. Generate multi-target labels
    3. Walk-forward training and validation
    4. Generate backtest results
    5. Produce final report
    """
    # 1. DATA LOADING
    logger.info("Loading data...")
    df_h4 = load_h4_data(config['data']['h4_file'])
    df_d1 = load_d1_data(config['data']['d1_file'])
    
    logger.info("Merging H4 and D1 data...")
    df = merge_h4_d1_safe(df_h4, df_d1)
    validate_no_leakage(df)
    
    # 2. FEATURE ENGINEERING
    logger.info("Engineering features...")
    df = engineer_features(df, config)
    
    # 3. LABEL GENERATION
    logger.info("Generating multi-target labels...")
    df = generate_multi_target_labels(df, config['labels'])
    
    # 4. WALK-FORWARD VALIDATION
    logger.info("Running walk-forward validation...")
    splits = walk_forward_split(df, config['validation'])
    
    all_results = []
    
    for split in splits:
        logger.info(f"Processing fold {split['fold']}...")
        
        # Split data
        df_train = df[split['train_idx']]
        df_cal = df[split['cal_idx']]
        df_test = df[split['test_idx']]
        
        # Train probability model
        model = ProbabilityModelTrainer(config['model'])
        model.train(df_train, df_cal)
        
        # Evaluate calibration
        cal_metrics = evaluate_calibration(df_test)
        
        # Run backtest
        backtest = run_backtest(df_test, model, config['strategy'])
        
        all_results.append({
            'fold': split['fold'],
            'calibration': cal_metrics,
            'backtest': backtest
        })
    
    # 5. AGGREGATE RESULTS
    logger.info("Aggregating results...")
    final_report = aggregate_results(all_results, config)
    
    # 6. GENERATE OUTPUTS
    save_models(model, config['output']['models_dir'])
    save_report(final_report, config['output']['report_file'])
    
    return final_report
```

## 9.3 Backtest Engine

```python
def run_backtest(
    df: pd.DataFrame,
    model: ProbabilityModelTrainer,
    strategy_config: dict
) -> dict:
    """
    Run backtest on test data.
    
    Simulates trading with dynamic exits and Kelly sizing.
    """
    decision_engine = TradeDecisionEngine(strategy_config)
    decision_engine.probability_model = model
    
    trades = []
    equity_curve = [strategy_config.get('initial_balance', 10000)]
    
    for idx, row in df.iterrows():
        # Check if this is a signal candle
        if not is_valid_signal(row, strategy_config['signal_filters']):
            continue
        
        # Get trade decision
        decision = decision_engine.evaluate_signal(
            features=row,
            current_atr_pips=row['atr_pips'],
            volatility_percentile=row['atr_percentile'],
            account_balance=equity_curve[-1]
        )
        
        if decision['action'] == 'NO_TRADE':
            continue
        
        # Simulate trade outcome
        outcome = simulate_trade_outcome(
            entry_price=row['close'],
            sl_pips=decision['stop_loss_pips'],
            tp_pips=decision['take_profit_pips'],
            future_data=get_future_candles(df, idx, max_bars=72)
        )
        
        # Record trade
        trade = {
            'entry_time': row['timestamp'],
            'entry_price': row['close'],
            'sl_pips': decision['stop_loss_pips'],
            'tp_pips': decision['take_profit_pips'],
            'position_lots': decision['position_lots'],
            'predicted_prob': decision['win_probability'],
            'predicted_ev': decision['expected_value'],
            'outcome': outcome['result'],  # WIN or LOSS
            'pips_result': outcome['pips'],
            'bars_held': outcome['bars_held'],
            'equity_after': equity_curve[-1] + outcome['pnl']
        }
        trades.append(trade)
        equity_curve.append(trade['equity_after'])
    
    # Calculate metrics
    return calculate_backtest_metrics(trades, equity_curve)
```

## 9.4 Performance Metrics

```python
def calculate_backtest_metrics(trades: List[dict], equity_curve: List[float]) -> dict:
    """
    Calculate comprehensive performance metrics.
    """
    if not trades:
        return {'error': 'No trades'}
    
    # Basic stats
    total_trades = len(trades)
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    
    win_rate = len(wins) / total_trades
    avg_win = np.mean([t['pips_result'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pips_result']) for t in losses]) if losses else 0
    
    # Profit factor
    gross_profit = sum(t['pips_result'] for t in wins)
    gross_loss = abs(sum(t['pips_result'] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expected value
    expected_value = np.mean([t['pips_result'] for t in trades])
    
    # Sharpe ratio (annualized)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)  # H4 = 6 per day
    
    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown)
    
    # Calibration accuracy
    predicted_probs = [t['predicted_prob'] for t in trades]
    actual_wins = [1 if t['outcome'] == 'WIN' else 0 for t in trades]
    calibration_error = np.mean(np.abs(np.array(predicted_probs) - np.array(actual_wins)))
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win_pips': avg_win,
        'avg_loss_pips': avg_loss,
        'profit_factor': profit_factor,
        'expected_value_pips': expected_value,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'final_equity': equity_curve[-1],
        'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
        'calibration_error': calibration_error
    }
```

## 9.5 Monte Carlo Simulation

```python
def run_monte_carlo(
    trades: List[dict],
    n_simulations: int = 1000,
    initial_balance: float = 10000
) -> dict:
    """
    Run Monte Carlo simulation for robustness testing.
    
    Shuffles trade order to test strategy's dependence on sequence.
    """
    results = []
    
    for _ in range(n_simulations):
        # Shuffle trades
        shuffled = trades.copy()
        np.random.shuffle(shuffled)
        
        # Simulate equity curve
        equity = [initial_balance]
        for trade in shuffled:
            pnl = trade['pips_result'] * trade['position_lots'] * 10  # $10 per pip per lot
            equity.append(equity[-1] + pnl)
        
        # Calculate metrics
        final_equity = equity[-1]
        peak = np.maximum.accumulate(equity)
        max_dd = np.max((peak - equity) / peak)
        
        results.append({
            'final_equity': final_equity,
            'max_drawdown': max_dd,
            'total_return': (final_equity - initial_balance) / initial_balance
        })
    
    # Calculate confidence intervals
    returns = [r['total_return'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    
    return {
        'n_simulations': n_simulations,
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'return_5th_percentile': np.percentile(returns, 5),
        'return_95th_percentile': np.percentile(returns, 95),
        'drawdown_mean': np.mean(drawdowns),
        'drawdown_95th_percentile': np.percentile(drawdowns, 95),
        'prob_profitable': np.mean([r > 0 for r in returns]),
        'prob_20pct_drawdown': np.mean([d > 0.20 for d in drawdowns])
    }
```

## 9.6 Final Report Format

```json
{
  "report_generated": "2026-03-10T15:00:00",
  "strategy_version": "Stella Alpha V2",
  
  "data_summary": {
    "date_range": "2000-01-01 to 2025-12-31",
    "total_candles": 41523,
    "signal_candles": 1847,
    "training_folds": 5
  },
  
  "model_performance": {
    "calibration": {
      "brier_score": 0.18,
      "expected_calibration_error": 0.042,
      "calibration_curves": {...}
    },
    "probability_accuracy": {
      "p_30_pip_accuracy": 0.95,
      "p_50_pip_accuracy": 0.93,
      "p_80_pip_accuracy": 0.91
    }
  },
  
  "backtest_results": {
    "aggregate": {
      "total_trades": 1423,
      "win_rate": 0.62,
      "avg_win_pips": 42.5,
      "avg_loss_pips": 35.2,
      "profit_factor": 1.78,
      "expected_value_pips": 8.4,
      "sharpe_ratio": 1.92,
      "max_drawdown": 0.12,
      "total_return": 1.85
    },
    "by_fold": [
      {"fold": 1, "win_rate": 0.60, "sharpe": 1.75},
      {"fold": 2, "win_rate": 0.63, "sharpe": 2.01},
      ...
    ],
    "by_regime": {
      "trending": {"trades": 612, "win_rate": 0.65, "ev": 10.2},
      "ranging": {"trades": 811, "win_rate": 0.59, "ev": 6.8}
    }
  },
  
  "robustness": {
    "monte_carlo": {
      "n_simulations": 1000,
      "return_95_ci": [0.65, 2.15],
      "max_dd_95th_pct": 0.18,
      "prob_profitable": 0.97
    },
    "parameter_sensitivity": "LOW",
    "regime_stability": "HIGH"
  },
  
  "strategy_config": {
    "signal_filters": {
      "min_bb_position": 0.95,
      "min_rsi": 70,
      "require_d1_alignment": true
    },
    "dynamic_sl": {
      "atr_multiple": 1.5,
      "mae_safety_factor": 1.2
    },
    "position_sizing": {
      "method": "half_kelly",
      "max_risk_pct": 0.10
    }
  },
  
  "recommendation": {
    "verdict": "DEPLOY",
    "confidence": "HIGH",
    "notes": [
      "Strategy shows consistent edge across all folds",
      "Calibration error within acceptable bounds",
      "Monte Carlo confirms robustness"
    ]
  }
}
```

## 9.7 Phase 5 Files to Create

```
FILES:
═════════════════════════════════════════════════════════════════════════════

src/validation/
├── backtest_engine.py      # Backtest simulation
├── performance_metrics.py  # Metric calculations
├── monte_carlo.py          # Monte Carlo simulation
├── regime_analysis.py      # Performance by regime
└── report_generator.py     # Report generation

run_stella_v2_pipeline.py   # Main entry point

OUTPUT:
artifacts/
├── final_report.json
├── equity_curve.csv
├── trade_log.csv
├── calibration_plots/
└── models/
```

---

# 10. CONFIGURATION SYSTEM

## 10.1 Master Configuration File

```yaml
# config/stella_v2_settings.yaml
# Stella Alpha V2: Institutional-Grade ML Trading System

# =============================================================================
# DATA SOURCES
# =============================================================================
data:
  h4_file: "data/EURUSD_H4_features.csv"
  d1_file: "data/EURUSD_D1_features.csv"
  pip_value: 0.0001
  timezone: "UTC"

# =============================================================================
# PHASE 1: ALPHA RESEARCH
# =============================================================================
research:
  enabled: true
  output_file: "artifacts/alpha_research_report.json"
  
  # Analysis parameters
  bb_buckets: [[0.90, 0.95], [0.95, 0.98], [0.98, 1.00], [1.00, 1.10]]
  rsi_buckets: [[65, 70], [70, 75], [75, 80], [80, 85], [85, 100]]
  volatility_percentiles: [0.25, 0.50, 0.75]
  
  # Statistical thresholds
  min_sample_size: 100
  significance_level: 0.05
  min_effect_size: 0.2

# =============================================================================
# PHASE 2: LABEL GENERATION
# =============================================================================
labels:
  max_holding_bars: 48
  
  # Binary targets (pips)
  binary_targets: [30, 50, 80, 100, 150]
  
  # Quality flag thresholds
  safe_trade_mae_ratio: 0.8  # MAE < TP * ratio
  fast_trade_bars: 12
  clean_winner_mfe_mae_ratio: 1.5

# =============================================================================
# PHASE 3: PROBABILITY MODEL
# =============================================================================
model:
  type: "lightgbm"
  
  targets:
    - "reached_30_pips"
    - "reached_50_pips"
    - "reached_80_pips"
  
  regression_targets:
    - "mfe_pips"
    - "mae_pips"
  
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    min_child_samples: 50
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 0.1
    reg_lambda: 0.1
    class_weight: "balanced"
    random_state: 42

  calibration:
    method: "sigmoid"  # or "isotonic"

# =============================================================================
# PHASE 4: STRATEGY
# =============================================================================
strategy:
  # Signal filters
  signal_filters:
    min_bb_position: 0.95
    min_rsi: 70
    require_d1_alignment: true
    min_volatility_percentile: 0.25
  
  # Dynamic stop loss
  dynamic_sl:
    atr_multiple: 1.5
    mae_safety_factor: 1.2
    min_sl_pips: 15
    max_sl_pips: 100
  
  # Take profit optimization
  take_profit:
    target_levels: [30, 40, 50, 60, 80, 100]
    min_rr_ratio: 1.0
    min_ev: 0.0
  
  # Position sizing
  position_sizing:
    method: "kelly"
    kelly_fraction: 0.5  # Half-Kelly
    max_risk_pct: 0.10
    min_risk_pct: 0.01

# =============================================================================
# PHASE 5: VALIDATION
# =============================================================================
validation:
  walk_forward:
    n_folds: 5
    calibration_pct: 0.15
    test_years: 2
  
  monte_carlo:
    enabled: true
    n_simulations: 1000
  
  acceptance_criteria:
    min_win_rate: 0.50
    min_profit_factor: 1.3
    min_sharpe: 1.0
    max_drawdown: 0.20
    min_trades: 200
    max_calibration_error: 0.10

# =============================================================================
# OUTPUT
# =============================================================================
output:
  artifacts_dir: "artifacts"
  models_dir: "artifacts/models"
  report_file: "artifacts/final_report.json"
  trade_log_file: "artifacts/trade_log.csv"
  equity_curve_file: "artifacts/equity_curve.csv"

# =============================================================================
# EXECUTION
# =============================================================================
execution:
  n_workers: 4
  random_state: 42
  verbose: true
  log_level: "INFO"
  log_file: "artifacts/stella_v2.log"
```

---

# 11. FILE STRUCTURE

```
stellaalpha_v2/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
│
├── config/
│   └── stella_v2_settings.yaml         # Master configuration
│
├── data/
│   ├── EURUSD_H4_features.csv          # H4 input data
│   └── EURUSD_D1_features.csv          # D1 input data
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Load CSV data
│   │   ├── data_merger.py              # Merge H4+D1 (leakage-safe)
│   │   ├── data_validator.py           # Data quality checks
│   │   └── feature_engineering.py      # Calculate derived features
│   │
│   ├── research/                       # PHASE 1
│   │   ├── __init__.py
│   │   ├── alpha_research.py           # Main research module
│   │   ├── forward_returns.py          # MFE/MAE calculation
│   │   ├── mean_reversion.py           # Mean reversion analysis
│   │   ├── volatility_analysis.py      # Volatility regime analysis
│   │   ├── session_analysis.py         # Session/time analysis
│   │   ├── mtf_analysis.py             # Multi-timeframe analysis
│   │   └── research_report.py          # Report generation
│   │
│   ├── labels/                         # PHASE 2
│   │   ├── __init__.py
│   │   ├── multi_target_labels.py      # Label generation
│   │   ├── forward_metrics.py          # MFE/MAE per candle
│   │   ├── quality_flags.py            # Safe/clean/fast flags
│   │   └── label_statistics.py         # Summary statistics
│   │
│   ├── models/                         # PHASE 3
│   │   ├── __init__.py
│   │   ├── probability_model.py        # Multi-target model
│   │   ├── calibration.py              # Probability calibration
│   │   ├── walk_forward.py             # Walk-forward CV
│   │   └── model_persistence.py        # Save/load models
│   │
│   ├── strategy/                       # PHASE 4
│   │   ├── __init__.py
│   │   ├── dynamic_exits.py            # Dynamic SL calculation
│   │   ├── ev_optimizer.py             # TP optimization
│   │   ├── position_sizing.py          # Kelly criterion
│   │   ├── trade_decision.py           # Decision engine
│   │   └── regime_detector.py          # Volatility regime
│   │
│   ├── evaluation/                     # PHASE 3 & 5
│   │   ├── __init__.py
│   │   ├── probability_metrics.py      # Brier score, ECE
│   │   ├── calibration_curves.py       # Calibration visualization
│   │   └── model_comparison.py         # Compare models
│   │
│   ├── validation/                     # PHASE 5
│   │   ├── __init__.py
│   │   ├── backtest_engine.py          # Backtest simulation
│   │   ├── performance_metrics.py      # Metric calculations
│   │   ├── monte_carlo.py              # Monte Carlo simulation
│   │   ├── regime_analysis.py          # Performance by regime
│   │   └── report_generator.py         # Final report
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                   # Logging utilities
│       ├── config_loader.py            # Load YAML config
│       └── visualization.py            # Plotting utilities
│
├── scripts/
│   ├── run_alpha_research.py           # Phase 1 entry point
│   ├── run_generate_labels.py          # Phase 2 entry point
│   ├── run_train_model.py              # Phase 3 entry point
│   ├── run_backtest.py                 # Phase 5 entry point
│   └── run_stella_v2_pipeline.py       # Complete pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_data_merger.py
│   ├── test_labels.py
│   ├── test_calibration.py
│   ├── test_dynamic_exits.py
│   └── test_backtest.py
│
└── artifacts/                          # Output directory
    ├── alpha_research_report.json
    ├── label_statistics.json
    ├── final_report.json
    ├── trade_log.csv
    ├── equity_curve.csv
    ├── models/
    │   ├── prob_model_fold_1.pkl
    │   ├── calibrator_fold_1.pkl
    │   └── ...
    └── plots/
        ├── calibration_curve.png
        └── equity_curve.png
```

---

# 12. COMMAND LINE INTERFACE

## 12.1 Phase 1: Alpha Research

```bash
python scripts/run_alpha_research.py \
    --config config/stella_v2_settings.yaml \
    --h4 data/EURUSD_H4_features.csv \
    --d1 data/EURUSD_D1_features.csv \
    --output artifacts/alpha_research_report.json

# Options:
#   --config    Path to configuration file
#   --h4        Path to H4 data (overrides config)
#   --d1        Path to D1 data (overrides config)
#   --output    Output report path
#   --verbose   Enable verbose logging
```

## 12.2 Phase 2: Generate Labels

```bash
python scripts/run_generate_labels.py \
    --config config/stella_v2_settings.yaml \
    --input data/EURUSD_H4_merged.csv \
    --output data/EURUSD_H4_labeled.csv \
    --max-hold 48

# Options:
#   --input     Merged H4+D1 data
#   --output    Output with labels
#   --max-hold  Maximum holding period in bars
#   --analyze   Print label statistics
```

## 12.3 Phase 3: Train Model

```bash
python scripts/run_train_model.py \
    --config config/stella_v2_settings.yaml \
    --input data/EURUSD_H4_labeled.csv \
    --output artifacts/models/

# Options:
#   --folds     Number of walk-forward folds
#   --calibrate Enable probability calibration
#   --evaluate  Evaluate on test set
```

## 12.4 Phase 5: Run Backtest

```bash
python scripts/run_backtest.py \
    --config config/stella_v2_settings.yaml \
    --model artifacts/models/prob_model.pkl \
    --data data/EURUSD_H4_labeled.csv \
    --output artifacts/backtest_report.json

# Options:
#   --monte-carlo   Run Monte Carlo simulation
#   --n-sims        Number of simulations
#   --plot          Generate plots
```

## 12.5 Complete Pipeline

```bash
python scripts/run_stella_v2_pipeline.py \
    --config config/stella_v2_settings.yaml \
    --all

# Options:
#   --all           Run all phases
#   --research      Run only Phase 1
#   --labels        Run only Phase 2
#   --train         Run only Phase 3
#   --validate      Run only Phase 5
#   --workers       Number of parallel workers
#   --verbose       Enable verbose output
```

---

# 13. OUTPUT ARTIFACTS

## 13.1 Research Output (Phase 1)

```
artifacts/
├── alpha_research_report.json    # Complete findings
├── mean_reversion_analysis.json  # BB/RSI analysis
├── volatility_analysis.json      # Regime analysis
├── mtf_analysis.json             # D1 alignment impact
└── research_summary.txt          # Human-readable summary
```

## 13.2 Model Output (Phase 3)

```
artifacts/models/
├── prob_model_reached_30.pkl     # Model for 30-pip target
├── prob_model_reached_50.pkl     # Model for 50-pip target
├── prob_model_reached_80.pkl     # Model for 80-pip target
├── regressor_mfe.pkl             # MFE regression model
├── regressor_mae.pkl             # MAE regression model
├── calibrator_30.pkl             # Calibrator for 30-pip model
├── calibrator_50.pkl             # Calibrator for 50-pip model
├── feature_columns.json          # Selected features
└── model_metadata.json           # Training info
```

## 13.3 Validation Output (Phase 5)

```
artifacts/
├── final_report.json             # Complete report
├── trade_log.csv                 # Every trade
├── equity_curve.csv              # Equity over time
├── monte_carlo_results.json      # MC simulation
├── fold_results.json             # Per-fold metrics
│
└── plots/
    ├── equity_curve.png
    ├── drawdown_chart.png
    ├── calibration_curve_30.png
    ├── calibration_curve_50.png
    ├── win_rate_by_regime.png
    └── ev_distribution.png
```

---

# 14. TESTING REQUIREMENTS

## 14.1 Unit Tests

```python
# tests/test_data_merger.py
def test_no_leakage():
    """Verify D1 data is always from previous day."""
    
def test_merge_coverage():
    """Verify all H4 rows have D1 data (except first ~50)."""

# tests/test_labels.py
def test_mfe_calculation():
    """Verify MFE is correctly calculated."""
    
def test_binary_targets():
    """Verify reached_X_pips flags are correct."""

# tests/test_calibration.py
def test_calibration_improves_brier():
    """Verify calibration reduces Brier score."""

# tests/test_dynamic_exits.py
def test_sl_bounds():
    """Verify SL respects min/max bounds."""
    
def test_ev_optimization():
    """Verify optimal TP maximizes EV."""

# tests/test_backtest.py
def test_trade_simulation():
    """Verify trade outcomes match price data."""
```

## 14.2 Integration Tests

```python
def test_full_pipeline():
    """Run complete pipeline on small dataset."""
    
def test_walk_forward():
    """Verify no data leakage in CV splits."""
    
def test_model_persistence():
    """Verify models save and load correctly."""
```

## 14.3 Acceptance Tests

```
□ Research report identifies statistically significant patterns
□ Model calibration error < 10%
□ Backtest win rate > 50%
□ Backtest profit factor > 1.3
□ Monte Carlo 95% CI is positive
□ Max drawdown < 20%
```

---

# 15. IMPLEMENTATION CHECKLIST

## Phase 1: Alpha Research (Day 1)

```
□ Create src/research/ directory structure
□ Implement forward_returns.py (MFE/MAE calculation)
□ Implement mean_reversion.py
□ Implement volatility_analysis.py
□ Implement session_analysis.py
□ Implement mtf_analysis.py
□ Implement research_report.py
□ Create run_alpha_research.py script
□ Test on actual data
□ Generate research report
```

## Phase 2: Labels (Day 1-2)

```
□ Create src/labels/ directory structure
□ Implement multi_target_labels.py
□ Implement forward_metrics.py
□ Implement quality_flags.py
□ Implement label_statistics.py
□ Create run_generate_labels.py script
□ Test label generation
□ Verify no NaN in critical columns
```

## Phase 3: Model (Day 2-3)

```
□ Create src/models/ directory structure
□ Implement probability_model.py
□ Implement calibration.py
□ Implement walk_forward.py
□ Implement model_persistence.py
□ Create run_train_model.py script
□ Train models on data
□ Evaluate calibration metrics
```

## Phase 4: Strategy (Day 3-4)

```
□ Create src/strategy/ directory structure
□ Implement dynamic_exits.py
□ Implement ev_optimizer.py
□ Implement position_sizing.py
□ Implement trade_decision.py
□ Implement regime_detector.py
□ Test decision engine
```

## Phase 5: Validation (Day 4-5)

```
□ Create src/validation/ directory structure
□ Implement backtest_engine.py
□ Implement performance_metrics.py
□ Implement monte_carlo.py
□ Implement regime_analysis.py
□ Implement report_generator.py
□ Create run_stella_v2_pipeline.py
□ Run complete validation
□ Generate final report
□ Review results
```

---

# 16. APPENDICES

## Appendix A: Feature List

```
H4 BASE FEATURES (from EA or calculated):
─────────────────────────────────────────
timestamp, open, high, low, close, volume
bb_position, bb_width_pct, rsi_value, atr_pct
volume_ratio, trend_strength, candle_body_pct, candle_rejection

H4 DERIVED FEATURES (calculated in Python):
───────────────────────────────────────────
rsi_slope_3, rsi_slope_5, price_change_5, price_change_10
bb_position_lag1, bb_position_lag2, rsi_lag1, rsi_lag2
atr_percentile, volume_percentile, hour, day_of_week
is_asian_session, is_london_session, is_ny_session, is_overlap_session
consecutive_bullish, consecutive_bearish
macd_line, macd_signal, macd_histogram, adx

D1 FEATURES (prefixed with d1_):
────────────────────────────────
d1_bb_position, d1_rsi_value, d1_atr_pct, d1_trend_strength
d1_trend_direction, d1_is_trending, d1_consecutive_bullish
d1_ma20_slope, d1_atr_percentile

CROSS-TIMEFRAME FEATURES (prefixed with mtf_):
──────────────────────────────────────────────
mtf_rsi_aligned, mtf_bb_aligned, mtf_trend_aligned
mtf_confluence_score, d1_supports_direction, d1_opposes_direction
```

## Appendix B: Statistical Formulas

```
COHEN'S D (Effect Size):
d = (μ1 - μ2) / σ_pooled
σ_pooled = sqrt((σ1² + σ2²) / 2)

BRIER SCORE:
BS = (1/N) Σ (p_i - o_i)²
Where p_i = predicted probability, o_i = actual outcome (0 or 1)

EXPECTED CALIBRATION ERROR:
ECE = Σ (|bin_i| / N) × |accuracy_i - confidence_i|

KELLY CRITERION:
f* = (p × b - q) / b
Where p = P(win), q = P(loss), b = win/loss ratio

SHARPE RATIO (Annualized):
SR = (μ_returns / σ_returns) × sqrt(periods_per_year)
For H4: periods_per_year = 252 × 6 = 1512
```

## Appendix C: Error Codes

```
ERR_001: Data file not found
ERR_002: Invalid timestamp format
ERR_003: Data leakage detected in merge
ERR_004: Insufficient data for analysis (< 100 samples)
ERR_005: Model training failed
ERR_006: Calibration failed (not enough data)
ERR_007: No valid trades in backtest
ERR_008: Configuration parameter missing
ERR_009: Feature column mismatch
ERR_010: Output directory not writable
```

---

# END OF SPECIFICATION

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  STELLA ALPHA V2 SPECIFICATION COMPLETE                                       ║
║  INSTITUTIONAL-GRADE ML TRADING SYSTEM                                        ║
║                                                                               ║
║  Document Version: 2.0.0                                                      ║
║  Total Sections: 16                                                           ║
║  Total Pages: ~100 (when formatted)                                           ║
║                                                                               ║
║  KEY COMPONENTS:                                                              ║
║  • Phase 1: Alpha Research (find where edge exists)                           ║
║  • Phase 2: Multi-Target Labels (rich price path data)                        ║
║  • Phase 3: Probability Model (calibrated predictions)                        ║
║  • Phase 4: Dynamic Strategy (adaptive TP/SL/sizing)                          ║
║  • Phase 5: Validation (comprehensive testing)                                ║
║                                                                               ║
║  TO IMPLEMENT:                                                                ║
║  Provide this document to any AI LLM and say:                                 ║
║  "Implement Stella Alpha V2 based on this specification.                      ║
║   Start with Phase 1: Alpha Research."                                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```



# STELLA ALPHA V2: FILE ARCHITECTURE
## Complete Directory Structure with File Descriptions

---

# DOCUMENT INFORMATION

| Field | Value |
|-------|-------|
| **Version** | 2.0.0 |
| **Status** | Ready for Implementation |
| **Purpose** | Complete file architecture for Stella Alpha V2 |

---

# LEGEND

```
STATUS INDICATORS:
══════════════════════════════════════════════════════════════════════════════

[NEW]       = New file created specifically for V2
[MODIFIED]  = Existing V1 file that needs modifications for V2
[KEPT]      = V1 file kept as-is, no changes needed
[DEPRECATED]= V1 file no longer needed in V2
[RENAMED]   = V1 file renamed/moved in V2

PRIORITY INDICATORS:
══════════════════════════════════════════════════════════════════════════════

[P1] = Phase 1 - Alpha Research (Day 1)
[P2] = Phase 2 - Label Generation (Day 1-2)
[P3] = Phase 3 - Probability Model (Day 2-3)
[P4] = Phase 4 - Dynamic Strategy (Day 3-4)
[P5] = Phase 5 - Validation & Integration (Day 4-5)
[CORE] = Core infrastructure, needed by all phases
```

---

# COMPLETE DIRECTORY STRUCTURE

```
stellaalpha_v2/
│
├── 📄 README.md                                    [MODIFIED] [CORE]
├── 📄 requirements.txt                             [MODIFIED] [CORE]
├── 📄 setup.py                                     [NEW] [CORE]
├── 📄 .gitignore                                   [NEW] [CORE]
│
├── 📄 STELLA_ALPHA_V2_SPECIFICATION.md             [NEW] [CORE]
├── 📄 STELLA_ALPHA_V2_FILE_ARCHITECTURE.md         [NEW] [CORE]
│
├── 📁 config/
│   ├── 📄 stella_v2_settings.yaml                  [NEW] [CORE]
│   └── 📄 logging_config.yaml                      [NEW] [CORE]
│
├── 📁 data/
│   ├── 📄 EURUSD_H4_features.csv                   [KEPT] [CORE]
│   ├── 📄 EURUSD_D1_features.csv                   [KEPT] [CORE]
│   └── 📄 .gitkeep                                 [NEW] [CORE]
│
├── 📁 src/
│   ├── 📄 __init__.py                              [MODIFIED] [CORE]
│   │
│   ├── 📁 data/                                    [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [CORE]
│   │   ├── 📄 data_loader.py                       [NEW] [CORE]
│   │   ├── 📄 data_merger.py                       [MODIFIED] [CORE]
│   │   ├── 📄 data_validator.py                    [NEW] [CORE]
│   │   └── 📄 feature_engineering.py               [MODIFIED] [CORE]
│   │
│   ├── 📁 research/                                [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P1]
│   │   ├── 📄 alpha_research.py                    [NEW] [P1]
│   │   ├── 📄 forward_returns.py                   [NEW] [P1]
│   │   ├── 📄 mean_reversion_analysis.py           [NEW] [P1]
│   │   ├── 📄 volatility_analysis.py               [NEW] [P1]
│   │   ├── 📄 session_analysis.py                  [NEW] [P1]
│   │   ├── 📄 mtf_analysis.py                      [NEW] [P1]
│   │   └── 📄 research_report.py                   [NEW] [P1]
│   │
│   ├── 📁 labels/                                  [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P2]
│   │   ├── 📄 multi_target_labels.py               [NEW] [P2]
│   │   ├── 📄 forward_metrics.py                   [NEW] [P2]
│   │   ├── 📄 quality_flags.py                     [NEW] [P2]
│   │   └── 📄 label_statistics.py                  [NEW] [P2]
│   │
│   ├── 📁 models/                                  [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P3]
│   │   ├── 📄 probability_model.py                 [NEW] [P3]
│   │   ├── 📄 calibration.py                       [NEW] [P3]
│   │   ├── 📄 model_trainer.py                     [NEW] [P3]
│   │   ├── 📄 walk_forward.py                      [NEW] [P3]
│   │   └── 📄 model_persistence.py                 [NEW] [P3]
│   │
│   ├── 📁 strategy/                                [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P4]
│   │   ├── 📄 dynamic_exits.py                     [NEW] [P4]
│   │   ├── 📄 ev_optimizer.py                      [NEW] [P4]
│   │   ├── 📄 position_sizing.py                   [NEW] [P4]
│   │   ├── 📄 trade_decision.py                    [NEW] [P4]
│   │   └── 📄 regime_detector.py                   [NEW] [P4]
│   │
│   ├── 📁 evaluation/                              [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P3]
│   │   ├── 📄 probability_metrics.py               [NEW] [P3]
│   │   ├── 📄 calibration_curves.py                [NEW] [P3]
│   │   └── 📄 model_comparison.py                  [NEW] [P3]
│   │
│   ├── 📁 validation/                              [NEW DIRECTORY]
│   │   ├── 📄 __init__.py                          [NEW] [P5]
│   │   ├── 📄 backtest_engine.py                   [NEW] [P5]
│   │   ├── 📄 performance_metrics.py               [NEW] [P5]
│   │   ├── 📄 monte_carlo.py                       [NEW] [P5]
│   │   ├── 📄 regime_analysis.py                   [NEW] [P5]
│   │   └── 📄 report_generator.py                  [NEW] [P5]
│   │
│   └── 📁 utils/                                   [NEW DIRECTORY]
│       ├── 📄 __init__.py                          [NEW] [CORE]
│       ├── 📄 logger.py                            [NEW] [CORE]
│       ├── 📄 config_loader.py                     [NEW] [CORE]
│       ├── 📄 visualization.py                     [NEW] [P5]
│       └── 📄 exceptions.py                        [NEW] [CORE]
│
├── 📁 scripts/
│   ├── 📄 run_alpha_research.py                    [NEW] [P1]
│   ├── 📄 run_generate_labels.py                   [NEW] [P2]
│   ├── 📄 run_train_model.py                       [NEW] [P3]
│   ├── 📄 run_optimize_strategy.py                 [NEW] [P4]
│   ├── 📄 run_backtest.py                          [NEW] [P5]
│   └── 📄 run_stella_v2_pipeline.py                [NEW] [P5]
│
├── 📁 tests/
│   ├── 📄 __init__.py                              [NEW] [CORE]
│   ├── 📄 conftest.py                              [NEW] [CORE]
│   ├── 📄 test_data_loader.py                      [NEW] [CORE]
│   ├── 📄 test_data_merger.py                      [NEW] [CORE]
│   ├── 📄 test_forward_returns.py                  [NEW] [P1]
│   ├── 📄 test_multi_target_labels.py              [NEW] [P2]
│   ├── 📄 test_probability_model.py                [NEW] [P3]
│   ├── 📄 test_calibration.py                      [NEW] [P3]
│   ├── 📄 test_dynamic_exits.py                    [NEW] [P4]
│   ├── 📄 test_position_sizing.py                  [NEW] [P4]
│   ├── 📄 test_backtest_engine.py                  [NEW] [P5]
│   └── 📄 test_integration.py                      [NEW] [P5]
│
├── 📁 notebooks/                                   [NEW DIRECTORY]
│   ├── 📄 01_data_exploration.ipynb                [NEW] [CORE]
│   ├── 📄 02_alpha_research.ipynb                  [NEW] [P1]
│   ├── 📄 03_label_analysis.ipynb                  [NEW] [P2]
│   ├── 📄 04_model_training.ipynb                  [NEW] [P3]
│   └── 📄 05_backtest_analysis.ipynb               [NEW] [P5]
│
├── 📁 artifacts/                                   [OUTPUT DIRECTORY]
│   ├── 📄 .gitkeep                                 [NEW] [CORE]
│   ├── 📁 research/                                [P1 OUTPUT]
│   ├── 📁 labels/                                  [P2 OUTPUT]
│   ├── 📁 models/                                  [P3 OUTPUT]
│   ├── 📁 strategy/                                [P4 OUTPUT]
│   ├── 📁 validation/                              [P5 OUTPUT]
│   └── 📁 plots/                                   [P5 OUTPUT]
│
└── 📁 archive/                                     [V1 FILES - REFERENCE ONLY]
    ├── 📄 run_pure_ml_stella_alpha.py              [DEPRECATED]
    ├── 📄 pure_ml_labels.py                        [DEPRECATED]
    ├── 📄 experiment.py                            [DEPRECATED]
    ├── 📄 training.py                              [DEPRECATED]
    ├── 📄 evaluation.py                            [DEPRECATED]
    ├── 📄 diagnose.py                              [DEPRECATED]
    ├── 📄 analyze_losses.py                        [DEPRECATED]
    ├── 📄 filter_recommendations.py                [DEPRECATED]
    ├── 📄 loss_analysis.py                         [DEPRECATED]
    ├── 📄 recommend_filters.py                     [DEPRECATED]
    ├── 📄 compare_versions.py                      [DEPRECATED]
    ├── 📄 tier_classification.py                   [DEPRECATED]
    ├── 📄 trade_recorder.py                        [DEPRECATED]
    ├── 📄 checkpoint_db.py                         [DEPRECATED]
    ├── 📄 statistical_validation.py                [DEPRECATED]
    ├── 📄 validate_leakage.py                      [DEPRECATED]
    ├── 📄 test_suite.py                            [DEPRECATED]
    └── 📄 pure_ml_stella_alpha_settings.yaml       [DEPRECATED]
```

---

# DETAILED FILE DESCRIPTIONS

## ROOT DIRECTORY FILES

### 📄 README.md
```
Status:     [MODIFIED] - Update from V1
Phase:      [CORE]
Purpose:    Project documentation and quick start guide

Description:
Main documentation file explaining:
- What Stella Alpha V2 is
- Installation instructions
- Quick start commands
- Link to full specification
- Architecture overview

Changes from V1:
- Complete rewrite for V2 architecture
- New philosophy explanation
- Updated installation instructions
- New command examples
```

### 📄 requirements.txt
```
Status:     [MODIFIED] - Update dependencies
Phase:      [CORE]
Purpose:    Python package dependencies

Description:
Lists all required Python packages:
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- lightgbm>=4.0.0
- scipy>=1.11.0
- pyyaml>=6.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- tqdm>=4.65.0
- pytest>=7.4.0

Changes from V1:
- Added calibration libraries
- Added visualization libraries
- Version updates
```

### 📄 setup.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Package installation and distribution

Description:
Python package setup file enabling:
- pip install -e . (development install)
- Package metadata
- Entry points for CLI commands
- Dependency specification
```

### 📄 .gitignore
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Git ignore patterns

Description:
Excludes from version control:
- artifacts/ (generated outputs)
- __pycache__/
- *.pyc
- .env
- *.pkl (model files)
- *.db (database files)
- .ipynb_checkpoints/
```

### 📄 STELLA_ALPHA_V2_SPECIFICATION.md
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Complete functional specification

Description:
The master specification document containing:
- Full system architecture
- All phase specifications
- Data formats
- Code examples
- Configuration templates
~100 pages of detailed specification
```

### 📄 STELLA_ALPHA_V2_FILE_ARCHITECTURE.md
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    This file - directory structure documentation

Description:
Complete file architecture with:
- Directory tree
- File descriptions
- Status indicators (NEW/MODIFIED/DEPRECATED)
- Phase assignments
```

---

## CONFIG DIRECTORY

### 📁 config/

### 📄 config/stella_v2_settings.yaml
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Master configuration file

Description:
Central YAML configuration containing:
- Data source paths
- Phase 1 research parameters
- Phase 2 label settings
- Phase 3 model configuration
- Phase 4 strategy parameters
- Phase 5 validation settings
- Output paths

Key sections:
- data: Input file paths
- research: Alpha research settings
- labels: Multi-target label config
- model: LightGBM parameters
- strategy: Dynamic exit settings
- validation: Walk-forward CV settings
- output: Artifact paths
```

### 📄 config/logging_config.yaml
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Logging configuration

Description:
Configures Python logging:
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Log formats
- File handlers
- Console handlers
- Per-module log levels
```

---

## DATA DIRECTORY

### 📁 data/

### 📄 data/EURUSD_H4_features.csv
```
Status:     [KEPT] - No changes
Phase:      [CORE]
Purpose:    H4 timeframe OHLCV data

Description:
Primary input data file containing:
- 25 years of EURUSD H4 candles (~41,000 rows)
- OHLCV columns
- Pre-computed indicators (BB, RSI, ATR)
- Exported from MetaTrader EA

Columns:
timestamp, open, high, low, close, volume,
bb_position, bb_width_pct, rsi_value, atr_pct,
volume_ratio, trend_strength, candle_body_pct, etc.
```

### 📄 data/EURUSD_D1_features.csv
```
Status:     [KEPT] - No changes
Phase:      [CORE]
Purpose:    D1 timeframe OHLCV data

Description:
Higher timeframe context data:
- 25 years of EURUSD daily candles (~6,500 rows)
- Same structure as H4 file
- Used for multi-timeframe alignment
```

---

## SRC DIRECTORY - CORE

### 📁 src/

### 📄 src/__init__.py
```
Status:     [MODIFIED]
Phase:      [CORE]
Purpose:    Package initialization

Description:
Makes src a Python package.
Exports key classes and functions.

Changes from V1:
- New module imports
- Updated exports
```

---

## SRC/DATA - Data Loading & Processing

### 📁 src/data/

### 📄 src/data/__init__.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Data subpackage initialization

Description:
Exports:
- DataLoader
- DataMerger
- DataValidator
- FeatureEngineer
```

### 📄 src/data/data_loader.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Load CSV data files

Description:
Responsible for:
- Loading H4 CSV with proper dtypes
- Loading D1 CSV with proper dtypes
- Timestamp parsing
- Initial data validation
- Memory optimization

Key functions:
- load_h4_data(filepath) -> pd.DataFrame
- load_d1_data(filepath) -> pd.DataFrame
- parse_timestamps(df, format) -> pd.DataFrame

Dependencies: pandas, numpy
```

### 📄 src/data/data_merger.py
```
Status:     [MODIFIED] - Major refactor from V1
Phase:      [CORE]
Purpose:    Merge H4 and D1 data without leakage

Description:
Critical module for combining timeframes:
- Leakage-safe D1 alignment (previous day only)
- merge_asof implementation
- Validation of merge correctness
- Column prefixing (d1_)

Key functions:
- merge_h4_d1_safe(df_h4, df_d1) -> pd.DataFrame
- validate_no_leakage(df_merged) -> bool
- get_merge_statistics(df_merged) -> dict

V1 Location: data_merger.py (root)
Changes from V1:
- Moved to src/data/
- Improved leakage validation
- Better error handling
- Added statistics reporting
```

### 📄 src/data/data_validator.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Validate data quality

Description:
Comprehensive data validation:
- Timestamp integrity checks
- Price integrity (high >= low, etc.)
- Volume validation
- NaN detection and reporting
- Outlier detection
- Gap detection

Key functions:
- validate_ohlcv(df) -> ValidationResult
- check_timestamps(df) -> List[str]
- detect_outliers(df, columns) -> pd.DataFrame
- generate_quality_report(df) -> dict
```

### 📄 src/data/feature_engineering.py
```
Status:     [MODIFIED] - Streamlined from V1
Phase:      [CORE]
Purpose:    Calculate derived features

Description:
Feature calculation for H4 and D1:
- RSI slopes and lags
- BB position derivatives
- Volume features
- Session indicators
- Cross-timeframe features

Key functions:
- engineer_h4_features(df) -> pd.DataFrame
- engineer_d1_features(df) -> pd.DataFrame
- engineer_cross_tf_features(df) -> pd.DataFrame
- get_feature_columns(df) -> List[str]

V1 Location: feature_engineering.py (root)
Changes from V1:
- Moved to src/data/
- Removed unused features
- Added cross-TF features
- Cleaner implementation
```

---

## SRC/RESEARCH - Phase 1: Alpha Research

### 📁 src/research/

### 📄 src/research/__init__.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Research subpackage initialization

Description:
Exports:
- AlphaResearcher
- ForwardReturnCalculator
- MeanReversionAnalyzer
- VolatilityAnalyzer
- SessionAnalyzer
- MTFAnalyzer
- ResearchReportGenerator
```

### 📄 src/research/alpha_research.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Main alpha research orchestrator

Description:
Coordinates all research analyses:
- Runs all sub-analyses
- Aggregates findings
- Generates recommendations
- Produces final report

Key classes:
- AlphaResearcher
  - run_full_research() -> ResearchReport
  - find_optimal_parameters() -> dict
  - generate_recommendations() -> List[str]

Dependencies: All other research modules
```

### 📄 src/research/forward_returns.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Calculate MFE/MAE for each candle

Description:
Core forward-looking calculations:
- Max Favorable Excursion (MFE)
- Max Adverse Excursion (MAE)
- Time to target
- Final return at horizon

Key functions:
- calculate_forward_returns(df, direction, horizon) -> pd.DataFrame
- calculate_mfe(prices, entry, direction) -> float
- calculate_mae(prices, entry, direction) -> float
- calculate_time_to_target(prices, entry, target) -> int

This is the foundation for all research and labels.
```

### 📄 src/research/mean_reversion_analysis.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Analyze mean reversion patterns

Description:
Statistical analysis of reversal patterns:
- BB position buckets analysis
- RSI level analysis
- Combined signal analysis
- Probability calculations
- Effect size computation

Key functions:
- analyze_bb_buckets(df) -> List[Finding]
- analyze_rsi_levels(df) -> List[Finding]
- analyze_combined_signals(df) -> List[Finding]
- compute_baseline_probabilities(df) -> dict
```

### 📄 src/research/volatility_analysis.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Analyze volatility regime effects

Description:
Volatility-based analysis:
- ATR percentile bucketing
- Performance by volatility regime
- Optimal SL by ATR multiple
- Volatility clustering detection

Key functions:
- analyze_volatility_regimes(df) -> List[Finding]
- find_optimal_atr_sl(df) -> dict
- detect_volatility_clusters(df) -> pd.DataFrame
```

### 📄 src/research/session_analysis.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Analyze session and time effects

Description:
Time-based pattern analysis:
- Session performance (Asian, London, NY, Overlap)
- Day of week effects
- Hour of day effects
- Statistical significance testing

Key functions:
- analyze_sessions(df) -> List[Finding]
- analyze_day_of_week(df) -> List[Finding]
- analyze_hour_of_day(df) -> List[Finding]
```

### 📄 src/research/mtf_analysis.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Analyze multi-timeframe effects

Description:
D1 alignment impact analysis:
- D1 trend direction impact
- D1 BB position impact
- Confluence scoring analysis
- Quantify alignment edge

Key functions:
- analyze_d1_trend_alignment(df) -> List[Finding]
- analyze_d1_bb_alignment(df) -> List[Finding]
- calculate_alignment_edge(df) -> float
```

### 📄 src/research/research_report.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Generate research output report

Description:
Report generation and formatting:
- JSON report generation
- Human-readable summary
- Recommendation synthesis
- Statistical summary tables

Key functions:
- generate_json_report(findings) -> dict
- generate_text_summary(findings) -> str
- save_report(report, filepath) -> None
```

---

## SRC/LABELS - Phase 2: Label Generation

### 📁 src/labels/

### 📄 src/labels/__init__.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Labels subpackage initialization

Description:
Exports:
- MultiTargetLabelGenerator
- ForwardMetricsCalculator
- QualityFlagGenerator
- LabelStatisticsCalculator
```

### 📄 src/labels/multi_target_labels.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Generate all label types for each candle

Description:
Main label generation module:
- Binary targets (reached_30_pips, etc.)
- Regression targets (mfe_pips, mae_pips)
- Multiple horizons (12, 24, 48, 72 bars)
- Coordinates other label modules

Key classes:
- MultiTargetLabelGenerator
  - generate_all_labels(df, config) -> pd.DataFrame
  - generate_binary_targets(df, thresholds) -> pd.DataFrame
  - generate_regression_targets(df) -> pd.DataFrame

This replaces V1's pure_ml_labels.py with richer labels.
```

### 📄 src/labels/forward_metrics.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Calculate per-candle forward metrics

Description:
Efficient forward metric calculation:
- Vectorized MFE/MAE computation
- Time-to-target calculation
- Final return calculation
- Multiple horizon support

Key functions:
- calculate_forward_metrics(df, horizon) -> pd.DataFrame
- vectorized_mfe(close, low, high, horizon) -> np.array
- vectorized_mae(close, low, high, horizon) -> np.array
```

### 📄 src/labels/quality_flags.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Generate trade quality flags

Description:
Quality classification for trades:
- safe_X_pips: Low MAE relative to target
- clean_winner: High MFE/MAE ratio
- fast_X_pips: Quick target achievement
- Composite quality score

Key functions:
- generate_quality_flags(df, config) -> pd.DataFrame
- calculate_safe_trade_flag(mfe, mae, target) -> int
- calculate_clean_winner_flag(mfe, mae) -> int
```

### 📄 src/labels/label_statistics.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Generate label summary statistics

Description:
Statistical summary of generated labels:
- Target distribution analysis
- Correlation matrix
- Class imbalance reporting
- Data quality checks

Key functions:
- calculate_label_statistics(df) -> dict
- generate_distribution_report(df) -> dict
- check_label_quality(df) -> List[str]
```

---

## SRC/MODELS - Phase 3: Probability Model

### 📁 src/models/

### 📄 src/models/__init__.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Models subpackage initialization

Description:
Exports:
- ProbabilityModel
- ModelCalibrator
- ModelTrainer
- WalkForwardCV
- ModelPersistence
```

### 📄 src/models/probability_model.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Multi-output probability prediction model

Description:
Core model architecture:
- Separate models per target
- Unified prediction interface
- Probability output (0-1)
- Feature importance tracking

Key classes:
- ProbabilityModel
  - fit(X_train, y_train, targets) -> None
  - predict_probabilities(X) -> dict
  - get_feature_importance() -> dict

This replaces V1's binary classification approach.
```

### 📄 src/models/calibration.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Probability calibration (Platt scaling, isotonic)

Description:
Ensure predicted probabilities are accurate:
- Platt scaling (sigmoid)
- Isotonic regression
- Calibration on held-out data
- Calibration quality metrics

Key classes:
- ModelCalibrator
  - fit(y_true, y_prob) -> None
  - calibrate(y_prob) -> np.array
  - evaluate_calibration() -> dict

Key functions:
- platt_scaling(y_true, y_prob) -> Callable
- isotonic_calibration(y_true, y_prob) -> Callable
```

### 📄 src/models/model_trainer.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Model training orchestration

Description:
Handles complete training pipeline:
- Feature selection
- Hyperparameter tuning
- Model fitting
- Calibration
- Evaluation

Key classes:
- ModelTrainer
  - train(df_train, df_cal, config) -> ProbabilityModel
  - tune_hyperparameters(X, y) -> dict
  - select_features(X, y) -> List[str]
```

### 📄 src/models/walk_forward.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Walk-forward cross-validation

Description:
Time-series appropriate validation:
- Expanding window splits
- Train/calibration/test separation
- No future data leakage
- Fold statistics

Key classes:
- WalkForwardCV
  - split(df, n_folds) -> List[Fold]
  - validate(model, df) -> List[FoldResult]
  
Key functions:
- create_walk_forward_splits(df, config) -> List[dict]
```

### 📄 src/models/model_persistence.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Save and load trained models

Description:
Model serialization:
- Pickle model saving
- JSON metadata saving
- Model loading with validation
- Version compatibility checking

Key functions:
- save_model(model, path, metadata) -> None
- load_model(path) -> ProbabilityModel
- save_feature_columns(columns, path) -> None
```

---

## SRC/STRATEGY - Phase 4: Dynamic Strategy

### 📁 src/strategy/

### 📄 src/strategy/__init__.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Strategy subpackage initialization

Description:
Exports:
- DynamicExitCalculator
- EVOptimizer
- PositionSizer
- TradeDecisionEngine
- RegimeDetector
```

### 📄 src/strategy/dynamic_exits.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Calculate dynamic stop loss levels

Description:
ATR-based dynamic stop loss:
- ATR multiple calculation
- MAE-based adjustment
- Volatility regime scaling
- Min/max bounds enforcement

Key classes:
- DynamicExitCalculator
  - calculate_sl(atr, predicted_mae, regime) -> float
  - calculate_tp_levels(probabilities) -> List[float]
  
Key functions:
- atr_based_sl(atr, multiple) -> float
- mae_based_sl(predicted_mae, safety_factor) -> float
```

### 📄 src/strategy/ev_optimizer.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Find EV-optimal take profit level

Description:
Expected value optimization:
- EV calculation for each TP level
- Constraint satisfaction (min R:R)
- Optimal TP selection
- Risk-adjusted ranking

Key classes:
- EVOptimizer
  - optimize_tp(probabilities, sl, constraints) -> dict
  - calculate_ev(p_win, tp, sl) -> float
  - rank_options(options) -> List[dict]
```

### 📄 src/strategy/position_sizing.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Kelly criterion position sizing

Description:
Optimal position sizing:
- Full Kelly calculation
- Fractional Kelly (half, quarter)
- Maximum risk constraints
- Position size in lots

Key classes:
- PositionSizer
  - calculate_kelly(p_win, rr_ratio) -> float
  - calculate_position_lots(kelly, sl, balance) -> float
  - apply_constraints(kelly, config) -> float
```

### 📄 src/strategy/trade_decision.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Complete trade decision engine

Description:
Combines all strategy components:
- Signal validation
- Dynamic SL calculation
- TP optimization
- Position sizing
- Final trade decision

Key classes:
- TradeDecisionEngine
  - evaluate_signal(features, context) -> TradeDecision
  - should_trade(decision) -> bool
  - format_trade_order(decision) -> dict
```

### 📄 src/strategy/regime_detector.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Detect current volatility regime

Description:
Market regime classification:
- ATR percentile calculation
- Trend strength detection
- Regime classification (HIGH/MEDIUM/LOW)
- Regime history tracking

Key classes:
- RegimeDetector
  - detect_regime(df) -> str
  - get_regime_history(df) -> pd.Series
  - calculate_regime_features(df) -> pd.DataFrame
```

---

## SRC/EVALUATION - Evaluation Metrics

### 📁 src/evaluation/

### 📄 src/evaluation/__init__.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Evaluation subpackage initialization

Description:
Exports:
- ProbabilityMetrics
- CalibrationCurves
- ModelComparison
```

### 📄 src/evaluation/probability_metrics.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Probability-specific evaluation metrics

Description:
Metrics for calibrated probabilities:
- Brier score
- Expected Calibration Error (ECE)
- Log loss
- Reliability diagrams data

Key functions:
- brier_score(y_true, y_prob) -> float
- expected_calibration_error(y_true, y_prob, n_bins) -> float
- calculate_calibration_curve(y_true, y_prob) -> dict
```

### 📄 src/evaluation/calibration_curves.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Calibration visualization

Description:
Generate calibration plots:
- Reliability diagram
- Confidence histogram
- Before/after calibration comparison

Key functions:
- plot_calibration_curve(y_true, y_prob, title) -> Figure
- plot_confidence_histogram(y_prob) -> Figure
- save_calibration_plots(results, output_dir) -> None
```

### 📄 src/evaluation/model_comparison.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Compare model variants

Description:
Statistical model comparison:
- A/B test between models
- Cross-validation comparison
- Feature importance comparison
- Performance summary tables

Key functions:
- compare_models(model_a, model_b, test_data) -> dict
- rank_models(models, metric) -> List[tuple]
```

---

## SRC/VALIDATION - Phase 5: Validation

### 📁 src/validation/

### 📄 src/validation/__init__.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Validation subpackage initialization

Description:
Exports:
- BacktestEngine
- PerformanceMetrics
- MonteCarloSimulator
- RegimeAnalyzer
- ReportGenerator
```

### 📄 src/validation/backtest_engine.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Run historical backtests

Description:
Complete backtesting simulation:
- Signal detection
- Trade execution simulation
- Outcome calculation
- Equity curve tracking
- Trade logging

Key classes:
- BacktestEngine
  - run(df, model, strategy_config) -> BacktestResult
  - simulate_trade(entry, sl, tp, future_data) -> TradeOutcome
  - generate_equity_curve(trades) -> pd.Series
```

### 📄 src/validation/performance_metrics.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Calculate trading performance metrics

Description:
Comprehensive trading metrics:
- Win rate, profit factor
- Sharpe, Sortino ratios
- Maximum drawdown
- Expected value
- Calmar ratio

Key functions:
- calculate_all_metrics(trades, equity_curve) -> dict
- calculate_sharpe_ratio(returns) -> float
- calculate_max_drawdown(equity_curve) -> float
- calculate_profit_factor(trades) -> float
```

### 📄 src/validation/monte_carlo.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Monte Carlo robustness testing

Description:
Randomized simulation:
- Trade order shuffling
- Confidence intervals
- Probability of ruin
- Drawdown distribution

Key classes:
- MonteCarloSimulator
  - run(trades, n_simulations) -> MonteCarloResult
  - calculate_confidence_intervals(results) -> dict
  - estimate_probability_of_ruin(results, threshold) -> float
```

### 📄 src/validation/regime_analysis.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Analyze performance by market regime

Description:
Regime-specific performance:
- Performance in trending vs ranging
- Performance in high vs low volatility
- Regime transition analysis
- Strategy robustness by regime

Key functions:
- analyze_by_regime(trades, regimes) -> dict
- compare_regime_performance(results) -> dict
- identify_weak_regimes(results) -> List[str]
```

### 📄 src/validation/report_generator.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Generate final validation report

Description:
Complete report generation:
- JSON report with all metrics
- PDF report (optional)
- Summary tables
- Recommendation generation

Key classes:
- ReportGenerator
  - generate_full_report(results) -> dict
  - generate_summary(results) -> str
  - save_report(report, filepath) -> None
```

---

## SRC/UTILS - Utilities

### 📁 src/utils/

### 📄 src/utils/__init__.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Utils subpackage initialization

Description:
Exports:
- Logger
- ConfigLoader
- Visualization
- StellaException
```

### 📄 src/utils/logger.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Logging utilities

Description:
Standardized logging:
- Console and file logging
- Colored output
- Progress tracking
- Timing utilities

Key functions:
- setup_logger(name, config) -> Logger
- log_progress(current, total, message) -> None
- log_timing(func) -> Callable  # Decorator
```

### 📄 src/utils/config_loader.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Load and validate configuration

Description:
Configuration management:
- YAML loading
- Schema validation
- Default value handling
- Environment variable support

Key functions:
- load_config(filepath) -> dict
- validate_config(config, schema) -> bool
- merge_with_defaults(config, defaults) -> dict
```

### 📄 src/utils/visualization.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Plotting utilities

Description:
Standard visualizations:
- Equity curve plots
- Drawdown charts
- Distribution plots
- Heatmaps

Key functions:
- plot_equity_curve(equity, trades) -> Figure
- plot_drawdown(equity) -> Figure
- plot_distribution(data, title) -> Figure
- save_all_plots(figures, output_dir) -> None
```

### 📄 src/utils/exceptions.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Custom exception classes

Description:
Project-specific exceptions:
- DataValidationError
- LeakageDetectedError
- ConfigurationError
- ModelTrainingError
- BacktestError
```

---

## SCRIPTS DIRECTORY

### 📁 scripts/

### 📄 scripts/run_alpha_research.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    CLI entry point for Phase 1

Description:
Command-line script to run alpha research:
- Argument parsing
- Config loading
- Research execution
- Report saving

Usage:
python scripts/run_alpha_research.py \
    --config config/stella_v2_settings.yaml \
    --output artifacts/research/
```

### 📄 scripts/run_generate_labels.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    CLI entry point for Phase 2

Description:
Generate multi-target labels:
- Load merged data
- Generate all labels
- Save labeled dataset
- Print statistics

Usage:
python scripts/run_generate_labels.py \
    --config config/stella_v2_settings.yaml \
    --output data/EURUSD_H4_labeled.csv
```

### 📄 scripts/run_train_model.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    CLI entry point for Phase 3

Description:
Train probability models:
- Walk-forward training
- Calibration
- Model saving
- Evaluation

Usage:
python scripts/run_train_model.py \
    --config config/stella_v2_settings.yaml \
    --output artifacts/models/
```

### 📄 scripts/run_optimize_strategy.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    CLI entry point for Phase 4

Description:
Optimize strategy parameters:
- Dynamic exit tuning
- Position sizing calibration
- Save strategy config

Usage:
python scripts/run_optimize_strategy.py \
    --config config/stella_v2_settings.yaml \
    --output artifacts/strategy/
```

### 📄 scripts/run_backtest.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    CLI entry point for backtesting

Description:
Run backtest on test data:
- Load model and strategy
- Execute backtest
- Generate metrics
- Save report

Usage:
python scripts/run_backtest.py \
    --config config/stella_v2_settings.yaml \
    --model artifacts/models/prob_model.pkl \
    --output artifacts/validation/
```

### 📄 scripts/run_stella_v2_pipeline.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Complete pipeline execution

Description:
Run all phases end-to-end:
- Phase 1: Research
- Phase 2: Labels
- Phase 3: Model
- Phase 4: Strategy
- Phase 5: Validation
- Final report

Usage:
python scripts/run_stella_v2_pipeline.py \
    --config config/stella_v2_settings.yaml \
    --all
```

---

## TESTS DIRECTORY

### 📁 tests/

### 📄 tests/__init__.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Test package initialization
```

### 📄 tests/conftest.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Pytest fixtures and configuration

Description:
Shared test fixtures:
- Sample data generators
- Mock objects
- Test configuration
```

### 📄 tests/test_data_loader.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Test data loading functions

Tests:
- CSV loading
- Timestamp parsing
- Error handling
```

### 📄 tests/test_data_merger.py
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Test D1 merge logic

Tests:
- Leakage prevention
- Merge correctness
- Edge cases
```

### 📄 tests/test_forward_returns.py
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Test MFE/MAE calculations

Tests:
- MFE calculation
- MAE calculation
- Time to target
```

### 📄 tests/test_multi_target_labels.py
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Test label generation

Tests:
- Binary targets
- Regression targets
- Quality flags
```

### 📄 tests/test_probability_model.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Test probability model

Tests:
- Model training
- Prediction output
- Feature handling
```

### 📄 tests/test_calibration.py
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Test calibration

Tests:
- Platt scaling
- Isotonic regression
- Calibration metrics
```

### 📄 tests/test_dynamic_exits.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Test dynamic exit logic

Tests:
- SL calculation
- TP optimization
- Bounds enforcement
```

### 📄 tests/test_position_sizing.py
```
Status:     [NEW]
Phase:      [P4]
Purpose:    Test Kelly criterion

Tests:
- Kelly calculation
- Fractional Kelly
- Constraints
```

### 📄 tests/test_backtest_engine.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Test backtest engine

Tests:
- Trade simulation
- Equity curve
- Metrics calculation
```

### 📄 tests/test_integration.py
```
Status:     [NEW]
Phase:      [P5]
Purpose:    End-to-end integration tests

Tests:
- Full pipeline on sample data
- No data leakage
- Reproducibility
```

---

## NOTEBOOKS DIRECTORY

### 📁 notebooks/

### 📄 notebooks/01_data_exploration.ipynb
```
Status:     [NEW]
Phase:      [CORE]
Purpose:    Interactive data exploration

Description:
Jupyter notebook for:
- Data loading and inspection
- Basic statistics
- Visualization
- Quality checks
```

### 📄 notebooks/02_alpha_research.ipynb
```
Status:     [NEW]
Phase:      [P1]
Purpose:    Interactive alpha research

Description:
Notebook for:
- Running research analyses
- Visualizing findings
- Parameter exploration
```

### 📄 notebooks/03_label_analysis.ipynb
```
Status:     [NEW]
Phase:      [P2]
Purpose:    Label exploration

Description:
Notebook for:
- Label distribution analysis
- Correlation analysis
- Quality assessment
```

### 📄 notebooks/04_model_training.ipynb
```
Status:     [NEW]
Phase:      [P3]
Purpose:    Interactive model development

Description:
Notebook for:
- Model experimentation
- Calibration analysis
- Feature importance
```

### 📄 notebooks/05_backtest_analysis.ipynb
```
Status:     [NEW]
Phase:      [P5]
Purpose:    Backtest results analysis

Description:
Notebook for:
- Backtest visualization
- Performance analysis
- Report generation
```

---

## ARTIFACTS DIRECTORY

### 📁 artifacts/

Output directory structure:

```
artifacts/
├── research/                   [P1 OUTPUT]
│   ├── alpha_research_report.json
│   ├── mean_reversion_analysis.json
│   ├── volatility_analysis.json
│   ├── session_analysis.json
│   └── mtf_analysis.json
│
├── labels/                     [P2 OUTPUT]
│   ├── label_statistics.json
│   └── EURUSD_H4_labeled.csv
│
├── models/                     [P3 OUTPUT]
│   ├── prob_model_fold_1.pkl
│   ├── prob_model_fold_2.pkl
│   ├── calibrator_fold_1.pkl
│   ├── feature_columns.json
│   └── model_metadata.json
│
├── strategy/                   [P4 OUTPUT]
│   └── strategy_config.json
│
├── validation/                 [P5 OUTPUT]
│   ├── final_report.json
│   ├── trade_log.csv
│   ├── equity_curve.csv
│   ├── monte_carlo_results.json
│   └── fold_results.json
│
└── plots/                      [P5 OUTPUT]
    ├── equity_curve.png
    ├── drawdown_chart.png
    ├── calibration_curve_30.png
    ├── calibration_curve_50.png
    └── regime_performance.png
```

---

## ARCHIVE DIRECTORY (V1 Files)

### 📁 archive/

Contains V1 files for reference only. These are **NOT used** in V2.

| File | V1 Purpose | Why Deprecated |
|------|------------|----------------|
| run_pure_ml_stella_alpha.py | Main V1 pipeline | Replaced by V2 modular pipeline |
| pure_ml_labels.py | Binary labels | Replaced by multi_target_labels.py |
| experiment.py | Run experiments | Replaced by walk_forward.py |
| training.py | Train model | Replaced by model_trainer.py |
| evaluation.py | Evaluate model | Replaced by probability_metrics.py |
| diagnose.py | Diagnose results | Replaced by report_generator.py |
| analyze_losses.py | Loss analysis | Concept replaced by probability approach |
| filter_recommendations.py | Generate filters | Not needed with probability model |
| loss_analysis.py | Loss patterns | Not needed with probability model |
| recommend_filters.py | CLI for filters | Not needed |
| compare_versions.py | Compare V3/V4 | Not needed |
| tier_classification.py | Classify tiers | Replaced by dynamic strategy |
| trade_recorder.py | Record trades | Integrated into backtest_engine |
| checkpoint_db.py | Resume capability | Simplified in V2 |
| statistical_validation.py | Statistical tests | Integrated into research modules |
| validate_leakage.py | Leakage validation | Integrated into data_merger |
| test_suite.py | V1 tests | Replaced by tests/ directory |
| pure_ml_stella_alpha_settings.yaml | V1 config | Replaced by stella_v2_settings.yaml |

---

# FILE COUNT SUMMARY

```
TOTAL FILES IN V2:
══════════════════════════════════════════════════════════════════════════════

ROOT FILES:                     6
CONFIG FILES:                   2
DATA FILES:                     2
SRC/DATA:                       5
SRC/RESEARCH:                   8
SRC/LABELS:                     5
SRC/MODELS:                     6
SRC/STRATEGY:                   6
SRC/EVALUATION:                 4
SRC/VALIDATION:                 6
SRC/UTILS:                      5
SCRIPTS:                        6
TESTS:                         13
NOTEBOOKS:                      5
─────────────────────────────────
TOTAL NEW/MODIFIED:            79 files

ARCHIVED (V1):                 18 files
```

---

# PHASE-BY-PHASE FILE CREATION ORDER

```
DAY 1 - PHASE 1 (Alpha Research):
══════════════════════════════════════════════════════════════════════════════
1. Create directory structure
2. Create config/stella_v2_settings.yaml
3. Create src/utils/logger.py
4. Create src/utils/config_loader.py
5. Create src/utils/exceptions.py
6. Create src/data/data_loader.py
7. Create src/data/data_merger.py
8. Create src/data/data_validator.py
9. Create src/research/forward_returns.py
10. Create src/research/mean_reversion_analysis.py
11. Create src/research/volatility_analysis.py
12. Create src/research/session_analysis.py
13. Create src/research/mtf_analysis.py
14. Create src/research/research_report.py
15. Create src/research/alpha_research.py
16. Create scripts/run_alpha_research.py
17. Create tests/test_forward_returns.py
18. Create tests/test_data_merger.py


DAY 1-2 - PHASE 2 (Labels):
══════════════════════════════════════════════════════════════════════════════
19. Create src/labels/forward_metrics.py
20. Create src/labels/quality_flags.py
21. Create src/labels/label_statistics.py
22. Create src/labels/multi_target_labels.py
23. Create scripts/run_generate_labels.py
24. Create tests/test_multi_target_labels.py


DAY 2-3 - PHASE 3 (Model):
══════════════════════════════════════════════════════════════════════════════
25. Create src/models/probability_model.py
26. Create src/models/calibration.py
27. Create src/models/walk_forward.py
28. Create src/models/model_persistence.py
29. Create src/models/model_trainer.py
30. Create src/evaluation/probability_metrics.py
31. Create src/evaluation/calibration_curves.py
32. Create src/evaluation/model_comparison.py
33. Create scripts/run_train_model.py
34. Create tests/test_probability_model.py
35. Create tests/test_calibration.py


DAY 3-4 - PHASE 4 (Strategy):
══════════════════════════════════════════════════════════════════════════════
36. Create src/strategy/regime_detector.py
37. Create src/strategy/dynamic_exits.py
38. Create src/strategy/ev_optimizer.py
39. Create src/strategy/position_sizing.py
40. Create src/strategy/trade_decision.py
41. Create scripts/run_optimize_strategy.py
42. Create tests/test_dynamic_exits.py
43. Create tests/test_position_sizing.py


DAY 4-5 - PHASE 5 (Validation):
══════════════════════════════════════════════════════════════════════════════
44. Create src/validation/backtest_engine.py
45. Create src/validation/performance_metrics.py
46. Create src/validation/monte_carlo.py
47. Create src/validation/regime_analysis.py
48. Create src/validation/report_generator.py
49. Create src/utils/visualization.py
50. Create scripts/run_backtest.py
51. Create scripts/run_stella_v2_pipeline.py
52. Create tests/test_backtest_engine.py
53. Create tests/test_integration.py


FINAL:
══════════════════════════════════════════════════════════════════════════════
54. Update README.md
55. Update requirements.txt
56. Create setup.py
57. Create .gitignore
58. Create all __init__.py files
59. Create notebook templates
60. Move V1 files to archive/
```

---

# END OF FILE ARCHITECTURE DOCUMENT

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  STELLA ALPHA V2 FILE ARCHITECTURE COMPLETE                                   ║
║                                                                               ║
║  Total Files:        79 (new/modified)                                        ║
║  Archived Files:     18 (V1 deprecated)                                       ║
║  Directories:        15                                                       ║
║                                                                               ║
║  Implementation Timeline: 5 days                                              ║
║                                                                               ║
║  TO IMPLEMENT:                                                                ║
║  Follow the phase-by-phase file creation order above.                         ║
║  Each file has clear purpose and dependencies documented.                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```
