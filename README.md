# Stella Alpha: Multi-Timeframe Pure ML Pipeline

**Version 5** of the trading ML pipeline, featuring multi-timeframe analysis and loss pattern detection.

## 🎯 What's New in Stella Alpha

### 1. Multi-Timeframe Analysis (MTF)
- **D1 Data Integration**: Uses daily (D1) data for higher timeframe context
- **Cross-Timeframe Features**: ~20 new features comparing H4 and D1
- **Safe Merge**: Strict leakage prevention (only uses completed D1 data)

### 2. High R:R Configurations (Runners)
- **Fixed SL=30 pips**: All configs have favorable risk:reward
- **TP Range 50-150 pips**: Focus on larger moves
- **Tier 0 Runners**: R:R >= 2.5:1 (need only 17-29% win rate)
- **Tier 1 Ideal**: R:R 1.67-2.49:1 (need only 30-37% win rate)

### 3. Loss Analysis System
- Record every trade with full feature context
- Compare winning vs losing trades statistically
- Identify patterns that predict losses
- Generate actionable filter recommendations

## 📊 Config Space

| Parameter | V4 | Stella Alpha |
|-----------|-----|--------------|
| TP Range | 30-80 pips | 50-150 pips |
| SL | 40 pips | 30 pips (fixed) |
| Hold | 12-30 bars | 12-72 bars |
| Total Configs | 24 | 66 |
| Min R:R | 0.6:1 | 1.67:1 |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pure_ml_stella_alpha.py -c config/pure_ml_stella_alpha_settings.yaml -i data/EURUSD_H4.csv --d1-input data/EURUSD_D1.csv -w 6

# Analyze results
python diagnose.py --db artifacts/pure_ml_stella_alpha.db

# Loss analysis
python analyze_losses.py --trades artifacts/trades_stella_alpha.db

# Filter recommendations
python recommend_filters.py --trades artifacts/trades_stella_alpha.db --tp 100 --sl 30
```

## 📁 Project Structure

```
stella-alpha/
├── run_pure_ml_stella_alpha.py    # Main entry point
├── diagnose.py                    # Diagnostic analysis
├── analyze_losses.py              # Loss analysis CLI
├── recommend_filters.py           # Filter recommendations CLI
├── compare_versions.py            # V4 vs Stella Alpha comparison
│
├── src/
│   ├── __init__.py
│   ├── data_merger.py             # D1 safe merge (leakage prevention)
│   ├── tier_classification.py     # Tier 0/1 classification
│   ├── checkpoint_db.py           # Enhanced checkpoint database
│   ├── feature_engineering.py     # H4 + D1 + MTF features
│   ├── statistical_validation.py  # Statistical tests
│   ├── pure_ml_labels.py          # Label generation
│   ├── experiment.py              # Parallel experiment runner
│   ├── training.py                # Model training
│   ├── evaluation.py              # Metrics calculation
│   ├── trade_recorder.py          # Trade logging for loss analysis
│   ├── loss_analysis.py           # Loss pattern detection
│   └── filter_recommendations.py  # Filter generation
│
├── config/
│   └── pure_ml_stella_alpha_settings.yaml
│
├── artifacts/
│   ├── pure_ml_stella_alpha.db    # Checkpoint database
│   ├── trades_stella_alpha.db     # Trade log for loss analysis
│   └── models/                    # Exported models per config
│
└── tests/
    └── test_data_merger.py
```

## 🔒 Data Leakage Prevention

**CRITICAL**: The D1 merge is the most important code for correctness.

```
For H4 candle at timestamp T:
→ Use D1 candle from floor(T.date) - 1 day
→ NEVER use same-day D1 data

Example:
H4: 2024-01-15 08:00 → Uses D1: 2024-01-14 ✓
H4: 2024-01-15 12:00 → Uses D1: 2024-01-14 ✓
H4: 2024-01-16 00:00 → Uses D1: 2024-01-15 ✓ (now available)
```

## 📈 Success Metrics

| Metric | V4 Baseline | Stella Alpha Target |
|--------|-------------|---------------------|
| Pass Rate | 39.6% | 50%+ |
| GOLD Configs | 1 | 3+ |
| Best EV | +4.17 | +8.0+ |
| TP >= 80 passing | 0 | 3+ |
| High R:R passing | 0 | 5+ |

## 🏆 Tier Classification

### Tier 0 - Runners 🚀
- R:R >= 2.5:1 (TP >= 75 with SL=30)
- One winner covers 2.5-5 losers
- Need only 17-29% win rate to profit

### Tier 1 - Ideal ⭐
- R:R 1.67:1 to 2.49:1 (TP 50-70 with SL=30)
- One winner covers 1.67-2.33 losers
- Need only 30-37% win rate to profit

## 📝 License

Private - All rights reserved
