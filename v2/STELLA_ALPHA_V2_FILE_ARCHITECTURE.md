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
