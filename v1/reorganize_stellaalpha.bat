@echo off
REM ============================================================================
REM STELLA ALPHA - FILE REORGANIZATION SCRIPT
REM ============================================================================
REM 
REM This script reorganizes the Stella Alpha project into the proper structure.
REM Run this from the stellaalpha root folder.
REM
REM BEFORE: Mixed flat structure
REM AFTER:  Organized src/, config/, tests/, artifacts/ structure
REM
REM ============================================================================

echo.
echo ============================================================================
echo   STELLA ALPHA - FILE REORGANIZATION
echo ============================================================================
echo.

REM Get current directory
set "ROOT=%CD%"
echo Current directory: %ROOT%
echo.

REM ============================================================================
REM STEP 1: Create directories if they don't exist
REM ============================================================================
echo [1/6] Creating directory structure...

if not exist "src" mkdir src
if not exist "config" mkdir config
if not exist "artifacts" mkdir artifacts
if not exist "tests" mkdir tests
if not exist "data" mkdir data

echo       - src/
echo       - config/
echo       - artifacts/
echo       - tests/
echo       - data/
echo.

REM ============================================================================
REM STEP 2: Move source modules to src/ (if they're in root)
REM ============================================================================
echo [2/6] Moving source modules to src/...

REM List of source modules that should be in src/
set "SRC_FILES=checkpoint_db.py data_merger.py evaluation.py experiment.py feature_engineering.py features.py filter_recommendations.py loss_analysis.py pure_ml_labels.py statistical_validation.py tier_classification.py trade_recorder.py training.py"

for %%f in (%SRC_FILES%) do (
    if exist "%%f" (
        echo       Moving %%f to src/
        move /Y "%%f" "src\%%f" >nul 2>&1
    )
)

REM Also move __init__.py to src if it exists in root and src doesn't have one
if exist "__init__.py" (
    if not exist "src\__init__.py" (
        echo       Moving __init__.py to src/
        move /Y "__init__.py" "src\__init__.py" >nul 2>&1
    ) else (
        echo       Keeping existing src/__init__.py
        del "__init__.py" >nul 2>&1
    )
)

echo.

REM ============================================================================
REM STEP 3: Move config files to config/
REM ============================================================================
echo [3/6] Moving config files to config/...

if exist "pure_ml_stella_alpha_settings.yaml" (
    echo       Moving pure_ml_stella_alpha_settings.yaml to config/
    move /Y "pure_ml_stella_alpha_settings.yaml" "config\pure_ml_stella_alpha_settings.yaml" >nul 2>&1
)

echo.

REM ============================================================================
REM STEP 4: Move test files to tests/
REM ============================================================================
echo [4/6] Moving test files to tests/...

if exist "test_suite.py" (
    echo       Moving test_suite.py to tests/
    move /Y "test_suite.py" "tests\test_suite.py" >nul 2>&1
)

if exist "validate_leakage.py" (
    echo       Moving validate_leakage.py to tests/
    move /Y "validate_leakage.py" "tests\validate_leakage.py" >nul 2>&1
)

REM Create tests/__init__.py if it doesn't exist
if not exist "tests\__init__.py" (
    echo # Tests package > "tests\__init__.py"
    echo       Created tests/__init__.py
)

echo.

REM ============================================================================
REM STEP 5: Clean up leftover files
REM ============================================================================
echo [5/6] Cleaning up...

REM Delete empty/leftover database files
if exist "v3-fast.db" (
    echo       Deleting v3-fast.db (empty leftover)
    del "v3-fast.db" >nul 2>&1
)

REM Delete root __pycache__ if exists
if exist "__pycache__" (
    echo       Removing root __pycache__/
    rmdir /S /Q "__pycache__" >nul 2>&1
)

echo.

REM ============================================================================
REM STEP 6: Create/update src/__init__.py with proper exports
REM ============================================================================
echo [6/6] Creating src/__init__.py with exports...

(
echo """
echo Stella Alpha - Source Package
echo ==============================
echo.
echo Multi-Timeframe Pure ML Pipeline for HIGH R:R SHORT trade prediction.
echo """
echo.
echo # Core modules
echo from .checkpoint_db import StellaAlphaCheckpointManager
echo from .data_merger import DataMerger, merge_h4_d1_safe, validate_no_leakage
echo from .evaluation import compute_metrics, optimize_threshold, check_acceptance_criteria
echo from .experiment import run_all_experiments, generate_config_space
echo from .feature_engineering import FeatureEngineering, get_feature_columns
echo from .features import run_rfe, run_rfe_with_importance, get_consensus_features
echo from .filter_recommendations import FilterRecommendationEngine
echo from .loss_analysis import LossAnalyzer
echo from .pure_ml_labels import precompute_all_labels, apply_precomputed_labels
echo from .statistical_validation import StatisticalFeatureFilter, apply_statistical_prefilter
echo from .tier_classification import classify_tier, Tier
echo from .trade_recorder import TradeRecorder
echo from .training import train_model, save_model
echo.
echo __all__ = [
echo     'StellaAlphaCheckpointManager',
echo     'DataMerger',
echo     'merge_h4_d1_safe',
echo     'validate_no_leakage',
echo     'compute_metrics',
echo     'optimize_threshold',
echo     'check_acceptance_criteria',
echo     'run_all_experiments',
echo     'generate_config_space',
echo     'FeatureEngineering',
echo     'get_feature_columns',
echo     'run_rfe',
echo     'run_rfe_with_importance',
echo     'get_consensus_features',
echo     'FilterRecommendationEngine',
echo     'LossAnalyzer',
echo     'precompute_all_labels',
echo     'apply_precomputed_labels',
echo     'StatisticalFeatureFilter',
echo     'apply_statistical_prefilter',
echo     'classify_tier',
echo     'Tier',
echo     'TradeRecorder',
echo     'train_model',
echo     'save_model',
echo ]
) > "src\__init__.py"

echo       Updated src/__init__.py
echo.

REM ============================================================================
REM DONE - Show final structure
REM ============================================================================
echo ============================================================================
echo   REORGANIZATION COMPLETE!
echo ============================================================================
echo.
echo Final structure:
echo.
echo   stellaalpha/
echo   ^|
echo   ^|-- run_pure_ml_stella_alpha.py    (main entry point)
echo   ^|-- diagnose.py                    (diagnostics CLI)
echo   ^|-- compare_versions.py            (V4 comparison CLI)
echo   ^|-- analyze_losses.py              (loss analysis CLI)
echo   ^|-- recommend_filters.py           (filter recommendations CLI)
echo   ^|-- requirements.txt
echo   ^|-- README.md
echo   ^|-- STELLA_ALPHA_SPECIFICATION.md
echo   ^|
echo   ^|-- config/
echo   ^|   ^`-- pure_ml_stella_alpha_settings.yaml
echo   ^|
echo   ^|-- src/
echo   ^|   ^|-- __init__.py
echo   ^|   ^|-- checkpoint_db.py
echo   ^|   ^|-- data_merger.py
echo   ^|   ^|-- evaluation.py
echo   ^|   ^|-- experiment.py
echo   ^|   ^|-- feature_engineering.py
echo   ^|   ^|-- features.py               (NEW)
echo   ^|   ^|-- filter_recommendations.py
echo   ^|   ^|-- loss_analysis.py
echo   ^|   ^|-- pure_ml_labels.py
echo   ^|   ^|-- statistical_validation.py
echo   ^|   ^|-- tier_classification.py
echo   ^|   ^|-- trade_recorder.py
echo   ^|   ^`-- training.py
echo   ^|
echo   ^|-- tests/
echo   ^|   ^|-- __init__.py
echo   ^|   ^|-- test_suite.py
echo   ^|   ^`-- validate_leakage.py
echo   ^|
echo   ^|-- artifacts/                     (output files go here)
echo   ^|
echo   ^`-- data/                          (put your CSV files here)
echo       ^|-- EURUSD_H4_features.csv
echo       ^`-- EURUSD_D1_features.csv
echo.
echo ============================================================================
echo.
echo Next steps:
echo   1. Put your H4 and D1 CSV files in the data/ folder
echo   2. Update config/pure_ml_stella_alpha_settings.yaml with correct paths
echo   3. Run: python run_pure_ml_stella_alpha.py --help
echo.
pause
