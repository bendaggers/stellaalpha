"""
Stella Alpha: Multi-Timeframe Pure ML Pipeline

Version 5 of the trading ML pipeline, featuring:
- D1 (Daily) data integration with H4
- Loss Analysis System
- Filter Recommendations
- High R:R Runner configs
- Tier Classification (Tier 0: Runners, Tier 1: Ideal)
"""

__version__ = "5.0.0"
__name__ = "stella_alpha"

from .data_merger import (
    load_h4_data,
    load_d1_data,
    merge_h4_d1_safe,
    validate_no_leakage,
    handle_missing_d1_data,
    MergeStats,
    LeakageValidationResult
)

from .tier_classification import (
    Tier,
    TierClassification,
    classify_tier,
    classify_all_configs,
    calculate_risk_reward,
    calculate_min_winrate,
    generate_stella_alpha_config_space
)

from .checkpoint_db import (
    StellaAlphaCheckpointManager,
    ConfigResult
)

__all__ = [
    # Data Merger
    'load_h4_data',
    'load_d1_data', 
    'merge_h4_d1_safe',
    'validate_no_leakage',
    'handle_missing_d1_data',
    'MergeStats',
    'LeakageValidationResult',
    
    # Tier Classification
    'Tier',
    'TierClassification',
    'classify_tier',
    'classify_all_configs',
    'calculate_risk_reward',
    'calculate_min_winrate',
    'generate_stella_alpha_config_space',
    
    # Checkpoint
    'StellaAlphaCheckpointManager',
    'ConfigResult'
]
