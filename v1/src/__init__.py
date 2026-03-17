"""
Stella Alpha - Source Package
==============================

Multi-Timeframe Pure ML Pipeline for HIGH R:R SHORT trade prediction.
"""

# Core modules
from .checkpoint_db import StellaAlphaCheckpointManager
from .data_merger import DataMerger, merge_h4_d1_safe, validate_no_leakage
from .evaluation import compute_metrics, optimize_threshold, check_acceptance_criteria
from .experiment import run_all_experiments, generate_config_space
from .feature_engineering import FeatureEngineering, get_feature_columns
from .features import run_rfe, run_rfe_with_importance, get_consensus_features
from .filter_recommendations import FilterRecommendationEngine
from .loss_analysis import LossAnalyzer
from .pure_ml_labels import precompute_all_labels, apply_precomputed_labels
from .statistical_validation import StatisticalFeatureFilter, apply_statistical_prefilter
from .tier_classification import classify_tier, Tier
from .trade_recorder import TradeRecorder
from .training import train_model, save_model

__all__ = [
    'StellaAlphaCheckpointManager',
    'DataMerger',
    'merge_h4_d1_safe',
    'validate_no_leakage',
    'compute_metrics',
    'optimize_threshold',
    'check_acceptance_criteria',
    'run_all_experiments',
    'generate_config_space',
    'FeatureEngineering',
    'get_feature_columns',
    'run_rfe',
    'run_rfe_with_importance',
    'get_consensus_features',
    'FilterRecommendationEngine',
    'LossAnalyzer',
    'precompute_all_labels',
    'apply_precomputed_labels',
    'StatisticalFeatureFilter',
    'apply_statistical_prefilter',
    'classify_tier',
    'Tier',
    'TradeRecorder',
    'train_model',
    'save_model',
]
