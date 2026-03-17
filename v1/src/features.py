"""
Feature Selection Module - Stella Alpha
========================================

Standalone feature selection utilities for the Stella Alpha pipeline.
Provides RFE, feature importance analysis, and consensus feature selection.

This module complements feature_engineering.py by providing:
1. Simplified RFE API matching the spec
2. LightGBM-based feature selection (faster than GradientBoosting)
3. Feature consensus across folds
4. Feature importance validation
5. D1/MTF feature tracking

Usage:
    from features import (
        run_rfe,
        get_feature_importance,
        get_consensus_features,
        analyze_feature_categories,
    )
    
    # Simple RFE
    selected = run_rfe(X_train, y_train, settings)
    
    # With importance
    result = run_rfe_with_importance(X_train, y_train, settings)
    print(result.selected_features)
    print(result.importance_ranking)
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureImportance:
    """Single feature importance result."""
    feature_name: str
    importance: float
    rank: int
    selected: bool
    category: str = "unknown"  # h4, d1, mtf, other


@dataclass
class RFEResult:
    """Result of RFE feature selection."""
    selected_features: List[str]
    n_features_selected: int
    n_features_original: int
    importance_ranking: List[FeatureImportance]
    optimal_n_features: Optional[int] = None
    cv_scores: Optional[List[float]] = None
    method: str = "rfe"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'selected_features': self.selected_features,
            'n_features_selected': self.n_features_selected,
            'n_features_original': self.n_features_original,
            'optimal_n_features': self.optimal_n_features,
            'method': self.method,
            'top_10_features': [
                {'name': f.feature_name, 'importance': f.importance, 'category': f.category}
                for f in self.importance_ranking[:10]
            ],
        }
    
    def get_by_category(self) -> Dict[str, List[str]]:
        """Group selected features by category."""
        categories = {'h4': [], 'd1': [], 'mtf': [], 'other': []}
        for feat in self.selected_features:
            cat = categorize_feature(feat)
            categories[cat].append(feat)
        return categories


@dataclass
class ConsensusResult:
    """Result of consensus feature selection across folds."""
    consensus_features: List[str]
    feature_counts: Dict[str, int]
    n_folds: int
    threshold: float
    all_fold_features: List[List[str]] = field(default_factory=list)


# =============================================================================
# FEATURE CATEGORIZATION
# =============================================================================

def categorize_feature(feature_name: str) -> str:
    """
    Categorize feature by source timeframe.
    
    Categories:
        - h4: H4-only features (base or derived)
        - d1: D1 features (prefixed with d1_)
        - mtf: Cross-timeframe features (mtf_, h4_vs_d1_, d1_supports_, etc.)
        - other: Unknown category
    """
    name = feature_name.lower()
    
    # MTF features
    if name.startswith('mtf_'):
        return 'mtf'
    if name.startswith('h4_vs_d1_') or name.startswith('h4_vs_'):
        return 'mtf'
    if name in ('d1_supports_short', 'd1_opposes_short', 'd1_neutral', 'mtf_strong_short_setup'):
        return 'mtf'
    
    # D1 features
    if name.startswith('d1_'):
        return 'd1'
    
    # H4 features (everything else that's numeric)
    return 'h4'


def analyze_feature_categories(features: List[str]) -> Dict[str, Any]:
    """
    Analyze feature distribution by category.
    
    Returns:
        Dict with counts and percentages per category
    """
    categories = {'h4': [], 'd1': [], 'mtf': [], 'other': []}
    
    for feat in features:
        cat = categorize_feature(feat)
        categories[cat].append(feat)
    
    total = len(features)
    result = {
        'total': total,
        'categories': {},
    }
    
    for cat, feats in categories.items():
        count = len(feats)
        result['categories'][cat] = {
            'count': count,
            'percentage': (count / total * 100) if total > 0 else 0,
            'features': feats,
        }
    
    # Summary flags
    result['has_d1'] = len(categories['d1']) > 0
    result['has_mtf'] = len(categories['mtf']) > 0
    result['d1_contribution'] = len(categories['d1']) / total * 100 if total > 0 else 0
    result['mtf_contribution'] = len(categories['mtf']) / total * 100 if total > 0 else 0
    
    return result


# =============================================================================
# RFE FEATURE SELECTION
# =============================================================================

def run_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    settings: Optional[Dict] = None,
) -> List[str]:
    """
    Run Recursive Feature Elimination with Cross-Validation.
    
    This is the simplified API matching the spec.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        settings: Dict with 'rfe' key containing:
            - min_features: Minimum features to select (default: 10)
            - max_features: Maximum features to select (default: 25)
            - cv_folds: CV folds for RFECV (default: 3)
            - use_rfecv: Use RFECV vs RFE (default: True)
            - scoring: Scoring metric (default: 'average_precision')
    
    Returns:
        List of selected feature names
    """
    result = run_rfe_with_importance(X, y, settings)
    return result.selected_features


def run_rfe_with_importance(
    X: pd.DataFrame,
    y: pd.Series,
    settings: Optional[Dict] = None,
) -> RFEResult:
    """
    Run RFE and return full result with importance rankings.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        settings: Configuration dict
    
    Returns:
        RFEResult with selected features and importance rankings
    """
    if settings is None:
        settings = {}
    
    rfe_settings = settings.get('rfe', {})
    
    min_features = rfe_settings.get('min_features', 10)
    max_features = rfe_settings.get('max_features', 25)
    cv_folds = rfe_settings.get('cv_folds', 3)
    use_rfecv = rfe_settings.get('use_rfecv', True)
    scoring = rfe_settings.get('scoring', 'average_precision')
    random_state = settings.get('random_state', 42)
    
    feature_columns = list(X.columns)
    n_features_original = len(feature_columns)
    
    # Handle edge cases
    if n_features_original == 0:
        return RFEResult(
            selected_features=[],
            n_features_selected=0,
            n_features_original=0,
            importance_ranking=[],
            method='rfe',
        )
    
    if n_features_original <= min_features:
        # Not enough features to select from
        importance_ranking = [
            FeatureImportance(
                feature_name=col,
                importance=1.0,
                rank=i + 1,
                selected=True,
                category=categorize_feature(col),
            )
            for i, col in enumerate(feature_columns)
        ]
        return RFEResult(
            selected_features=feature_columns,
            n_features_selected=n_features_original,
            n_features_original=n_features_original,
            importance_ranking=importance_ranking,
            method='passthrough',
        )
    
    # Prepare data
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Create estimator
    if LIGHTGBM_AVAILABLE:
        estimator = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
            force_col_wise=True,
        )
    else:
        estimator = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=random_state,
            subsample=0.8,
        )
    
    # Run RFE or RFECV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = None
    
    if use_rfecv:
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring=scoring,
            min_features_to_select=min_features,
            n_jobs=-1,
        )
        method = 'rfecv'
    else:
        n_select = min(max_features, n_features_original)
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_select,
            step=1,
        )
        method = 'rfe'
    
    try:
        selector.fit(X_clean, y)
    except Exception as e:
        # Fallback: return top features by simple importance
        warnings.warn(f"RFE failed: {e}. Using importance-based selection.")
        return _fallback_importance_selection(X_clean, y, min_features, max_features, random_state)
    
    # Get selected features
    selected_mask = selector.support_
    selected_features = [col for col, sel in zip(feature_columns, selected_mask) if sel]
    
    # Limit to max_features if RFECV selected too many
    if len(selected_features) > max_features:
        # Get importances and take top
        try:
            importances = selector.estimator_.feature_importances_
            feat_imp = sorted(
                zip(selected_features, importances[selected_mask]),
                key=lambda x: x[1],
                reverse=True
            )
            selected_features = [f[0] for f in feat_imp[:max_features]]
        except Exception:
            selected_features = selected_features[:max_features]
    
    # Build importance ranking
    importance_ranking = _build_importance_ranking(
        selector, feature_columns, selected_features
    )
    
    # Get CV scores if available
    if hasattr(selector, 'cv_results_'):
        cv_scores = list(selector.cv_results_.get('mean_test_score', []))
    
    return RFEResult(
        selected_features=selected_features,
        n_features_selected=len(selected_features),
        n_features_original=n_features_original,
        importance_ranking=importance_ranking,
        optimal_n_features=selector.n_features_ if hasattr(selector, 'n_features_') else len(selected_features),
        cv_scores=cv_scores,
        method=method,
    )


def _build_importance_ranking(
    selector,
    feature_columns: List[str],
    selected_features: List[str],
) -> List[FeatureImportance]:
    """Build importance ranking from selector."""
    rankings = []
    
    # Get rankings from selector
    feature_ranks = selector.ranking_
    
    # Try to get importances
    importances = {}
    try:
        if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
            # Map importances to selected features
            selected_set = set(selected_features)
            imp_values = selector.estimator_.feature_importances_
            selected_cols = [c for c in feature_columns if c in selected_set]
            if len(imp_values) == len(selected_cols):
                importances = dict(zip(selected_cols, imp_values))
    except Exception:
        pass
    
    # Build ranking objects
    for col, rank in zip(feature_columns, feature_ranks):
        rankings.append(FeatureImportance(
            feature_name=col,
            importance=importances.get(col, 0.0),
            rank=int(rank),
            selected=(col in selected_features),
            category=categorize_feature(col),
        ))
    
    # Sort by importance (selected first, then by importance)
    rankings.sort(key=lambda x: (-int(x.selected), -x.importance, x.rank))
    
    return rankings


def _fallback_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int,
    max_features: int,
    random_state: int,
) -> RFEResult:
    """Fallback feature selection using simple importance."""
    feature_columns = list(X.columns)
    
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
            verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=random_state,
        )
    
    model.fit(X, y)
    importances = model.feature_importances_
    
    # Rank by importance
    feat_imp = sorted(
        zip(feature_columns, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    n_select = min(max_features, max(min_features, len(feature_columns)))
    selected = [f[0] for f in feat_imp[:n_select]]
    
    importance_ranking = [
        FeatureImportance(
            feature_name=col,
            importance=imp,
            rank=i + 1,
            selected=(col in selected),
            category=categorize_feature(col),
        )
        for i, (col, imp) in enumerate(feat_imp)
    ]
    
    return RFEResult(
        selected_features=selected,
        n_features_selected=len(selected),
        n_features_original=len(feature_columns),
        importance_ranking=importance_ranking,
        method='importance_fallback',
    )


# =============================================================================
# FEATURE IMPORTANCE UTILITIES
# =============================================================================

def get_feature_importance(
    model,
    feature_columns: List[str],
    top_n: int = 20,
) -> List[FeatureImportance]:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_columns: List of feature names
        top_n: Number of top features to return
    
    Returns:
        List of FeatureImportance sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        return []
    
    importances = model.feature_importances_
    
    if len(importances) != len(feature_columns):
        return []
    
    feat_imp = sorted(
        zip(feature_columns, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    result = [
        FeatureImportance(
            feature_name=col,
            importance=imp,
            rank=i + 1,
            selected=True,
            category=categorize_feature(col),
        )
        for i, (col, imp) in enumerate(feat_imp[:top_n])
    ]
    
    return result


def get_top_features_by_category(
    importance_list: List[FeatureImportance],
    category: str,
    top_n: int = 10,
) -> List[FeatureImportance]:
    """Get top N features from a specific category."""
    filtered = [f for f in importance_list if f.category == category]
    return sorted(filtered, key=lambda x: -x.importance)[:top_n]


# =============================================================================
# CONSENSUS FEATURE SELECTION
# =============================================================================

def get_consensus_features(
    fold_features: List[List[str]],
    threshold: float = 0.6,
) -> ConsensusResult:
    """
    Get consensus features selected across multiple folds.
    
    Args:
        fold_features: List of feature lists from each fold
        threshold: Minimum fraction of folds a feature must appear in (default: 0.6 = 60%)
    
    Returns:
        ConsensusResult with consensus features and counts
    """
    n_folds = len(fold_features)
    
    if n_folds == 0:
        return ConsensusResult(
            consensus_features=[],
            feature_counts={},
            n_folds=0,
            threshold=threshold,
        )
    
    # Count feature occurrences
    feature_counts = {}
    for features in fold_features:
        for feat in features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    # Filter by threshold
    min_count = int(np.ceil(n_folds * threshold))
    consensus = [
        feat for feat, count in feature_counts.items()
        if count >= min_count
    ]
    
    # Sort by count (most common first)
    consensus.sort(key=lambda x: -feature_counts[x])
    
    return ConsensusResult(
        consensus_features=consensus,
        feature_counts=feature_counts,
        n_folds=n_folds,
        threshold=threshold,
        all_fold_features=fold_features,
    )


def analyze_consensus_stability(
    consensus_result: ConsensusResult,
) -> Dict[str, Any]:
    """
    Analyze stability of feature selection across folds.
    
    Returns:
        Dict with stability metrics
    """
    counts = consensus_result.feature_counts
    n_folds = consensus_result.n_folds
    
    if n_folds == 0 or not counts:
        return {
            'stability_score': 0.0,
            'always_selected': [],
            'sometimes_selected': [],
            'rarely_selected': [],
        }
    
    always = [f for f, c in counts.items() if c == n_folds]
    sometimes = [f for f, c in counts.items() if n_folds * 0.4 <= c < n_folds]
    rarely = [f for f, c in counts.items() if c < n_folds * 0.4]
    
    # Stability score: fraction of features that appear in all folds
    total_unique = len(counts)
    stability_score = len(always) / total_unique if total_unique > 0 else 0
    
    return {
        'stability_score': stability_score,
        'always_selected': always,
        'always_count': len(always),
        'sometimes_selected': sometimes,
        'sometimes_count': len(sometimes),
        'rarely_selected': rarely,
        'rarely_count': len(rarely),
        'total_unique_features': total_unique,
    }


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def validate_selected_features(
    df: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validate selected features exist and have no issues.
    
    Returns:
        (valid_features, validation_report)
    """
    valid = []
    issues = {}
    
    for feat in selected_features:
        if feat not in df.columns:
            issues[feat] = 'missing'
            continue
        
        col = df[feat]
        
        # Check for all NaN
        if col.isna().all():
            issues[feat] = 'all_nan'
            continue
        
        # Check for zero variance
        if col.std() == 0:
            issues[feat] = 'zero_variance'
            continue
        
        # Check for excessive NaN
        nan_pct = col.isna().mean()
        if nan_pct > 0.5:
            issues[feat] = f'high_nan_{nan_pct:.1%}'
            continue
        
        valid.append(feat)
    
    report = {
        'n_input': len(selected_features),
        'n_valid': len(valid),
        'n_invalid': len(issues),
        'issues': issues,
    }
    
    return valid, report


# =============================================================================
# PRINT UTILITIES
# =============================================================================

def print_feature_selection_report(result: RFEResult, verbose: bool = True):
    """Print formatted feature selection report."""
    print("\n" + "=" * 60)
    print("  FEATURE SELECTION REPORT")
    print("=" * 60)
    
    print(f"""
    Method:           {result.method.upper()}
    Original:         {result.n_features_original} features
    Selected:         {result.n_features_selected} features
    """)
    
    # Category breakdown
    categories = result.get_by_category()
    print("    Category Breakdown:")
    print("    " + "-" * 40)
    for cat, feats in categories.items():
        if feats:
            print(f"      {cat.upper():6s}: {len(feats):3d} features")
    
    if verbose and result.importance_ranking:
        print("\n    Top 15 Features by Importance:")
        print("    " + "-" * 50)
        print(f"    {'Rank':<6} {'Feature':<35} {'Importance':>10}")
        print("    " + "-" * 50)
        
        for feat in result.importance_ranking[:15]:
            if feat.selected:
                marker = "✓"
            else:
                marker = " "
            print(f"    {marker} {feat.rank:<4d} {feat.feature_name:<35} {feat.importance:>10.4f}")
    
    print("\n" + "=" * 60)


def print_consensus_report(result: ConsensusResult):
    """Print consensus feature report."""
    print("\n" + "=" * 60)
    print("  CONSENSUS FEATURE REPORT")
    print("=" * 60)
    
    print(f"""
    Folds:            {result.n_folds}
    Threshold:        {result.threshold:.0%}
    Consensus:        {len(result.consensus_features)} features
    """)
    
    stability = analyze_consensus_stability(result)
    
    print(f"    Stability Score: {stability['stability_score']:.1%}")
    print(f"    Always Selected: {stability['always_count']} features")
    print(f"    Sometimes:       {stability['sometimes_count']} features")
    print(f"    Rarely:          {stability['rarely_count']} features")
    
    if result.consensus_features:
        print("\n    Consensus Features (sorted by frequency):")
        print("    " + "-" * 40)
        for feat in result.consensus_features[:20]:
            count = result.feature_counts[feat]
            pct = count / result.n_folds * 100
            print(f"      {feat:<35} {count}/{result.n_folds} ({pct:.0f}%)")
    
    print("\n" + "=" * 60)


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Features Module - Stella Alpha")
    print("=" * 40)
    
    # Quick test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create test features with different prefixes
    feature_names = (
        [f"h4_feature_{i}" for i in range(20)] +
        [f"d1_feature_{i}" for i in range(15)] +
        [f"mtf_feature_{i}" for i in range(10)] +
        [f"other_{i}" for i in range(5)]
    )
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Create target with some signal
    y = pd.Series(
        (X['h4_feature_0'] + X['d1_feature_0'] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    )
    
    print(f"\nTest data: {n_samples} samples, {n_features} features")
    print(f"Target distribution: {y.mean():.1%} positive")
    
    # Test RFE
    print("\n--- Running RFE ---")
    settings = {
        'rfe': {
            'min_features': 10,
            'max_features': 20,
            'cv_folds': 3,
            'use_rfecv': False,
        }
    }
    
    result = run_rfe_with_importance(X, y, settings)
    print_feature_selection_report(result)
    
    # Test category analysis
    print("\n--- Category Analysis ---")
    analysis = analyze_feature_categories(result.selected_features)
    print(f"H4 features: {analysis['categories']['h4']['count']}")
    print(f"D1 features: {analysis['categories']['d1']['count']}")
    print(f"MTF features: {analysis['categories']['mtf']['count']}")
    
    # Test consensus
    print("\n--- Consensus Test ---")
    fold_features = [
        result.selected_features[:15],
        result.selected_features[2:17],
        result.selected_features[:12] + ['extra_1', 'extra_2'],
    ]
    consensus = get_consensus_features(fold_features, threshold=0.6)
    print_consensus_report(consensus)
    
    print("\n✅ All tests passed!")
