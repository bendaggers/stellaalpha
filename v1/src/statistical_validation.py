"""
Statistical Validation Framework - STELLA ALPHA (Phase 2)

PURPOSE:
Use statistical tests throughout the pipeline to ensure:
1. Features have REAL predictive power (not noise)
2. Model performance differences are SIGNIFICANT (not luck)
3. Results are RELIABLE and will generalize to live trading

STAGES:
───────
Stage 1: Feature Pre-Filter    – T-test + Cohen's d before RFE (~296 → ~100 features)
Stage 2: Feature Importance    – Validate LightGBM top features have real predictive power
Stage 3: Fold Stability        – Check CV is consistent (no regime overfitting)
Stage 4: Config Comparison     – Paired t-test to confirm best config is truly better

SIGNIFICANCE THRESHOLDS:
────────────────────────
P-value < 0.001:  Highly significant (***)
P-value < 0.01:   Very significant (**)
P-value < 0.05:   Significant (*)
P-value >= 0.05:  Not significant

EFFECT SIZE (Cohen's d):
────────────────────────
d > 0.8:  Large effect
d > 0.5:  Medium effect
d > 0.2:  Small effect
d < 0.2:  Negligible effect
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureSignificance:
    """Statistical significance results for a single feature."""
    feature_name: str
    wins_mean: float
    losses_mean: float
    wins_std: float
    losses_std: float
    t_statistic: float
    p_value: float
    effect_size: float           # Cohen's d
    is_significant: bool         # p < threshold
    is_practically_significant: bool   # effect_size >= min_effect_size
    recommendation: str          # "KEEP", "MAYBE", "DROP"

    @property
    def significance_stars(self) -> str:
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        return ""

    @property
    def effect_label(self) -> str:
        if self.effect_size >= 0.8:
            return "LARGE"
        elif self.effect_size >= 0.5:
            return "MEDIUM"
        elif self.effect_size >= 0.2:
            return "SMALL"
        return "NEGLIGIBLE"


@dataclass
class ValidatedFeatureImportance:
    """Feature importance with statistical validation."""
    feature_name: str
    lgbm_importance: float       # From LightGBM
    lgbm_rank: int               # Rank by importance
    p_value: float               # Statistical significance
    effect_size: float           # Practical significance (Cohen's d)
    is_validated: bool           # Both important AND statistically significant
    confidence: str              # "HIGH", "MEDIUM", "LOW", "UNVALIDATED"


@dataclass
class FoldStabilityResult:
    """Results of fold stability analysis for a single metric."""
    metric_name: str
    fold_values: List[float]
    mean: float
    std: float
    cv: float                    # Coefficient of variation (std/mean)
    is_stable: bool              # CV < 0.15 (15%)
    stability_grade: str         # "A", "B", "C", "F"
    interpretation: str


@dataclass
class ConfigComparisonResult:
    """Statistical comparison between two configs."""
    config_a: str
    config_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float            # mean_a - mean_b
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence: str              # "HIGH", "MEDIUM", "LOW"
    interpretation: str


@dataclass
class FeatureComparison:
    """Feature value comparison between wins and losses."""
    feature_name: str
    wins_mean: float
    wins_std: float
    losses_mean: float
    losses_std: float
    difference: float
    pct_difference: float
    t_statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    direction: str               # "higher_in_losses" or "higher_in_wins"


# =============================================================================
# STAGE 1: FEATURE PRE-FILTER
# =============================================================================

class StatisticalFeatureFilter:
    """
    Filter features based on statistical significance before RFE.

    Reduces ~296 features to ~80-120 significant ones, making
    RFE faster and removing noise that could cause overfitting.

    Usage:
        filter = StatisticalFeatureFilter()
        kept, dropped, results = filter.filter_features(df, feature_cols, 'label')
        filter.print_report()
    """

    def __init__(
        self,
        p_threshold: float = 0.05,
        min_effect_size: float = 0.2,
        min_samples: int = 100,
    ):
        """
        Args:
            p_threshold: P-value cutoff for significance (default 0.05)
            min_effect_size: Minimum Cohen's d (default 0.2 = small effect)
            min_samples: Minimum samples in each group (wins/losses)
        """
        self.p_threshold = p_threshold
        self.min_effect_size = min_effect_size
        self.min_samples = min_samples
        self.results: List[FeatureSignificance] = []

    def analyze_feature(
        self,
        wins: np.ndarray,
        losses: np.ndarray,
        feature_name: str,
    ) -> FeatureSignificance:
        """
        Analyze a single feature for statistical significance.

        Uses Welch's t-test (unequal variances) and Cohen's d effect size.
        """
        # Remove NaN
        wins = wins[~np.isnan(wins)]
        losses = losses[~np.isnan(losses)]

        # Not enough data
        if len(wins) < self.min_samples or len(losses) < self.min_samples:
            return FeatureSignificance(
                feature_name=feature_name,
                wins_mean=0.0, losses_mean=0.0,
                wins_std=0.0, losses_std=0.0,
                t_statistic=0.0, p_value=1.0, effect_size=0.0,
                is_significant=False,
                is_practically_significant=False,
                recommendation="DROP",
            )

        wins_mean = float(np.mean(wins))
        wins_std = float(np.std(wins))
        losses_mean = float(np.mean(losses))
        losses_std = float(np.std(losses))

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(wins, losses, equal_var=False)
        t_stat = float(t_stat)
        p_value = float(p_value)

        # Cohen's d effect size
        pooled_std = float(np.sqrt((wins_std ** 2 + losses_std ** 2) / 2))
        effect_size = abs(wins_mean - losses_mean) / pooled_std if pooled_std > 0 else 0.0

        is_significant = p_value < self.p_threshold
        is_practical = effect_size >= self.min_effect_size

        # Recommendation
        if is_significant and is_practical:
            recommendation = "KEEP"
        elif is_significant or effect_size >= 0.15:
            recommendation = "MAYBE"
        else:
            recommendation = "DROP"

        return FeatureSignificance(
            feature_name=feature_name,
            wins_mean=wins_mean,
            losses_mean=losses_mean,
            wins_std=wins_std,
            losses_std=losses_std,
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            is_practically_significant=is_practical,
            recommendation=recommendation,
        )

    def filter_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = 'label',
    ) -> Tuple[List[str], List[str], List[FeatureSignificance]]:
        """
        Filter features based on statistical significance.

        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names to test
            label_column: Name of the binary label column (1=win, 0=loss)

        Returns:
            (kept_features, dropped_features, all_results)
        """
        wins_mask = df[label_column] == 1
        losses_mask = df[label_column] == 0

        kept: List[str] = []
        dropped: List[str] = []
        self.results = []

        for feature in feature_columns:
            if feature not in df.columns:
                continue

            wins = df.loc[wins_mask, feature].values.astype(float)
            losses = df.loc[losses_mask, feature].values.astype(float)

            result = self.analyze_feature(wins, losses, feature)
            self.results.append(result)

            if result.recommendation in ("KEEP", "MAYBE"):
                kept.append(feature)
            else:
                dropped.append(feature)

        return kept, dropped, self.results

    def print_report(self, top_n: int = 30) -> None:
        """Print feature pre-filter report to console."""
        if not self.results:
            print("  No results to display. Run filter_features() first.")
            return

        print("\n" + "=" * 70)
        print("  STATISTICAL FEATURE PRE-FILTER REPORT")
        print("=" * 70)

        sorted_results = sorted(self.results, key=lambda x: x.effect_size, reverse=True)

        kept = [r for r in sorted_results if r.recommendation in ("KEEP", "MAYBE")]
        dropped = [r for r in sorted_results if r.recommendation == "DROP"]
        keep_only = [r for r in kept if r.recommendation == "KEEP"]

        print(f"\n  SUMMARY:")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Total features analyzed:  {len(self.results)}")
        print(f"  KEEP  (p<{self.p_threshold} & d>={self.min_effect_size}):  {len(keep_only)}")
        print(f"  MAYBE (borderline):        {len(kept) - len(keep_only)}")
        print(f"  DROP  (not significant):   {len(dropped)}")

        print(f"\n  TOP {min(top_n, len(sorted_results))} FEATURES BY EFFECT SIZE:")
        print(f"  {'Feature':<38} {'P-Value':>10} {'Stars':>5} {'Effect':>8} {'Size':>10} {'Action':>7}")
        print(f"  " + "─" * 82)

        for r in sorted_results[:top_n]:
            stars = r.significance_stars
            label = r.effect_label[:3]
            print(
                f"  {r.feature_name:<38} {r.p_value:>10.4f} {stars:>5} "
                f"{r.effect_size:>8.3f} {label:>10} {r.recommendation:>7}"
            )

        if dropped:
            print(f"\n  DROPPED ({len(dropped)} features removed as noise):")
            for r in sorted(dropped, key=lambda x: x.effect_size, reverse=True)[:10]:
                print(f"    {r.feature_name:<38} p={r.p_value:.4f}  d={r.effect_size:.3f}")
            if len(dropped) > 10:
                print(f"    ... and {len(dropped) - 10} more")

        print("\n" + "=" * 70)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary (for logging)."""
        kept = [r for r in self.results if r.recommendation in ("KEEP", "MAYBE")]
        dropped = [r for r in self.results if r.recommendation == "DROP"]
        return {
            'total_analyzed': len(self.results),
            'kept': len(kept),
            'dropped': len(dropped),
            'kept_features': [r.feature_name for r in kept],
            'dropped_features': [r.feature_name for r in dropped],
        }


# =============================================================================
# STAGE 2: FEATURE IMPORTANCE VALIDATION
# =============================================================================

def validate_feature_importance(
    model,
    df_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
) -> List[ValidatedFeatureImportance]:
    """
    Validate LightGBM feature importance with statistical tests.

    A feature might have high LightGBM importance but no real
    predictive power (possible overfitting to noise). This function
    cross-references importance with statistical significance.

    Args:
        model: Trained LightGBM model
        df_test: Test set features
        y_test: Test set labels
        feature_columns: Feature column names

    Returns:
        List of ValidatedFeatureImportance sorted by lgbm_rank
    """
    # Get LightGBM importances
    lgbm_importance = dict(zip(feature_columns, model.feature_importances_))
    sorted_features = sorted(lgbm_importance.items(), key=lambda x: x[1], reverse=True)

    stat_filter = StatisticalFeatureFilter(p_threshold=0.05, min_effect_size=0.1, min_samples=30)

    results: List[ValidatedFeatureImportance] = []

    for rank, (feature, importance) in enumerate(sorted_features, 1):
        if feature not in df_test.columns:
            continue

        wins = df_test.loc[y_test == 1, feature].values.astype(float)
        losses = df_test.loc[y_test == 0, feature].values.astype(float)

        stat_result = stat_filter.analyze_feature(wins, losses, feature)

        is_validated = stat_result.is_significant and importance > 0

        if is_validated and stat_result.effect_size > 0.5:
            confidence = "HIGH"
        elif is_validated and stat_result.effect_size > 0.2:
            confidence = "MEDIUM"
        elif stat_result.is_significant:
            confidence = "LOW"
        else:
            confidence = "UNVALIDATED"

        results.append(ValidatedFeatureImportance(
            feature_name=feature,
            lgbm_importance=importance,
            lgbm_rank=rank,
            p_value=stat_result.p_value,
            effect_size=stat_result.effect_size,
            is_validated=is_validated,
            confidence=confidence,
        ))

    return results


def print_validated_importance(
    results: List[ValidatedFeatureImportance],
    top_n: int = 25,
) -> None:
    """Print validated feature importance report."""
    print("\n" + "=" * 80)
    print("  VALIDATED FEATURE IMPORTANCE")
    print("=" * 80)

    conf_icon = {"HIGH": "✅", "MEDIUM": "⚠️ ", "LOW": "❓", "UNVALIDATED": "❌"}

    print(f"\n  {'Rank':<6} {'Feature':<32} {'Importance':>12} {'P-Value':>10} {'Effect':>8} {'Status':>14}")
    print(f"  " + "─" * 85)

    for r in results[:top_n]:
        icon = conf_icon.get(r.confidence, "  ")
        print(
            f"  {r.lgbm_rank:<6} {r.feature_name:<32} {r.lgbm_importance:>12.1f} "
            f"{r.p_value:>10.4f} {r.effect_size:>8.3f}  {icon} {r.confidence:>10}"
        )

    validated = [r for r in results if r.is_validated]
    high_conf = [r for r in results if r.confidence == "HIGH"]

    print(f"\n  VALIDATION SUMMARY:")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total features:     {len(results)}")
    print(f"  Validated:          {len(validated)} ({100*len(validated)/max(len(results),1):.1f}%)")
    print(f"  High confidence:    {len(high_conf)}")

    if len(validated) < len(results) * 0.5:
        print(f"\n  ⚠️  WARNING: Less than 50% of features validated!")
        print(f"     Model may be overfitting to noise features.")

    # Highlight D1 and MTF features
    d1_validated = [r for r in validated if r.feature_name.startswith('d1_')]
    mtf_validated = [r for r in validated if r.feature_name.startswith(('mtf_', 'h4_vs_d1_'))]

    if d1_validated:
        print(f"\n  D1 VALIDATED FEATURES ({len(d1_validated)}):")
        for r in d1_validated[:5]:
            print(f"    {r.feature_name:<35} rank={r.lgbm_rank}, d={r.effect_size:.3f}")

    if mtf_validated:
        print(f"\n  MTF VALIDATED FEATURES ({len(mtf_validated)}):")
        for r in mtf_validated[:5]:
            print(f"    {r.feature_name:<35} rank={r.lgbm_rank}, d={r.effect_size:.3f}")


# =============================================================================
# STAGE 3: FOLD STABILITY TESTING
# =============================================================================

def test_fold_stability(
    fold_results: List[Dict],
    metrics: Optional[List[str]] = None,
) -> List[FoldStabilityResult]:
    """
    Test if model metrics are stable across folds.

    Uses Coefficient of Variation (CV = std/mean).
    Unstable results suggest regime-dependence or overfitting.

    Args:
        fold_results: List of dicts with metric values per fold
        metrics: Metric names to check (default: precision, ev, recall)

    Returns:
        List of FoldStabilityResult per metric
    """
    if metrics is None:
        metrics = ['precision', 'ev', 'recall', 'f1', 'auc_pr']

    results: List[FoldStabilityResult] = []

    for metric in metrics:
        values = []
        for fold in fold_results:
            v = fold.get(metric)
            if v is not None and not np.isnan(float(v)):
                values.append(float(v))

        if len(values) < 2:
            continue

        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-9 else float('inf')

        # Grade
        if cv < 0.10:
            grade = "A"
            is_stable = True
            interpretation = "Excellent stability - consistent across time periods"
        elif cv < 0.20:
            grade = "B"
            is_stable = True
            interpretation = "Good stability - minor variation across folds"
        elif cv < 0.35:
            grade = "C"
            is_stable = False
            interpretation = "Moderate instability - may be regime-dependent"
        else:
            grade = "F"
            is_stable = False
            interpretation = "High instability - model unreliable across time periods"

        results.append(FoldStabilityResult(
            metric_name=metric,
            fold_values=values,
            mean=mean_val,
            std=std_val,
            cv=cv,
            is_stable=is_stable,
            stability_grade=grade,
            interpretation=interpretation,
        ))

    return results


def print_fold_stability(results: List[FoldStabilityResult]) -> None:
    """Print fold stability report."""
    print("\n" + "=" * 70)
    print("  FOLD STABILITY ANALYSIS")
    print("=" * 70)

    grade_icon = {"A": "✅", "B": "✅", "C": "⚠️ ", "F": "❌"}

    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8} {'CV%':>8} {'Grade':>7} {'Status':>8}")
    print(f"  " + "─" * 60)

    for r in results:
        icon = grade_icon.get(r.stability_grade, "  ")
        fold_str = ", ".join(f"{v:.3f}" for v in r.fold_values)
        print(
            f"  {r.metric_name:<15} {r.mean:>8.4f} {r.std:>8.4f} {r.cv*100:>7.1f}% "
            f"  {icon} {r.stability_grade:>4}"
        )
        print(f"    Folds: [{fold_str}]")
        print(f"    → {r.interpretation}")

    stable = [r for r in results if r.is_stable]
    print(f"\n  STABILITY SUMMARY: {len(stable)}/{len(results)} metrics stable")

    if len(stable) < len(results):
        unstable = [r.metric_name for r in results if not r.is_stable]
        print(f"  ⚠️  Unstable metrics: {', '.join(unstable)}")
        print(f"     Consider: more data, fewer features, or regime-aware modeling")


# =============================================================================
# STAGE 4: CONFIG COMPARISON
# =============================================================================

def compare_configs_significance(
    config_a_folds: List[float],
    config_b_folds: List[float],
    config_a_name: str,
    config_b_name: str,
    metric: str = 'ev',
) -> ConfigComparisonResult:
    """
    Paired t-test to compare two configs across CV folds.

    Ensures the difference between configs is statistically real,
    not just due to random variation in the data splits.

    Args:
        config_a_folds: Metric values for config A across folds
        config_b_folds: Metric values for config B across folds
        config_a_name: Config A identifier
        config_b_name: Config B identifier
        metric: Name of metric being compared

    Returns:
        ConfigComparisonResult
    """
    a = np.array(config_a_folds, dtype=float)
    b = np.array(config_b_folds, dtype=float)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a, b)
    t_stat = float(t_stat)
    p_value = float(p_value)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    difference = mean_a - mean_b

    is_significant = p_value < 0.05

    if p_value < 0.001:
        confidence = "HIGH"
    elif p_value < 0.01:
        confidence = "MEDIUM"
    elif p_value < 0.05:
        confidence = "LOW"
    else:
        confidence = "NONE"

    if is_significant and difference > 0:
        interpretation = f"{config_a_name} is significantly BETTER than {config_b_name} (p={p_value:.4f})"
    elif is_significant and difference < 0:
        interpretation = f"{config_b_name} is significantly BETTER than {config_a_name} (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference between configs (p={p_value:.4f}) - choose based on other criteria"

    return ConfigComparisonResult(
        config_a=config_a_name,
        config_b=config_b_name,
        metric=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=difference,
        t_statistic=t_stat,
        p_value=p_value,
        is_significant=is_significant,
        confidence=confidence,
        interpretation=interpretation,
    )


def compare_top_configs(
    all_results: List[Dict],
    metric: str = 'ev_mean',
    top_n: int = 5,
) -> None:
    """
    Compare top N configs and print significance matrix.

    Args:
        all_results: List of config result dicts (from checkpoint DB)
        metric: Metric to compare by
        top_n: Number of top configs to compare
    """
    print("\n" + "=" * 70)
    print("  CONFIG COMPARISON MATRIX")
    print("  (Statistical Significance of Differences)")
    print("=" * 70)

    sorted_configs = sorted(all_results, key=lambda x: x.get(metric, -999), reverse=True)[:top_n]

    print(f"\n  Comparing top {len(sorted_configs)} configs by {metric}:")
    for i, config in enumerate(sorted_configs, 1):
        print(f"    #{i}  {config.get('config_id', '?'):25s}  {metric}={config.get(metric, 0):+.2f}")

    if len(sorted_configs) < 2:
        return

    best = sorted_configs[0]
    print(f"\n  Is #{1} ({best.get('config_id')}) significantly better?")
    print(f"  " + "─" * 50)

    for i, other in enumerate(sorted_configs[1:], 2):
        ev_diff = best.get(metric, 0) - other.get(metric, 0)
        # Without fold-level data we can only report the difference
        print(f"    vs #{i} ({other.get('config_id')}): Δ{metric}={ev_diff:+.3f}")

    print(f"\n  NOTE: For full paired t-test, provide fold-level results.")


# =============================================================================
# LOSS ANALYSIS HELPERS (used by loss_analysis.py)
# =============================================================================

def compare_feature_wins_losses(
    wins: pd.DataFrame,
    losses: pd.DataFrame,
    features_to_check: Optional[List[str]] = None,
) -> List[FeatureComparison]:
    """
    Compare feature distributions between winning and losing trades.

    Args:
        wins: DataFrame of winning trade rows
        losses: DataFrame of losing trade rows
        features_to_check: Feature columns to compare (None = auto-detect)

    Returns:
        List of FeatureComparison sorted by significance and effect size
    """
    if features_to_check is None:
        features_to_check = [
            # H4 features
            'rsi_value', 'bb_position', 'trend_strength',
            'atr_pct', 'volume_ratio', 'is_high_activity',
            'consecutive_bullish', 'exhaustion_score',
            'adx', 'rsi_slope_3', 'bb_position_slope_3',
            # D1 features
            'd1_rsi_value', 'd1_bb_position', 'd1_trend_strength',
            'd1_trend_direction', 'd1_atr_percentile', 'd1_consecutive_bullish',
            'd1_rsi_slope_3', 'd1_is_trending', 'd1_adx',
            # MTF features
            'mtf_confluence_score', 'mtf_rsi_aligned', 'mtf_bb_aligned',
            'mtf_trend_aligned', 'd1_supports_short', 'd1_opposes_short',
            'mtf_strong_short_setup', 'h4_vs_d1_rsi',
            # Other
            'model_probability',
        ]

    comparisons: List[FeatureComparison] = []

    for feature in features_to_check:
        if feature not in wins.columns or feature not in losses.columns:
            continue

        wins_vals = wins[feature].dropna()
        losses_vals = losses[feature].dropna()

        if len(wins_vals) < 30 or len(losses_vals) < 30:
            continue

        wins_mean = float(wins_vals.mean())
        wins_std = float(wins_vals.std())
        losses_mean = float(losses_vals.mean())
        losses_std = float(losses_vals.std())

        t_stat, p_value = stats.ttest_ind(wins_vals.values, losses_vals.values, equal_var=False)

        pooled_std = float(np.sqrt((wins_std ** 2 + losses_std ** 2) / 2))
        effect_size = abs(wins_mean - losses_mean) / pooled_std if pooled_std > 0 else 0.0

        pct_diff = (losses_mean - wins_mean) / abs(wins_mean) * 100 if wins_mean != 0 else 0.0

        comparisons.append(FeatureComparison(
            feature_name=feature,
            wins_mean=wins_mean,
            wins_std=wins_std,
            losses_mean=losses_mean,
            losses_std=losses_std,
            difference=losses_mean - wins_mean,
            pct_difference=pct_diff,
            t_statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            effect_size=effect_size,
            direction="higher_in_losses" if losses_mean > wins_mean else "higher_in_wins",
        ))

    comparisons.sort(key=lambda x: (not x.is_significant, -abs(x.effect_size)))
    return comparisons


def print_feature_comparison_report(comparisons: List[FeatureComparison], top_n: int = 20) -> None:
    """Print feature comparison report (wins vs losses)."""
    print("\n" + "=" * 80)
    print("  FEATURE COMPARISON: WINS vs LOSSES")
    print("=" * 80)

    sig = [c for c in comparisons if c.is_significant]
    not_sig = [c for c in comparisons if not c.is_significant]

    print(f"\n  Total features compared: {len(comparisons)}")
    print(f"  Significant differences: {len(sig)}")
    print(f"  No significant diff:     {len(not_sig)}")

    print(f"\n  {'Feature':<32} {'Wins μ':>9} {'Loss μ':>9} {'Δ%':>7} {'P-Value':>10} {'Stars':>5} {'Effect':>8}")
    print(f"  " + "─" * 85)

    for c in comparisons[:top_n]:
        stars = "***" if c.p_value < 0.001 else "**" if c.p_value < 0.01 else "*" if c.p_value < 0.05 else ""
        direction_arrow = "▲" if c.direction == "higher_in_losses" else "▼"
        print(
            f"  {c.feature_name:<32} {c.wins_mean:>9.4f} {c.losses_mean:>9.4f} "
            f"{c.pct_difference:>+6.1f}% {c.p_value:>10.4f} {stars:>5} {c.effect_size:>7.3f}{direction_arrow}"
        )

    if sig:
        print(f"\n  KEY FINDINGS (significant differences):")
        for c in sig[:10]:
            if c.direction == "higher_in_losses":
                msg = f"    {c.feature_name}: HIGHER in losses (+{abs(c.pct_difference):.1f}%) → potential filter candidate"
            else:
                msg = f"    {c.feature_name}: LOWER in losses ({c.pct_difference:.1f}%) → may help when high"
            print(msg)


# =============================================================================
# INTEGRATION HELPER: apply pre-filter in pipeline
# =============================================================================

def apply_statistical_prefilter(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str = 'label',
    p_threshold: float = 0.05,
    min_effect_size: float = 0.15,
    verbose: bool = True,
) -> Tuple[List[str], List[FeatureSignificance]]:
    """
    Convenience function to apply statistical pre-filter in the pipeline.

    Call this AFTER feature engineering and BEFORE RFE.

    Args:
        df: DataFrame with features + label
        feature_columns: All feature column names (~296)
        label_column: Binary label column name
        p_threshold: P-value cutoff
        min_effect_size: Cohen's d cutoff
        verbose: Print report

    Returns:
        (kept_features, all_results)
        kept_features is the filtered list (~80-120) to pass to RFE
    """
    sf = StatisticalFeatureFilter(
        p_threshold=p_threshold,
        min_effect_size=min_effect_size,
    )

    kept, dropped, results = sf.filter_features(df, feature_columns, label_column)

    if verbose:
        sf.print_report()

    return kept, results
