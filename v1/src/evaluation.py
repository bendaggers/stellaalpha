"""
Evaluation module - Stella Alpha.

This module handles:
1. Computing classification metrics
2. Computing trading metrics (Expected Value)
3. Threshold optimization
4. Fold aggregation

CHANGES FROM V4:
────────────────
- check_acceptance_criteria() is now R:R-aware.
  Instead of a fixed min_precision floor (e.g. 0.55), it checks whether
  precision > min_winrate_required for the given TP/SL ratio.
  This correctly handles high R:R configs (TP=100, SL=30) that only need
  ~23% win rate to be profitable, not 55%.

- Added compute_edge_above_breakeven() helper.
- Added compute_min_winrate() helper.
- All other functions (compute_metrics, optimize_threshold,
  aggregate_fold_metrics, etc.) are unchanged from V4.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# =============================================================================
# DATA CLASSES  (unchanged from V4)
# =============================================================================

@dataclass
class MetricsBundle:
    """Complete metrics for a model evaluation."""
    precision: float
    recall: float
    f1_score: float
    auc_pr: float
    roc_auc: Optional[float]
    expected_value: float
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_pr': self.auc_pr,
            'roc_auc': self.roc_auc,
            'expected_value': self.expected_value,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_rate,
            'threshold': self.threshold,
        }


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    optimal_threshold: float
    expected_value: float
    precision: float
    recall: float
    f1_score: float
    trade_count: int
    win_count: int
    loss_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimal_threshold': self.optimal_threshold,
            'expected_value': self.expected_value,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
        }


@dataclass
class RegimeMetrics:
    """Metrics broken down by regime."""
    regime: str
    precision: float
    recall: float
    expected_value: float
    trade_count: int
    win_count: int
    loss_count: int
    pct_of_total: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime,
            'precision': self.precision,
            'recall': self.recall,
            'expected_value': self.expected_value,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'pct_of_total': self.pct_of_total,
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics across folds."""
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    auc_pr_mean: float
    auc_pr_std: float
    ev_mean: float
    ev_std: float
    total_trades: int
    n_folds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'precision': {'mean': self.precision_mean, 'std': self.precision_std},
            'recall': {'mean': self.recall_mean, 'std': self.recall_std},
            'f1_score': {'mean': self.f1_mean, 'std': self.f1_std},
            'auc_pr': {'mean': self.auc_pr_mean, 'std': self.auc_pr_std},
            'expected_value': {'mean': self.ev_mean, 'std': self.ev_std},
            'total_trades': self.total_trades,
            'n_folds': self.n_folds,
        }


# =============================================================================
# METRICS COMPUTATION  (unchanged from V4)
# =============================================================================

def compute_expected_value(
    precision: float,
    tp_pips: float,
    sl_pips: float,
) -> float:
    """
    Compute expected value per trade.

    EV = (win_rate * avg_win) - (loss_rate * avg_loss)
    """
    win_rate = precision
    loss_rate = 1.0 - precision
    return (win_rate * tp_pips) - (loss_rate * sl_pips)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    tp_pips: float,
    sl_pips: float,
    threshold: float,
) -> MetricsBundle:
    """Compute all metrics for a prediction set."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return MetricsBundle(
            precision=0.0, recall=0.0, f1_score=0.0,
            auc_pr=0.0, roc_auc=None,
            expected_value=0.0, trade_count=0,
            win_count=0, loss_count=0,
            win_rate=0.0, threshold=threshold,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc_pr = average_precision_score(y_true, y_proba)
        except Exception:
            auc_pr = 0.0

        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc_auc = None

    trade_count = int(y_pred.sum())
    win_count = int(((y_pred == 1) & (y_true == 1)).sum())
    loss_count = int(((y_pred == 1) & (y_true == 0)).sum())
    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    ev = compute_expected_value(float(prec), tp_pips, sl_pips)

    return MetricsBundle(
        precision=float(prec),
        recall=float(rec),
        f1_score=float(f1),
        auc_pr=float(auc_pr),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        expected_value=float(ev),
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=float(win_rate),
        threshold=threshold,
    )


# =============================================================================
# THRESHOLD OPTIMIZATION  (unchanged from V4)
# =============================================================================

def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    tp_pips: float,
    sl_pips: float,
    min_threshold: float = 0.40,
    max_threshold: float = 0.80,
    step: float = 0.05,
    min_trades: int = 20,
) -> ThresholdResult:
    """
    Optimize probability threshold to maximise Expected Value.
    """
    best_result = None
    best_ev = float('-inf')

    thresholds = np.arange(min_threshold, max_threshold + step / 2, step)

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        trade_count = int(y_pred.sum())

        if trade_count < min_trades:
            continue

        win_count = int(((y_pred == 1) & (y_true == 1)).sum())
        loss_count = int(((y_pred == 1) & (y_true == 0)).sum())
        precision = win_count / trade_count if trade_count > 0 else 0.0
        ev = compute_expected_value(precision, tp_pips, sl_pips)

        if ev > best_ev:
            best_ev = ev

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

            best_result = ThresholdResult(
                optimal_threshold=float(threshold),
                expected_value=float(ev),
                precision=float(precision),
                recall=float(rec),
                f1_score=float(f1),
                trade_count=trade_count,
                win_count=win_count,
                loss_count=loss_count,
            )

    if best_result is None:
        return ThresholdResult(
            optimal_threshold=0.50,
            expected_value=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            trade_count=0,
            win_count=0,
            loss_count=0,
        )

    return best_result


# =============================================================================
# AGGREGATION  (unchanged from V4)
# =============================================================================

def aggregate_fold_metrics(fold_metrics: List[MetricsBundle]) -> AggregateMetrics:
    """Aggregate metrics across folds."""
    if not fold_metrics:
        return AggregateMetrics(
            precision_mean=0.0, precision_std=0.0,
            recall_mean=0.0, recall_std=0.0,
            f1_mean=0.0, f1_std=0.0,
            auc_pr_mean=0.0, auc_pr_std=0.0,
            ev_mean=0.0, ev_std=0.0,
            total_trades=0, n_folds=0,
        )

    precisions = [m.precision for m in fold_metrics]
    recalls = [m.recall for m in fold_metrics]
    f1s = [m.f1_score for m in fold_metrics]
    auc_prs = [m.auc_pr for m in fold_metrics]
    evs = [m.expected_value for m in fold_metrics]
    total_trades = sum(m.trade_count for m in fold_metrics)

    return AggregateMetrics(
        precision_mean=float(np.mean(precisions)),
        precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)),
        recall_std=float(np.std(recalls)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
        auc_pr_mean=float(np.mean(auc_prs)),
        auc_pr_std=float(np.std(auc_prs)),
        ev_mean=float(np.mean(evs)),
        ev_std=float(np.std(evs)),
        total_trades=total_trades,
        n_folds=len(fold_metrics),
    )


def get_consensus_threshold(
    fold_results: List[ThresholdResult],
    method: str = 'median',
) -> float:
    """Get consensus threshold from multiple folds."""
    if not fold_results:
        return 0.5
    thresholds = [r.optimal_threshold for r in fold_results]
    if method == 'mean':
        return float(np.mean(thresholds))
    return float(np.median(thresholds))


def compute_regime_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime: np.ndarray,
    tp_pips: float,
    sl_pips: float,
) -> Dict[str, RegimeMetrics]:
    """Compute metrics broken down by regime."""
    results = {}
    total_trades = int(y_pred.sum())

    for r in np.unique(regime):
        mask = regime == r
        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]
        trade_count = int(y_pred_r.sum())
        if trade_count == 0:
            continue

        win_count = int(((y_pred_r == 1) & (y_true_r == 1)).sum())
        loss_count = int(((y_pred_r == 1) & (y_true_r == 0)).sum())
        precision = win_count / trade_count if trade_count > 0 else 0.0

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            rec = recall_score(y_true_r, y_pred_r, zero_division=0)

        ev = compute_expected_value(precision, tp_pips, sl_pips)

        results[str(r)] = RegimeMetrics(
            regime=str(r),
            precision=float(precision),
            recall=float(rec),
            expected_value=float(ev),
            trade_count=trade_count,
            win_count=win_count,
            loss_count=loss_count,
            pct_of_total=100.0 * trade_count / total_trades if total_trades > 0 else 0.0,
        )

    return results


# =============================================================================
# STELLA ALPHA: R:R-AWARE ACCEPTANCE CRITERIA  (MODIFIED from V4)
# =============================================================================

def compute_min_winrate(tp_pips: float, sl_pips: float) -> float:
    """
    Compute the minimum win rate needed to break even for a given R:R.

    Formula: min_winrate = SL / (TP + SL)

    Examples:
        TP=30, SL=30  → 50.0% break-even win rate
        TP=50, SL=30  → 37.5% break-even win rate
        TP=100, SL=30 → 23.1% break-even win rate
        TP=150, SL=30 → 16.7% break-even win rate
    """
    return sl_pips / (tp_pips + sl_pips)


def compute_edge_above_breakeven(precision: float, tp_pips: float, sl_pips: float) -> float:
    """
    Compute the edge above breakeven win rate.

    Positive = model has edge above what's needed to be profitable.
    This is more meaningful than raw precision for comparing across R:R ratios.
    """
    min_wr = compute_min_winrate(tp_pips, sl_pips)
    return precision - min_wr


def check_acceptance_criteria(
    aggregate_metrics: AggregateMetrics,
    tp_pips: float,
    sl_pips: float,
    min_edge_above_breakeven: float = 0.05,
    min_trades_per_fold: int = 30,
    min_expected_value: float = 0.0,
    max_precision_cv: float = 0.30,
) -> Tuple[bool, List[str]]:
    """
    Check if model meets acceptance criteria.

    STELLA ALPHA CHANGE from V4:
    ────────────────────────────
    V4 used a fixed min_precision = 0.55 regardless of TP/SL.
    This incorrectly rejected high R:R configs (TP=100, SL=30) that are
    profitable at 25% win rate but need 55% to pass the old check.

    Stella Alpha instead checks:
      precision > min_winrate_required + min_edge_above_breakeven

    This means:
      TP=50,  SL=30 → needs precision > 37.5% + 5% = 42.5%
      TP=100, SL=30 → needs precision > 23.1% + 5% = 28.1%
      TP=150, SL=30 → needs precision > 16.7% + 5% = 21.7%

    A 5% edge above breakeven ensures we have a real edge, not just noise.

    Args:
        aggregate_metrics: Aggregated fold metrics
        tp_pips: Take profit in pips (used to compute R:R)
        sl_pips: Stop loss in pips (used to compute R:R)
        min_edge_above_breakeven: Minimum precision edge above break-even (default 5%)
        min_trades_per_fold: Minimum average trades per fold
        min_expected_value: Minimum expected value in pips (default 0 = profitable)
        max_precision_cv: Maximum precision coefficient of variation (stability)

    Returns:
        (passed: bool, rejection_reasons: List[str])
    """
    passed = True
    reasons = []

    min_wr = compute_min_winrate(tp_pips, sl_pips)
    min_precision_required = min_wr + min_edge_above_breakeven

    # ── 1. Precision must exceed breakeven + edge ─────────────────────────
    if aggregate_metrics.precision_mean <= min_precision_required:
        passed = False
        reasons.append(
            f"precision {aggregate_metrics.precision_mean:.3f} <= "
            f"min_required {min_precision_required:.3f} "
            f"(breakeven={min_wr:.3f} + edge={min_edge_above_breakeven:.3f})"
        )

    # ── 2. Expected Value must be positive ───────────────────────────────
    if aggregate_metrics.ev_mean <= min_expected_value:
        passed = False
        reasons.append(
            f"EV {aggregate_metrics.ev_mean:.2f} <= {min_expected_value:.2f} pips"
        )

    # ── 3. Minimum trade volume (statistical significance) ────────────────
    avg_trades = aggregate_metrics.total_trades / max(aggregate_metrics.n_folds, 1)
    if avg_trades < min_trades_per_fold:
        passed = False
        reasons.append(
            f"avg_trades_per_fold {avg_trades:.0f} < {min_trades_per_fold}"
        )

    # ── 4. Precision stability across folds ──────────────────────────────
    if aggregate_metrics.precision_mean > 0:
        cv = aggregate_metrics.precision_std / aggregate_metrics.precision_mean
        if cv > max_precision_cv:
            passed = False
            reasons.append(
                f"precision_cv {cv:.3f} > {max_precision_cv:.3f} (unstable across folds)"
            )

    return passed, reasons


def get_rejection_reasons(
    aggregate_metrics: AggregateMetrics,
    tp_pips: float,
    sl_pips: float,
    min_edge_above_breakeven: float = 0.05,
    min_trades_per_fold: int = 30,
    min_expected_value: float = 0.0,
    max_precision_cv: float = 0.30,
) -> List[str]:
    """
    Get rejection reasons without returning pass/fail.
    Convenience wrapper around check_acceptance_criteria.
    """
    _, reasons = check_acceptance_criteria(
        aggregate_metrics=aggregate_metrics,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        min_edge_above_breakeven=min_edge_above_breakeven,
        min_trades_per_fold=min_trades_per_fold,
        min_expected_value=min_expected_value,
        max_precision_cv=max_precision_cv,
    )
    return reasons
