"""
Loss Analysis Module - Stella Alpha

Analyzes patterns in losing trades to understand WHY trades fail
and surface actionable insights for filter generation.

ANALYSES PERFORMED:
1. Feature comparisons (wins vs losses) — t-test + Cohen's d
2. Session analysis — which sessions have lower win rates
3. Confidence analysis — how model probability correlates with outcome
4. D1 alignment analysis — impact of D1 trend on win rate
5. MTF confluence analysis — impact of multi-timeframe agreement
6. Optimal confidence threshold — find the probability cutoff that maximises net EV

Usage:
    from loss_analysis import LossAnalyzer
    analyzer = LossAnalyzer("artifacts/trades_stella_alpha.db")
    report = analyzer.analyze(config_id="cfg_0042")   # or None for all
    analyzer.print_report(report)
    analyzer.save_report(report, "artifacts/loss_analysis_stella_alpha.json")
"""

import sqlite3
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureComparison:
    """Statistical comparison of a single feature between wins and losses."""
    feature_name: str
    wins_mean: float
    wins_std: float
    losses_mean: float
    losses_std: float
    difference: float           # losses_mean - wins_mean
    pct_difference: float       # percentage difference relative to wins_mean
    t_statistic: float
    p_value: float
    is_significant: bool        # p < 0.05
    effect_size: float          # Cohen's d (absolute)
    direction: str              # "higher_in_losses" or "higher_in_wins"


@dataclass
class SessionAnalysis:
    """Win/loss breakdown by trading session."""
    session: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    loss_rate: float
    is_problematic: bool        # win rate significantly below overall average


@dataclass
class ConfidenceAnalysis:
    """Win/loss breakdown by model confidence bucket."""
    confidence_bucket: str      # e.g. "0.50-0.55"
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_confidence: float


@dataclass
class D1AlignmentAnalysis:
    """Impact of D1 trend alignment on outcomes."""
    category: str               # e.g. "D1 Supports Short", "D1 Opposes Short"
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    pct_of_all_trades: float


@dataclass
class LossAnalysisReport:
    """Complete loss analysis output for one config (or all configs)."""
    config_id: str
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float

    # Feature analysis
    feature_comparisons: List[FeatureComparison]
    significant_features: List[FeatureComparison]

    # Broken-down analyses
    session_analysis: List[SessionAnalysis]
    confidence_analysis: List[ConfidenceAnalysis]
    d1_alignment_analysis: List[D1AlignmentAnalysis]

    # Aggregated impact summaries (used by filter engine)
    d1_alignment_impact: Dict[str, Any]
    mtf_confluence_impact: Dict[str, Any]

    # Key findings
    top_loss_predictors: List[str]         # feature names, sorted by effect size
    problematic_sessions: List[str]
    optimal_confidence_threshold: float

    # Meta
    generated_at: str = ""


# =============================================================================
# LOSS ANALYZER
# =============================================================================

class LossAnalyzer:
    """
    Loads trades from SQLite and runs statistical analyses.
    """

    # Features to compare between wins and losses
    _COMPARE_FEATURES = [
        # H4
        "h4_rsi_value", "h4_bb_position", "h4_trend_strength",
        "h4_atr_pct", "h4_volume_ratio", "h4_hour",
        "h4_consecutive_bullish", "h4_exhaustion_score",
        "h4_macd_histogram", "h4_adx",
        # D1
        "d1_rsi_value", "d1_bb_position", "d1_trend_strength",
        "d1_trend_direction", "d1_atr_percentile", "d1_consecutive_bullish",
        # Cross-TF
        "mtf_confluence_score",
        # Model
        "model_probability",
    ]

    _SESSION_COLS = {
        "Asian":   "h4_is_asian_session",
        "London":  "h4_is_london_session",
        "New York":"h4_is_ny_session",
        "Overlap": "h4_is_overlap_session",
    }

    def __init__(self, db_path: str = "artifacts/trades_stella_alpha.db"):
        self.db_path = Path(db_path)
        self.trades_df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────

    def load_trades(self, config_id: Optional[str] = None) -> pd.DataFrame:
        """Load trades from the database into a DataFrame."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Trade database not found: {self.db_path}")

        conn = sqlite3.connect(str(self.db_path))
        if config_id:
            query = "SELECT * FROM trades WHERE config_id = ?"
            df = pd.read_sql_query(query, conn, params=(config_id,))
        else:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()

        self.trades_df = df
        return df

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        config_id: Optional[str] = None,
        min_samples: int = 30,
    ) -> LossAnalysisReport:
        """
        Run the complete loss analysis.

        Args:
            config_id: Filter to a specific config, or None for all trades combined.
            min_samples: Minimum wins/losses required to compute a comparison.

        Returns:
            LossAnalysisReport with all analyses.
        """
        if self.trades_df is None:
            self.load_trades(config_id)

        df = self.trades_df
        if config_id:
            df = df[df["config_id"] == config_id].copy()

        if len(df) == 0:
            raise ValueError(f"No trades found for config_id={config_id!r}")

        wins   = df[df["outcome"] == "WIN"]
        losses = df[df["outcome"] == "LOSS"]

        # 1. Feature comparisons
        comparisons   = self._compare_features(wins, losses, min_samples)
        significant   = [c for c in comparisons if c.is_significant]

        # 2. Session analysis
        session_analysis     = self._analyze_sessions(df)
        problematic_sessions = [s.session for s in session_analysis if s.is_problematic]

        # 3. Confidence analysis
        confidence_analysis = self._analyze_confidence(df)
        optimal_threshold   = self._find_optimal_confidence(df)

        # 4. D1 alignment analysis
        d1_analysis      = self._analyze_d1_alignment(df)
        d1_impact_summary = self._d1_impact_summary(df)

        # 5. MTF confluence analysis
        mtf_impact = self._analyze_mtf_confluence(df)

        import datetime as _dt
        return LossAnalysisReport(
            config_id=config_id or "ALL",
            total_trades=len(df),
            total_wins=len(wins),
            total_losses=len(losses),
            win_rate=len(wins) / len(df) if len(df) > 0 else 0.0,
            feature_comparisons=comparisons,
            significant_features=significant,
            session_analysis=session_analysis,
            confidence_analysis=confidence_analysis,
            d1_alignment_analysis=d1_analysis,
            d1_alignment_impact=d1_impact_summary,
            mtf_confluence_impact=mtf_impact,
            top_loss_predictors=[c.feature_name for c in significant[:10]],
            problematic_sessions=problematic_sessions,
            optimal_confidence_threshold=optimal_threshold,
            generated_at=_dt.datetime.utcnow().isoformat(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Analysis methods
    # ──────────────────────────────────────────────────────────────────────

    def _compare_features(
        self,
        wins: pd.DataFrame,
        losses: pd.DataFrame,
        min_samples: int,
    ) -> List[FeatureComparison]:
        comparisons: List[FeatureComparison] = []

        for feature in self._COMPARE_FEATURES:
            if feature not in wins.columns:
                continue

            w_vals = wins[feature].dropna()
            l_vals = losses[feature].dropna()

            if len(w_vals) < min_samples or len(l_vals) < min_samples:
                continue

            w_mean, w_std = float(w_vals.mean()), float(w_vals.std())
            l_mean, l_std = float(l_vals.mean()), float(l_vals.std())

            # Welch's t-test (doesn't assume equal variance)
            t_stat, p_value = scipy_stats.ttest_ind(w_vals, l_vals, equal_var=False)

            # Cohen's d
            pooled_std = np.sqrt((w_std ** 2 + l_std ** 2) / 2.0)
            d = (l_mean - w_mean) / pooled_std if pooled_std > 0 else 0.0

            pct_diff = (
                (l_mean - w_mean) / abs(w_mean) * 100.0 if w_mean != 0 else 0.0
            )

            comparisons.append(FeatureComparison(
                feature_name=feature,
                wins_mean=w_mean,
                wins_std=w_std,
                losses_mean=l_mean,
                losses_std=l_std,
                difference=l_mean - w_mean,
                pct_difference=float(pct_diff),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=bool(p_value < 0.05),
                effect_size=float(abs(d)),
                direction="higher_in_losses" if l_mean > w_mean else "higher_in_wins",
            ))

        # Sort: significant first, then by absolute effect size descending
        comparisons.sort(key=lambda c: (not c.is_significant, -c.effect_size))
        return comparisons

    def _analyze_sessions(self, df: pd.DataFrame) -> List[SessionAnalysis]:
        overall_wr = (df["outcome"] == "WIN").mean() if len(df) > 0 else 0.5
        # Flag a session as problematic if win rate is >10 pp below average
        problem_threshold = overall_wr - 0.10

        results: List[SessionAnalysis] = []
        for session_name, col in self._SESSION_COLS.items():
            if col not in df.columns:
                continue
            subset = df[df[col] == 1]
            if len(subset) < 10:
                continue
            wins   = int((subset["outcome"] == "WIN").sum())
            losses = int((subset["outcome"] == "LOSS").sum())
            total  = wins + losses
            wr     = wins / total if total > 0 else 0.0
            results.append(SessionAnalysis(
                session=session_name,
                total_trades=total,
                wins=wins,
                losses=losses,
                win_rate=wr,
                loss_rate=1.0 - wr,
                is_problematic=wr < problem_threshold,
            ))

        # Also add an "Unknown/No Session" bucket
        if any(c in df.columns for c in self._SESSION_COLS.values()):
            session_cols = [c for c in self._SESSION_COLS.values() if c in df.columns]
            no_session = df[(df[session_cols] == 0).all(axis=1)]
            if len(no_session) >= 10:
                wins   = int((no_session["outcome"] == "WIN").sum())
                losses = int((no_session["outcome"] == "LOSS").sum())
                total  = wins + losses
                wr     = wins / total if total > 0 else 0.0
                results.append(SessionAnalysis(
                    session="Off-Hours",
                    total_trades=total,
                    wins=wins,
                    losses=losses,
                    win_rate=wr,
                    loss_rate=1.0 - wr,
                    is_problematic=wr < problem_threshold,
                ))

        results.sort(key=lambda s: s.win_rate)
        return results

    def _analyze_confidence(self, df: pd.DataFrame) -> List[ConfidenceAnalysis]:
        if "model_probability" not in df.columns:
            return []

        buckets = [
            (0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
            (0.65, 0.70), (0.70, 0.80), (0.80, 1.01),
        ]
        results: List[ConfidenceAnalysis] = []

        for lo, hi in buckets:
            subset = df[
                (df["model_probability"] >= lo) &
                (df["model_probability"] < hi)
            ]
            if len(subset) < 5:
                continue
            wins   = int((subset["outcome"] == "WIN").sum())
            losses = int((subset["outcome"] == "LOSS").sum())
            total  = wins + losses
            wr     = wins / total if total > 0 else 0.0
            results.append(ConfidenceAnalysis(
                confidence_bucket=f"{lo:.2f}-{hi:.2f}",
                total_trades=total,
                wins=wins,
                losses=losses,
                win_rate=wr,
                avg_confidence=float(subset["model_probability"].mean()),
            ))

        return results

    def _find_optimal_confidence(self, df: pd.DataFrame) -> float:
        """
        Find the probability threshold above which win rate is maximised
        while keeping at least 10% of trades.

        Simple grid search over thresholds [0.50 … 0.85].
        """
        if "model_probability" not in df.columns or len(df) < 30:
            return 0.50

        min_trades = max(int(len(df) * 0.10), 10)
        best_wr    = 0.0
        best_thresh = 0.50

        for t in np.arange(0.50, 0.86, 0.01):
            subset = df[df["model_probability"] >= t]
            if len(subset) < min_trades:
                break
            wr = (subset["outcome"] == "WIN").mean()
            if wr > best_wr:
                best_wr    = wr
                best_thresh = float(t)

        return round(best_thresh, 2)

    def _analyze_d1_alignment(self, df: pd.DataFrame) -> List[D1AlignmentAnalysis]:
        total = len(df)
        if total == 0:
            return []

        categories: List[Tuple[str, pd.Series]] = []

        if "d1_supports_short" in df.columns:
            mask = df["d1_supports_short"] == 1
            categories.append(("D1 Supports Short", mask))
            categories.append(("D1 Does NOT Support Short", ~mask))

        if "d1_opposes_short" in df.columns:
            mask = df["d1_opposes_short"] == 1
            categories.append(("D1 Opposes Short", mask))

        if "d1_is_trending_up" in df.columns:
            categories.append(("D1 Trending Up", df["d1_is_trending_up"] == 1))
            categories.append(("D1 NOT Trending Up", df["d1_is_trending_up"] == 0))

        results: List[D1AlignmentAnalysis] = []
        for cat_name, mask in categories:
            subset = df[mask]
            if len(subset) < 10:
                continue
            wins   = int((subset["outcome"] == "WIN").sum())
            losses = int((subset["outcome"] == "LOSS").sum())
            t      = wins + losses
            wr     = wins / t if t > 0 else 0.0
            results.append(D1AlignmentAnalysis(
                category=cat_name,
                total_trades=t,
                wins=wins,
                losses=losses,
                win_rate=wr,
                pct_of_all_trades=100.0 * t / total,
            ))

        results.sort(key=lambda r: r.win_rate, reverse=True)
        return results

    def _d1_impact_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summary dict consumed by FilterRecommendationEngine."""
        summary: Dict[str, Any] = {}
        for col in ("d1_supports_short", "d1_opposes_short", "d1_is_trending_up"):
            if col not in df.columns:
                continue
            for val, label in [(1, "when_true"), (0, "when_false")]:
                subset = df[df[col] == val]
                if len(subset) < 5:
                    continue
                wins = (subset["outcome"] == "WIN").sum()
                t    = len(subset)
                summary.setdefault(col, {})[label] = {
                    "total": t,
                    "wins": int(wins),
                    "win_rate": float(wins / t),
                }
        return summary

    def _analyze_mtf_confluence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Breakdown of win rates by MTF confluence level."""
        if "mtf_confluence_score" not in df.columns:
            return {}

        buckets = {
            "high_confluence":    df["mtf_confluence_score"] >= 0.75,
            "medium_confluence":  (df["mtf_confluence_score"] >= 0.50) & (df["mtf_confluence_score"] < 0.75),
            "low_confluence":     df["mtf_confluence_score"] < 0.50,
        }

        result: Dict[str, Any] = {}
        for label, mask in buckets.items():
            subset = df[mask]
            if len(subset) < 5:
                continue
            wins = (subset["outcome"] == "WIN").sum()
            t    = len(subset)
            result[label] = {
                "total": t,
                "wins": int(wins),
                "win_rate": float(wins / t),
                "avg_score": float(subset["mtf_confluence_score"].mean()),
            }
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Output helpers
    # ──────────────────────────────────────────────────────────────────────

    def print_report(self, report: LossAnalysisReport) -> None:
        """Print a formatted report to stdout."""
        line = "=" * 70
        dline = "─" * 50

        print(f"\n{line}")
        print(f"  LOSS ANALYSIS REPORT  |  Config: {report.config_id}")
        print(f"{line}")
        print(f"  Total trades : {report.total_trades:,}")
        print(f"  Wins         : {report.total_wins:,}  ({report.win_rate*100:.1f}%)")
        print(f"  Losses       : {report.total_losses:,}  ({(1-report.win_rate)*100:.1f}%)")

        # ── Feature comparisons ─────────────────────────────────────────
        print(f"\n  SIGNIFICANT FEATURE DIFFERENCES  (wins vs losses)")
        print(f"  {dline}")
        if report.significant_features:
            header = f"  {'Feature':<32} {'Wins':>8} {'Losses':>8} {'Δ%':>7} {'p':>7} {'d':>6}"
            print(header)
            for c in report.significant_features[:15]:
                arrow = "▲" if c.direction == "higher_in_losses" else "▼"
                print(
                    f"  {c.feature_name:<32} "
                    f"{c.wins_mean:>8.3f} "
                    f"{c.losses_mean:>8.3f} "
                    f"{c.pct_difference:>+7.1f}% "
                    f"{c.p_value:>7.4f} "
                    f"{c.effect_size:>6.3f} {arrow}"
                )
        else:
            print("  No significant feature differences found.")

        # ── Session analysis ────────────────────────────────────────────
        print(f"\n  SESSION ANALYSIS")
        print(f"  {dline}")
        if report.session_analysis:
            for s in report.session_analysis:
                flag = " ⚠️ PROBLEMATIC" if s.is_problematic else ""
                print(
                    f"  {s.session:<12} "
                    f"Win: {s.win_rate*100:5.1f}%  "
                    f"({s.total_trades:,} trades){flag}"
                )
        else:
            print("  No session data available.")

        # ── Confidence analysis ─────────────────────────────────────────
        print(f"\n  CONFIDENCE ANALYSIS")
        print(f"  {dline}")
        if report.confidence_analysis:
            for c in report.confidence_analysis:
                print(
                    f"  Prob {c.confidence_bucket}  "
                    f"Win: {c.win_rate*100:5.1f}%  "
                    f"({c.total_trades:,} trades)"
                )
        else:
            print("  No confidence data available.")

        # ── D1 alignment ────────────────────────────────────────────────
        print(f"\n  D1 ALIGNMENT IMPACT")
        print(f"  {dline}")
        if report.d1_alignment_analysis:
            for d in report.d1_alignment_analysis:
                print(
                    f"  {d.category:<30} "
                    f"Win: {d.win_rate*100:5.1f}%  "
                    f"({d.total_trades:,} trades, {d.pct_of_all_trades:.0f}%)"
                )
        else:
            print("  No D1 alignment data available.")

        # ── MTF confluence ───────────────────────────────────────────────
        print(f"\n  MTF CONFLUENCE IMPACT")
        print(f"  {dline}")
        mtf = report.mtf_confluence_impact
        if mtf:
            for label, data in mtf.items():
                print(
                    f"  {label:<22} "
                    f"Win: {data['win_rate']*100:5.1f}%  "
                    f"({data['total']:,} trades)"
                )
        else:
            print("  No MTF confluence data available.")

        # ── Key findings ─────────────────────────────────────────────────
        print(f"\n  KEY FINDINGS")
        print(f"  {dline}")
        if report.top_loss_predictors:
            print(f"  Top loss predictors : {', '.join(report.top_loss_predictors[:5])}")
        if report.problematic_sessions:
            print(f"  Problematic sessions: {', '.join(report.problematic_sessions)}")
        print(f"  Optimal confidence threshold: {report.optimal_confidence_threshold:.2f}")
        print(f"\n{line}\n")

    def save_report(
        self,
        report: LossAnalysisReport,
        output_path: str,
    ) -> None:
        """Serialize report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _ser(obj: Any) -> Any:
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=_ser)

        print(f"Report saved → {path}")
