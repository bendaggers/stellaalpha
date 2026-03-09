"""
Filter Recommendation Engine - Stella Alpha

Generates actionable filter rules from trade history by:
1. Building candidate filters (session, confidence, D1 alignment, RSI, MTF, BB)
2. Simulating each filter's impact on historical P&L
3. Ranking by net pip improvement
4. Testing combinations of top filters

EXAMPLE OUTPUT:
  RANK 1 ─ "Skip when D1 opposes short"
           Removes 312 losses, misses 41 wins → Net +14,820 pips (+23.4%)
           STRONG RECOMMENDATION

  RANK 2 ─ "Require confidence >= 0.60"
           Removes 198 losses, misses 87 wins → Net +7,020 pips (+11.1%)
           MODERATE RECOMMENDATION

Usage:
    from filter_recommendations import FilterRecommendationEngine
    engine = FilterRecommendationEngine(trades_df, config={'tp_pips': 100, 'sl_pips': 30})
    recommendations = engine.generate_recommendations()
    engine.print_recommendations(recommendations)
    engine.save_recommendations(recommendations, "artifacts/filter_recommendations.json")
"""

import json
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FilterRule:
    """
    A single filter rule: feature operator threshold.

    Example: FilterRule('d1_opposes_short', '==', 0, "Skip when D1 opposes short")
    → Keep only trades where d1_opposes_short == 0
    """
    feature: str
    operator: str       # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    description: str

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask of rows that PASS this filter (i.e. are allowed to trade)."""
        if self.feature not in df.columns:
            return pd.Series([True] * len(df), index=df.index)

        col = df[self.feature]
        ops = {
            '>':  col >  self.threshold,
            '<':  col <  self.threshold,
            '>=': col >= self.threshold,
            '<=': col <= self.threshold,
            '==': col == self.threshold,
            '!=': col != self.threshold,
        }
        return ops.get(self.operator, pd.Series([True] * len(df), index=df.index))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature":     self.feature,
            "operator":    self.operator,
            "threshold":   self.threshold,
            "description": self.description,
        }


@dataclass
class FilterImpact:
    """Full P&L impact analysis of applying a filter."""
    filter_rule: FilterRule

    # Before filter
    total_trades_before:  int
    wins_before:          int
    losses_before:        int
    win_rate_before:      float
    total_pips_before:    float

    # After filter
    total_trades_after:   int
    wins_after:           int
    losses_after:         int
    win_rate_after:       float
    total_pips_after:     float

    # Delta
    trades_removed:       int
    wins_removed:         int
    losses_removed:       int
    pips_saved:           float   # losses_removed × sl_pips
    pips_lost:            float   # wins_removed  × tp_pips
    net_pips_improvement: float   # pips_saved - pips_lost
    pct_improvement:      float   # net / |total_before| * 100

    # Quality
    loss_removal_rate:    float   # losses_removed / losses_before
    win_preservation_rate: float  # wins_after / wins_before
    efficiency_ratio:     float   # losses_removed / wins_removed (∞ if no wins removed)


@dataclass
class FilterRecommendation:
    """A ranked filter with its impact and a human-readable verdict."""
    rank:           int
    filter_rule:    FilterRule
    impact:         FilterImpact
    recommendation: str   # "STRONG", "MODERATE", "WEAK", "NOT RECOMMENDED"
    reasoning:      str


@dataclass
class FilterCombinationResult:
    """Result of testing multiple filters applied together."""
    filters:              List[FilterRule]
    combined_description: str
    impact:               FilterImpact
    recommendation:       str
    reasoning:            str


# =============================================================================
# FILTER RECOMMENDATION ENGINE
# =============================================================================

class FilterRecommendationEngine:
    """
    Generates, evaluates, and ranks filter recommendations.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        config: Dict[str, Any],
    ):
        """
        Args:
            trades_df: DataFrame loaded from trades_stella_alpha.db
            config: Dict with keys 'tp_pips' and 'sl_pips'
        """
        self.trades_df = trades_df.copy()
        self.tp = float(config.get("tp_pips", 30))
        self.sl = float(config.get("sl_pips", 30))

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def generate_recommendations(
        self,
        top_n: int = 20,
        min_net_improvement: float = 0.0,
    ) -> List[FilterRecommendation]:
        """
        Generate and rank individual filter recommendations.

        Args:
            top_n: Maximum number of recommendations to return.
            min_net_improvement: Only include filters with net pips > this value.

        Returns:
            Ranked list of FilterRecommendation objects.
        """
        candidates = self._generate_candidate_filters()

        impacts = []
        for rule in candidates:
            impact = self._evaluate_filter(rule)
            if impact.net_pips_improvement > min_net_improvement:
                impacts.append(impact)

        # Sort by net pips improvement descending
        impacts.sort(key=lambda x: x.net_pips_improvement, reverse=True)

        recommendations = []
        for rank, impact in enumerate(impacts[:top_n], start=1):
            rec = self._create_recommendation(rank, impact)
            recommendations.append(rec)

        return recommendations

    def test_filter_combination(
        self,
        filters: List[FilterRule],
    ) -> FilterCombinationResult:
        """
        Evaluate several filters applied simultaneously (logical AND).
        """
        df = self.trades_df
        mask = pd.Series([True] * len(df), index=df.index)
        for f in filters:
            mask = mask & f.apply(df)

        impact = self._compute_impact_from_mask(
            mask,
            FilterRule(
                feature="COMBINED",
                operator="AND",
                threshold=float(len(filters)),
                description=" AND ".join(f.description for f in filters),
            )
        )

        strength, reasoning = self._judge(impact)
        return FilterCombinationResult(
            filters=filters,
            combined_description=impact.filter_rule.description,
            impact=impact,
            recommendation=strength,
            reasoning=reasoning,
        )

    def test_top_combinations(
        self,
        recommendations: List[FilterRecommendation],
        max_filters: int = 3,
        top_singles: int = 5,
    ) -> List[FilterCombinationResult]:
        """
        Test pairs and triples from the top individual recommendations.
        """
        import itertools
        top_rules = [r.filter_rule for r in recommendations[:top_singles]]
        results: List[FilterCombinationResult] = []

        for size in range(2, max_filters + 1):
            for combo in itertools.combinations(top_rules, size):
                try:
                    result = self.test_filter_combination(list(combo))
                    if result.impact.net_pips_improvement > 0:
                        results.append(result)
                except Exception:
                    continue

        results.sort(key=lambda r: r.impact.net_pips_improvement, reverse=True)
        return results

    def print_recommendations(
        self,
        recommendations: List[FilterRecommendation],
        combinations: Optional[List[FilterCombinationResult]] = None,
    ) -> None:
        """Print a formatted summary of recommendations."""
        line  = "=" * 70
        dline = "─" * 50

        print(f"\n{line}")
        print(f"  FILTER RECOMMENDATIONS  |  TP={self.tp:.0f}  SL={self.sl:.0f}")
        print(f"{line}")

        baseline_trades = recommendations[0].impact.total_trades_before if recommendations else 0
        baseline_wr     = recommendations[0].impact.win_rate_before      if recommendations else 0.0
        print(f"  Baseline  {baseline_trades:,} trades  |  Win rate: {baseline_wr*100:.1f}%")
        print()

        for rec in recommendations:
            imp = rec.impact
            strength_icon = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🟠"}.get(
                rec.recommendation, "🔴"
            )
            print(
                f"  RANK {rec.rank:2d}  {strength_icon} {rec.recommendation:<14} "
                f"Net: {imp.net_pips_improvement:+,.0f} pips  ({imp.pct_improvement:+.1f}%)"
            )
            print(f"          \"{rec.filter_rule.description}\"")
            print(
                f"          Removes {imp.losses_removed:,} losses  "
                f"Misses {imp.wins_removed:,} wins  "
                f"Efficiency: {imp.efficiency_ratio:.1f}:1  "
                f"Win rate after: {imp.win_rate_after*100:.1f}%"
            )
            print(f"          {rec.reasoning}")
            print()

        if combinations:
            print(f"  TOP FILTER COMBINATIONS")
            print(f"  {dline}")
            for i, combo in enumerate(combinations[:5], 1):
                strength_icon = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🟠"}.get(
                    combo.recommendation, "🔴"
                )
                print(
                    f"  COMBO {i}  {strength_icon} {combo.recommendation:<14} "
                    f"Net: {combo.impact.net_pips_improvement:+,.0f} pips"
                )
                print(f"          \"{combo.combined_description}\"")
                print()

        print(line)

    def save_recommendations(
        self,
        recommendations: List[FilterRecommendation],
        output_path: str,
        combinations: Optional[List[FilterCombinationResult]] = None,
    ) -> None:
        """Serialize recommendations to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _ser(obj: Any) -> Any:
            if isinstance(obj, FilterRule):
                return obj.to_dict()
            if isinstance(obj, FilterImpact):
                d = asdict(obj)
                d["filter_rule"] = obj.filter_rule.to_dict()
                return d
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return str(obj)

        output = {
            "config": {"tp_pips": self.tp, "sl_pips": self.sl},
            "recommendations": [
                {
                    "rank":           r.rank,
                    "filter":         r.filter_rule.to_dict(),
                    "impact":         _impact_to_dict(r.impact),
                    "recommendation": r.recommendation,
                    "reasoning":      r.reasoning,
                }
                for r in recommendations
            ],
        }

        if combinations:
            output["combinations"] = [
                {
                    "filters":      [f.to_dict() for f in c.filters],
                    "description":  c.combined_description,
                    "impact":       _impact_to_dict(c.impact),
                    "recommendation": c.recommendation,
                    "reasoning":    c.reasoning,
                }
                for c in combinations
            ]

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(output, fp, indent=2, default=_ser)

        print(f"Recommendations saved → {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Candidate filter generation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_candidate_filters(self) -> List[FilterRule]:
        """Build the full list of candidate filter rules."""
        filters: List[FilterRule] = []

        # ── Session filters ───────────────────────────────────────────────
        filters += [
            FilterRule("h4_is_asian_session",  "==", 0, "Avoid Asian session"),
            FilterRule("h4_is_overlap_session","==", 1, "Only trade London/NY overlap"),
            FilterRule("h4_is_london_session", "==", 1, "Only trade London session"),
            FilterRule("h4_is_ny_session",     "==", 1, "Only trade NY session"),
        ]

        # ── Confidence filters ────────────────────────────────────────────
        for thresh in [0.55, 0.58, 0.60, 0.63, 0.65, 0.70, 0.75]:
            filters.append(FilterRule(
                "model_probability", ">=", thresh,
                f"Require model confidence >= {thresh:.2f}",
            ))

        # ── D1 alignment filters ──────────────────────────────────────────
        filters += [
            FilterRule("d1_opposes_short",   "==", 0, "Skip when D1 opposes short"),
            FilterRule("d1_supports_short",  "==", 1, "Only when D1 supports short"),
            FilterRule("d1_is_trending_up",  "==", 0, "Skip D1 uptrend (counter-trend risk)"),
            FilterRule("d1_trend_direction", "<=", 0.3, "D1 not strongly bullish"),
        ]

        # ── H4 RSI filters ────────────────────────────────────────────────
        for thresh in [60, 65, 68, 70, 72, 75]:
            filters.append(FilterRule(
                "h4_rsi_value", ">=", thresh,
                f"Require H4 RSI >= {thresh} (overbought signal)",
            ))

        # ── D1 RSI filters ────────────────────────────────────────────────
        for thresh in [50, 55, 60, 65]:
            filters.append(FilterRule(
                "d1_rsi_value", ">=", thresh,
                f"Require D1 RSI >= {thresh}",
            ))

        # ── MTF confluence filters ────────────────────────────────────────
        for thresh in [0.50, 0.60, 0.70, 0.75]:
            filters.append(FilterRule(
                "mtf_confluence_score", ">=", thresh,
                f"Require MTF confluence >= {thresh:.2f}",
            ))

        # ── MTF alignment flags ───────────────────────────────────────────
        filters += [
            FilterRule("mtf_rsi_aligned",   "==", 1, "Require H4/D1 RSI aligned"),
            FilterRule("mtf_trend_aligned",  "==", 1, "Require H4/D1 trend aligned"),
            FilterRule("mtf_bb_aligned",     "==", 1, "Require H4/D1 BB aligned"),
            FilterRule("mtf_strong_short_setup", "==", 1, "Require strong MTF short setup"),
        ]

        # ── H4 BB position filters ────────────────────────────────────────
        for thresh in [0.80, 0.85, 0.88, 0.90, 0.93, 0.95]:
            filters.append(FilterRule(
                "h4_bb_position", ">=", thresh,
                f"Require H4 BB position >= {thresh:.2f} (near upper band)",
            ))

        # ── ADX / trend strength ─────────────────────────────────────────
        for thresh in [15, 20, 25]:
            filters.append(FilterRule(
                "h4_adx", ">=", thresh,
                f"Require H4 ADX >= {thresh} (trending market)",
            ))

        # ── Volume filters ────────────────────────────────────────────────
        for thresh in [0.8, 1.0, 1.2]:
            filters.append(FilterRule(
                "h4_volume_ratio", ">=", thresh,
                f"Require H4 volume ratio >= {thresh:.1f}",
            ))

        # ── Exhaustion / consecutive candles ─────────────────────────────
        for thresh in [2, 3]:
            filters.append(FilterRule(
                "h4_consecutive_bullish", ">=", thresh,
                f"Require >= {thresh} consecutive bullish H4 candles",
            ))
        for thresh in [0.3, 0.5, 0.7]:
            filters.append(FilterRule(
                "h4_exhaustion_score", ">=", thresh,
                f"Require H4 exhaustion score >= {thresh:.1f}",
            ))

        # ── Only include filters whose feature column exists ──────────────
        available_cols = set(self.trades_df.columns)
        filters = [f for f in filters if f.feature in available_cols]

        return filters

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_filter(self, rule: FilterRule) -> FilterImpact:
        mask = rule.apply(self.trades_df)
        return self._compute_impact_from_mask(mask, rule)

    def _compute_impact_from_mask(
        self,
        mask: pd.Series,
        rule: FilterRule,
    ) -> FilterImpact:
        df = self.trades_df

        total_before = len(df)
        wins_before  = int((df["outcome"] == "WIN").sum())
        losses_before = int((df["outcome"] == "LOSS").sum())
        pips_before  = float(df["pips_result"].sum())

        df_after      = df[mask]
        total_after   = len(df_after)
        wins_after    = int((df_after["outcome"] == "WIN").sum())
        losses_after  = int((df_after["outcome"] == "LOSS").sum())
        pips_after    = float(df_after["pips_result"].sum())

        trades_removed  = total_before  - total_after
        wins_removed    = wins_before   - wins_after
        losses_removed  = losses_before - losses_after

        pips_saved  = losses_removed * self.sl
        pips_lost   = wins_removed   * self.tp
        net_pips    = pips_saved - pips_lost
        pct_impr    = (
            (net_pips / abs(pips_before) * 100.0)
            if pips_before != 0 else 0.0
        )
        eff_ratio   = (
            losses_removed / wins_removed
            if wins_removed > 0 else float("inf")
        )

        return FilterImpact(
            filter_rule=rule,
            total_trades_before=total_before,
            wins_before=wins_before,
            losses_before=losses_before,
            win_rate_before=wins_before / total_before if total_before > 0 else 0.0,
            total_pips_before=pips_before,
            total_trades_after=total_after,
            wins_after=wins_after,
            losses_after=losses_after,
            win_rate_after=wins_after / total_after if total_after > 0 else 0.0,
            total_pips_after=pips_after,
            trades_removed=trades_removed,
            wins_removed=wins_removed,
            losses_removed=losses_removed,
            pips_saved=pips_saved,
            pips_lost=pips_lost,
            net_pips_improvement=net_pips,
            pct_improvement=pct_impr,
            loss_removal_rate=losses_removed / losses_before if losses_before > 0 else 0.0,
            win_preservation_rate=wins_after / wins_before if wins_before > 0 else 0.0,
            efficiency_ratio=eff_ratio,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Verdict logic
    # ──────────────────────────────────────────────────────────────────────

    def _judge(self, impact: FilterImpact) -> Tuple[str, str]:
        """Return (recommendation label, reasoning string)."""
        eff  = impact.efficiency_ratio
        net  = impact.net_pips_improvement
        pct  = impact.pct_improvement

        if eff >= 3.0 and net > 1000:
            strength = "STRONG"
            reasoning = (
                f"Removes {impact.losses_removed:,} losses for only "
                f"{impact.wins_removed:,} missed wins ({eff:.1f}:1 ratio)"
            )
        elif eff >= 2.0 and net > 300:
            strength = "MODERATE"
            reasoning = (
                f"Good efficiency ({eff:.1f}:1) with solid net improvement "
                f"of {net:+,.0f} pips ({pct:+.1f}%)"
            )
        elif net > 0:
            strength = "WEAK"
            reasoning = (
                f"Positive but marginal improvement ({net:+,.0f} pips, "
                f"efficiency {eff:.1f}:1)"
            )
        else:
            strength = "NOT RECOMMENDED"
            reasoning = "Filter removes more winning trades than losing ones"

        return strength, reasoning

    def _create_recommendation(
        self,
        rank: int,
        impact: FilterImpact,
    ) -> FilterRecommendation:
        strength, reasoning = self._judge(impact)
        return FilterRecommendation(
            rank=rank,
            filter_rule=impact.filter_rule,
            impact=impact,
            recommendation=strength,
            reasoning=reasoning,
        )


# =============================================================================
# HELPERS
# =============================================================================

def _impact_to_dict(impact: FilterImpact) -> Dict[str, Any]:
    """Flatten FilterImpact to a JSON-safe dict."""
    return {
        "filter": impact.filter_rule.to_dict(),
        "before": {
            "total_trades": impact.total_trades_before,
            "wins":         impact.wins_before,
            "losses":       impact.losses_before,
            "win_rate":     round(impact.win_rate_before, 4),
            "total_pips":   round(impact.total_pips_before, 2),
        },
        "after": {
            "total_trades": impact.total_trades_after,
            "wins":         impact.wins_after,
            "losses":       impact.losses_after,
            "win_rate":     round(impact.win_rate_after, 4),
            "total_pips":   round(impact.total_pips_after, 2),
        },
        "delta": {
            "trades_removed":       impact.trades_removed,
            "wins_removed":         impact.wins_removed,
            "losses_removed":       impact.losses_removed,
            "pips_saved":           round(impact.pips_saved, 2),
            "pips_lost":            round(impact.pips_lost, 2),
            "net_pips_improvement": round(impact.net_pips_improvement, 2),
            "pct_improvement":      round(impact.pct_improvement, 2),
        },
        "quality": {
            "loss_removal_rate":    round(impact.loss_removal_rate, 4),
            "win_preservation_rate":round(impact.win_preservation_rate, 4),
            "efficiency_ratio":     round(impact.efficiency_ratio, 2)
                                    if impact.efficiency_ratio != float("inf") else 9999,
        },
    }
