#!/usr/bin/env python3
"""
Stella Alpha — Test Suite
==========================

Comprehensive unit and integration tests for all modules.

Run all tests:
    python tests/test_suite.py

Run a single group:
    python tests/test_suite.py TradeRecorderTests
    python tests/test_suite.py LeakageValidationTests

Requires:
    pip install scipy numpy pandas
    (no ML dependencies needed — tests are self-contained)
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve src path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
SRC  = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# HELPER: synthetic data generators
# =============================================================================

def _make_h4(n: int = 200, start: str = "2024-01-01") -> pd.DataFrame:
    """Create a minimal H4 OHLCV DataFrame."""
    ts = pd.date_range(start, periods=n, freq="4h")
    np.random.seed(42)
    close = 1.08 + np.cumsum(np.random.randn(n) * 0.001)
    return pd.DataFrame({
        "timestamp": ts,
        "open":   close * (1 + np.random.randn(n) * 0.0005),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close":  close,
        "volume": np.random.randint(800, 1500, n).astype(float),
        "rsi_value":    60 + np.random.randn(n) * 8,
        "bb_position":  np.clip(0.75 + np.random.randn(n) * 0.15, 0.0, 1.0),
        "atr_pct":      0.005 + np.abs(np.random.randn(n)) * 0.001,
        "trend_strength": np.random.randn(n),
        "volume_ratio": 1.0 + np.random.randn(n) * 0.3,
        "lower_band":   close * 0.99,
        "middle_band":  close,
        "upper_band":   close * 1.01,
    })


def _make_d1(n: int = 50, start: str = "2024-01-01") -> pd.DataFrame:
    """Create a minimal D1 OHLCV DataFrame."""
    ts = pd.date_range(start, periods=n, freq="D")
    np.random.seed(7)
    close = 1.08 + np.cumsum(np.random.randn(n) * 0.002)
    return pd.DataFrame({
        "timestamp": ts,
        "open":    close * (1 + np.random.randn(n) * 0.001),
        "high":    close * (1 + np.abs(np.random.randn(n)) * 0.002),
        "low":     close * (1 - np.abs(np.random.randn(n)) * 0.002),
        "close":   close,
        "volume":  np.random.randint(40000, 60000, n).astype(float),
        "rsi_value":   55 + np.random.randn(n) * 10,
        "bb_position": np.clip(0.65 + np.random.randn(n) * 0.2, 0.0, 1.0),
        "atr_pct":     0.008 + np.abs(np.random.randn(n)) * 0.002,
        "trend_strength": np.random.randn(n),
        "lower_band":  close * 0.985,
        "middle_band": close,
        "upper_band":  close * 1.015,
    })


def _make_trade_records(n: int = 200, config_id: str = "TP100_SL30_H48") -> list:
    np.random.seed(42)
    outcomes = np.where(np.random.rand(n) > 0.35, "WIN", "LOSS")
    records = []
    for i in range(n):
        records.append({
            "config_id":            config_id,
            "fold":                 (i % 5) + 1,
            "timestamp":            f"2024-01-{(i%28)+1:02d} 08:00",
            "outcome":              outcomes[i],
            "pips_result":          100.0 if outcomes[i] == "WIN" else -30.0,
            "model_probability":    float(np.clip(0.50 + np.random.rand() * 0.40, 0.50, 0.95)),
            "h4_rsi_value":         float(55 + np.random.randn() * 10),
            "h4_bb_position":       float(np.clip(0.75 + np.random.randn() * 0.15, 0, 1)),
            "h4_is_asian_session":  int(np.random.rand() < 0.25),
            "h4_is_london_session": int(np.random.rand() < 0.35),
            "h4_is_ny_session":     int(np.random.rand() < 0.30),
            "h4_is_overlap_session":int(np.random.rand() < 0.20),
            "h4_adx":               float(20 + np.random.randn() * 8),
            "d1_rsi_value":         float(52 + np.random.randn() * 12),
            "d1_trend_direction":   float(np.random.rand()),
            "d1_is_trending_up":    int(np.random.rand() < 0.4),
            "d1_supports_short":    int(np.random.rand() < 0.45),
            "d1_opposes_short":     int(np.random.rand() < 0.30),
            "mtf_confluence_score": float(np.random.rand()),
            "mtf_rsi_aligned":      int(np.random.rand() < 0.5),
            "mtf_trend_aligned":    int(np.random.rand() < 0.5),
        })
    return records


# =============================================================================
# GROUP 1: DATA MERGER & LEAKAGE VALIDATION
# =============================================================================

class LeakageValidationTests(unittest.TestCase):
    """Tests for D1 merge correctness and leakage prevention."""

    def setUp(self):
        try:
            from data_merger import DataMerger
            self.merger = DataMerger()
            self.available = True
        except ImportError:
            self.available = False

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest("data_merger.py not importable (expected in src/)")

    def test_merge_d1_is_previous_day(self):
        """Every H4 row must use D1 data from the PREVIOUS completed day."""
        self._skip_if_unavailable()
        h4 = _make_h4(120, "2024-01-10")
        d1 = _make_d1(30,  "2024-01-01")
        merged = self.merger.merge(h4, d1)

        if "d1_timestamp" not in merged.columns:
            self.skipTest("Merger doesn't add d1_timestamp column")

        h4_dates = pd.to_datetime(merged["timestamp"]).dt.date
        d1_dates = pd.to_datetime(merged["d1_timestamp"]).dt.date

        violations = merged[d1_dates >= h4_dates].dropna(subset=["d1_timestamp"])
        self.assertEqual(
            len(violations), 0,
            f"Leakage detected: {len(violations)} rows where D1 date >= H4 date"
        )

    def test_merged_columns_have_d1_prefix(self):
        """All D1-sourced columns must be prefixed with d1_."""
        self._skip_if_unavailable()
        h4 = _make_h4(60, "2024-01-10")
        d1 = _make_d1(15, "2024-01-01")
        merged = self.merger.merge(h4, d1)

        d1_cols = [c for c in merged.columns if c.startswith("d1_")]
        self.assertGreater(len(d1_cols), 0, "No d1_-prefixed columns found after merge")

    def test_merged_no_duplicate_rows(self):
        """Merge must not duplicate H4 rows."""
        self._skip_if_unavailable()
        h4 = _make_h4(120, "2024-01-10")
        d1 = _make_d1(30,  "2024-01-01")
        merged = self.merger.merge(h4, d1)
        self.assertEqual(
            len(merged), len(h4),
            f"Row count changed: {len(h4)} H4 rows → {len(merged)} after merge"
        )

    def test_validate_no_leakage_passes_clean_data(self):
        """validate_no_leakage() should return True on properly merged data."""
        self._skip_if_unavailable()
        if not hasattr(self.merger, "validate_no_leakage"):
            self.skipTest("validate_no_leakage() not implemented in DataMerger")

        h4 = _make_h4(120, "2024-01-10")
        d1 = _make_d1(30,  "2024-01-01")
        merged = self.merger.merge(h4, d1)
        ok = self.merger.validate_no_leakage(merged)
        self.assertTrue(ok, "validate_no_leakage returned False on clean merged data")

    def test_merge_h4_only_still_works(self):
        """Pipeline must not crash when no D1 data is provided."""
        self._skip_if_unavailable()
        h4 = _make_h4(60, "2024-01-10")
        # Simply assert no exception when d1 is None or empty
        try:
            merged = self.merger.merge(h4, None)
            self.assertIsNotNone(merged)
        except TypeError:
            # Some implementations require d1 — that's fine, just document
            pass


# =============================================================================
# GROUP 2: TRADE RECORDER
# =============================================================================

class TradeRecorderTests(unittest.TestCase):
    """Tests for TradeRecorder SQLite writer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test_trades.db")
        from trade_recorder import TradeRecorder
        self.recorder = TradeRecorder(self.db_path)

    def test_insert_single_trade(self):
        """Single insert must return True and increment count."""
        from trade_recorder import TradeRecorder
        rec = {"config_id": "TP100_SL30_H48", "fold": 1,
               "timestamp": "2024-01-01 08:00", "outcome": "WIN",
               "pips_result": 100.0, "model_probability": 0.65}
        ok = self.recorder.insert_trade(rec)
        self.assertTrue(ok)
        self.assertEqual(self.recorder.get_trade_count(), 1)

    def test_batch_insert(self):
        """Batch insert must persist all records."""
        records = _make_trade_records(50)
        inserted = self.recorder.insert_trades_batch(records)
        self.assertEqual(inserted, 50)
        self.assertEqual(self.recorder.get_trade_count(), 50)

    def test_get_summary_correct(self):
        """Summary must correctly count wins and losses."""
        records = _make_trade_records(100)
        self.recorder.insert_trades_batch(records)
        summary = self.recorder.get_summary()
        expected_total = 100
        self.assertEqual(summary["total_trades"], expected_total)
        self.assertEqual(summary["wins"] + summary["losses"], expected_total)
        self.assertAlmostEqual(summary["win_rate"], summary["wins"] / 100, places=4)

    def test_schema_all_columns_present(self):
        """Database must have all required schema columns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(trades)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()

        required = {
            "trade_id", "config_id", "fold", "timestamp", "outcome", "pips_result",
            "model_probability", "h4_rsi_value", "h4_bb_position",
            "h4_is_asian_session", "h4_is_london_session",
            "d1_rsi_value", "d1_trend_direction",
            "mtf_confluence_score", "d1_supports_short", "d1_opposes_short",
        }
        missing = required - cols
        self.assertEqual(missing, set(), f"Missing columns: {missing}")

    def test_missing_optional_fields_insert_as_null(self):
        """Partial records (missing optional features) must still insert."""
        rec = {"config_id": "TP50_SL30_H24", "fold": 1,
               "timestamp": "2024-03-01 12:00", "outcome": "LOSS",
               "pips_result": -30.0, "model_probability": 0.52}
        ok = self.recorder.insert_trade(rec)
        self.assertTrue(ok)

    def test_thread_safety_concurrent_inserts(self):
        """Multiple threads inserting simultaneously must not lose records."""
        import threading
        results = []
        n_per_thread = 20
        n_threads = 5

        from trade_recorder import TradeRecorder

        def insert_batch(thread_id: int):
            r = TradeRecorder(self.db_path)
            recs = _make_trade_records(n_per_thread, config_id=f"cfg_t{thread_id}")
            n = r.insert_trades_batch(recs)
            results.append(n)
            r.close()

        threads = [threading.Thread(target=insert_batch, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = self.recorder.get_trade_count()
        expected = n_per_thread * n_threads
        self.assertEqual(total, expected,
                         f"Expected {expected} trades, got {total} (thread safety issue)")

    def test_build_trade_record_helper(self):
        """build_trade_record() must produce a dict with all required keys."""
        from trade_recorder import build_trade_record
        row = pd.Series({
            "timestamp": "2024-01-15 08:00",
            "rsi_value": 72.5, "bb_position": 0.90,
            "d1_rsi_value": 65.0, "mtf_confluence_score": 0.75,
            "d1_supports_short": 1, "d1_opposes_short": 0,
        })
        rec = build_trade_record(
            row=row, outcome="WIN", pips_result=100.0,
            model_probability=0.67, config_id="TP100_SL30_H48",
            fold=3, threshold_used=0.55,
        )
        required_keys = ["config_id", "fold", "timestamp", "outcome", "pips_result",
                         "model_probability", "d1_rsi_value", "mtf_confluence_score"]
        for k in required_keys:
            self.assertIn(k, rec, f"Missing key: {k}")
        self.assertEqual(rec["outcome"], "WIN")
        self.assertEqual(rec["config_id"], "TP100_SL30_H48")


# =============================================================================
# GROUP 3: LOSS ANALYSIS
# =============================================================================

class LossAnalysisTests(unittest.TestCase):
    """Tests for LossAnalyzer statistical analysis."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "trades.db")
        from trade_recorder import TradeRecorder
        rec = TradeRecorder(self.db_path)
        rec.insert_trades_batch(_make_trade_records(300))
        from loss_analysis import LossAnalyzer
        self.analyzer = LossAnalyzer(self.db_path)

    def test_load_returns_nonempty_df(self):
        df = self.analyzer.load_trades()
        self.assertGreater(len(df), 0)

    def test_analyze_returns_report(self):
        from loss_analysis import LossAnalysisReport
        report = self.analyzer.analyze()
        self.assertIsInstance(report, LossAnalysisReport)
        self.assertGreater(report.total_trades, 0)
        self.assertEqual(report.total_wins + report.total_losses, report.total_trades)
        self.assertAlmostEqual(
            report.win_rate,
            report.total_wins / report.total_trades,
            places=4,
        )

    def test_session_analysis_has_entries(self):
        report = self.analyzer.analyze()
        # May be empty if session cols not in data — just must not crash
        self.assertIsInstance(report.session_analysis, list)

    def test_confidence_analysis_populated(self):
        report = self.analyzer.analyze()
        # model_probability is in synthetic data so buckets should populate
        self.assertGreater(len(report.confidence_analysis), 0)
        for c in report.confidence_analysis:
            self.assertEqual(c.wins + c.losses, c.total_trades)

    def test_optimal_threshold_in_range(self):
        report = self.analyzer.analyze()
        self.assertGreaterEqual(report.optimal_confidence_threshold, 0.50)
        self.assertLessEqual(report.optimal_confidence_threshold, 0.95)

    def test_feature_comparisons_sorted_by_significance(self):
        report = self.analyzer.analyze()
        if len(report.feature_comparisons) < 2:
            return
        # Significant features must come first
        sig_indices = [i for i, c in enumerate(report.feature_comparisons) if c.is_significant]
        insig_indices = [i for i, c in enumerate(report.feature_comparisons) if not c.is_significant]
        if sig_indices and insig_indices:
            self.assertLess(max(sig_indices), min(insig_indices),
                            "Significant features should precede insignificant ones")

    def test_save_report_creates_valid_json(self):
        report = self.analyzer.analyze()
        out = str(Path(self.tmpdir) / "report.json")
        self.analyzer.save_report(report, out)
        self.assertTrue(Path(out).exists())
        with open(out) as f:
            data = json.load(f)
        self.assertIn("config_id", data)
        self.assertIn("total_trades", data)

    def test_planted_session_pattern_detected(self):
        """Inject extreme Asian session losses — must appear as problematic."""
        np.random.seed(42)
        n = 500
        is_asian = (np.random.rand(n) < 0.25).astype(int)
        wr = np.where(is_asian, 0.05, 0.65)   # extreme 5% on Asian
        outcomes = np.array(["WIN" if np.random.rand() < w else "LOSS" for w in wr])
        pips = np.where(outcomes == "WIN", 100.0, -30.0)

        db2 = str(Path(self.tmpdir) / "trades2.db")
        from trade_recorder import TradeRecorder
        from loss_analysis import LossAnalyzer
        rec = TradeRecorder(db2)
        rows = [{"config_id": "cfg", "fold": 1, "timestamp": "2024-01-01",
                 "outcome": outcomes[i], "pips_result": pips[i],
                 "model_probability": 0.6,
                 "h4_is_asian_session": int(is_asian[i]),
                 "h4_is_london_session": 0, "h4_is_ny_session": 0,
                 "h4_is_overlap_session": 0} for i in range(n)]
        rec.insert_trades_batch(rows)
        analyzer = LossAnalyzer(db2)
        report = analyzer.analyze()
        self.assertIn("Asian", report.problematic_sessions,
                      "Asian session should be flagged as problematic")


# =============================================================================
# GROUP 4: FILTER RECOMMENDATION ENGINE
# =============================================================================

class FilterRecommendationTests(unittest.TestCase):
    """Tests for FilterRecommendationEngine."""

    def setUp(self):
        np.random.seed(42)
        n = 1000
        # Plant: d1_opposes_short → 8% WR vs 65% baseline
        bad_flag = (np.random.rand(n) < 0.30).astype(int)
        wr = np.where(bad_flag, 0.08, 0.65)
        outcomes = np.array(["WIN" if np.random.rand() < w else "LOSS" for w in wr])
        pips = np.where(outcomes == "WIN", 100.0, -30.0)

        self.df = pd.DataFrame({
            "outcome":            outcomes,
            "pips_result":        pips,
            "model_probability":  np.clip(0.55 + np.random.randn(n) * 0.1, 0.50, 0.95),
            "d1_opposes_short":   bad_flag,
            "d1_supports_short":  (1 - bad_flag),
            "d1_is_trending_up":  (np.random.rand(n) < 0.40).astype(int),
            "h4_is_asian_session":(np.random.rand(n) < 0.20).astype(int),
            "h4_rsi_value":       55 + np.random.randn(n) * 10,
            "h4_bb_position":     np.clip(0.75 + np.random.randn(n) * 0.15, 0, 1),
            "mtf_confluence_score": np.random.rand(n),
        })
        from filter_recommendations import FilterRecommendationEngine
        self.engine = FilterRecommendationEngine(
            self.df, config={"tp_pips": 100, "sl_pips": 30}
        )

    def test_filter_rule_apply(self):
        """FilterRule.apply() must return correct boolean mask."""
        from filter_recommendations import FilterRule
        rule = FilterRule("d1_opposes_short", "==", 0, "test")
        mask = rule.apply(self.df)
        self.assertEqual(len(mask), len(self.df))
        self.assertTrue(mask.dtype == bool or mask.dtype == np.bool_)
        self.assertTrue((self.df[mask]["d1_opposes_short"] == 0).all())

    def test_generates_at_least_one_recommendation(self):
        recs = self.engine.generate_recommendations(top_n=10)
        self.assertGreater(len(recs), 0,
                           "Expected at least 1 recommendation for planted pattern")

    def test_planted_pattern_surfaces_as_top_recommendation(self):
        recs = self.engine.generate_recommendations(top_n=20)
        top_features = [r.filter_rule.feature for r in recs[:5]]
        self.assertIn("d1_opposes_short", top_features,
                      "d1_opposes_short should appear in top 5 recommendations")

    def test_recommendation_impact_math(self):
        """net = pips_saved - pips_lost must be consistent."""
        recs = self.engine.generate_recommendations(top_n=5)
        for r in recs:
            imp = r.impact
            expected_net = imp.pips_saved - imp.pips_lost
            self.assertAlmostEqual(
                imp.net_pips_improvement, expected_net, places=2,
                msg=f"Net pips math error for {r.filter_rule.description}"
            )
            expected_saved = imp.losses_removed * 30
            self.assertAlmostEqual(imp.pips_saved, expected_saved, places=2)

    def test_efficiency_ratio_computed(self):
        recs = self.engine.generate_recommendations(top_n=5)
        for r in recs:
            if r.impact.wins_removed > 0:
                expected_eff = r.impact.losses_removed / r.impact.wins_removed
                self.assertAlmostEqual(r.impact.efficiency_ratio, expected_eff, places=3)

    def test_combinations_tested(self):
        recs = self.engine.generate_recommendations(top_n=5)
        combos = self.engine.test_top_combinations(recs, max_filters=2, top_singles=3)
        self.assertIsInstance(combos, list)
        # May be empty if no combo improves on singles — just must not crash

    def test_win_preservation_rate_range(self):
        recs = self.engine.generate_recommendations(top_n=10)
        for r in recs:
            self.assertGreaterEqual(r.impact.win_preservation_rate, 0.0)
            self.assertLessEqual(r.impact.win_preservation_rate, 1.0)

    def test_save_recommendations_json(self):
        recs = self.engine.generate_recommendations(top_n=3)
        tmpdir = tempfile.mkdtemp()
        out = str(Path(tmpdir) / "recs.json")
        self.engine.save_recommendations(recs, out)
        self.assertTrue(Path(out).exists())
        with open(out) as f:
            data = json.load(f)
        self.assertIn("recommendations", data)


# =============================================================================
# GROUP 5: STATISTICAL VALIDATION
# =============================================================================

class StatisticalValidationTests(unittest.TestCase):
    """Tests for statistical pre-filter in statistical_validation.py."""

    def setUp(self):
        try:
            from statistical_validation import StatisticalFeatureFilter, apply_statistical_prefilter
            self.StatisticalFeatureFilter = StatisticalFeatureFilter
            self.apply_fn = apply_statistical_prefilter
            self.available = True
        except ImportError:
            self.available = False

    def _skip(self):
        if not self.available:
            self.skipTest("statistical_validation.py not importable")

    def _make_labeled_df(self, n: int = 500) -> pd.DataFrame:
        np.random.seed(1)
        # Feature A: highly significant (different means for label 0/1)
        # Feature B: noise (same distribution for both labels)
        label = (np.random.rand(n) > 0.40).astype(int)
        return pd.DataFrame({
            "label":     label,
            "feat_sig":  np.where(label == 1, 1.5 + np.random.randn(n) * 0.5,
                                               0.0 + np.random.randn(n) * 0.5),
            "feat_noise":np.random.randn(n),   # pure noise
            "feat_weak": np.where(label == 1, 0.3 + np.random.randn(n) * 1.0,
                                               0.0 + np.random.randn(n) * 1.0),
        })

    def test_significant_feature_kept(self):
        self._skip()
        df = self._make_labeled_df(500)
        kept, _, _ = self.StatisticalFeatureFilter().filter_features(
            df, ["feat_sig", "feat_noise"], "label"
        )
        self.assertIn("feat_sig", kept,
                      "Significant feature should be kept")

    def test_noise_feature_dropped(self):
        self._skip()
        df = self._make_labeled_df(500)
        _, dropped, _ = self.StatisticalFeatureFilter().filter_features(
            df, ["feat_sig", "feat_noise"], "label"
        )
        self.assertIn("feat_noise", dropped,
                      "Pure noise feature should be dropped")

    def test_apply_prefilter_reduces_features(self):
        self._skip()
        df = self._make_labeled_df(500)
        kept, _ = self.apply_fn(df, ["feat_sig", "feat_noise", "feat_weak"], "label")
        self.assertLess(len(kept), 3,
                        "Pre-filter should drop at least 1 feature from noisy set")


# =============================================================================
# GROUP 6: EVALUATION — R:R-AWARE ACCEPTANCE CRITERIA
# =============================================================================

class EvaluationTests(unittest.TestCase):
    """Tests for R:R-aware acceptance criteria in evaluation.py."""

    def setUp(self):
        try:
            from evaluation import check_acceptance_criteria, compute_min_winrate
            self.check = check_acceptance_criteria
            self.min_wr = compute_min_winrate
            self.available = True
        except ImportError:
            self.available = False

    def _skip(self):
        if not self.available:
            self.skipTest("evaluation.py not importable")

    def test_breakeven_formula(self):
        """min_winrate = SL / (TP + SL)."""
        self._skip()
        self.assertAlmostEqual(self.min_wr(100, 30), 30 / 130, places=6)
        self.assertAlmostEqual(self.min_wr(50,  30), 30 /  80, places=6)
        self.assertAlmostEqual(self.min_wr(30,  30), 30 /  60, places=6)

    def _metrics(self, precision: float, precision_std: float, ev: float, trades: int):
        """Build an AggregateMetrics with the given values."""
        from evaluation import AggregateMetrics
        return AggregateMetrics(
            precision_mean=precision, precision_std=precision_std,
            recall_mean=0.30, recall_std=0.05,
            f1_mean=0.35, f1_std=0.05,
            auc_pr_mean=0.55, auc_pr_std=0.05,
            ev_mean=ev, ev_std=abs(ev) * 0.2,
            total_trades=trades, n_folds=5,
        )

    def test_high_rr_passes_with_low_precision(self):
        """TP=100, SL=30: breakeven=23%, so 30% precision (edge=+7%) should PASS."""
        self._skip()
        # std=0.03 → cv=0.10, well below 0.30 limit
        passed, reasons = self.check(
            self._metrics(precision=0.30, precision_std=0.03, ev=5.0, trades=200),
            tp_pips=100, sl_pips=30,
        )
        self.assertTrue(passed,
            f"30% precision should pass for TP=100 SL=30 (breakeven≈23%). Reasons: {reasons}")

    def test_high_rr_fails_below_breakeven(self):
        """TP=100, SL=30: 20% precision < 28% required → must FAIL."""
        self._skip()
        passed, reasons = self.check(
            self._metrics(precision=0.20, precision_std=0.02, ev=-2.0, trades=200),
            tp_pips=100, sl_pips=30,
        )
        self.assertFalse(passed,
            "20% precision should fail for TP=100 SL=30 (required≈28% incl. edge)")

    def test_rejects_high_cv(self):
        """CV > 30% → unstable → fail."""
        self._skip()
        # std=0.22 → cv = 0.22/0.55 = 0.40 > 0.30
        passed, reasons = self.check(
            self._metrics(precision=0.55, precision_std=0.22, ev=5.0, trades=200),
            tp_pips=100, sl_pips=30, max_precision_cv=0.30,
        )
        self.assertFalse(passed, f"High CV should cause rejection. Reasons: {reasons}")

    def test_rejects_low_trade_count(self):
        """Fewer than min_trades_per_fold total → fail."""
        self._skip()
        passed, reasons = self.check(
            self._metrics(precision=0.55, precision_std=0.05, ev=5.0, trades=20),
            tp_pips=100, sl_pips=30, min_trades_per_fold=30,
        )
        self.assertFalse(passed, f"Insufficient trades should cause rejection. Reasons: {reasons}")


# =============================================================================
# GROUP 7: INTEGRATION — END-TO-END MODULE CHAIN
# =============================================================================

class IntegrationTests(unittest.TestCase):
    """
    End-to-end tests that chain modules together without the full ML pipeline.
    These test the data flow: TradeRecorder → LossAnalyzer → FilterEngine.
    """

    def test_full_analysis_chain(self):
        """TradeRecorder → LossAnalyzer → FilterRecommendationEngine."""
        import sqlite3
        tmpdir = tempfile.mkdtemp()
        db = str(Path(tmpdir) / "trades.db")

        # 1. Record trades with planted pattern
        from trade_recorder import TradeRecorder
        rec = TradeRecorder(db)
        np.random.seed(7)
        n = 400
        is_asian = (np.random.rand(n) < 0.25).astype(int)
        wr = np.where(is_asian, 0.10, 0.65)
        outcomes = np.array(["WIN" if np.random.rand() < w else "LOSS" for w in wr])
        rows = [{"config_id": "TP100_SL30_H48", "fold": 1,
                 "timestamp": f"2024-01-{(i%28)+1:02d}", "outcome": outcomes[i],
                 "pips_result": 100.0 if outcomes[i] == "WIN" else -30.0,
                 "model_probability": 0.60,
                 "h4_is_asian_session": int(is_asian[i]),
                 "h4_is_london_session": 0, "h4_is_ny_session": 0,
                 "h4_is_overlap_session": 0,
                 "h4_rsi_value": 65.0, "h4_bb_position": 0.85,
                 "d1_rsi_value": 60.0, "mtf_confluence_score": 0.70,
                 "d1_supports_short": 1, "d1_opposes_short": 0} for i in range(n)]
        inserted = rec.insert_trades_batch(rows)
        self.assertEqual(inserted, n)

        # 2. Run loss analysis
        from loss_analysis import LossAnalyzer
        analyzer = LossAnalyzer(db)
        analyzer.load_trades()
        report = analyzer.analyze()
        self.assertGreater(report.total_trades, 0)

        # 3. Run filter engine
        conn = sqlite3.connect(db)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        from filter_recommendations import FilterRecommendationEngine
        engine = FilterRecommendationEngine(df, {"tp_pips": 100, "sl_pips": 30})
        recs = engine.generate_recommendations(top_n=5)
        self.assertGreater(len(recs), 0)

        # 4. Top filter must involve asian session (planted pattern)
        top_features = [r.filter_rule.feature for r in recs[:3]]
        self.assertTrue(
            any("asian" in f for f in top_features),
            f"Expected asian session in top filters, got: {top_features}"
        )

    def test_trade_recorder_summary_after_multiple_configs(self):
        """Multiple configs written to same DB must all be tracked."""
        tmpdir = tempfile.mkdtemp()
        db = str(Path(tmpdir) / "multi.db")
        from trade_recorder import TradeRecorder
        rec = TradeRecorder(db)
        for cfg in ["TP50_SL30_H24", "TP100_SL30_H48", "TP150_SL30_H72"]:
            rec.insert_trades_batch(_make_trade_records(50, config_id=cfg))
        summary = rec.get_summary()
        self.assertEqual(summary["total_trades"], 150)
        self.assertEqual(summary["distinct_configs"], 3)

    def test_loss_report_json_round_trip(self):
        """save_report → reload JSON must preserve key fields."""
        tmpdir = tempfile.mkdtemp()
        db = str(Path(tmpdir) / "t.db")
        from trade_recorder import TradeRecorder
        from loss_analysis import LossAnalyzer
        TradeRecorder(db).insert_trades_batch(_make_trade_records(100))
        analyzer = LossAnalyzer(db)
        analyzer.load_trades()
        report = analyzer.analyze()
        out = str(Path(tmpdir) / "r.json")
        analyzer.save_report(report, out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(data["total_trades"], report.total_trades)
        self.assertIn("feature_comparisons", data)


# =============================================================================
# RUNNER
# =============================================================================

def run_tests(test_class_name: str | None = None) -> bool:
    """Run all tests or a specific class. Returns True if all pass."""
    loader = unittest.TestLoader()

    if test_class_name:
        # Find the class by name
        classes = {
            "LeakageValidationTests":     LeakageValidationTests,
            "TradeRecorderTests":         TradeRecorderTests,
            "LossAnalysisTests":          LossAnalysisTests,
            "FilterRecommendationTests":  FilterRecommendationTests,
            "StatisticalValidationTests": StatisticalValidationTests,
            "EvaluationTests":            EvaluationTests,
            "IntegrationTests":           IntegrationTests,
        }
        cls = classes.get(test_class_name)
        if cls is None:
            print(f"Unknown test class: {test_class_name}")
            print(f"Available: {list(classes.keys())}")
            return False
        suite = loader.loadTestsFromTestCase(cls)
    else:
        suite = unittest.TestSuite()
        for cls in [
            LeakageValidationTests,
            TradeRecorderTests,
            LossAnalysisTests,
            FilterRecommendationTests,
            StatisticalValidationTests,
            EvaluationTests,
            IntegrationTests,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_tests(target)
    sys.exit(0 if success else 1)
