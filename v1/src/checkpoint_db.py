"""
Enhanced Checkpoint Database for Stella Alpha

Stores progress to allow resuming interrupted runs.
Adds tier classification, trade recording, and multi-model export support.

PRIMARY KEY: (tp_pips, sl_pips, max_holding_bars)
Database: pure_ml_stella_alpha.db

NEW IN STELLA ALPHA:
- Tier classification (0=Runner, 1=Ideal)
- Risk/Reward metrics
- Edge above breakeven
- Model storage for multi-model export
- Per-fold results storage
"""

import sqlite3
import threading
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ConfigResult:
    """Complete result for a config."""
    config_id: str
    tp_pips: int
    sl_pips: int
    max_holding_bars: int
    status: str
    tier: int
    tier_name: str
    ev_mean: float
    ev_std: float
    precision_mean: float
    precision_std: float
    precision_cv: float
    recall_mean: float
    f1_mean: float
    auc_pr_mean: float
    total_trades: int
    selected_features: List[str]
    n_features: int
    consensus_threshold: float
    classification: Optional[str]
    rejection_reasons: Optional[List[str]]
    risk_reward_ratio: float
    min_winrate_required: float
    edge_above_breakeven: float
    execution_time_seconds: float


class StellaAlphaCheckpointManager:
    """
    Enhanced checkpoint manager for Stella Alpha pipeline.
    
    PRIMARY KEY: (tp_pips, sl_pips, max_holding_bars)
    
    Features:
    - Saves completed configs to SQLite
    - Stores trained models for multi-model export
    - Stores per-fold results
    - Allows resuming interrupted runs
    - Thread-safe for parallel workers (WAL mode)
    - Tier classification support
    """
    
    def __init__(self, db_path: str = "artifacts/pure_ml_stella_alpha.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._completed_params_cache: Optional[Set[Tuple]] = None
        self._cache_lock = threading.Lock()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                isolation_level=None
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-16000")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Create tables with enhanced schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        
        # Main results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS completed (
                -- Primary key (config identifier)
                tp_pips INTEGER NOT NULL,
                sl_pips INTEGER NOT NULL,
                max_holding_bars INTEGER NOT NULL,
                config_id TEXT,
                
                -- Status
                status TEXT NOT NULL,
                tier INTEGER,
                tier_name TEXT,
                
                -- Metrics (aggregated across folds)
                ev_mean REAL,
                ev_std REAL,
                precision_mean REAL,
                precision_std REAL,
                precision_cv REAL,
                recall_mean REAL,
                f1_mean REAL,
                auc_pr_mean REAL,
                total_trades INTEGER,
                
                -- Feature selection
                selected_features TEXT,
                n_features INTEGER,
                consensus_threshold REAL,
                
                -- Classification
                classification TEXT,
                
                -- Rejection info (if failed)
                rejection_reasons TEXT,
                
                -- Risk/Reward (NEW)
                risk_reward_ratio REAL,
                min_winrate_required REAL,
                edge_above_breakeven REAL,
                
                -- Execution metadata
                execution_time REAL,
                timestamp TEXT,
                
                PRIMARY KEY (tp_pips, sl_pips, max_holding_bars)
            )
        """)
        
        # Models table (for multi-model export)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT NOT NULL,
                fold INTEGER NOT NULL,
                model_blob BLOB NOT NULL,
                threshold REAL,
                precision REAL,
                ev REAL,
                saved_at TEXT,
                
                UNIQUE(config_id, fold)
            )
        """)
        
        # Per-fold results (for detailed analysis)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fold_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT NOT NULL,
                fold INTEGER NOT NULL,
                
                -- Fold-specific metrics
                precision REAL,
                recall REAL,
                f1 REAL,
                ev REAL,
                auc_pr REAL,
                threshold REAL,
                n_trades INTEGER,
                
                -- Fold date range
                train_start TEXT,
                train_end TEXT,
                test_start TEXT,
                test_end TEXT,
                
                UNIQUE(config_id, fold)
            )
        """)
        
        # Indexes for fast queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON completed(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON completed(tier)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ev ON completed(ev_mean DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_classification ON completed(classification)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rr ON completed(risk_reward_ratio DESC)")
        
        conn.commit()
        conn.close()
    
    def _get_completed_params(self) -> Set[Tuple]:
        """Get all completed parameter combinations as a set for fast lookup."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT tp_pips, sl_pips, max_holding_bars 
            FROM completed
        """)
        return {(row[0], row[1], row[2]) for row in cursor.fetchall()}
    
    def is_params_completed(self, tp: int, sl: int, hold: int) -> bool:
        """Check if this parameter combination was already tested."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT 1 FROM completed 
            WHERE tp_pips = ? AND sl_pips = ? AND max_holding_bars = ?
            LIMIT 1
        """, (tp, sl, hold))
        return cursor.fetchone() is not None
    
    def get_pending_configs(self, all_configs: List) -> List:
        """
        Get configs that need processing.
        Checks by PARAMETERS (tp, sl, hold).
        """
        if not all_configs:
            return []
        
        with self._cache_lock:
            if self._completed_params_cache is None:
                self._completed_params_cache = self._get_completed_params()
            completed_params = self._completed_params_cache
        
        pending = []
        for config in all_configs:
            param_tuple = (
                config.tp_pips if hasattr(config, 'tp_pips') else config[0],
                config.sl_pips if hasattr(config, 'sl_pips') else config[1],
                config.max_holding_bars if hasattr(config, 'max_holding_bars') else config[2]
            )
            if param_tuple not in completed_params:
                pending.append(config)
        
        return pending
    
    def mark_completed(
        self,
        config_id: str,
        status: str,
        tp_pips: int,
        sl_pips: int,
        max_holding_bars: int,
        tier: int = None,
        tier_name: str = None,
        ev_mean: float = None,
        ev_std: float = None,
        precision_mean: float = None,
        precision_std: float = None,
        recall_mean: float = None,
        f1_mean: float = None,
        auc_pr_mean: float = None,
        total_trades: int = None,
        selected_features: List[str] = None,
        consensus_threshold: float = None,
        classification: str = None,
        rejection_reasons: List[str] = None,
        risk_reward_ratio: float = None,
        min_winrate_required: float = None,
        edge_above_breakeven: float = None,
        execution_time: float = None
    ):
        """
        Mark a config as completed with all metrics.
        Uses INSERT OR REPLACE to handle duplicates.
        """
        conn = self._get_conn()
        
        features_json = json.dumps(selected_features) if selected_features else None
        reasons_json = json.dumps(rejection_reasons) if rejection_reasons else None
        n_features = len(selected_features) if selected_features else 0
        
        # Calculate precision CV
        precision_cv = None
        if precision_mean and precision_std and precision_mean > 0:
            precision_cv = precision_std / precision_mean
        
        conn.execute("""
            INSERT OR REPLACE INTO completed (
                tp_pips, sl_pips, max_holding_bars,
                config_id, status, tier, tier_name,
                ev_mean, ev_std, precision_mean, precision_std, precision_cv,
                recall_mean, f1_mean, auc_pr_mean, total_trades,
                selected_features, n_features, consensus_threshold,
                classification, rejection_reasons,
                risk_reward_ratio, min_winrate_required, edge_above_breakeven,
                execution_time, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tp_pips, sl_pips, max_holding_bars,
            config_id, status, tier, tier_name,
            ev_mean, ev_std, precision_mean, precision_std, precision_cv,
            recall_mean, f1_mean, auc_pr_mean, total_trades,
            features_json, n_features, consensus_threshold,
            classification, reasons_json,
            risk_reward_ratio, min_winrate_required, edge_above_breakeven,
            execution_time, datetime.now().isoformat()
        ))
        
        # Invalidate cache
        with self._cache_lock:
            self._completed_params_cache = None
    
    def save_model(
        self,
        config_id: str,
        fold: int,
        model,
        threshold: float,
        precision: float,
        ev: float
    ):
        """Save trained model to database."""
        conn = self._get_conn()
        model_bytes = pickle.dumps(model)
        
        conn.execute("""
            INSERT OR REPLACE INTO models 
            (config_id, fold, model_blob, threshold, precision, ev, saved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            config_id, fold, model_bytes, threshold, precision, ev,
            datetime.now().isoformat()
        ))
    
    def load_model(self, config_id: str, fold: Optional[int] = None):
        """
        Load model from database.
        
        Args:
            config_id: Config identifier
            fold: Specific fold, or None for best fold
        """
        conn = self._get_conn()
        
        if fold is not None:
            cursor = conn.execute(
                "SELECT model_blob FROM models WHERE config_id = ? AND fold = ?",
                (config_id, fold)
            )
        else:
            # Get model with best precision
            cursor = conn.execute(
                "SELECT model_blob FROM models WHERE config_id = ? ORDER BY precision DESC LIMIT 1",
                (config_id,)
            )
        
        result = cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        return None
    
    def save_fold_result(
        self,
        config_id: str,
        fold: int,
        precision: float,
        recall: float,
        f1: float,
        ev: float,
        auc_pr: float,
        threshold: float,
        n_trades: int,
        train_start: str = None,
        train_end: str = None,
        test_start: str = None,
        test_end: str = None
    ):
        """Save per-fold results."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO fold_results (
                config_id, fold, precision, recall, f1, ev, auc_pr, 
                threshold, n_trades, train_start, train_end, test_start, test_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config_id, fold, precision, recall, f1, ev, auc_pr,
            threshold, n_trades, train_start, train_end, test_start, test_end
        ))
    
    def get_all_results(self) -> List[Dict]:
        """Get all completed results."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM completed ORDER BY ev_mean DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_passed_results(self) -> List[Dict]:
        """Get only passing results."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM completed 
            WHERE status = 'passed' OR status = 'PASSED'
            ORDER BY ev_mean DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_tier_results(self, tier: int) -> List[Dict]:
        """Get results for a specific tier."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM completed 
            WHERE tier = ? AND (status = 'passed' OR status = 'PASSED')
            ORDER BY ev_mean DESC
        """, (tier,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_gold_configs(self) -> List[Dict]:
        """Get GOLD classified configs."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM completed 
            WHERE classification = 'GOLD'
            ORDER BY ev_mean DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_progress_stats(self, total_configs: int = None) -> Dict[str, Any]:
        """Get progress statistics."""
        conn = self._get_conn()
        
        total = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        passed = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE status = 'passed' OR status = 'PASSED'"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE status != 'passed' AND status != 'PASSED'"
        ).fetchone()[0]
        
        gold = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE classification = 'GOLD'"
        ).fetchone()[0]
        silver = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE classification = 'SILVER'"
        ).fetchone()[0]
        bronze = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE classification = 'BRONZE'"
        ).fetchone()[0]
        
        tier_0 = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE tier = 0 AND (status = 'passed' OR status = 'PASSED')"
        ).fetchone()[0]
        tier_1 = conn.execute(
            "SELECT COUNT(*) FROM completed WHERE tier = 1 AND (status = 'passed' OR status = 'PASSED')"
        ).fetchone()[0]
        
        best = conn.execute("""
            SELECT config_id, ev_mean, tp_pips, sl_pips, max_holding_bars, tier
            FROM completed 
            WHERE ev_mean IS NOT NULL 
            ORDER BY ev_mean DESC LIMIT 1
        """).fetchone()
        
        db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        
        return {
            'total_configs': total_configs or total,
            'completed': total,
            'pending': (total_configs - total) if total_configs else 0,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'gold': gold,
            'silver': silver,
            'bronze': bronze,
            'tier_0_runners': tier_0,
            'tier_1_ideal': tier_1,
            'best_config': dict(best) if best else None,
            'database_size_mb': db_size
        }
    
    def count_pending(self, all_configs: List) -> Tuple[int, int, int]:
        """
        Count how many configs are pending vs already completed.
        Returns: (total, completed, pending)
        """
        with self._cache_lock:
            if self._completed_params_cache is None:
                self._completed_params_cache = self._get_completed_params()
            completed_params = self._completed_params_cache
        
        completed_count = 0
        pending_count = 0
        
        for config in all_configs:
            param_tuple = (
                config.tp_pips if hasattr(config, 'tp_pips') else config[0],
                config.sl_pips if hasattr(config, 'sl_pips') else config[1],
                config.max_holding_bars if hasattr(config, 'max_holding_bars') else config[2]
            )
            if param_tuple in completed_params:
                completed_count += 1
            else:
                pending_count += 1
        
        return len(all_configs), completed_count, pending_count
    
    def get_best_configs(self, limit: int = 10) -> List[Dict]:
        """Get top N configs by EV."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM completed 
            WHERE (status = 'passed' OR status = 'PASSED') AND ev_mean IS NOT NULL
            ORDER BY ev_mean DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def clear_all(self):
        """Clear all checkpoint data (use with caution!)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM completed")
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM fold_results")
        with self._cache_lock:
            self._completed_params_cache = None
        print(f"Cleared all checkpoints from {self.db_path}")
    
    def export_to_csv(self, filepath: str) -> None:
        """Export all results to CSV."""
        import csv
        
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT 
                config_id, tp_pips, sl_pips, max_holding_bars,
                status, tier, tier_name, classification,
                ev_mean, ev_std, precision_mean, precision_std, 
                total_trades, consensus_threshold, n_features,
                risk_reward_ratio, min_winrate_required, edge_above_breakeven,
                selected_features
            FROM completed 
            ORDER BY ev_mean DESC
        """)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'config_id', 'tp_pips', 'sl_pips', 'max_holding_bars',
                'status', 'tier', 'tier_name', 'classification',
                'ev_mean', 'ev_std', 'precision_mean', 'precision_std',
                'total_trades', 'consensus_threshold', 'n_features',
                'risk_reward_ratio', 'min_winrate_required', 'edge_above_breakeven',
                'selected_features'
            ])
            for row in cursor.fetchall():
                writer.writerow(row)
        
        print(f"Exported to {filepath}")
    
    def close(self):
        """Close thread-local connection."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn


def print_resume_status(stats: Dict[str, Any], total_configs: int):
    """Print formatted resume status."""
    print("\n" + "=" * 60)
    print("  RESUMING FROM CHECKPOINT")
    print("=" * 60)
    print(f"""
    Completed: {stats['completed']}/{total_configs} ({100*stats['completed']/total_configs:.1f}%)
    Passed: {stats['passed']} | Failed: {stats['failed']}
    
    Classifications:
      🥇 GOLD:   {stats['gold']}
      🥈 SILVER: {stats['silver']}
      🥉 BRONZE: {stats['bronze']}
    
    Tiers:
      🚀 Tier 0 (Runners): {stats['tier_0_runners']}
      ⭐ Tier 1 (Ideal):   {stats['tier_1_ideal']}
    
    Remaining: {stats['pending']} configs
    """)
    print("=" * 60)


if __name__ == '__main__':
    # Test the checkpoint manager
    print("Testing StellaAlphaCheckpointManager...")
    
    db = StellaAlphaCheckpointManager("test_stella_alpha.db")
    
    # Test marking completed
    db.mark_completed(
        config_id="TP100_SL30_H48",
        status="passed",
        tp_pips=100,
        sl_pips=30,
        max_holding_bars=48,
        tier=0,
        tier_name="TIER 0 - RUNNER",
        ev_mean=15.2,
        ev_std=2.1,
        precision_mean=0.38,
        precision_std=0.05,
        risk_reward_ratio=3.33,
        min_winrate_required=0.231,
        edge_above_breakeven=0.149,
        classification="GOLD"
    )
    
    # Test retrieval
    results = db.get_all_results()
    print(f"Results stored: {len(results)}")
    
    stats = db.get_progress_stats(66)
    print(f"Stats: {stats}")
    
    # Cleanup test file
    import os
    os.remove("test_stella_alpha.db")
    print("✅ Checkpoint manager test passed")
