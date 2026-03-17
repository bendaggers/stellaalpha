"""
Trade Recorder - Stella Alpha

Records every trade decision (win/loss) with full feature context
into trades_stella_alpha.db for downstream loss analysis.

DESIGN GOALS:
- Thread-safe: multiple parallel workers can write concurrently
- Best-effort: failures never block the experiment pipeline
- Rich context: H4 + D1 + MTF features captured at entry
- Batch-friendly: supports single and batch inserts

Usage (from experiment.py):
    recorder = TradeRecorder("artifacts/trades_stella_alpha.db")
    recorder.insert_trade({...})
    recorder.insert_trades_batch([{...}, {...}])
"""

import sqlite3
import threading
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


# =============================================================================
# TRADE RECORD DATACLASS
# =============================================================================

@dataclass
class TradeRecord:
    """
    Complete record of a single trade decision.
    All feature fields are Optional — missing features are stored as NULL.
    """
    # Identification
    config_id: str
    fold: int
    timestamp: str

    # Outcome
    outcome: str            # "WIN" or "LOSS"
    pips_result: float      # +tp_pips or -sl_pips
    model_probability: float

    # Optional meta
    threshold_used: Optional[float] = None
    bars_held: Optional[int] = None

    # H4 features
    h4_rsi_value: Optional[float] = None
    h4_bb_position: Optional[float] = None
    h4_trend_strength: Optional[float] = None
    h4_atr_pct: Optional[float] = None
    h4_volume_ratio: Optional[float] = None
    h4_hour: Optional[float] = None
    h4_day_of_week: Optional[int] = None
    h4_is_asian_session: Optional[int] = None
    h4_is_london_session: Optional[int] = None
    h4_is_ny_session: Optional[int] = None
    h4_is_overlap_session: Optional[int] = None
    h4_consecutive_bullish: Optional[int] = None
    h4_exhaustion_score: Optional[float] = None
    h4_macd_histogram: Optional[float] = None
    h4_adx: Optional[float] = None

    # D1 features
    d1_rsi_value: Optional[float] = None
    d1_bb_position: Optional[float] = None
    d1_trend_strength: Optional[float] = None
    d1_trend_direction: Optional[float] = None
    d1_is_trending_up: Optional[int] = None
    d1_is_trending_down: Optional[int] = None
    d1_atr_percentile: Optional[float] = None
    d1_consecutive_bullish: Optional[int] = None

    # Cross-TF features
    mtf_confluence_score: Optional[float] = None
    mtf_rsi_aligned: Optional[int] = None
    mtf_bb_aligned: Optional[int] = None
    mtf_trend_aligned: Optional[int] = None
    d1_supports_short: Optional[int] = None
    d1_opposes_short: Optional[int] = None
    mtf_strong_short_setup: Optional[int] = None
    h4_vs_d1_rsi: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# TRADE RECORDER
# =============================================================================

class TradeRecorder:
    """
    Thread-safe SQLite-backed trade recorder.

    Each parallel worker gets its own connection via threading.local().
    WAL mode enables concurrent writes without blocking.
    """

    # Ordered list of all columns (matches CREATE TABLE below)
    _COLUMNS = [
        "config_id", "fold", "timestamp", "outcome", "pips_result",
        "model_probability", "threshold_used", "bars_held",
        "h4_rsi_value", "h4_bb_position", "h4_trend_strength",
        "h4_atr_pct", "h4_volume_ratio", "h4_hour", "h4_day_of_week",
        "h4_is_asian_session", "h4_is_london_session", "h4_is_ny_session",
        "h4_is_overlap_session", "h4_consecutive_bullish",
        "h4_exhaustion_score", "h4_macd_histogram", "h4_adx",
        "d1_rsi_value", "d1_bb_position", "d1_trend_strength",
        "d1_trend_direction", "d1_is_trending_up", "d1_is_trending_down",
        "d1_atr_percentile", "d1_consecutive_bullish",
        "mtf_confluence_score", "mtf_rsi_aligned", "mtf_bb_aligned",
        "mtf_trend_aligned", "d1_supports_short", "d1_opposes_short",
        "mtf_strong_short_setup", "h4_vs_d1_rsi",
    ]

    def __init__(self, db_path: str = "artifacts/trades_stella_alpha.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ──────────────────────────────────────────────────────────────────────
    # DB initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create tables and indexes (idempotent)."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id            INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Identification
                config_id           TEXT    NOT NULL,
                fold                INTEGER NOT NULL,
                timestamp           TEXT    NOT NULL,

                -- Outcome
                outcome             TEXT    NOT NULL,
                pips_result         REAL    NOT NULL,
                model_probability   REAL    NOT NULL,
                threshold_used      REAL,
                bars_held           INTEGER,

                -- H4 features at entry
                h4_rsi_value        REAL,
                h4_bb_position      REAL,
                h4_trend_strength   REAL,
                h4_atr_pct          REAL,
                h4_volume_ratio     REAL,
                h4_hour             REAL,
                h4_day_of_week      INTEGER,
                h4_is_asian_session  INTEGER,
                h4_is_london_session INTEGER,
                h4_is_ny_session     INTEGER,
                h4_is_overlap_session INTEGER,
                h4_consecutive_bullish INTEGER,
                h4_exhaustion_score REAL,
                h4_macd_histogram   REAL,
                h4_adx              REAL,

                -- D1 features at entry
                d1_rsi_value        REAL,
                d1_bb_position      REAL,
                d1_trend_strength   REAL,
                d1_trend_direction  REAL,
                d1_is_trending_up   INTEGER,
                d1_is_trending_down INTEGER,
                d1_atr_percentile   REAL,
                d1_consecutive_bullish INTEGER,

                -- Cross-TF features
                mtf_confluence_score REAL,
                mtf_rsi_aligned      INTEGER,
                mtf_bb_aligned       INTEGER,
                mtf_trend_aligned    INTEGER,
                d1_supports_short    INTEGER,
                d1_opposes_short     INTEGER,
                mtf_strong_short_setup INTEGER,
                h4_vs_d1_rsi         REAL,

                created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome    ON trades(outcome)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_config     ON trades(config_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fold       ON trades(fold)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_probability ON trades(model_probability)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_config_outcome ON trades(config_id, outcome)")

        conn.commit()
        conn.close()

    # ──────────────────────────────────────────────────────────────────────
    # Thread-local connection
    # ──────────────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def insert_trade(self, record: Dict[str, Any]) -> bool:
        """
        Insert a single trade record.

        Accepts a plain dict — keys should match column names.
        Unknown keys are silently ignored. Missing keys → NULL.

        Returns True on success, False on failure.
        """
        try:
            row = self._dict_to_row(record)
            placeholders = ", ".join(["?"] * len(self._COLUMNS))
            cols = ", ".join(self._COLUMNS)
            sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
            conn = self._get_conn()
            conn.execute(sql, row)
            conn.commit()
            return True
        except Exception:
            return False

    def insert_trades_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Insert multiple trade records in a single transaction.

        Returns the number of successfully inserted records.
        """
        if not records:
            return 0
        try:
            rows = [self._dict_to_row(r) for r in records]
            placeholders = ", ".join(["?"] * len(self._COLUMNS))
            cols = ", ".join(self._COLUMNS)
            sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
            conn = self._get_conn()
            conn.executemany(sql, rows)
            conn.commit()
            return len(rows)
        except Exception:
            return 0

    def get_trade_count(
        self,
        config_id: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> int:
        """Return count of trades, optionally filtered."""
        try:
            sql = "SELECT COUNT(*) FROM trades WHERE 1=1"
            params: List[Any] = []
            if config_id:
                sql += " AND config_id = ?"
                params.append(config_id)
            if outcome:
                sql += " AND outcome = ?"
                params.append(outcome)
            conn = self._get_conn()
            cursor = conn.execute(sql, params)
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def get_summary(self) -> Dict[str, Any]:
        """Return a quick summary of the trades database."""
        try:
            conn = self._get_conn()
            total    = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            wins     = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome='WIN'").fetchone()[0]
            losses   = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome='LOSS'").fetchone()[0]
            configs  = conn.execute("SELECT COUNT(DISTINCT config_id) FROM trades").fetchone()[0]
            pips_sum = conn.execute("SELECT SUM(pips_result) FROM trades").fetchone()[0] or 0.0
            return {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total if total > 0 else 0.0,
                "distinct_configs": configs,
                "total_pips": float(pips_sum),
            }
        except Exception:
            return {}

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _dict_to_row(self, record: Dict[str, Any]) -> tuple:
        """Convert a dict to an ordered tuple matching _COLUMNS."""
        return tuple(
            self._safe_value(record.get(col))
            for col in self._COLUMNS
        )

    @staticmethod
    def _safe_value(v: Any) -> Any:
        """Coerce numpy types to Python scalars; keep None as-is."""
        if v is None:
            return None
        try:
            import numpy as np
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.bool_,)):
                return int(v)
        except ImportError:
            pass
        return v


# =============================================================================
# CONVENIENCE FUNCTION (used by experiment.py hook)
# =============================================================================

def build_trade_record(
    row: Any,                  # pandas Series (single dataframe row)
    outcome: str,              # "WIN" or "LOSS"
    pips_result: float,
    model_probability: float,
    config_id: str,
    fold: int,
    threshold_used: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a trade record dict from a dataframe row.

    Used by experiment.py's _record_trades_for_result() hook.
    All feature lookups use .get() so missing columns → None.
    """

    def g(name: str, default=None):
        """Safe getter for both Series and dict-like objects."""
        try:
            val = row.get(name, default) if hasattr(row, 'get') else getattr(row, name, default)
            if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'float' and val != val):
                return default
            return val
        except Exception:
            return default

    return {
        "config_id":            config_id,
        "fold":                 int(fold),
        "timestamp":            str(g("timestamp", "")),
        "outcome":              outcome,
        "pips_result":          float(pips_result),
        "model_probability":    float(model_probability),
        "threshold_used":       float(threshold_used) if threshold_used is not None else None,
        # H4
        "h4_rsi_value":         g("rsi_value"),
        "h4_bb_position":       g("bb_position"),
        "h4_trend_strength":    g("trend_strength"),
        "h4_atr_pct":           g("atr_pct"),
        "h4_volume_ratio":      g("volume_ratio"),
        "h4_hour":              g("hour_sin") or g("hour"),
        "h4_day_of_week":       g("day_of_week"),
        "h4_is_asian_session":  g("is_asian") or g("is_asian_session"),
        "h4_is_london_session": g("is_london") or g("is_london_session"),
        "h4_is_ny_session":     g("is_ny") or g("is_ny_session"),
        "h4_is_overlap_session":g("is_overlap") or g("is_overlap_session"),
        "h4_consecutive_bullish": g("consecutive_bullish"),
        "h4_exhaustion_score":  g("exhaustion_score"),
        "h4_macd_histogram":    g("macd_histogram"),
        "h4_adx":               g("adx"),
        # D1
        "d1_rsi_value":         g("d1_rsi_value"),
        "d1_bb_position":       g("d1_bb_position"),
        "d1_trend_strength":    g("d1_trend_strength"),
        "d1_trend_direction":   g("d1_trend_direction"),
        "d1_is_trending_up":    g("d1_is_trending_up"),
        "d1_is_trending_down":  g("d1_is_trending_down"),
        "d1_atr_percentile":    g("d1_atr_percentile"),
        "d1_consecutive_bullish": g("d1_consecutive_bullish"),
        # MTF
        "mtf_confluence_score": g("mtf_confluence_score"),
        "mtf_rsi_aligned":      g("mtf_rsi_aligned"),
        "mtf_bb_aligned":       g("mtf_bb_aligned"),
        "mtf_trend_aligned":    g("mtf_trend_aligned"),
        "d1_supports_short":    g("d1_supports_short"),
        "d1_opposes_short":     g("d1_opposes_short"),
        "mtf_strong_short_setup": g("mtf_strong_short_setup"),
        "h4_vs_d1_rsi":         g("h4_vs_d1_rsi"),
    }
