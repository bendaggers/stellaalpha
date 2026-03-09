#!/usr/bin/env python3
"""
recommend_filters.py — Stella Alpha Filter Recommendation CLI
==============================================================

Generates and ranks filter rules that could improve net P&L by
avoiding losing trades, based on the recorded trade history.

Reads from:  artifacts/trades_stella_alpha.db
Writes to:   artifacts/filter_recommendations.json  (optional)

How it works:
  1. Load all trades (or a specific config) from the database
  2. Build ~50 candidate filter rules (session, confidence, D1 alignment,
     RSI thresholds, MTF confluence, BB position, ADX, volume, exhaustion)
  3. Simulate each filter: count wins removed vs losses removed
  4. Rank by net pip improvement:  pips_saved(losses×SL) − pips_lost(wins×TP)
  5. Test combinations of top filters
  6. Print results + optionally save to JSON

Usage examples:
    # Minimal — reads best config TP/SL from trading_config_stella_alpha.json
    python recommend_filters.py

    # Explicit TP/SL
    python recommend_filters.py --tp 100 --sl 30

    # Specific config only
    python recommend_filters.py --tp 100 --sl 30 --config TP100_SL30_H48

    # Test combinations of top 5 single filters
    python recommend_filters.py --tp 100 --sl 30 --combinations --top-singles 5

    # Save to JSON
    python recommend_filters.py --tp 100 --sl 30 --output artifacts/filters.json

    # Full run with all options
    python recommend_filters.py \\
        --trades artifacts/trades_stella_alpha.db \\
        --tp 100 --sl 30 \\
        --top-n 25 \\
        --combinations \\
        --output artifacts/filter_recommendations.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR    = SCRIPT_DIR / "src"
for p in (str(SCRIPT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  pip install pandas")
    sys.exit(1)

try:
    from filter_recommendations import FilterRecommendationEngine, FilterRule
except ImportError:
    print("ERROR: filter_recommendations.py not found.")
    print("Expected location: src/filter_recommendations.py  OR  same directory as this script.")
    sys.exit(1)


# =============================================================================
# HELPERS
# =============================================================================

def _load_best_config(artifacts_dir: str) -> dict:
    """Try to read TP/SL from trading_config_stella_alpha.json."""
    cfg_path = Path(artifacts_dir) / "trading_config_stella_alpha.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data = json.load(f)
            bc = data.get("best_config", {})
            return {
                "tp_pips": int(bc.get("tp_pips", 0)),
                "sl_pips": int(bc.get("sl_pips", 0)),
                "config_id": bc.get("config_id", None),
            }
        except Exception:
            pass
    return {}


def _load_trades(db_path: str, config_id: str | None) -> pd.DataFrame:
    """Load trades DataFrame from SQLite, optionally filtered by config."""
    conn = sqlite3.connect(db_path)
    if config_id:
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE config_id = ?", conn, params=(config_id,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df


def _print_header(tp: int, sl: int, n_trades: int, n_wins: int, n_losses: int, config_id: str | None):
    wr = 100.0 * n_wins / n_trades if n_trades > 0 else 0.0
    rr = tp / max(sl, 1)
    breakeven = 100.0 * sl / max(tp + sl, 1)
    print()
    print("=" * 70)
    print("  STELLA ALPHA — FILTER RECOMMENDATIONS")
    print("=" * 70)
    print(f"  Config     : {config_id or 'ALL combined'}")
    print(f"  TP / SL    : {tp} pips / {sl} pips  (R:R {rr:.2f}:1)")
    print(f"  Breakeven  : {breakeven:.1f}% win rate")
    print(f"  Trades     : {n_trades:,}  |  Wins: {n_wins:,}  |  Losses: {n_losses:,}")
    print(f"  Win rate   : {wr:.1f}%")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stella Alpha — Filter Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recommend_filters.py --tp 100 --sl 30
  python recommend_filters.py --tp 100 --sl 30 --config TP100_SL30_H48
  python recommend_filters.py --tp 100 --sl 30 --combinations --top-singles 5
  python recommend_filters.py --tp 100 --sl 30 --output artifacts/filter_recommendations.json
""",
    )

    p.add_argument(
        "--trades", "-t",
        default="artifacts/trades_stella_alpha.db",
        help="Path to trades database",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Take-profit pips for P&L calculation (auto-read from trading_config if omitted)",
    )
    p.add_argument(
        "--sl",
        type=int,
        default=30,
        help="Stop-loss pips (default: 30)",
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Limit analysis to a specific config_id (e.g. TP100_SL30_H48)",
    )
    p.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Artifacts directory (used to auto-read TP from trading_config if --tp not given)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top single-filter recommendations to show (default: 20)",
    )
    p.add_argument(
        "--min-net",
        type=float,
        default=0.0,
        help="Minimum net pip improvement for a filter to be included (default: 0)",
    )
    p.add_argument(
        "--combinations",
        action="store_true",
        help="Also test pairs/triples of top single filters",
    )
    p.add_argument(
        "--top-singles",
        type=int,
        default=5,
        help="How many top singles to combine (default: 5, used with --combinations)",
    )
    p.add_argument(
        "--max-combo-size",
        type=int,
        default=3,
        help="Max filter combination size (default: 3)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save recommendations to this JSON path",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validate database ────────────────────────────────────────────────
    db_path = args.trades
    if not Path(db_path).exists():
        print(f"ERROR: Trades database not found: {db_path}")
        print("Run the main pipeline first to generate trade records.")
        sys.exit(1)

    # ── Resolve TP ───────────────────────────────────────────────────────
    tp = args.tp
    sl = args.sl
    auto_config_id = None

    if tp is None:
        best = _load_best_config(args.artifacts_dir)
        if best.get("tp_pips"):
            tp            = best["tp_pips"]
            sl            = best.get("sl_pips", sl)
            auto_config_id = best.get("config_id")
            print(f"Auto-detected TP={tp} SL={sl} from trading_config_stella_alpha.json")
        else:
            print("ERROR: --tp is required (could not auto-read from trading_config_stella_alpha.json).")
            print("Example: python recommend_filters.py --tp 100 --sl 30")
            sys.exit(1)

    config_id = args.config or auto_config_id

    # ── Load trades ──────────────────────────────────────────────────────
    print(f"\nLoading trades from: {db_path}")
    try:
        trades_df = _load_trades(db_path, config_id)
    except Exception as e:
        print(f"ERROR loading trades: {e}")
        sys.exit(1)

    if len(trades_df) == 0:
        print(f"ERROR: No trades found{f' for config {config_id!r}' if config_id else ''}.")
        sys.exit(1)

    n_trades = len(trades_df)
    n_wins   = int((trades_df["outcome"] == "WIN").sum())
    n_losses = int((trades_df["outcome"] == "LOSS").sum())

    _print_header(tp, sl, n_trades, n_wins, n_losses, config_id)

    # ── Generate recommendations ─────────────────────────────────────────
    engine = FilterRecommendationEngine(
        trades_df,
        config={"tp_pips": tp, "sl_pips": sl},
    )

    print(f"\nGenerating filter recommendations…")
    recs = engine.generate_recommendations(
        top_n=args.top_n,
        min_net_improvement=args.min_net,
    )

    combos = None
    if args.combinations and recs:
        print(f"Testing combinations of top {args.top_singles} filters…")
        combos = engine.test_top_combinations(
            recs,
            max_filters=args.max_combo_size,
            top_singles=args.top_singles,
        )

    # ── Print ────────────────────────────────────────────────────────────
    engine.print_recommendations(recs, combos)

    if not recs:
        print()
        print("  No filters showed positive net improvement for this TP/SL config.")
        print(f"  (TP={tp}, SL={sl}: each avoided loss saves {sl} pips,")
        print(f"   each missed win costs {tp} pips — filters need high loss concentration)")
        print()

    # ── Save ─────────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        engine.save_recommendations(recs, str(out), combos)
        print(f"\nSaved → {out}")

    print()


if __name__ == "__main__":
    main()
