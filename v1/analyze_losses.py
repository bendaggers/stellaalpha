#!/usr/bin/env python3
"""
analyze_losses.py — Stella Alpha Loss Analysis CLI
====================================================

Standalone tool to analyze patterns in winning vs losing trades
after the main pipeline has completed.

Reads from:   artifacts/trades_stella_alpha.db  (written by experiment.py)
Writes to:    artifacts/loss_analysis_stella_alpha.json  (optional)

Usage examples:
    # Analyze all configs combined
    python analyze_losses.py

    # Analyze one specific config
    python analyze_losses.py --config TP100_SL30_H48

    # Show only top-N loss predictors
    python analyze_losses.py --top 15

    # Save report to JSON
    python analyze_losses.py --output artifacts/loss_analysis.json

    # List all config IDs in the database, then exit
    python analyze_losses.py --list-configs

    # Custom trades DB path
    python analyze_losses.py --trades path/to/trades.db --config TP50_SL30_H36

    # Full run: all configs + JSON output + show filter hints
    python analyze_losses.py --all --output artifacts/loss_analysis_stella_alpha.json --filters
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths so the script works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR    = SCRIPT_DIR / "src"
for p in (str(SCRIPT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from loss_analysis import LossAnalyzer
except ImportError:
    print("ERROR: loss_analysis.py not found.")
    print("Expected location: src/loss_analysis.py  OR  same directory as this script.")
    sys.exit(1)


# =============================================================================
# HELPERS
# =============================================================================

def list_configs(db_path: str) -> None:
    """Print all distinct config_ids in the trades database."""
    if not Path(db_path).exists():
        print(f"ERROR: Trades database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT config_id, COUNT(*) as n, "
        "SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins "
        "FROM trades GROUP BY config_id ORDER BY config_id"
    ).fetchall()
    conn.close()

    if not rows:
        print("No trades found in database.")
        return

    print(f"\n  {'Config ID':<28} {'Trades':>8} {'Wins':>8} {'Win%':>7}")
    print(f"  {'─'*55}")
    for config_id, n, wins in rows:
        wr = 100.0 * wins / n if n > 0 else 0.0
        print(f"  {config_id:<28} {n:>8,} {wins:>8,} {wr:>6.1f}%")
    print(f"\n  Total configs: {len(rows)}")


def db_trade_count(db_path: str) -> int:
    """Return total number of trades in database."""
    try:
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def print_filter_hints(report) -> None:
    """Print a short actionable filter hint section after the report."""
    print("\n" + "=" * 70)
    print("  QUICK FILTER HINTS")
    print("=" * 70)

    if report.optimal_confidence_threshold > 0.50:
        print(f"\n  📌 Confidence filter:")
        print(f"     Require model_probability >= {report.optimal_confidence_threshold:.2f}")
        print(f"     (Current: all trades ≥ 0.50)")

    if report.problematic_sessions:
        print(f"\n  📌 Session filter:")
        for s in report.problematic_sessions:
            col = s.lower().replace(" ", "_")
            print(f"     Avoid {s} session  →  filter: h4_is_{col}_session == 0")

    if report.top_loss_predictors:
        print(f"\n  📌 Top loss-predicting features (consider threshold filters on these):")
        for i, feat in enumerate(report.top_loss_predictors[:5], 1):
            cmp = report.feature_comparisons
            matched = [c for c in cmp if c.feature_name == feat]
            if matched:
                c = matched[0]
                direction = "higher" if c.direction == "higher_in_losses" else "lower"
                print(f"     {i}. {feat:<35}  losses have {direction} values "
                      f"(Δ={c.pct_difference:+.1f}%, p={c.p_value:.4f})")
            else:
                print(f"     {i}. {feat}")

    print()
    print("  Run  recommend_filters.py  for full P&L-simulated filter ranking.")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stella Alpha — Loss Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_losses.py
  python analyze_losses.py --config TP100_SL30_H48
  python analyze_losses.py --all --output artifacts/loss_analysis.json
  python analyze_losses.py --list-configs
  python analyze_losses.py --trades custom/path/trades.db --top 20 --filters
""",
    )

    p.add_argument(
        "--trades", "-t",
        default="artifacts/trades_stella_alpha.db",
        help="Path to trades database (default: artifacts/trades_stella_alpha.db)",
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Specific config_id to analyze (e.g. TP100_SL30_H48)",
    )
    p.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze ALL configs combined (default if --config not given)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save report as JSON to this path",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of significant features to show (default: 20)",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Min trades required per outcome to compute feature comparison (default: 30)",
    )
    p.add_argument(
        "--list-configs",
        action="store_true",
        help="List all config_ids in the database and exit",
    )
    p.add_argument(
        "--filters",
        action="store_true",
        help="Print quick filter hints after the report",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    db_path = args.trades

    # ── list-configs shortcut ────────────────────────────────────────────
    if args.list_configs:
        list_configs(db_path)
        return

    # ── validate DB ──────────────────────────────────────────────────────
    if not Path(db_path).exists():
        print(f"ERROR: Trades database not found: {db_path}")
        print()
        print("Run the main pipeline first:")
        print("  python run_pure_ml_stella_alpha.py ...")
        print()
        print("Or check the path with: --trades <path>")
        sys.exit(1)

    n_trades = db_trade_count(db_path)
    if n_trades == 0:
        print(f"ERROR: Trades database is empty: {db_path}")
        print("Loss analysis requires recorded trades (loss_analysis.enabled must be true in config).")
        sys.exit(1)

    print(f"\nTrades database : {db_path}")
    print(f"Total trades    : {n_trades:,}")

    # ── run analysis ─────────────────────────────────────────────────────
    config_id = args.config if args.config else None
    label     = config_id or "ALL configs combined"

    print(f"Analyzing       : {label}\n")

    try:
        analyzer = LossAnalyzer(db_path)
        analyzer.load_trades(config_id=config_id)
        report   = analyzer.analyze(config_id=config_id, min_samples=args.min_samples)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        if "No trades found" in str(e):
            print()
            print("Available configs:")
            list_configs(db_path)
        sys.exit(1)

    # ── print report ─────────────────────────────────────────────────────
    analyzer.print_report(report)

    # ── filter hints ─────────────────────────────────────────────────────
    if args.filters:
        print_filter_hints(report)

    # ── save JSON ────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        analyzer.save_report(report, str(out))
        print(f"\nReport saved → {out}")

    print()


if __name__ == "__main__":
    main()
