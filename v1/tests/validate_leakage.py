#!/usr/bin/env python3
"""
validate_leakage.py — Stella Alpha Data Leakage & Quality Validator
=====================================================================

Standalone tool to verify:
  1. No D1 data leakage into H4 (D1 date must be STRICTLY before H4 date)
  2. H4 CSV quality (column presence, NaN check, date range, monotonic time)
  3. D1 CSV quality (same checks)
  4. Post-merge quality (NaN rates, column counts, feature completeness)
  5. Optional: check for weekend / holiday gaps

Run BEFORE the main pipeline to catch data problems early.

Usage:
    # Validate H4 + D1 separately (pre-merge)
    python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv

    # Validate an already-merged CSV
    python validate_leakage.py --merged data/merged.csv

    # Full check: validate inputs AND perform a test merge to check for leakage
    python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --test-merge

    # Use strict mode (any warning → non-zero exit code)
    python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --strict

    # Save report to JSON
    python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --output report.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── colour helpers ────────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        GREEN = RED = YELLOW = CYAN = ""
    class Style:
        BRIGHT = RESET_ALL = ""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    detail:  str = ""
    severity: str = "ERROR"   # ERROR | WARN | INFO


@dataclass
class ValidationReport:
    checks:        List[CheckResult] = field(default_factory=list)
    errors:        int = 0
    warnings:      int = 0
    infos:         int = 0
    overall_passed: bool = True

    def add(self, check: CheckResult):
        self.checks.append(check)
        if not check.passed:
            if check.severity == "ERROR":
                self.errors += 1
                self.overall_passed = False
            elif check.severity == "WARN":
                self.warnings += 1
        else:
            self.infos += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "errors":   self.errors,
            "warnings": self.warnings,
            "checks": [
                {"name": c.name, "passed": c.passed,
                 "message": c.message, "detail": c.detail, "severity": c.severity}
                for c in self.checks
            ],
        }


# =============================================================================
# INDIVIDUAL CHECKS
# =============================================================================

H4_REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]
D1_REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]
H4_FEATURE_COLS = [
    "rsi_value", "bb_position", "atr_pct", "volume_ratio",
    "lower_band", "upper_band", "trend_strength",
]
D1_FEATURE_COLS = [
    "rsi_value", "bb_position", "atr_pct", "trend_strength",
]


def check_required_columns(df: pd.DataFrame, required: List[str], label: str) -> CheckResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return CheckResult(
            name=f"{label}: required columns",
            passed=False,
            message=f"Missing {len(missing)} required column(s)",
            detail=f"Missing: {missing}",
            severity="ERROR",
        )
    return CheckResult(
        name=f"{label}: required columns",
        passed=True,
        message=f"All {len(required)} required columns present",
    )


def check_feature_columns(df: pd.DataFrame, expected: List[str], label: str) -> CheckResult:
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    pct = 100.0 * len(present) / len(expected)
    if len(missing) > 0:
        return CheckResult(
            name=f"{label}: feature columns",
            passed=False,
            message=f"{len(present)}/{len(expected)} expected feature columns present ({pct:.0f}%)",
            detail=f"Missing features: {missing}",
            severity="WARN",
        )
    return CheckResult(
        name=f"{label}: feature columns",
        passed=True,
        message=f"All {len(expected)} expected feature columns present",
    )


def check_timestamp_monotonic(df: pd.DataFrame, ts_col: str, label: str) -> CheckResult:
    if ts_col not in df.columns:
        return CheckResult(name=f"{label}: timestamp monotonic", passed=False,
                           message="Timestamp column missing", severity="ERROR")
    ts = pd.to_datetime(df[ts_col])
    dups = ts.duplicated().sum()
    not_mono = (ts.diff().dropna() <= pd.Timedelta(0)).sum()
    if dups > 0:
        return CheckResult(
            name=f"{label}: timestamp monotonic",
            passed=False,
            message=f"{dups} duplicate timestamps found",
            severity="ERROR",
        )
    if not_mono > 0:
        return CheckResult(
            name=f"{label}: timestamp monotonic",
            passed=False,
            message=f"{not_mono} non-monotonic timestamp steps",
            severity="ERROR",
        )
    return CheckResult(
        name=f"{label}: timestamp monotonic",
        passed=True,
        message=f"Timestamps are strictly monotonic ({len(ts):,} rows)",
    )


def check_nan_rates(df: pd.DataFrame, label: str, max_nan_pct: float = 5.0) -> CheckResult:
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) == 0:
        return CheckResult(name=f"{label}: NaN rates", passed=True,
                           message="No numeric columns to check")
    nan_pcts = numeric.isnull().mean() * 100
    bad = nan_pcts[nan_pcts > max_nan_pct]
    if len(bad) > 0:
        worst = bad.nlargest(3)
        detail = "  |  ".join(f"{c}: {v:.1f}%" for c, v in worst.items())
        return CheckResult(
            name=f"{label}: NaN rates",
            passed=False,
            message=f"{len(bad)} columns exceed {max_nan_pct}% NaN threshold",
            detail=detail,
            severity="WARN",
        )
    overall_nan = float(nan_pcts.max())
    return CheckResult(
        name=f"{label}: NaN rates",
        passed=True,
        message=f"Max NaN rate: {overall_nan:.2f}% (threshold: {max_nan_pct}%)",
    )


def check_ohlcv_sanity(df: pd.DataFrame, label: str) -> CheckResult:
    """high >= close >= low, high >= open >= low."""
    needed = {"open", "high", "low", "close"} - set(df.columns)
    if needed:
        return CheckResult(name=f"{label}: OHLC sanity", passed=True,
                           message="OHLC columns not all present — skipping sanity check",
                           severity="INFO")
    violations = (
        (df["high"] < df["low"]) |
        (df["close"] < df["low"]) | (df["close"] > df["high"]) |
        (df["open"]  < df["low"]) | (df["open"]  > df["high"])
    ).sum()
    if violations > 0:
        pct = 100.0 * violations / len(df)
        return CheckResult(
            name=f"{label}: OHLC sanity",
            passed=False,
            message=f"{violations} rows ({pct:.2f}%) fail high>=close>=low constraint",
            severity="ERROR",
        )
    return CheckResult(
        name=f"{label}: OHLC sanity",
        passed=True,
        message="OHLCV sanity: all rows pass high≥close/open≥low",
    )


def check_row_count(df: pd.DataFrame, min_rows: int, label: str) -> CheckResult:
    if len(df) < min_rows:
        return CheckResult(
            name=f"{label}: row count",
            passed=False,
            message=f"Only {len(df):,} rows — minimum recommended is {min_rows:,}",
            severity="WARN",
        )
    return CheckResult(
        name=f"{label}: row count",
        passed=True,
        message=f"{len(df):,} rows (minimum {min_rows:,})",
    )


def check_date_range_overlap(h4_df: pd.DataFrame, d1_df: pd.DataFrame,
                              ts_col: str = "timestamp") -> CheckResult:
    """D1 date range must substantially overlap H4 date range."""
    h4_min = pd.to_datetime(h4_df[ts_col]).min()
    h4_max = pd.to_datetime(h4_df[ts_col]).max()
    d1_min = pd.to_datetime(d1_df[ts_col]).min()
    d1_max = pd.to_datetime(d1_df[ts_col]).max()

    # D1 should start at least 1 day before H4
    if d1_min > h4_min:
        days_short = (d1_min - h4_min).days
        return CheckResult(
            name="D1/H4: date range overlap",
            passed=False,
            message=f"D1 starts {days_short}d AFTER H4 — no context for early H4 rows",
            detail=f"H4: {h4_min.date()} → {h4_max.date()}  |  D1: {d1_min.date()} → {d1_max.date()}",
            severity="WARN",
        )

    # D1 should reach at least 90% of the H4 range
    h4_span = (h4_max - h4_min).days
    d1_coverage = (min(d1_max, h4_max) - max(d1_min, h4_min)).days
    if h4_span > 0 and (d1_coverage / h4_span) < 0.90:
        return CheckResult(
            name="D1/H4: date range overlap",
            passed=False,
            message=f"D1 only covers {100*d1_coverage/h4_span:.0f}% of H4 date range",
            severity="WARN",
        )

    return CheckResult(
        name="D1/H4: date range overlap",
        passed=True,
        message=f"H4: {h4_min.date()}→{h4_max.date()}  D1: {d1_min.date()}→{d1_max.date()}",
    )


def check_leakage(merged_df: pd.DataFrame) -> CheckResult:
    """
    CRITICAL: For every merged row, d1_timestamp.date() must be STRICTLY
    before the H4 timestamp.date().  Any row where d1 date >= h4 date is a leak.
    """
    if "d1_timestamp" not in merged_df.columns:
        return CheckResult(
            name="Leakage: d1_timestamp < h4_timestamp",
            passed=True,
            message="d1_timestamp column not present — leakage check skipped",
            severity="INFO",
        )

    valid = merged_df.dropna(subset=["d1_timestamp"])
    if len(valid) == 0:
        return CheckResult(
            name="Leakage: d1_timestamp < h4_timestamp",
            passed=True,
            message="No rows have d1_timestamp — nothing to validate",
            severity="INFO",
        )

    h4_dates = pd.to_datetime(valid["timestamp"]).dt.date
    d1_dates = pd.to_datetime(valid["d1_timestamp"]).dt.date
    violations = valid[d1_dates >= h4_dates]

    if len(violations) > 0:
        pct = 100.0 * len(violations) / len(valid)
        examples = violations[["timestamp", "d1_timestamp"]].head(5).to_string(index=False)
        return CheckResult(
            name="Leakage: d1_timestamp < h4_timestamp",
            passed=False,
            message=f"🚨 DATA LEAKAGE: {len(violations)} rows ({pct:.2f}%) where D1 date >= H4 date",
            detail=f"First violations:\n{examples}",
            severity="ERROR",
        )

    return CheckResult(
        name="Leakage: d1_timestamp < h4_timestamp",
        passed=True,
        message=f"✅ No leakage: all {len(valid):,} rows have d1_timestamp < h4_timestamp",
    )


def check_merged_d1_coverage(merged_df: pd.DataFrame) -> CheckResult:
    """At least 80% of merged rows should have D1 data."""
    if "d1_timestamp" not in merged_df.columns:
        return CheckResult(name="Merge: D1 coverage", passed=True,
                           message="d1_timestamp not present — skipping coverage check",
                           severity="INFO")
    coverage = merged_df["d1_timestamp"].notna().mean()
    if coverage < 0.80:
        return CheckResult(
            name="Merge: D1 coverage",
            passed=False,
            message=f"Only {coverage*100:.1f}% of H4 rows have D1 data (< 80%)",
            severity="WARN",
        )
    return CheckResult(
        name="Merge: D1 coverage",
        passed=True,
        message=f"D1 coverage: {coverage*100:.1f}% of merged rows",
    )


def check_d1_column_count(merged_df: pd.DataFrame) -> CheckResult:
    d1_cols = [c for c in merged_df.columns if c.startswith("d1_")]
    if len(d1_cols) < 5:
        return CheckResult(
            name="Merge: d1_ column count",
            passed=False,
            message=f"Only {len(d1_cols)} d1_-prefixed columns — expected at least 5",
            severity="WARN",
        )
    return CheckResult(
        name="Merge: d1_ column count",
        passed=True,
        message=f"{len(d1_cols)} d1_-prefixed columns present",
    )


# =============================================================================
# SUMMARY PRINTER
# =============================================================================

def print_report(report: ValidationReport, label: str = ""):
    W = 68
    print(f"\n{'='*W}")
    title = f"STELLA ALPHA DATA VALIDATION{' — '+label if label else ''}"
    print(f"  {title}")
    print(f"{'='*W}")

    for c in report.checks:
        if c.passed:
            icon  = f"{Fore.GREEN}  ✅{Style.RESET_ALL}"
        elif c.severity == "WARN":
            icon  = f"{Fore.YELLOW}  ⚠️ {Style.RESET_ALL}"
        else:
            icon  = f"{Fore.RED}  ❌{Style.RESET_ALL}"
        print(f"{icon} {c.name}")
        print(f"      {c.message}")
        if c.detail:
            for line in c.detail.strip().split("\n"):
                print(f"         {line}")

    print(f"\n{'─'*W}")
    e_col = Fore.RED   if report.errors   > 0 else Fore.GREEN
    w_col = Fore.YELLOW if report.warnings > 0 else Fore.GREEN
    print(f"  Errors   : {e_col}{report.errors}{Style.RESET_ALL}")
    print(f"  Warnings : {w_col}{report.warnings}{Style.RESET_ALL}")
    print(f"  Passed   : {report.infos}")
    print()
    if report.overall_passed and report.warnings == 0:
        print(f"  {Fore.GREEN}{Style.BRIGHT}✅ ALL CHECKS PASSED — data is clean{Style.RESET_ALL}")
    elif report.overall_passed:
        print(f"  {Fore.YELLOW}⚠️  PASSED with {report.warnings} warning(s) — review above{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}{Style.BRIGHT}❌ VALIDATION FAILED — fix errors before running the pipeline{Style.RESET_ALL}")
    print(f"{'='*W}\n")


# =============================================================================
# VALIDATORS
# =============================================================================

def validate_csv(path: str, required_cols: List[str], feature_cols: List[str],
                 label: str, min_rows: int = 500) -> ValidationReport:
    report = ValidationReport()

    # Load
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        report.add(CheckResult(
            name=f"{label}: file load",
            passed=True,
            message=f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path}",
        ))
    except Exception as e:
        report.add(CheckResult(
            name=f"{label}: file load",
            passed=False, message=str(e), severity="ERROR",
        ))
        return report

    # Normalise timestamp
    ts_col = "timestamp"
    if ts_col in df.columns:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
        except Exception:
            pass

    report.add(check_row_count(df, min_rows, label))
    report.add(check_required_columns(df, required_cols, label))
    report.add(check_feature_columns(df, feature_cols, label))
    report.add(check_timestamp_monotonic(df, ts_col, label))
    report.add(check_nan_rates(df, label))
    report.add(check_ohlcv_sanity(df, label))

    return report, df


def validate_merged_csv(path: str) -> ValidationReport:
    report = ValidationReport()
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        report.add(CheckResult(
            name="Merged: file load",
            passed=True,
            message=f"Loaded {len(df):,} rows × {len(df.columns)} columns",
        ))
    except Exception as e:
        report.add(CheckResult(
            name="Merged: file load",
            passed=False, message=str(e), severity="ERROR",
        ))
        return report

    for col in ["timestamp", "d1_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    report.add(check_leakage(df))
    report.add(check_merged_d1_coverage(df))
    report.add(check_d1_column_count(df))
    report.add(check_nan_rates(df, "Merged"))
    report.add(check_ohlcv_sanity(df, "Merged"))
    return report


def test_merge_and_validate(h4_df: pd.DataFrame,
                             d1_df: pd.DataFrame) -> ValidationReport:
    """Attempt a test merge (if data_merger available) and check for leakage."""
    report = ValidationReport()

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from data_merger import DataMerger
        merger = DataMerger()
        merged = merger.merge(h4_df.copy(), d1_df.copy())
        report.add(CheckResult(
            name="Test merge",
            passed=True,
            message=f"Merge produced {len(merged):,} rows, {len(merged.columns)} columns",
        ))
        report.add(check_leakage(merged))
        report.add(check_merged_d1_coverage(merged))
        report.add(check_d1_column_count(merged))
    except ImportError:
        report.add(CheckResult(
            name="Test merge",
            passed=True,
            message="data_merger.py not found in src/ — skipping test merge",
            severity="INFO",
        ))
    except Exception as e:
        report.add(CheckResult(
            name="Test merge",
            passed=False,
            message=f"Merge failed: {e}",
            severity="ERROR",
        ))

    return report


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stella Alpha — Data Leakage & Quality Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate H4 + D1 inputs
  python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv

  # Validate pre-merged file
  python validate_leakage.py --merged data/merged.csv

  # Full: validate inputs + test merge for leakage
  python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --test-merge

  # Strict: warnings also cause non-zero exit
  python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --strict

  # Save JSON report
  python validate_leakage.py --h4 data/EURUSD_H4.csv --d1 data/EURUSD_D1.csv --output report.json
""",
    )
    p.add_argument("--h4",          default=None, help="Path to H4 CSV")
    p.add_argument("--d1",          default=None, help="Path to D1 CSV")
    p.add_argument("--merged",      default=None, help="Path to already-merged CSV")
    p.add_argument("--test-merge",  action="store_true",
                   help="Perform a test merge of H4+D1 and check for leakage")
    p.add_argument("--strict",      action="store_true",
                   help="Treat warnings as errors (non-zero exit if any warning)")
    p.add_argument("--output", "-o",default=None, help="Save JSON report to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_reports: Dict[str, ValidationReport] = {}
    h4_df = d1_df = None

    # ── H4 ─────────────────────────────────────────────────────────────────
    if args.h4:
        if not Path(args.h4).exists():
            print(f"ERROR: H4 file not found: {args.h4}")
            sys.exit(1)
        report_h4, h4_df = validate_csv(
            args.h4, H4_REQUIRED, H4_FEATURE_COLS, "H4", min_rows=500
        )
        print_report(report_h4, f"H4 — {args.h4}")
        all_reports["h4"] = report_h4

    # ── D1 ─────────────────────────────────────────────────────────────────
    if args.d1:
        if not Path(args.d1).exists():
            print(f"ERROR: D1 file not found: {args.d1}")
            sys.exit(1)
        report_d1, d1_df = validate_csv(
            args.d1, D1_REQUIRED, D1_FEATURE_COLS, "D1", min_rows=100
        )
        print_report(report_d1, f"D1 — {args.d1}")
        all_reports["d1"] = report_d1

    # ── Date range overlap ─────────────────────────────────────────────────
    if h4_df is not None and d1_df is not None:
        overlap_report = ValidationReport()
        overlap_report.add(check_date_range_overlap(h4_df, d1_df))
        print_report(overlap_report, "H4 / D1 date range overlap")
        all_reports["overlap"] = overlap_report

    # ── Test merge ─────────────────────────────────────────────────────────
    if args.test_merge and h4_df is not None and d1_df is not None:
        merge_report = test_merge_and_validate(h4_df, d1_df)
        print_report(merge_report, "Test Merge — leakage check")
        all_reports["test_merge"] = merge_report

    # ── Merged CSV ─────────────────────────────────────────────────────────
    if args.merged:
        if not Path(args.merged).exists():
            print(f"ERROR: Merged file not found: {args.merged}")
            sys.exit(1)
        merged_report = validate_merged_csv(args.merged)
        print_report(merged_report, f"Merged — {args.merged}")
        all_reports["merged"] = merged_report

    if not all_reports:
        print("Nothing to validate. Provide --h4, --d1, or --merged.")
        sys.exit(1)

    # ── Save JSON ───────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined = {k: v.to_dict() for k, v in all_reports.items()}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"Report saved → {out}")

    # ── Exit code ───────────────────────────────────────────────────────────
    has_errors   = any(r.errors   > 0 for r in all_reports.values())
    has_warnings = any(r.warnings > 0 for r in all_reports.values())

    if has_errors:
        sys.exit(1)
    if args.strict and has_warnings:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
