#!/usr/bin/env python3
"""
compare_versions.py — V4 vs Stella Alpha Comparison Tool
==========================================================

Side-by-side statistical comparison of V4 and Stella Alpha experiment results.

What it compares:
  • Pass rates and config counts
  • EV and precision distributions (mean, best, worst)
  • High R:R config counts (TP > SL)
  • Tier 0 / Tier 1 breakdown (Stella Alpha only)
  • GOLD / SILVER / BRONZE classification
  • D1 and MTF feature contribution in Stella Alpha
  • Statistical significance of EV improvement (Welch's t-test)
  • Feature overlap and new features introduced

Usage examples:
    python compare_versions.py \\
        --v4    ../version-4/artifacts/pure_ml.db \\
        --stella artifacts/pure_ml_stella_alpha.db

    # Specify output file
    python compare_versions.py \\
        --v4 ../version-4/artifacts/pure_ml.db \\
        --stella artifacts/pure_ml_stella_alpha.db \\
        --output artifacts/v4_comparison_report.json

    # Verbose: also show top-10 configs from each version
    python compare_versions.py \\
        --v4 ../version-4/artifacts/pure_ml.db \\
        --stella artifacts/pure_ml_stella_alpha.db \\
        --verbose
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = WHITE = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VersionSummary:
    version:          str
    db_path:          str
    total_configs:    int
    passed:           int
    failed:           int
    pass_rate:        float
    # EV
    best_ev:          float
    mean_ev:          float
    worst_ev:         float
    # Precision
    best_prec:        float
    mean_prec:        float
    # R:R
    high_rr_passed:   int       # TP > SL (R:R >= 1)
    max_rr:           float
    # Classifications
    gold:             int
    silver:           int
    bronze:           int
    # Tiers (Stella Alpha only)
    tier_0:           int
    tier_1:           int
    # Features
    all_features:     List[str]
    d1_features:      List[str]
    mtf_features:     List[str]
    # Raw passed rows (for stat tests)
    passed_rows:      List[Dict]


# =============================================================================
# DB LOADER
# =============================================================================

def load_db(db_path: str, version_label: str) -> VersionSummary:
    """Load all passed/failed results from a checkpoint database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM completed").fetchall()
    conn.close()

    all_rows  = [dict(r) for r in rows]
    passed_rows = []
    failed_rows = []

    for r in all_rows:
        status = str(r.get("status", "")).upper()
        if status == "PASSED":
            passed_rows.append(r)
        else:
            failed_rows.append(r)

    def _float(r, k, default=0.0):
        v = r.get(k)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default

    def _int(r, k, default=0):
        v = r.get(k)
        try:
            return int(v) if v is not None else default
        except Exception:
            return default

    def _features(r) -> List[str]:
        f = r.get("selected_features", "[]") or "[]"
        if isinstance(f, str):
            try:
                return json.loads(f)
            except Exception:
                return []
        return list(f)

    evs   = [_float(r, "ev_mean")        for r in passed_rows] or [0.0]
    precs = [_float(r, "precision_mean") for r in passed_rows] or [0.0]

    # R:R for each passed config
    rrs = []
    high_rr = 0
    max_rr  = 0.0
    for r in passed_rows:
        tp = _int(r, "tp_pips", 1)
        sl = _int(r, "sl_pips", 1)
        rr = tp / max(sl, 1)
        rrs.append(rr)
        if rr >= 1.0:
            high_rr += 1
        max_rr = max(max_rr, rr)

    # All selected features across passed configs
    all_features_set: Dict[str, int] = {}
    for r in passed_rows:
        for f in _features(r):
            all_features_set[f] = all_features_set.get(f, 0) + 1

    sorted_feats = sorted(all_features_set.keys(),
                          key=lambda k: all_features_set[k], reverse=True)
    d1_feats  = [f for f in sorted_feats if f.startswith("d1_")]
    mtf_feats = [f for f in sorted_feats if f.startswith("mtf_")]

    return VersionSummary(
        version=version_label,
        db_path=db_path,
        total_configs=len(all_rows),
        passed=len(passed_rows),
        failed=len(failed_rows),
        pass_rate=len(passed_rows) / max(len(all_rows), 1),
        best_ev=max(evs),
        mean_ev=mean(evs),
        worst_ev=min(evs),
        best_prec=max(precs),
        mean_prec=mean(precs),
        high_rr_passed=high_rr,
        max_rr=max_rr,
        gold=sum(1 for r in passed_rows if r.get("classification") == "GOLD"),
        silver=sum(1 for r in passed_rows if r.get("classification") == "SILVER"),
        bronze=sum(1 for r in passed_rows if r.get("classification") == "BRONZE"),
        tier_0=sum(1 for r in passed_rows if _int(r, "tier") == 0),
        tier_1=sum(1 for r in passed_rows if _int(r, "tier") == 1),
        all_features=sorted_feats,
        d1_features=d1_feats,
        mtf_features=mtf_feats,
        passed_rows=passed_rows,
    )


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def compare(v4: VersionSummary, sa: VersionSummary) -> Dict[str, Any]:
    """Compute deltas and stat-test results."""

    def _delta(a, b) -> float:
        try:
            return float(b) - float(a)
        except Exception:
            return 0.0

    def _pct_delta(a, b) -> float:
        try:
            return (float(b) - float(a)) * 100.0
        except Exception:
            return 0.0

    v4_evs = [float(r.get("ev_mean") or 0) for r in v4.passed_rows] or [0.0]
    sa_evs = [float(r.get("ev_mean") or 0) for r in sa.passed_rows] or [0.0]

    t_stat = p_value = None
    if HAS_SCIPY and len(v4_evs) >= 3 and len(sa_evs) >= 3:
        t_stat, p_value = scipy_stats.ttest_ind(sa_evs, v4_evs, equal_var=False)
        t_stat  = float(t_stat)
        p_value = float(p_value)

    # Feature overlap
    v4_set = set(v4.all_features)
    sa_set = set(sa.all_features)
    shared = v4_set & sa_set
    new_sa = sa_set - v4_set

    return {
        "pass_rate_delta_pp":    _pct_delta(v4.pass_rate, sa.pass_rate),
        "best_ev_delta":         _delta(v4.best_ev, sa.best_ev),
        "mean_ev_delta":         _delta(v4.mean_ev, sa.mean_ev),
        "best_prec_delta_pp":    _pct_delta(v4.best_prec, sa.best_prec),
        "high_rr_delta":         sa.high_rr_passed - v4.high_rr_passed,
        "gold_delta":            sa.gold - v4.gold,
        "t_statistic":           t_stat,
        "p_value":               p_value,
        "ev_improvement_significant": (p_value is not None and p_value < 0.05),
        "features_shared":       len(shared),
        "features_new_in_stella":len(new_sa),
        "d1_features_selected":  len(sa.d1_features),
        "mtf_features_selected": len(sa.mtf_features),
    }


# =============================================================================
# PRINTING
# =============================================================================

W = 70

def _hdr(title: str):
    print(f"\n{Fore.CYAN}{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}{Style.RESET_ALL}")


def _sub(title: str):
    print(f"\n{Fore.YELLOW}  ── {title} ──{Style.RESET_ALL}")


def _delta_str(v: float, unit: str = "", positive_good: bool = True) -> str:
    if v > 0:
        color = Fore.GREEN if positive_good else Fore.RED
        return f"{color}+{v:.2f}{unit}{Style.RESET_ALL}"
    elif v < 0:
        color = Fore.RED if positive_good else Fore.GREEN
        return f"{color}{v:.2f}{unit}{Style.RESET_ALL}"
    else:
        return f"  {v:.2f}{unit}"


def print_comparison(v4: VersionSummary, sa: VersionSummary, deltas: Dict, verbose: bool):

    _hdr("V4  vs  STELLA ALPHA — COMPARISON REPORT")

    # ── Overall stats ─────────────────────────────────────────────────────
    _sub("OVERALL RESULTS")
    print(f"\n  {'Metric':<30} {'V4':>12} {'Stella Alpha':>14} {'Delta':>12}")
    print(f"  {'─'*72}")

    rows = [
        ("Configs tested",    v4.total_configs,    sa.total_configs,    None,  ""),
        ("Passed",            v4.passed,            sa.passed,            None,  ""),
        ("Pass rate",         v4.pass_rate*100,     sa.pass_rate*100,     deltas["pass_rate_delta_pp"], "pp"),
        ("Best EV (pips)",    v4.best_ev,           sa.best_ev,           deltas["best_ev_delta"],      " pips"),
        ("Mean EV (pips)",    v4.mean_ev,           sa.mean_ev,           deltas["mean_ev_delta"],      " pips"),
        ("Best precision",    v4.best_prec*100,     sa.best_prec*100,     deltas["best_prec_delta_pp"], "pp"),
        ("Mean precision",    v4.mean_prec*100,     sa.mean_prec*100,     None,  "%"),
        ("High R:R passed",   v4.high_rr_passed,    sa.high_rr_passed,    deltas["high_rr_delta"],      ""),
        ("Max R:R ratio",     v4.max_rr,            sa.max_rr,            None,  ":1"),
        ("GOLD configs",      v4.gold,              sa.gold,              deltas["gold_delta"],          ""),
        ("SILVER configs",    v4.silver,            sa.silver,            None,  ""),
        ("BRONZE configs",    v4.bronze,            sa.bronze,            None,  ""),
    ]

    for label, v4_val, sa_val, delta, unit in rows:
        v4_fmt  = f"{v4_val:.1f}{unit}" if isinstance(v4_val, float) else f"{v4_val}{unit}"
        sa_fmt  = f"{sa_val:.1f}{unit}" if isinstance(sa_val, float) else f"{sa_val}{unit}"
        if delta is not None:
            d_fmt = _delta_str(delta, unit)
        else:
            try:
                d = float(sa_val) - float(v4_val)
                d_fmt = _delta_str(d, unit) if d != 0 else "—"
            except Exception:
                d_fmt = "—"
        print(f"  {label:<30} {v4_fmt:>12} {sa_fmt:>14} {d_fmt:>12}")

    # ── Tier breakdown (Stella Alpha only) ────────────────────────────────
    _sub("TIER BREAKDOWN (Stella Alpha)")
    print(f"""
  🚀 Tier 0 Runners  (R:R ≥ 2.5:1) : {sa.tier_0}
  ⭐ Tier 1 Ideal    (R:R 1.67-2.5) : {sa.tier_1}
  V4 had no tier system (all configs R:R < 1:1 typically)
""")

    # ── Statistical significance ──────────────────────────────────────────
    _sub("STATISTICAL SIGNIFICANCE (EV improvement)")
    if HAS_SCIPY and deltas["t_statistic"] is not None:
        sig = deltas["ev_improvement_significant"]
        p   = deltas["p_value"]
        t   = deltas["t_statistic"]
        sig_str = f"{Fore.GREEN}✅ YES (p={p:.4f} < 0.05){Style.RESET_ALL}" if sig \
                  else f"{Fore.YELLOW}❌ NO  (p={p:.4f} ≥ 0.05){Style.RESET_ALL}"
        print(f"\n  Welch's t-test (Stella EV vs V4 EV):")
        print(f"  t-statistic : {t:.3f}")
        print(f"  p-value     : {p:.4f}")
        print(f"  Significant : {sig_str}")
    else:
        print("\n  scipy not available or insufficient data for t-test.")

    # ── D1 / MTF feature contribution ─────────────────────────────────────
    _sub("MULTI-TIMEFRAME FEATURE CONTRIBUTION")
    print(f"""
  Unique features in V4 passed configs    : {len(v4.all_features)}
  Unique features in Stella passed configs : {len(sa.all_features)}
  Features shared by both versions        : {deltas['features_shared']}
  NEW features introduced in Stella Alpha : {deltas['features_new_in_stella']}

  D1  features selected (Stella Alpha)   : {deltas['d1_features_selected']}
  MTF features selected (Stella Alpha)   : {deltas['mtf_features_selected']}
""")
    if deltas["d1_features_selected"] > 0 or deltas["mtf_features_selected"] > 0:
        print(f"  {Fore.GREEN}✅ Multi-timeframe data IS contributing to model selection{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}⚠️  Multi-timeframe data was NOT selected by RFE in passed configs{Style.RESET_ALL}")

    if sa.d1_features:
        _sub("Top D1 Features (Stella Alpha)")
        for f in sa.d1_features[:10]:
            print(f"    • {f}")
    if sa.mtf_features:
        _sub("Top MTF Features (Stella Alpha)")
        for f in sa.mtf_features[:10]:
            print(f"    • {f}")

    # ── Verbose: top configs ───────────────────────────────────────────────
    if verbose:
        for summary, label in [(v4, "V4"), (sa, "Stella Alpha")]:
            _sub(f"Top 10 Passed Configs — {label}")
            sorted_p = sorted(summary.passed_rows, key=lambda r: float(r.get("ev_mean") or 0), reverse=True)
            print(f"\n  {'Config ID':<28} {'EV':>8} {'Prec':>8} {'Trades':>8}")
            print(f"  {'─'*55}")
            for r in sorted_p[:10]:
                print(f"  {str(r.get('config_id','')):<28} "
                      f"{float(r.get('ev_mean') or 0):>+8.2f} "
                      f"{float(r.get('precision_mean') or 0)*100:>7.1f}% "
                      f"{int(r.get('total_trades') or 0):>8,}")

    # ── Final verdict ──────────────────────────────────────────────────────
    _hdr("VERDICT")

    improvements = []
    concerns     = []

    if deltas["mean_ev_delta"] > 0:
        improvements.append(f"Mean EV improved by {deltas['mean_ev_delta']:+.2f} pips")
    else:
        concerns.append(f"Mean EV regressed by {deltas['mean_ev_delta']:.2f} pips")

    if deltas["high_rr_delta"] > 0:
        improvements.append(f"Unlocked {deltas['high_rr_delta']} more high R:R configs")
    elif deltas["high_rr_delta"] == 0:
        concerns.append("No improvement in high R:R config count")

    if sa.tier_0 > 0:
        improvements.append(f"{sa.tier_0} Tier 0 Runner config(s) found")

    if deltas["d1_features_selected"] > 0 or deltas["mtf_features_selected"] > 0:
        improvements.append("D1/MTF features contributing to model")
    else:
        concerns.append("D1/MTF features not yet selected by RFE")

    if deltas["ev_improvement_significant"]:
        improvements.append("EV improvement is statistically significant (p < 0.05)")

    if improvements:
        print(f"\n  {Fore.GREEN}IMPROVEMENTS{Style.RESET_ALL}")
        for imp in improvements:
            print(f"  ✅ {imp}")

    if concerns:
        print(f"\n  {Fore.YELLOW}AREAS TO INVESTIGATE{Style.RESET_ALL}")
        for con in concerns:
            print(f"  ⚠️  {con}")

    # Overall rating
    score = len(improvements) - len(concerns)
    print()
    if score >= 3:
        print(f"  {Fore.GREEN}🎉 STELLA ALPHA IS A CLEAR IMPROVEMENT OVER V4{Style.RESET_ALL}")
    elif score > 0:
        print(f"  {Fore.YELLOW}📈 STELLA ALPHA SHOWS MARGINAL IMPROVEMENT — continue tuning{Style.RESET_ALL}")
    elif score == 0:
        print(f"  {Fore.YELLOW}🔄 MIXED RESULTS — some areas improved, some need work{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}📉 V4 PERFORMED BETTER — review Stella Alpha configuration{Style.RESET_ALL}")
    print()


def save_report(v4: VersionSummary, sa: VersionSummary, deltas: Dict, output_path: str) -> None:
    """Save comparison to JSON."""
    report = {
        "v4": {
            "db_path":         v4.db_path,
            "total_configs":   v4.total_configs,
            "passed":          v4.passed,
            "pass_rate":       round(v4.pass_rate, 4),
            "best_ev":         round(v4.best_ev, 4),
            "mean_ev":         round(v4.mean_ev, 4),
            "best_prec":       round(v4.best_prec, 4),
            "high_rr_passed":  v4.high_rr_passed,
            "gold":            v4.gold,
            "silver":          v4.silver,
        },
        "stella_alpha": {
            "db_path":         sa.db_path,
            "total_configs":   sa.total_configs,
            "passed":          sa.passed,
            "pass_rate":       round(sa.pass_rate, 4),
            "best_ev":         round(sa.best_ev, 4),
            "mean_ev":         round(sa.mean_ev, 4),
            "best_prec":       round(sa.best_prec, 4),
            "high_rr_passed":  sa.high_rr_passed,
            "gold":            sa.gold,
            "silver":          sa.silver,
            "tier_0":          sa.tier_0,
            "tier_1":          sa.tier_1,
            "d1_features":     sa.d1_features,
            "mtf_features":    sa.mtf_features,
        },
        "deltas": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in deltas.items()},
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {out}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stella Alpha — V4 vs Stella Alpha Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_versions.py \\
      --v4    ../version-4/artifacts/pure_ml.db \\
      --stella artifacts/pure_ml_stella_alpha.db

  python compare_versions.py \\
      --v4 ../version-4/artifacts/pure_ml.db \\
      --stella artifacts/pure_ml_stella_alpha.db \\
      --output artifacts/v4_comparison_report.json --verbose
""",
    )
    p.add_argument("--v4",     required=True,  help="Path to V4 checkpoint database (pure_ml.db)")
    p.add_argument("--stella", required=True,  help="Path to Stella Alpha database (pure_ml_stella_alpha.db)")
    p.add_argument("--output", "-o", default=None, help="Save JSON report to this path")
    p.add_argument("--verbose", "-v", action="store_true", help="Show top-10 configs from each version")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    for path, label in [(args.v4, "V4"), (args.stella, "Stella Alpha")]:
        if not Path(path).exists():
            print(f"ERROR: {label} database not found: {path}")
            sys.exit(1)

    print(f"\nLoading V4          : {args.v4}")
    try:
        v4 = load_db(args.v4, "V4")
    except Exception as e:
        print(f"ERROR loading V4 database: {e}")
        sys.exit(1)

    print(f"Loading Stella Alpha: {args.stella}")
    try:
        sa = load_db(args.stella, "Stella Alpha")
    except Exception as e:
        print(f"ERROR loading Stella Alpha database: {e}")
        sys.exit(1)

    print(f"V4 configs: {v4.total_configs} total, {v4.passed} passed")
    print(f"SA configs: {sa.total_configs} total, {sa.passed} passed")

    deltas = compare(v4, sa)
    print_comparison(v4, sa, deltas, verbose=args.verbose)

    if args.output:
        save_report(v4, sa, deltas, args.output)


if __name__ == "__main__":
    main()
