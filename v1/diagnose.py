#!/usr/bin/env python3
"""
Stella Alpha Diagnostic Analyzer
==================================

Comprehensive analysis of experiment results after pipeline completion.

Sections:
  1.  Overview & pass/fail stats
  2.  Tier breakdown (Tier 0 Runners vs Tier 1 Ideal)
  3.  GOLD configurations (detailed)
  4.  SILVER configurations
  5.  All passed configs (sorted by EV)
  6.  Rejection analysis
  7.  Near-miss detection (R:R-aware)
  8.  Parameter patterns (TP, SL, Hold, R:R)
  9.  Feature importance (most-selected features)
  10. D1 & MTF feature analysis (NEW)
  11. Recommendations & final verdict

Usage:
    python diagnose.py
    python diagnose.py --db artifacts/pure_ml_stella_alpha.db
    python diagnose.py --db artifacts/pure_ml_stella_alpha.db --top 15
    python diagnose.py --db artifacts/pure_ml_stella_alpha.db --compare-v4 ../version-4/artifacts/pure_ml.db
"""

import argparse
import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = MAGENTA = WHITE = BLUE = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class ConfigResult:
    tp_pips:            int
    sl_pips:            int
    max_holding_bars:   int
    config_id:          str
    status:             str
    tier:               int
    tier_name:          str
    ev_mean:            float
    ev_std:             float
    precision_mean:     float
    precision_std:      float
    precision_cv:       float
    recall_mean:        float
    f1_mean:            float
    auc_pr_mean:        float
    total_trades:       int
    selected_features:  List[str]
    n_features:         int
    consensus_threshold:float
    classification:     Optional[str]
    rejection_reasons:  List[str]
    risk_reward_ratio:  float
    min_winrate_required: float
    edge_above_breakeven: float
    execution_time:     float

    @property
    def is_passed(self) -> bool:
        return self.status.upper() == "PASSED"

    @property
    def is_gold(self) -> bool:
        return self.classification == "GOLD"

    @property
    def is_silver(self) -> bool:
        return self.classification == "SILVER"

    @property
    def is_bronze(self) -> bool:
        return self.classification == "BRONZE"

    @property
    def sharpe_like(self) -> float:
        return self.ev_mean / max(self.ev_std, 0.001)

    def short_str(self) -> str:
        return f"TP={self.tp_pips} SL={self.sl_pips} H={self.max_holding_bars}"


# =============================================================================
# DB LOADER
# =============================================================================

def load_results(db_path: str) -> List[ConfigResult]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM completed").fetchall()
    conn.close()

    results = []
    for row in rows:
        d = dict(row)

        features = d.get("selected_features", "[]") or "[]"
        if isinstance(features, str):
            try:
                features = json.loads(features)
            except Exception:
                features = []

        reasons = d.get("rejection_reasons", "[]") or "[]"
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = [reasons] if reasons else []

        tp = int(d.get("tp_pips", 0))
        sl = int(d.get("sl_pips", 0))

        results.append(ConfigResult(
            tp_pips=tp,
            sl_pips=sl,
            max_holding_bars=int(d.get("max_holding_bars", 0)),
            config_id=str(d.get("config_id", "")),
            status=str(d.get("status", "UNKNOWN")),
            tier=int(d.get("tier", -1)),
            tier_name=str(d.get("tier_name", "")),
            ev_mean=float(d.get("ev_mean") or 0),
            ev_std=float(d.get("ev_std") or 0),
            precision_mean=float(d.get("precision_mean") or 0),
            precision_std=float(d.get("precision_std") or 0),
            precision_cv=float(d.get("precision_cv") or 0),
            recall_mean=float(d.get("recall_mean") or 0),
            f1_mean=float(d.get("f1_mean") or 0),
            auc_pr_mean=float(d.get("auc_pr_mean") or 0),
            total_trades=int(d.get("total_trades") or 0),
            selected_features=features,
            n_features=int(d.get("n_features") or len(features)),
            consensus_threshold=float(d.get("consensus_threshold") or 0.5),
            classification=d.get("classification"),
            rejection_reasons=reasons,
            risk_reward_ratio=float(d.get("risk_reward_ratio") or (tp / max(sl, 1))),
            min_winrate_required=float(d.get("min_winrate_required") or (sl / max(tp + sl, 1))),
            edge_above_breakeven=float(d.get("edge_above_breakeven") or 0),
            execution_time=float(d.get("execution_time") or 0),
        ))

    return results


# =============================================================================
# PRINT HELPERS
# =============================================================================

W = 72

def _hdr(title: str):
    print(f"\n{Fore.CYAN}{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}{Style.RESET_ALL}")


def _sub(title: str):
    print(f"\n{Fore.YELLOW}  ── {title} ──{Style.RESET_ALL}")


def _bar(pct: float, width: int = 20) -> str:
    filled = int(width * pct / 100)
    return Fore.CYAN + "█" * filled + Style.RESET_ALL + "░" * (width - filled)


# =============================================================================
# SECTION 1: OVERVIEW
# =============================================================================

def analyze_overview(results: List[ConfigResult]):
    _hdr("STELLA ALPHA — DIAGNOSTIC REPORT")

    total   = len(results)
    passed  = [r for r in results if r.is_passed]
    failed  = [r for r in results if not r.is_passed]
    gold    = [r for r in passed if r.is_gold]
    silver  = [r for r in passed if r.is_silver]
    bronze  = [r for r in passed if r.is_bronze]
    tier0   = [r for r in passed if r.tier == 0]
    tier1   = [r for r in passed if r.tier == 1]

    print(f"""
  RESULTS SUMMARY
  {'─'*50}
  Total Configs Tested  : {total}
  {Fore.GREEN}✓ PASSED              : {len(passed):4d}  ({100*len(passed)/max(total,1):.1f}%){Style.RESET_ALL}
  {Fore.RED}✗ FAILED              : {len(failed):4d}  ({100*len(failed)/max(total,1):.1f}%){Style.RESET_ALL}

  CLASSIFICATIONS
  {'─'*50}
  🥇 GOLD               : {len(gold)}
  🥈 SILVER             : {len(silver)}
  🥉 BRONZE             : {len(bronze)}

  TIER BREAKDOWN (Stella Alpha)
  {'─'*50}
  🚀 Tier 0 Runners     : {len(tier0)}   (R:R >= 2.5:1)
  ⭐ Tier 1 Ideal       : {len(tier1)}   (R:R 1.67–2.49:1)
""")

    if passed:
        evs   = [r.ev_mean for r in passed]
        precs = [r.precision_mean for r in passed]
        rrs   = [r.risk_reward_ratio for r in passed]
        print(f"""
  PERFORMANCE SUMMARY (passed configs)
  {'─'*50}
  Expected Value (pips)  Best={max(evs):+.2f}  Mean={statistics.mean(evs):+.2f}  Worst={min(evs):+.2f}
  Precision              Best={max(precs)*100:.1f}%  Mean={statistics.mean(precs)*100:.1f}%  Worst={min(precs)*100:.1f}%
  R:R Ratio              Best={max(rrs):.2f}  Mean={statistics.mean(rrs):.2f}  Worst={min(rrs):.2f}
""")


# =============================================================================
# SECTION 2: TIER ANALYSIS
# =============================================================================

def analyze_tiers(results: List[ConfigResult]):
    _hdr("TIER ANALYSIS")

    passed = [r for r in results if r.is_passed]
    tier0  = [r for r in passed if r.tier == 0]
    tier1  = [r for r in passed if r.tier == 1]

    for tier_label, tier_set, threshold in [
        ("🚀 TIER 0 — RUNNERS (R:R >= 2.5:1)", tier0, 2.5),
        ("⭐ TIER 1 — IDEAL   (R:R 1.67–2.49:1)", tier1, 1.67),
    ]:
        _sub(tier_label)
        if not tier_set:
            print(f"  No {tier_label.split('—')[1].strip()} configs found.")
            continue

        sorted_t = sorted(tier_set, key=lambda r: r.ev_mean, reverse=True)
        print(f"\n  {'Config':<22} {'R:R':>6} {'EV':>8} {'Prec':>8} {'Edge':>8} {'Trades':>8}  Class")
        print(f"  {'─'*70}")
        for r in sorted_t[:10]:
            cls_icon = {"GOLD": "🥇", "SILVER": "🥈", "BRONZE": "🥉"}.get(r.classification or "", "  ")
            print(
                f"  {r.short_str():<22} "
                f"{r.risk_reward_ratio:>6.2f} "
                f"{r.ev_mean:>+8.2f} "
                f"{r.precision_mean*100:>7.1f}% "
                f"{r.edge_above_breakeven*100:>+7.1f}% "
                f"{r.total_trades:>8,}  {cls_icon}"
            )


# =============================================================================
# SECTION 3 & 4: GOLD / SILVER
# =============================================================================

def analyze_gold(results: List[ConfigResult]):
    _hdr("🥇 GOLD CONFIGURATIONS")
    gold = [r for r in results if r.is_gold]
    if not gold:
        print(f"\n  {Fore.YELLOW}No GOLD configs found.{Style.RESET_ALL}")
        print("  GOLD criteria: edge >10%, CV <12%, trades >500, EV >8 pips")
        return

    for i, r in enumerate(sorted(gold, key=lambda x: x.ev_mean, reverse=True), 1):
        print(f"""
  {Fore.YELLOW}{'─'*60}{Style.RESET_ALL}
  #{i}  {Fore.GREEN}{Style.BRIGHT}{r.short_str()}{Style.RESET_ALL}  [{r.tier_name}]
  {Fore.YELLOW}{'─'*60}{Style.RESET_ALL}

  Trade params : TP={r.tp_pips}  SL={r.sl_pips}  Hold={r.max_holding_bars}  R:R={r.risk_reward_ratio:.2f}:1
  Threshold    : {r.consensus_threshold:.2f}

  {Fore.GREEN}EV           : {r.ev_mean:+.2f} pips  (±{r.ev_std:.2f}){Style.RESET_ALL}
  Sharpe-like  : {r.sharpe_like:.2f}
  {Fore.GREEN}Precision    : {r.precision_mean*100:.1f}%  (±{r.precision_std*100:.1f}%){Style.RESET_ALL}
  Precision CV : {r.precision_cv*100:.1f}%
  Recall       : {r.recall_mean*100:.1f}%
  F1           : {r.f1_mean*100:.1f}%
  AUC-PR       : {r.auc_pr_mean:.3f}

  Min win rate : {r.min_winrate_required*100:.1f}%
  {Fore.GREEN}Edge         : {r.edge_above_breakeven*100:+.1f}% above breakeven{Style.RESET_ALL}
  Total trades : {r.total_trades:,}
  Features     : {', '.join(r.selected_features[:10])}{'...' if len(r.selected_features) > 10 else ''}
""")


def analyze_silver(results: List[ConfigResult], top_n: int = 10):
    _hdr("🥈 SILVER CONFIGURATIONS")
    silver = sorted([r for r in results if r.is_silver], key=lambda r: r.ev_mean, reverse=True)
    if not silver:
        print(f"\n  {Fore.YELLOW}No SILVER configs found.{Style.RESET_ALL}")
        return

    print(f"\n  {'Config':<22} {'R:R':>6} {'EV':>8} {'Prec':>8} {'CV':>7} {'Edge':>8} {'Trades':>8}")
    print(f"  {'─'*72}")
    for r in silver[:top_n]:
        print(
            f"  {r.short_str():<22} "
            f"{r.risk_reward_ratio:>6.2f} "
            f"{r.ev_mean:>+8.2f} "
            f"{r.precision_mean*100:>7.1f}% "
            f"{r.precision_cv*100:>6.1f}% "
            f"{r.edge_above_breakeven*100:>+7.1f}% "
            f"{r.total_trades:>8,}"
        )


# =============================================================================
# SECTION 5: ALL PASSED
# =============================================================================

def analyze_passed(results: List[ConfigResult], top_n: int = 20):
    _hdr("ALL PASSED CONFIGURATIONS (sorted by EV)")
    passed = sorted([r for r in results if r.is_passed], key=lambda r: r.ev_mean, reverse=True)
    if not passed:
        print(f"  {Fore.RED}No passed configs.{Style.RESET_ALL}")
        return

    print(f"\n  {'#':<4} {'Config':<22} {'Tier':>4} {'R:R':>6} {'EV':>8} {'Prec':>8} {'Thresh':>7} {'Trades':>8}  Class")
    print(f"  {'─'*80}")
    for i, r in enumerate(passed[:top_n], 1):
        cls_icon = {"GOLD": "🥇", "SILVER": "🥈", "BRONZE": "🥉"}.get(r.classification or "", "  ")
        tier_str = f"T{r.tier}" if r.tier >= 0 else " -"
        print(
            f"  {i:<4} {r.short_str():<22} "
            f"{tier_str:>4} "
            f"{r.risk_reward_ratio:>6.2f} "
            f"{r.ev_mean:>+8.2f} "
            f"{r.precision_mean*100:>7.1f}% "
            f"{r.consensus_threshold:>7.2f} "
            f"{r.total_trades:>8,}  {cls_icon}"
        )
    if len(passed) > top_n:
        print(f"\n  … and {len(passed) - top_n} more.")


# =============================================================================
# SECTION 6: REJECTION ANALYSIS
# =============================================================================

def analyze_rejected(results: List[ConfigResult]):
    _hdr("✗ REJECTION ANALYSIS")
    failed = [r for r in results if not r.is_passed]
    if not failed:
        print(f"\n  {Fore.GREEN}All configs passed!{Style.RESET_ALL}")
        return

    reason_counts: Dict[str, int] = {}
    for r in failed:
        for reason in r.rejection_reasons:
            key = reason.split("(")[0].strip()[:50]
            reason_counts[key] = reason_counts.get(key, 0) + 1

    print(f"\n  Total Failed: {len(failed)}")
    _sub("Rejection Reasons")
    print(f"\n  {'Reason':<52} {'Count':>6} {'%':>6}")
    print(f"  {'─'*68}")
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(failed)
        print(f"  {reason:<52} {count:>6} {pct:>5.1f}%")


# =============================================================================
# SECTION 7: NEAR MISSES (R:R-AWARE)
# =============================================================================

def analyze_near_misses(results: List[ConfigResult]):
    _hdr("⚠️  NEAR MISSES")
    failed = [r for r in results if not r.is_passed]

    near = []
    for r in failed:
        # Near-miss: had positive edge (above breakeven) but failed another criterion
        has_edge = r.edge_above_breakeven > 0
        ev_close  = r.ev_mean > -5.0
        has_trades = r.total_trades >= 50
        if has_edge and ev_close and has_trades:
            near.append(r)

    if not near:
        print(f"\n  No near-miss configs found.")
        return

    print(f"\n  Found {len(near)} configs that had positive edge but still failed:\n")
    print(f"  {'Config':<22} {'R:R':>6} {'EV':>8} {'Prec':>8} {'Edge':>8} {'Issue'}")
    print(f"  {'─'*75}")

    near.sort(key=lambda r: r.ev_mean, reverse=True)
    for r in near[:15]:
        issues = []
        if r.ev_mean <= 0:
            issues.append(f"EV {r.ev_mean:+.2f}≤0")
        if r.precision_cv > 0.20:
            issues.append(f"CV {r.precision_cv*100:.0f}%>20%")
        if r.total_trades < 100:
            issues.append(f"trades {r.total_trades}<100")
        print(
            f"  {r.short_str():<22} "
            f"{r.risk_reward_ratio:>6.2f} "
            f"{r.ev_mean:>+8.2f} "
            f"{r.precision_mean*100:>7.1f}% "
            f"{r.edge_above_breakeven*100:>+7.1f}%  "
            f"{' | '.join(issues)}"
        )


# =============================================================================
# SECTION 8: PARAMETER PATTERNS
# =============================================================================

def analyze_parameter_patterns(results: List[ConfigResult]):
    _hdr("📊 PARAMETER PATTERNS")
    passed = [r for r in results if r.is_passed]
    failed = [r for r in results if not r.is_passed]

    for label, values, attr in [
        ("TP (pips)",         sorted(set(r.tp_pips for r in results)),          "tp_pips"),
        ("SL (pips)",         sorted(set(r.sl_pips for r in results)),          "sl_pips"),
        ("Max Hold (bars)",   sorted(set(r.max_holding_bars for r in results)), "max_holding_bars"),
    ]:
        _sub(label)
        print(f"\n  {'Value':<12} {'Passed':>8} {'Failed':>8} {'Pass%':>8} {'AvgEV':>9}")
        print(f"  {'─'*50}")
        for val in values:
            p = [r for r in passed if getattr(r, attr) == val]
            f = [r for r in failed if getattr(r, attr) == val]
            t = len(p) + len(f)
            pct  = 100 * len(p) / t if t > 0 else 0
            avev = statistics.mean([r.ev_mean for r in p]) if p else 0
            color = Fore.GREEN if pct > 50 else Fore.RED if pct < 20 else Fore.YELLOW
            print(
                f"  {val:<12} {len(p):>8} {len(f):>8} "
                f"{color}{pct:>7.1f}%{Style.RESET_ALL} {avev:>+9.2f}"
            )

    # R:R buckets
    _sub("Risk : Reward Ratio")
    buckets = [
        ("Runner   (>2.5:1)", lambda r: r.risk_reward_ratio > 2.5),
        ("Ideal  (1.67–2.5)", lambda r: 1.67 <= r.risk_reward_ratio <= 2.5),
        ("Moderate(1.0–1.67)", lambda r: 1.0 <= r.risk_reward_ratio < 1.67),
        ("Low      (<1.0:1)", lambda r: r.risk_reward_ratio < 1.0),
    ]
    print(f"\n  {'R:R Bucket':<22} {'Passed':>8} {'Failed':>8} {'Pass%':>8} {'AvgEV':>9}")
    print(f"  {'─'*58}")
    for name, cond in buckets:
        p = [r for r in passed if cond(r)]
        f = [r for r in failed if cond(r)]
        t = len(p) + len(f)
        if t == 0:
            continue
        pct  = 100 * len(p) / t
        avev = statistics.mean([r.ev_mean for r in p]) if p else 0
        color = Fore.GREEN if pct > 50 else Fore.RED if pct < 20 else Fore.YELLOW
        print(
            f"  {name:<22} {len(p):>8} {len(f):>8} "
            f"{color}{pct:>7.1f}%{Style.RESET_ALL} {avev:>+9.2f}"
        )


# =============================================================================
# SECTION 9: FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(results: List[ConfigResult]):
    _hdr("🔧 FEATURE IMPORTANCE")
    passed = [r for r in results if r.is_passed and r.selected_features]
    if not passed:
        print("  No passed configs with feature data.")
        return

    counts: Dict[str, int] = {}
    for r in passed:
        for f in r.selected_features:
            counts[f] = counts.get(f, 0) + 1

    print(f"\n  Features selected across {len(passed)} passed configs:\n")
    print(f"  {'Feature':<38} {'Count':>6} {'%':>6}  {'Bar'}")
    print(f"  {'─'*65}")
    for feat, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:25]:
        pct = 100 * count / len(passed)
        print(f"  {feat:<38} {count:>6} {pct:>5.1f}%  {_bar(pct, 16)}")


# =============================================================================
# SECTION 10: D1 & MTF FEATURE ANALYSIS (NEW)
# =============================================================================

def analyze_d1_mtf_features(results: List[ConfigResult]):
    _hdr("🌐 D1 & MTF FEATURE ANALYSIS")
    passed = [r for r in results if r.is_passed and r.selected_features]
    if not passed:
        print("  No passed configs with feature data.")
        return

    # Aggregate counts by prefix
    all_features: Dict[str, int] = {}
    for r in passed:
        for f in r.selected_features:
            all_features[f] = all_features.get(f, 0) + 1

    total_selections = sum(all_features.values())

    h4_feats  = {k: v for k, v in all_features.items() if not k.startswith("d1_") and not k.startswith("mtf_")}
    d1_feats  = {k: v for k, v in all_features.items() if k.startswith("d1_")}
    mtf_feats = {k: v for k, v in all_features.items() if k.startswith("mtf_")}

    print(f"""
  FEATURE CATEGORY BREAKDOWN
  {'─'*50}
  H4 features selected  : {sum(h4_feats.values()):4d} ({100*sum(h4_feats.values())/max(total_selections,1):.1f}%)
  D1 features selected  : {sum(d1_feats.values()):4d} ({100*sum(d1_feats.values())/max(total_selections,1):.1f}%)
  MTF features selected : {sum(mtf_feats.values()):4d} ({100*sum(mtf_feats.values())/max(total_selections,1):.1f}%)
""")

    if d1_feats:
        _sub("Top D1 Features")
        print(f"\n  {'Feature':<38} {'Count':>6} {'%':>6}")
        print(f"  {'─'*55}")
        for feat, count in sorted(d1_feats.items(), key=lambda x: x[1], reverse=True)[:15]:
            pct = 100 * count / len(passed)
            print(f"  {feat:<38} {count:>6} {pct:>5.1f}%  {_bar(pct, 12)}")
    else:
        print(f"\n  {Fore.YELLOW}  ⚠️  No D1 features selected in any passed config.{Style.RESET_ALL}")
        print("     D1 features may need more data or different engineering.")

    if mtf_feats:
        _sub("Top MTF (Cross-Timeframe) Features")
        print(f"\n  {'Feature':<38} {'Count':>6} {'%':>6}")
        print(f"  {'─'*55}")
        for feat, count in sorted(mtf_feats.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = 100 * count / len(passed)
            print(f"  {feat:<38} {count:>6} {pct:>5.1f}%  {_bar(pct, 12)}")
    else:
        print(f"\n  {Fore.YELLOW}  ⚠️  No MTF features selected in any passed config.{Style.RESET_ALL}")

    # Impact of D1/MTF features on performance
    d1_users  = [r for r in passed if any(f.startswith("d1_")  for f in r.selected_features)]
    mtf_users = [r for r in passed if any(f.startswith("mtf_") for f in r.selected_features)]
    h4_only   = [r for r in passed if not any(f.startswith(("d1_", "mtf_")) for f in r.selected_features)]

    _sub("Performance by Feature Set Used")
    for label, group in [("D1 features used", d1_users), ("MTF features used", mtf_users), ("H4 only", h4_only)]:
        if not group:
            continue
        avg_ev = statistics.mean(r.ev_mean for r in group)
        avg_pr = statistics.mean(r.precision_mean for r in group)
        print(f"\n  {label:<22}  n={len(group):>3}  AvgEV={avg_ev:+.2f}  AvgPrec={avg_pr*100:.1f}%")


# =============================================================================
# SECTION 11: RECOMMENDATIONS & VERDICT
# =============================================================================

def generate_recommendations(results: List[ConfigResult]):
    _hdr("💡 RECOMMENDATIONS & VERDICT")

    passed = [r for r in results if r.is_passed]
    failed = [r for r in results if not r.is_passed]
    gold   = [r for r in passed if r.is_gold]
    tier0  = [r for r in passed if r.tier == 0]
    tier1  = [r for r in passed if r.tier == 1]

    recs = []

    # Tier 0 findings
    if tier0:
        best_t0 = max(tier0, key=lambda r: r.ev_mean)
        recs.append({
            "priority": "HIGH",
            "finding": f"{len(tier0)} RUNNER config(s) found (Tier 0, R:R ≥ 2.5:1)",
            "action": f"Deploy best Tier 0: {best_t0.short_str()} "
                      f"EV={best_t0.ev_mean:+.2f} pips, thresh={best_t0.consensus_threshold:.2f}",
        })
    elif tier1:
        recs.append({
            "priority": "MEDIUM",
            "finding": f"No Tier 0 Runners. {len(tier1)} Tier 1 Ideal config(s) found",
            "action": "Use best Tier 1 config; try expanding TP range to find Tier 0",
        })
    else:
        recs.append({
            "priority": "LOW",
            "finding": "No high R:R configs passed",
            "action": "Review TP range in config. Current TP values may be too aggressive.",
        })

    # GOLD configs
    if gold:
        best_g = max(gold, key=lambda r: r.ev_mean)
        recs.append({
            "priority": "HIGH",
            "finding": f"GOLD config found: {best_g.short_str()}",
            "action": f"Recommended for deployment — EV={best_g.ev_mean:+.2f}, "
                      f"edge={best_g.edge_above_breakeven*100:+.1f}%",
        })

    # D1/MTF feature effectiveness
    d1_in_passed = [r for r in passed if any(f.startswith("d1_") for f in r.selected_features)]
    if d1_in_passed:
        pct = 100 * len(d1_in_passed) / max(len(passed), 1)
        recs.append({
            "priority": "LOW",
            "finding": f"D1 features appear in {pct:.0f}% of passed configs",
            "action": "Multi-timeframe data is contributing — D1 features worth retaining.",
        })
    else:
        recs.append({
            "priority": "MEDIUM",
            "finding": "D1 features not selected by RFE in any passed config",
            "action": "Consider lowering stat pre-filter threshold or reviewing D1 feature engineering.",
        })

    # Pass rate
    pass_rate = 100 * len(passed) / max(len(results), 1)
    if pass_rate < 20:
        recs.append({
            "priority": "HIGH",
            "finding": f"Low pass rate: {pass_rate:.1f}%",
            "action": "Consider relaxing acceptance criteria (min_edge or max_cv) or expanding config space.",
        })

    # Print
    for rec in recs:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec["priority"], "⚪")
        print(f"""
  {icon} [{rec['priority']}]
     Finding : {rec['finding']}
     Action  : {rec['action']}""")

    # ── Final Verdict ─────────────────────────────────────────────────────
    _sub("FINAL VERDICT")
    if gold:
        best = max(gold, key=lambda r: r.ev_mean)
        print(f"""
  {Fore.GREEN}✅ READY FOR PAPER TRADING{Style.RESET_ALL}

  Recommended Config : {best.short_str()}  [{best.tier_name}]
  Classification     : {best.classification}
  Expected Value     : {best.ev_mean:+.2f} pips/trade
  Precision          : {best.precision_mean*100:.1f}%
  Edge above b/e     : {best.edge_above_breakeven*100:+.1f}%
  Threshold          : {best.consensus_threshold:.2f}
  R:R                : {best.risk_reward_ratio:.2f}:1
""")
    elif passed:
        best = max(passed, key=lambda r: r.ev_mean)
        print(f"""
  {Fore.YELLOW}⚠️  PROCEED WITH CAUTION{Style.RESET_ALL}

  Best Config        : {best.short_str()}  [{best.tier_name}]
  Expected Value     : {best.ev_mean:+.2f} pips/trade
  Precision          : {best.precision_mean*100:.1f}%

  Suggestions:
  — Paper trade before going live
  — Run loss analysis to identify filters
  — Monitor first 100 trades closely
""")
    else:
        print(f"""
  {Fore.RED}❌ NOT READY FOR TRADING{Style.RESET_ALL}

  No profitable configurations found.
  Next Steps:
  1. Expand config space (wider TP/Hold ranges)
  2. Check data quality and D1 leakage validation
  3. Lower acceptance thresholds to understand near-misses
  4. Review feature engineering output
""")


# =============================================================================
# V4 COMPARISON (optional)
# =============================================================================

def compare_with_v4(stella_results: List[ConfigResult], v4_db_path: str):
    _hdr("📈 V4 vs STELLA ALPHA COMPARISON")

    if not Path(v4_db_path).exists():
        print(f"  V4 database not found: {v4_db_path}")
        return

    try:
        v4_results = load_results(v4_db_path)
    except Exception as e:
        print(f"  Failed to load V4 results: {e}")
        return

    v4_passed = [r for r in v4_results if r.is_passed]
    sa_passed = [r for r in stella_results if r.is_passed]

    def _stats(lst):
        evs   = [r.ev_mean for r in lst] or [0]
        precs = [r.precision_mean for r in lst] or [0]
        return max(evs), statistics.mean(evs), max(precs), statistics.mean(precs)

    v4_best_ev, v4_avg_ev, v4_best_pr, v4_avg_pr = _stats(v4_passed)
    sa_best_ev, sa_avg_ev, sa_best_pr, sa_avg_pr = _stats(sa_passed)

    v4_pass_rate = 100 * len(v4_passed) / max(len(v4_results), 1)
    sa_pass_rate = 100 * len(sa_passed) / max(len(stella_results), 1)

    print(f"""
  {'Metric':<28} {'V4':>10} {'Stella Alpha':>14} {'Delta':>10}
  {'─'*65}
  Configs tested         {len(v4_results):>10} {len(stella_results):>14}
  Pass rate              {v4_pass_rate:>9.1f}% {sa_pass_rate:>13.1f}%  {sa_pass_rate-v4_pass_rate:>+9.1f}pp
  Best EV (pips)         {v4_best_ev:>+10.2f} {sa_best_ev:>+14.2f}  {sa_best_ev-v4_best_ev:>+10.2f}
  Mean EV (pips)         {v4_avg_ev:>+10.2f} {sa_avg_ev:>+14.2f}  {sa_avg_ev-v4_avg_ev:>+10.2f}
  Best precision         {v4_best_pr*100:>9.1f}% {sa_best_pr*100:>13.1f}%  {(sa_best_pr-v4_best_pr)*100:>+9.1f}pp
  GOLD configs           {len([r for r in v4_passed if r.is_gold]):>10} {len([r for r in sa_passed if r.is_gold]):>14}
""")

    # High R:R comparison
    v4_high_rr = [r for r in v4_passed if r.risk_reward_ratio >= 1.67]
    sa_high_rr = [r for r in sa_passed if r.risk_reward_ratio >= 1.67]
    print(f"  High R:R passed (≥1.67:1) V4={len(v4_high_rr)}  Stella={len(sa_high_rr)}  "
          f"Δ={len(sa_high_rr)-len(v4_high_rr):+d}")

    # D1/MTF contribution in Stella Alpha
    d1_selected = sum(
        len([f for f in r.selected_features if f.startswith("d1_")])
        for r in sa_passed
    )
    mtf_selected = sum(
        len([f for f in r.selected_features if f.startswith("mtf_")])
        for r in sa_passed
    )
    print(f"\n  D1 features selected across all passed configs  : {d1_selected}")
    print(f"  MTF features selected across all passed configs : {mtf_selected}")

    if d1_selected + mtf_selected > 0:
        print(f"\n  {Fore.GREEN}✅ Multi-timeframe data IS contributing to model selection{Style.RESET_ALL}")
    else:
        print(f"\n  {Fore.YELLOW}⚠️  Multi-timeframe data was NOT selected by RFE{Style.RESET_ALL}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stella Alpha Diagnostic Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db",         "-d", default="artifacts/pure_ml_stella_alpha.db",
                        help="Path to checkpoint database")
    parser.add_argument("--top",        "-t", type=int, default=10,
                        help="Number of top configs to show in detailed views")
    parser.add_argument("--compare-v4",       default=None,
                        help="Path to V4 database for comparison")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        print("Run the pipeline first to generate results.")
        return

    print(f"\nLoading results from: {args.db}")
    results = load_results(args.db)
    if not results:
        print("No results found in database.")
        return
    print(f"Loaded {len(results)} configurations.\n")

    analyze_overview(results)
    analyze_tiers(results)
    analyze_gold(results)
    analyze_silver(results, top_n=args.top)
    analyze_passed(results, top_n=args.top)
    analyze_rejected(results)
    analyze_near_misses(results)
    analyze_parameter_patterns(results)
    analyze_feature_importance(results)
    analyze_d1_mtf_features(results)
    generate_recommendations(results)

    if args.compare_v4:
        compare_with_v4(results, args.compare_v4)

    print(f"\n{'='*W}")
    print(f"  Analysis complete.")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
