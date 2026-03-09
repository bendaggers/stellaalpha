"""
Tier Classification System for Stella Alpha

Classifies trading configurations into tiers based on Risk:Reward ratio.

STELLA ALPHA CONFIG SPACE:
- SL is fixed at 30 pips
- TP ranges from 50 to 150 pips
- Minimum R:R is 1.67:1 (TP=50/SL=30)

TIER DEFINITIONS:
─────────────────
TIER 0 - RUNNERS 🚀 (R:R >= 2.5:1)
    TP >= 75 with SL=30
    One winner covers 2.5-5 losers
    Min win rate needed: 17-29%

TIER 1 - IDEAL ⭐ (R:R 1.67:1 to 2.49:1)
    TP 50-70 with SL=30
    One winner covers 1.67-2.33 losers
    Min win rate needed: 30-37%

NOTE: With fixed SL=30 and minimum TP=50:
      ALL configs have R:R >= 1.67:1
      No unfavorable configs possible!
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json


class Tier(Enum):
    """Trade tier classification."""
    TIER_0_RUNNER = 0   # R:R >= 2.5:1 (TP >= 75)
    TIER_1_IDEAL = 1    # R:R 1.67:1 to 2.49:1 (TP 50-70)


@dataclass
class TierClassification:
    """Tier classification result for a config."""
    config_id: str
    tp_pips: int
    sl_pips: int
    max_hold: int
    risk_reward_ratio: float
    tier: Tier
    tier_name: str
    tier_emoji: str
    min_winrate_needed: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'config_id': self.config_id,
            'tp_pips': self.tp_pips,
            'sl_pips': self.sl_pips,
            'max_hold': self.max_hold,
            'risk_reward_ratio': self.risk_reward_ratio,
            'tier': self.tier.value,
            'tier_name': self.tier_name,
            'tier_emoji': self.tier_emoji,
            'min_winrate_needed': self.min_winrate_needed,
            'description': self.description
        }


def calculate_risk_reward(tp_pips: int, sl_pips: int) -> float:
    """Calculate R:R ratio (TP/SL)."""
    return tp_pips / sl_pips if sl_pips > 0 else 0


def calculate_min_winrate(tp_pips: int, sl_pips: int) -> float:
    """
    Calculate minimum win rate needed to break even.
    
    Formula: SL / (TP + SL)
    
    Example:
        TP=100, SL=30: 30/(100+30) = 23.1%
        TP=50, SL=30: 30/(50+30) = 37.5%
    """
    return sl_pips / (tp_pips + sl_pips) if (tp_pips + sl_pips) > 0 else 0


def classify_tier(tp_pips: int, sl_pips: int, max_hold: int) -> TierClassification:
    """
    Classify a config into a tier based on R:R ratio.
    
    Stella Alpha Config Space:
    - SL is fixed at 30 pips
    - TP ranges from 50 to 150 pips
    - Minimum R:R is 1.67:1 (TP=50/SL=30)
    
    Tier 0 (Runner):  R:R >= 2.5:1 (TP >= 75)
    Tier 1 (Ideal):   R:R 1.67:1 to 2.49:1 (TP 50-70)
    
    Args:
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
        max_hold: Maximum holding period in bars
        
    Returns:
        TierClassification with tier info
    """
    rr = calculate_risk_reward(tp_pips, sl_pips)
    min_wr = calculate_min_winrate(tp_pips, sl_pips)
    config_id = f"TP{tp_pips}_SL{sl_pips}_H{max_hold}"
    
    if rr >= 2.5:
        return TierClassification(
            config_id=config_id,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            max_hold=max_hold,
            risk_reward_ratio=rr,
            tier=Tier.TIER_0_RUNNER,
            tier_name="TIER 0 - RUNNER",
            tier_emoji="🚀",
            min_winrate_needed=min_wr,
            description=f"Risk {sl_pips} to make {tp_pips} ({rr:.2f}:1) - Need only {min_wr*100:.1f}% win rate"
        )
    else:  # R:R >= 1.67 (minimum in Stella Alpha)
        return TierClassification(
            config_id=config_id,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            max_hold=max_hold,
            risk_reward_ratio=rr,
            tier=Tier.TIER_1_IDEAL,
            tier_name="TIER 1 - IDEAL",
            tier_emoji="⭐",
            min_winrate_needed=min_wr,
            description=f"Risk {sl_pips} to make {tp_pips} ({rr:.2f}:1) - Need {min_wr*100:.1f}% win rate"
        )


def classify_all_configs(configs: List[tuple]) -> Dict[str, List[TierClassification]]:
    """
    Classify all configs and organize by tier.
    
    Args:
        configs: List of (tp_pips, sl_pips, max_hold) tuples
        
    Returns:
        Dictionary with 'tier_0' and 'tier_1' lists
    """
    result = {
        'tier_0': [],
        'tier_1': []
    }
    
    for tp, sl, hold in configs:
        classification = classify_tier(tp, sl, hold)
        
        if classification.tier == Tier.TIER_0_RUNNER:
            result['tier_0'].append(classification)
        else:
            result['tier_1'].append(classification)
    
    # Sort each tier by R:R ratio (descending)
    result['tier_0'].sort(key=lambda x: x.risk_reward_ratio, reverse=True)
    result['tier_1'].sort(key=lambda x: x.risk_reward_ratio, reverse=True)
    
    return result


def generate_stella_alpha_config_space() -> List[tuple]:
    """
    Generate the full Stella Alpha config space.
    
    Returns:
        List of (tp_pips, sl_pips, max_hold) tuples
    """
    tp_values = list(range(50, 151, 10))  # 50, 60, 70, ..., 150 (11 values)
    sl_values = [30]  # Fixed at 30 pips
    hold_values = list(range(12, 73, 12))  # 12, 24, 36, 48, 60, 72 (6 values)
    
    configs = []
    for tp in tp_values:
        for sl in sl_values:
            for hold in hold_values:
                configs.append((tp, sl, hold))
    
    return configs


def print_tier_summary(classifications: Dict[str, List[TierClassification]]):
    """Print summary of tier classifications."""
    tier_0 = classifications['tier_0']
    tier_1 = classifications['tier_1']
    
    print("\n" + "=" * 70)
    print("  STELLA ALPHA TIER CLASSIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"""
    CONFIG SPACE:
    ─────────────────────────────────────────
    TP Range:       50-150 pips (step 10) = 11 values
    SL Fixed:       30 pips
    Hold Range:     12-72 bars (step 12) = 6 values
    Total Configs:  {len(tier_0) + len(tier_1)} configs
    
    TIER DISTRIBUTION:
    ─────────────────────────────────────────
    🚀 TIER 0 - RUNNERS (R:R >= 2.5:1):  {len(tier_0)} configs
    ⭐ TIER 1 - IDEAL (R:R 1.67-2.49:1): {len(tier_1)} configs
    
    KEY GUARANTEE:
    ─────────────────────────────────────────
    WORST case: TP=50 SL=30 (1.67:1 R:R) - Need only 37% win rate
    BEST case:  TP=150 SL=30 (5.0:1 R:R) - Need only 17% win rate
    
    ALL configs have favorable R:R (minimum 1.67:1) ✅
    """)
    
    # Show Tier 0 details
    if tier_0:
        print("    🚀 TIER 0 - RUNNERS (Best R:R first):")
        print("    " + "─" * 60)
        for c in tier_0[:5]:
            print(f"       {c.config_id}: R:R={c.risk_reward_ratio:.2f}:1, "
                  f"Need {c.min_winrate_needed*100:.1f}% WR")
        if len(tier_0) > 5:
            print(f"       ... and {len(tier_0) - 5} more Runner configs")
    
    # Show Tier 1 details
    if tier_1:
        print(f"\n    ⭐ TIER 1 - IDEAL:")
        print("    " + "─" * 60)
        for c in tier_1[:3]:
            print(f"       {c.config_id}: R:R={c.risk_reward_ratio:.2f}:1, "
                  f"Need {c.min_winrate_needed*100:.1f}% WR")
        if len(tier_1) > 3:
            print(f"       ... and {len(tier_1) - 3} more Ideal configs")
    
    print("\n" + "=" * 70)


def get_tier_color(tier: Tier) -> str:
    """Get ANSI color code for tier."""
    if tier == Tier.TIER_0_RUNNER:
        return "\033[92m"  # Green
    else:
        return "\033[93m"  # Yellow


def format_tier_badge(tier: Tier) -> str:
    """Format tier as a colored badge string."""
    if tier == Tier.TIER_0_RUNNER:
        return "🚀 RUNNER"
    else:
        return "⭐ IDEAL"


# =============================================================================
# MATH EXPLANATION: WHY RUNNERS ARE BETTER
# =============================================================================

def explain_runner_math():
    """Print explanation of why high R:R (Runner) configs are better."""
    print("\n" + "=" * 70)
    print("  MATH: WHY RUNNERS ARE BETTER")
    print("=" * 70)
    
    print("""
    SCENARIO A: High Win Rate, Low R:R (V4 Style)
    ──────────────────────────────────────────────
    TP=30, SL=50, Win Rate=67%
    100 trades:
      67 wins × 30 pips = +2,010 pips
      33 losses × 50 pips = -1,650 pips
      NET = +360 pips
      
    SCENARIO B: Lower Win Rate, High R:R (Runner Strategy)
    ──────────────────────────────────────────────────────
    TP=100, SL=30, Win Rate=45%
    100 trades:
      45 wins × 100 pips = +4,500 pips
      55 losses × 30 pips = -1,650 pips
      NET = +2,850 pips  ← 8x MORE PROFITABLE!
    
    SCENARIO C: Dream Runner
    ────────────────────────
    TP=150, SL=30, Win Rate=35%
    100 trades:
      35 wins × 150 pips = +5,250 pips
      65 losses × 30 pips = -1,950 pips
      NET = +3,300 pips  ← EVEN BETTER!
    
    BREAKEVEN WIN RATES BY R:R:
    ───────────────────────────
    TP=30  SL=50  (0.6:1)  → Need 62.5% to break even
    TP=50  SL=30  (1.67:1) → Need 37.5% to break even
    TP=80  SL=30  (2.67:1) → Need 27.3% to break even
    TP=100 SL=30  (3.33:1) → Need 23.1% to break even
    TP=150 SL=30  (5:1)    → Need 16.7% to break even
    
    CONCLUSION:
    With D1 confluence, if we can predict 100+ pip moves with even 
    35-45% accuracy, we have a MUCH more profitable system than 
    predicting 30-pip moves with 65% accuracy.
    """)
    print("=" * 70)


if __name__ == '__main__':
    # Generate full config space
    configs = generate_stella_alpha_config_space()
    print(f"\nGenerated {len(configs)} configs")
    
    # Classify all configs
    classifications = classify_all_configs(configs)
    
    # Print summary
    print_tier_summary(classifications)
    
    # Show math explanation
    explain_runner_math()
