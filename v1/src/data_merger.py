"""
D1 Safe Merge Module for Stella Alpha

CRITICAL: This is the most important code in Stella Alpha.
Any bug here invalidates all results.

PURPOSE:
Merge D1 (Daily) data into H4 (4-hour) data WITHOUT data leakage.

RULE: For each H4 candle, use ONLY the most recent COMPLETED D1 candle.
      This means using the PREVIOUS day's D1 data.

EXAMPLE:
┌─────────────────────┬────────────────────┐
│ H4 Timestamp        │ D1 Data From       │
├─────────────────────┼────────────────────┤
│ 2024-01-15 04:00    │ 2024-01-14         │
│ 2024-01-15 08:00    │ 2024-01-14         │
│ 2024-01-15 12:00    │ 2024-01-14         │
│ 2024-01-15 16:00    │ 2024-01-14         │
│ 2024-01-15 20:00    │ 2024-01-14         │
│ 2024-01-16 00:00    │ 2024-01-15         │ ← D1 Jan 15 now available
│ 2024-01-16 04:00    │ 2024-01-15         │
└─────────────────────┴────────────────────┘
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class MergeStats:
    """Statistics about the H4-D1 merge."""
    total_h4_rows: int
    h4_date_range: str
    d1_date_range: str
    d1_columns_added: int
    rows_with_d1_data: int
    rows_without_d1_data: int
    rows_dropped: int
    final_rows: int


@dataclass
class LeakageValidationResult:
    """Result of data leakage validation."""
    is_valid: bool
    violations_count: int
    violations_sample: Optional[pd.DataFrame]
    message: str


def load_h4_data(
    filepath: str,
    timestamp_col: str = 'timestamp',
    timestamp_format: str = '%Y.%m.%d %H:%M:%S'
) -> pd.DataFrame:
    """
    Load H4 data from CSV.
    
    Args:
        filepath: Path to H4 CSV file
        timestamp_col: Name of timestamp column
        timestamp_format: Format of timestamp
        
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    
    # Parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    return df


def load_d1_data(
    filepath: str,
    timestamp_col: str = 'timestamp',
    timestamp_format: str = '%Y.%m.%d %H:%M:%S'
) -> pd.DataFrame:
    """
    Load D1 data from CSV.
    
    Args:
        filepath: Path to D1 CSV file
        timestamp_col: Name of timestamp column
        timestamp_format: Format of timestamp
        
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    
    # Parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    return df


def merge_h4_d1_safe(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    validate: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, MergeStats]:
    """
    Merge D1 data into H4 data WITHOUT data leakage.
    
    RULE: For each H4 candle, use ONLY the most recent
          COMPLETED D1 candle (previous day).
    
    Args:
        df_h4: H4 dataframe with 'timestamp' column
        df_d1: D1 dataframe with 'timestamp' column
        validate: If True, run leakage validation
        verbose: If True, print progress
        
    Returns:
        Tuple of (merged DataFrame, MergeStats)
        
    Raises:
        ValueError: If data leakage is detected
    """
    df_h4 = df_h4.copy()
    df_d1 = df_d1.copy()
    
    initial_h4_rows = len(df_h4)
    
    if verbose:
        print(f"   Merging H4 ({len(df_h4):,} rows) with D1 ({len(df_d1):,} rows)...")
    
    # 1. Ensure timestamps are datetime
    df_h4['timestamp'] = pd.to_datetime(df_h4['timestamp'])
    df_d1['timestamp'] = pd.to_datetime(df_d1['timestamp'])
    
    # 2. Get DATE of each H4 candle (normalized to midnight)
    df_h4['_h4_date'] = df_h4['timestamp'].dt.normalize()
    
    # 3. D1 data becomes available AFTER the day closes
    #    So a D1 candle dated Jan 15 is available starting Jan 16 00:00
    #    We need to use Jan 14's D1 data for H4 candles on Jan 15 (before midnight)
    df_d1['_d1_close_date'] = df_d1['timestamp'].dt.normalize()
    df_d1['_d1_available_from'] = df_d1['_d1_close_date'] + pd.Timedelta(days=1)
    
    # 4. Rename D1 columns with 'd1_' prefix (except helper columns)
    helper_cols = ['_d1_close_date', '_d1_available_from']
    d1_cols_to_rename = [c for c in df_d1.columns if c not in ['timestamp'] + helper_cols]
    rename_map = {c: f'd1_{c}' for c in d1_cols_to_rename}
    rename_map['timestamp'] = 'd1_timestamp'
    df_d1 = df_d1.rename(columns=rename_map)
    
    # 5. Sort both dataframes by time
    df_h4 = df_h4.sort_values('timestamp').reset_index(drop=True)
    df_d1 = df_d1.sort_values('_d1_available_from').reset_index(drop=True)
    
    # 6. Merge using merge_asof
    #    For each H4 row, find the most recent D1 where available_from <= h4_date
    df_merged = pd.merge_asof(
        df_h4,
        df_d1.drop(columns=['_d1_close_date']),
        left_on='_h4_date',
        right_on='_d1_available_from',
        direction='backward'  # Use most recent PAST D1
    )
    
    # 7. Clean up helper columns
    df_merged = df_merged.drop(columns=['_h4_date', '_d1_available_from'], errors='ignore')
    
    # 8. Count D1 columns added
    d1_columns = [c for c in df_merged.columns if c.startswith('d1_')]
    rows_with_d1 = df_merged['d1_timestamp'].notna().sum()
    rows_without_d1 = df_merged['d1_timestamp'].isna().sum()
    
    # 9. Validate no leakage
    if validate:
        validation_result = validate_no_leakage(df_merged)
        if not validation_result.is_valid:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: {validation_result.violations_count} rows have future D1 data!\n"
                f"{validation_result.message}\n"
                f"First violations:\n{validation_result.violations_sample}"
            )
        if verbose:
            print(f"   ✓ Leakage validation PASSED")
    
    # 10. Create stats
    stats = MergeStats(
        total_h4_rows=initial_h4_rows,
        h4_date_range=f"{df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}",
        d1_date_range=f"{df_merged['d1_timestamp'].min()} to {df_merged['d1_timestamp'].max()}",
        d1_columns_added=len(d1_columns),
        rows_with_d1_data=rows_with_d1,
        rows_without_d1_data=rows_without_d1,
        rows_dropped=0,  # Not dropping yet
        final_rows=len(df_merged)
    )
    
    if verbose:
        print(f"   D1 columns added: {len(d1_columns)}")
        print(f"   Rows with D1 data: {rows_with_d1:,}")
        print(f"   Rows without D1 data: {rows_without_d1:,}")
    
    return df_merged, stats


def validate_no_leakage(df_merged: pd.DataFrame) -> LeakageValidationResult:
    """
    Validate that no D1 data comes from same day or future.
    
    RULE: d1_timestamp.date() must be < timestamp.date()
          (D1 date must be strictly BEFORE H4 date)
    
    Exception: H4 candles at exactly 00:00 CAN use previous day's D1
               because that D1 just became available at midnight.
    
    Returns:
        LeakageValidationResult with validation status
    """
    df = df_merged.copy()
    
    # Only check rows with D1 data
    df = df[df['d1_timestamp'].notna()]
    
    if len(df) == 0:
        return LeakageValidationResult(
            is_valid=True,
            violations_count=0,
            violations_sample=None,
            message="No D1 data to validate"
        )
    
    h4_date = pd.to_datetime(df['timestamp']).dt.date
    d1_date = pd.to_datetime(df['d1_timestamp']).dt.date
    
    # D1 date must be strictly before H4 date
    # This ensures we're using COMPLETED D1 data
    violations = df[d1_date >= h4_date]
    
    is_valid = len(violations) == 0
    
    if is_valid:
        return LeakageValidationResult(
            is_valid=True,
            violations_count=0,
            violations_sample=None,
            message="All D1 data is from previous days - no leakage"
        )
    else:
        return LeakageValidationResult(
            is_valid=False,
            violations_count=len(violations),
            violations_sample=violations[['timestamp', 'd1_timestamp']].head(5),
            message=f"Found {len(violations)} rows where D1 date >= H4 date"
        )


def handle_missing_d1_data(
    df_merged: pd.DataFrame,
    strategy: str = 'drop',
    verbose: bool = True
) -> Tuple[pd.DataFrame, int]:
    """
    Handle rows where D1 data is missing.
    
    Args:
        df_merged: Merged DataFrame
        strategy: 'drop' to remove rows, 'keep' to keep with NaN
        verbose: If True, print progress
        
    Returns:
        Tuple of (processed DataFrame, rows_dropped)
    """
    initial_rows = len(df_merged)
    
    if strategy == 'drop':
        # Drop rows without D1 timestamp
        df_result = df_merged.dropna(subset=['d1_timestamp']).copy()
        
        # Also drop rows with NaN in critical D1 features
        d1_features = [c for c in df_result.columns 
                      if c.startswith('d1_') and c != 'd1_timestamp']
        
        # Drop rows where ALL d1 features are NaN
        if d1_features:
            df_result = df_result.dropna(subset=d1_features, how='all')
        
        df_result = df_result.reset_index(drop=True)
        rows_dropped = initial_rows - len(df_result)
        
        if verbose:
            print(f"   Dropped {rows_dropped:,} rows without valid D1 data")
            print(f"   Remaining rows: {len(df_result):,}")
        
        return df_result, rows_dropped
    
    else:
        # Keep all rows, D1 features will be NaN
        return df_merged, 0


# =============================================================================
# DATAMERGER CLASS - Object-Oriented Interface
# =============================================================================

class DataMerger:
    """
    Class wrapper for D1 merge functionality.
    
    Provides an object-oriented interface to merge_h4_d1_safe().
    Used by run_pure_ml_stella_alpha.py.
    
    Usage:
        merger = DataMerger()
        df_merged = merger.merge(df_h4, df_d1)
        is_valid = merger.validate_no_leakage(df_merged)
        merger.print_summary()
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize DataMerger.
        
        Args:
            verbose: If True, print progress messages
        """
        self.verbose = verbose
        self.last_stats: Optional[MergeStats] = None
        self.last_validation: Optional[LeakageValidationResult] = None
    
    def merge(
        self,
        df_h4: pd.DataFrame,
        df_d1: pd.DataFrame,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Merge D1 data into H4 data safely (no leakage).
        
        Args:
            df_h4: H4 DataFrame with 'timestamp' column
            df_d1: D1 DataFrame with 'timestamp' column
            validate: If True, validate no data leakage
            
        Returns:
            Merged DataFrame with d1_ prefixed columns
        """
        # Call the standalone function
        df_merged, stats = merge_h4_d1_safe(
            df_h4=df_h4,
            df_d1=df_d1,
            validate=validate,
            verbose=self.verbose,
        )
        
        self.last_stats = stats
        
        # Handle missing D1 data (drop rows without D1)
        df_merged, dropped = handle_missing_d1_data(
            df_merged,
            strategy='drop',
            verbose=self.verbose,
        )
        
        # Update stats with dropped count
        self.last_stats = MergeStats(
            total_h4_rows=stats.total_h4_rows,
            h4_date_range=stats.h4_date_range,
            d1_date_range=stats.d1_date_range,
            d1_columns_added=stats.d1_columns_added,
            rows_with_d1_data=stats.rows_with_d1_data,
            rows_without_d1_data=stats.rows_without_d1_data,
            rows_dropped=dropped,
            final_rows=len(df_merged),
        )
        
        if self.verbose:
            print(f"   Final merged rows: {len(df_merged):,}")
        
        return df_merged
    
    def validate_no_leakage(self, df: pd.DataFrame) -> bool:
        """
        Validate that merged DataFrame has no data leakage.
        
        Args:
            df: Merged DataFrame to validate
            
        Returns:
            True if no leakage, False otherwise
        """
        result = validate_no_leakage(df)
        self.last_validation = result
        return result.is_valid
    
    def get_stats(self) -> Optional[MergeStats]:
        """Return stats from last merge operation."""
        return self.last_stats
    
    def get_validation_result(self) -> Optional[LeakageValidationResult]:
        """Return validation result from last validation."""
        return self.last_validation
    
    def print_summary(self):
        """Print summary of last merge."""
        if self.last_stats:
            print_merge_summary(self.last_stats)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash of the dataframe for cache validation."""
    hash_str = f"{len(df)}_{df['close'].iloc[0]:.6f}_{df['close'].iloc[-1]:.6f}"
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def print_merge_summary(stats: MergeStats):
    """Print formatted merge summary."""
    print("\n" + "=" * 60)
    print("  H4-D1 MERGE SUMMARY")
    print("=" * 60)
    print(f"""
    H4 Rows (input):      {stats.total_h4_rows:,}
    H4 Date Range:        {stats.h4_date_range}
    D1 Date Range:        {stats.d1_date_range}
    D1 Columns Added:     {stats.d1_columns_added}
    
    Rows with D1 data:    {stats.rows_with_d1_data:,}
    Rows without D1:      {stats.rows_without_d1_data:,}
    Rows dropped:         {stats.rows_dropped:,}
    Final rows:           {stats.final_rows:,}
    """)
    print("=" * 60)


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_merge_leakage_prevention():
    """Unit test for data leakage prevention."""
    print("\n" + "=" * 60)
    print("  RUNNING LEAKAGE PREVENTION TESTS")
    print("=" * 60)
    
    # Create test H4 data (6 candles per day)
    h4_timestamps = pd.date_range('2024-01-15 00:00', periods=18, freq='4h')
    h4_data = pd.DataFrame({
        'timestamp': h4_timestamps,
        'open': [1.08] * 18,
        'high': [1.09] * 18,
        'low': [1.07] * 18,
        'close': [1.08] * 18,
        'volume': [1000] * 18
    })
    
    # Create test D1 data
    d1_timestamps = pd.date_range('2024-01-13', periods=5, freq='D')
    d1_data = pd.DataFrame({
        'timestamp': d1_timestamps,
        'open': [1.07, 1.075, 1.08, 1.085, 1.09],
        'high': [1.08, 1.085, 1.09, 1.095, 1.10],
        'low': [1.06, 1.065, 1.07, 1.075, 1.08],
        'close': [1.075, 1.08, 1.085, 1.09, 1.095],
        'volume': [50000, 51000, 52000, 53000, 54000],
        'rsi_value': [55, 60, 65, 70, 75]
    })
    
    # Merge
    merged, stats = merge_h4_d1_safe(h4_data, d1_data, validate=True, verbose=False)
    
    # Test 1: H4 on Jan 15 should use D1 from Jan 14
    jan15_rows = merged[merged['timestamp'].dt.date == pd.Timestamp('2024-01-15').date()]
    
    test1_passed = True
    for _, row in jan15_rows.iterrows():
        if pd.notna(row['d1_timestamp']):
            d1_date = pd.to_datetime(row['d1_timestamp']).date()
            expected_date = pd.Timestamp('2024-01-14').date()
            if d1_date != expected_date:
                print(f"   ✗ FAIL: H4 on Jan 15 uses D1 from {d1_date}, expected {expected_date}")
                test1_passed = False
    
    if test1_passed:
        print("   ✓ Test 1 PASSED: Jan 15 H4 candles use Jan 14 D1 data")
    
    # Test 2: H4 on Jan 16 should use D1 from Jan 15
    jan16_rows = merged[merged['timestamp'].dt.date == pd.Timestamp('2024-01-16').date()]
    
    test2_passed = True
    for _, row in jan16_rows.iterrows():
        if pd.notna(row['d1_timestamp']):
            d1_date = pd.to_datetime(row['d1_timestamp']).date()
            expected_date = pd.Timestamp('2024-01-15').date()
            if d1_date != expected_date:
                print(f"   ✗ FAIL: H4 on Jan 16 uses D1 from {d1_date}, expected {expected_date}")
                test2_passed = False
    
    if test2_passed:
        print("   ✓ Test 2 PASSED: Jan 16 H4 candles use Jan 15 D1 data")
    
    # Test 3: D1 columns should be prefixed
    d1_cols = [c for c in merged.columns if c.startswith('d1_')]
    if len(d1_cols) >= 5:  # At least timestamp + 4 data columns
        print(f"   ✓ Test 3 PASSED: D1 columns prefixed correctly ({len(d1_cols)} columns)")
    else:
        print(f"   ✗ FAIL: Expected at least 5 D1 columns, got {len(d1_cols)}")
    
    # Test 4: Validation should pass
    validation_result = validate_no_leakage(merged)
    if validation_result.is_valid:
        print("   ✓ Test 4 PASSED: No leakage detected")
    else:
        print(f"   ✗ FAIL: Leakage detected - {validation_result.message}")
    
    # Test 5: Test DataMerger class
    print("\n   Testing DataMerger class...")
    merger = DataMerger(verbose=False)
    df_merged = merger.merge(h4_data, d1_data, validate=True)
    is_valid = merger.validate_no_leakage(df_merged)
    
    if is_valid and len(df_merged) > 0:
        print("   ✓ Test 5 PASSED: DataMerger class works correctly")
    else:
        print("   ✗ FAIL: DataMerger class failed")
    
    print("\n" + "=" * 60)
    
    return test1_passed and test2_passed and validation_result.is_valid


if __name__ == '__main__':
    # Run tests
    success = test_merge_leakage_prevention()
    
    if success:
        print("\n✅ All leakage prevention tests PASSED")
    else:
        print("\n❌ Some tests FAILED")
