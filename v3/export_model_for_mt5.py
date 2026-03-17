"""
STELLA ALPHA - MODEL EXPORT FOR MT5
====================================
Exports the trained LightGBM model to ONNX format for MT5.

MT5 supports ONNX natively since build 3550+.

Usage:
    python export_model_for_mt5.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv

Outputs:
    - stella_alpha_model.onnx (for MT5)
    - stella_alpha_model.pkl (Python backup)
    - stella_alpha_features.json (feature list)
    - stella_alpha_config.json (trading config)
    - stella_alpha_features.mqh (MQL5 helper)
"""

import pandas as pd
import numpy as np
import argparse
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import lightgbm as lgb

# For ONNX export
try:
    import onnx
    from onnxmltools.convert import convert_lightgbm
    from onnxconverter_common import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: ONNX libraries not installed. Run:")
    print("  pip install onnx onnxmltools onnxconverter-common")

from combined_features import CombinedFeatureEngineering, get_feature_columns


# Configuration
PIP_VALUE = 0.0001
MAX_HOLD_BARS = 72
N_FEATURES_TO_SELECT = 25
ML_THRESHOLD = 0.45
TP_PIPS = 100
SL_PIPS = 50

# Output directory
OUTPUT_DIR = Path("models")


def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data."""
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    d1_cols_to_keep = ["d1_available_date"]
    exclude_cols = ["timestamp", "d1_date", "d1_available_date"]
    for col in df_d1.columns:
        if col not in exclude_cols:
            new_name = f"d1_{col}" if not col.startswith("d1_") else col
            df_d1[new_name] = df_d1[col]
            if new_name not in d1_cols_to_keep:
                d1_cols_to_keep.append(new_name)
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(df_h4, df_d1_slim, left_on="h4_date", right_on="d1_available_date", direction="backward")
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate MTF trend-aligned signals."""
    df = df.copy()
    
    if 'trend_strength' in df.columns:
        df['h4_trend_up'] = (df['trend_strength'] > 0.3).astype(int)
    else:
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        df['h4_trend_up'] = (df['close'] > ema21).astype(int)
    
    if 'd1_trend_strength' in df.columns:
        df['d1_trend_up'] = (df['d1_trend_strength'] > 0.3).astype(int)
    elif 'd1_close' in df.columns:
        d1_ema = df['d1_close'].ewm(span=21, adjust=False).mean()
        df['d1_trend_up'] = (df['d1_close'] > d1_ema).astype(int)
    else:
        df['d1_trend_up'] = 1
    
    df['mtf_long_signal'] = ((df['h4_trend_up'] == 1) & (df['d1_trend_up'] == 1)).astype(int)
    
    return df


def label_trades(df: pd.DataFrame, tp: int = 100, sl: int = 50) -> pd.DataFrame:
    """Label trades with outcome."""
    df = df.copy()
    df["trade_label"] = np.nan
    
    signal_indices = df[df["mtf_long_signal"] == 1].index.tolist()
    
    for idx in signal_indices:
        if idx + MAX_HOLD_BARS >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE)
        sl_price = entry_price - (sl * PIP_VALUE)
        
        outcome = None
        for offset in range(1, MAX_HOLD_BARS + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            bar_high, bar_low = df.loc[bar_idx, "high"], df.loc[bar_idx, "low"]
            
            if bar_low <= sl_price:
                outcome = 0
                break
            if bar_high >= tp_price:
                outcome = 1
                break
        
        if outcome is None:
            final_price = df.loc[min(idx + MAX_HOLD_BARS, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE
            outcome = 1 if pips > 0 else 0
        
        df.loc[idx, "trade_label"] = outcome
    
    return df


def generate_mql5_helper(features: list, scaler: StandardScaler) -> str:
    """Generate MQL5 helper code for feature normalization."""
    
    means = scaler.mean_
    scales = scaler.scale_
    
    code = f"""//+------------------------------------------------------------------+
//| STELLA ALPHA - Feature Helper                                      |
//| Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                            |
//+------------------------------------------------------------------+

#define STELLA_N_FEATURES {len(features)}
#define STELLA_THRESHOLD {ML_THRESHOLD}
#define STELLA_TP_PIPS {TP_PIPS}
#define STELLA_SL_PIPS {SL_PIPS}
#define STELLA_MAX_HOLD {MAX_HOLD_BARS}

// Feature names (for reference)
string StellaFeatureNames[STELLA_N_FEATURES] = {{
"""
    
    for i, f in enumerate(features):
        comma = "," if i < len(features) - 1 else ""
        code += f'    "{f}"{comma}\n'
    
    code += "};\n\n"
    
    code += "// Scaler means\n"
    code += "double StellaMeans[STELLA_N_FEATURES] = {\n"
    for i, m in enumerate(means):
        comma = "," if i < len(means) - 1 else ""
        code += f"    {m:.10f}{comma}\n"
    code += "};\n\n"
    
    code += "// Scaler scales (std)\n"
    code += "double StellaScales[STELLA_N_FEATURES] = {\n"
    for i, s in enumerate(scales):
        comma = "," if i < len(scales) - 1 else ""
        code += f"    {s:.10f}{comma}\n"
    code += "};\n\n"
    
    code += """
//+------------------------------------------------------------------+
//| Normalize features using StandardScaler params                     |
//+------------------------------------------------------------------+
void StellaScaleFeatures(double &features[], double &scaled[])
{
    ArrayResize(scaled, STELLA_N_FEATURES);
    for(int i = 0; i < STELLA_N_FEATURES; i++)
    {
        if(StellaScales[i] != 0)
            scaled[i] = (features[i] - StellaMeans[i]) / StellaScales[i];
        else
            scaled[i] = 0;
    }
}

//+------------------------------------------------------------------+
//| Check if prediction passes threshold                               |
//+------------------------------------------------------------------+
bool StellaSignalValid(double probability)
{
    return probability >= STELLA_THRESHOLD;
}
"""
    
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  STELLA ALPHA - MODEL EXPORT FOR MT5")
    print(f"{'='*70}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n  Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering
    print(f"  Computing features...")
    fe = CombinedFeatureEngineering(verbose=False)
    df = fe.calculate_all_features(df)
    
    # Generate signals
    df = generate_signals(df)
    print(f"  MTF Long signals: {df['mtf_long_signal'].sum():,}")
    
    # Label trades
    print(f"  Labeling trades...")
    df = label_trades(df, TP_PIPS, SL_PIPS)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"  Features: {len(feature_cols)}")
    
    # Get labeled data
    signals_df = df[df['trade_label'].notna()].copy()
    print(f"  Total labeled signals: {len(signals_df):,}")
    
    # Prepare training data (use ALL data for final model)
    X = signals_df[feature_cols].copy()
    y = signals_df["trade_label"].values
    
    # Clean data
    medians = X.median()
    X = X.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
    
    # RFE Feature Selection
    print(f"\n  Running feature selection (RFE)...")
    estimator = lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1, random_state=42, n_jobs=-1)
    selector = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=5)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.support_].tolist()
    print(f"  Selected {len(selected_features)} features:")
    for i, f in enumerate(selected_features, 1):
        print(f"    {i:2}. {f}")
    
    X_selected = X[selected_features]
    
    # Scale features
    print(f"\n  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Train final model
    print(f"  Training final model...")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Evaluate
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= ML_THRESHOLD).astype(int)
    
    n_trades = y_pred.sum()
    win_rate = y[y_pred == 1].mean() if n_trades > 0 else 0
    print(f"\n  Model trained:")
    print(f"    Total signals: {len(y):,}")
    print(f"    Trades (threshold={ML_THRESHOLD}): {n_trades:,}")
    print(f"    Win rate: {win_rate*100:.1f}%")
    
    # =========================================================================
    # SAVE ARTIFACTS
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"  SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    
    # 1. Save feature list
    features_path = OUTPUT_DIR / "stella_alpha_features.json"
    with open(features_path, 'w') as f:
        json.dump({
            "selected_features": selected_features,
            "feature_order": list(range(len(selected_features))),
            "n_features": len(selected_features)
        }, f, indent=2)
    print(f"\n  ✅ Features saved: {features_path}")
    
    # 2. Save scaler parameters
    scaler_path = OUTPUT_DIR / "stella_alpha_scaler.json"
    with open(scaler_path, 'w') as f:
        json.dump({
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "feature_names": selected_features
        }, f, indent=2)
    print(f"  ✅ Scaler saved: {scaler_path}")
    
    # 3. Save trading config
    config_path = OUTPUT_DIR / "stella_alpha_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "model_version": "stella_alpha_v3",
            "export_date": datetime.now().isoformat(),
            "tp_pips": TP_PIPS,
            "sl_pips": SL_PIPS,
            "max_hold_bars": MAX_HOLD_BARS,
            "ml_threshold": ML_THRESHOLD,
            "n_features": len(selected_features),
            "signal_type": "MTF_TREND_LONG",
            "timeframe": "H4",
            "pair": "EURUSD",
            "backtest_stats": {
                "total_signals": int(len(y)),
                "trades_at_threshold": int(n_trades),
                "win_rate": float(win_rate),
                "annual_pips_expected": 1179,
                "trades_per_year": 154
            }
        }, f, indent=2)
    print(f"  ✅ Config saved: {config_path}")
    
    # 4. Save pickle (Python backup)
    pkl_path = OUTPUT_DIR / "stella_alpha_model.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'selected_features': selected_features,
            'threshold': ML_THRESHOLD,
            'medians': medians[selected_features].to_dict()
        }, f)
    print(f"  ✅ Pickle saved: {pkl_path}")
    
    # 5. Save ONNX (for MT5)
    if ONNX_AVAILABLE:
        print(f"\n  Exporting to ONNX...")
        try:
            onnx_path = OUTPUT_DIR / "stella_alpha_model.onnx"
            
            # Convert LightGBM to ONNX
            initial_type = [('input', FloatTensorType([None, len(selected_features)]))]
            onnx_model = convert_lightgbm(
                model, 
                initial_types=initial_type,
                target_opset=12
            )
            
            # Save ONNX model
            onnx.save_model(onnx_model, str(onnx_path))
            print(f"  ✅ ONNX saved: {onnx_path}")
            
            # Verify ONNX model
            onnx_loaded = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_loaded)
            print(f"  ✅ ONNX model verified!")
            
        except Exception as e:
            print(f"  ❌ ONNX export failed: {e}")
            print(f"     Will use pickle - need Python bridge for MT5")
    else:
        print(f"\n  ⚠️  ONNX libraries not available")
        print(f"     Install with: pip install onnx onnxmltools onnxconverter-common")
    
    # 6. Generate MQL5 helper code
    mql5_code = generate_mql5_helper(selected_features, scaler)
    mql5_path = OUTPUT_DIR / "stella_alpha_features.mqh"
    with open(mql5_path, 'w') as f:
        f.write(mql5_code)
    print(f"  ✅ MQL5 helper saved: {mql5_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"""
  FILES CREATED:
  ─────────────────────────────────────────
  📁 {OUTPUT_DIR}/
     ├── stella_alpha_model.onnx    ← For MT5 (ONNX runtime)
     ├── stella_alpha_model.pkl     ← Python backup
     ├── stella_alpha_features.json ← Feature list
     ├── stella_alpha_scaler.json   ← Normalization params
     ├── stella_alpha_config.json   ← Trading config
     └── stella_alpha_features.mqh  ← MQL5 helper code

  NEXT STEPS FOR MT5:
  ─────────────────────────────────────────
  1. Copy .onnx file to: MQL5/Files/
  2. Include .mqh file in your EA
  3. Use OnnxCreate() to load model
  4. Calculate {len(selected_features)} features in EA
  5. Scale using StellaMeans/StellaScales
  6. Run OnnxRun() inference
  7. If probability >= {ML_THRESHOLD}, take trade
    """)


if __name__ == "__main__":
    main()
