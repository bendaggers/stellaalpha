//+------------------------------------------------------------------+
//| STELLA ALPHA - Feature Helper                                      |
//| Auto-generated on 2026-03-17 08:18:28                            |
//+------------------------------------------------------------------+

#define STELLA_N_FEATURES 25
#define STELLA_THRESHOLD 0.45
#define STELLA_TP_PIPS 100
#define STELLA_SL_PIPS 50
#define STELLA_MAX_HOLD 72

// Feature names (for reference)
string StellaFeatureNames[STELLA_N_FEATURES] = {
    "adx",
    "atr_pct",
    "bb_position_rolling_std",
    "d1_atr_pct",
    "d1_bb_position",
    "d1_bb_width_pct",
    "d1_candle_body_pct",
    "d1_consecutive_bullish",
    "d1_middle_band",
    "d1_rsi_value",
    "d1_time_since_last_touch",
    "d1_trend_strength",
    "d1_upper_band",
    "h4_position_in_d1_range",
    "h4_vs_d1_trend",
    "rsi_roc_5",
    "rsi_rolling_max",
    "rsi_rolling_min",
    "rsi_rolling_std",
    "support_distance_pct",
    "time_since_last_touch",
    "trend_slope_3",
    "trend_strength",
    "upper_band",
    "volatility_ratio"
};

// Scaler means
double StellaMeans[STELLA_N_FEATURES] = {
    36.7909808949,
    0.3187228814,
    0.1895688873,
    0.7565594099,
    0.7092966783,
    3.4052165778,
    0.0236387285,
    4.9155595813,
    1.2473553301,
    61.1940437896,
    40.6092995169,
    2.8345446935,
    1.2685379014,
    1.3379778255,
    -1.5499881305,
    -0.6543931009,
    61.9752575053,
    45.3554305170,
    5.6705947850,
    13.7901850125,
    22.7373188406,
    -0.0016420780,
    1.2845565630,
    1.2655366683,
    1.0101556612
};

// Scaler scales (std)
double StellaScales[STELLA_N_FEATURES] = {
    16.3059014619,
    0.0902171395,
    0.0862089108,
    0.1754002211,
    0.2645416505,
    1.3702976625,
    0.4971232723,
    7.9262537582,
    0.1557354599,
    9.3291056935,
    26.8357923768,
    1.5350596664,
    0.1583493846,
    22.7063124437,
    1.6848168104,
    10.1202474041,
    10.9826836806,
    11.3539583320,
    2.3377528940,
    9.3430695848,
    25.3808994439,
    0.0234664656,
    0.7474805697,
    0.1582002404,
    0.1184305096
};


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
