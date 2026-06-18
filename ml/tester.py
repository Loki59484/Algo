import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.upstox_methods import UpstoxClient
from ml.database_builder import TESTING_DATA_PATH
ustox = UpstoxClient()

# =====================================================================
# 1. HARDCODE YOUR BEST PARAMETERS HERE
# =====================================================================
BEST_PARAMS = {
    'rsi_min': 47,
    'adx_min': 61,
    'use_ema': True,
    'use_supertrend': False,
    'req_active_slope': True,
    'use_macd': False,
    'bb_max_width': 0.014498753585377278,
    'sl_atr': 1.5,
    'target_atr': 8.0
}

if not TESTING_DATA_PATH.exists():
     raise FileNotFoundError(f"Could not find {TESTING_DATA_PATH}. Did you run the database_builder.py script?")
 
df = pd.read_parquet(TESTING_DATA_PATH)

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# Pre-extract base arrays
DATES = df.index.date
CE_HIGH = df['ce_high'].values
CE_LOW = df['ce_low'].values
CE_CLOSE = df['ce_close'].values

# Extract PE arrays for bearish trades
PE_HIGH = df['pe_high'].values
PE_LOW = df['pe_low'].values
PE_CLOSE = df['pe_close'].values

# Extract Trend & Filter arrays
SPOT_CLOSE = df['close'].values
EMA_200 = df['EMA_200'].values
SUPERTD = df['SUPERTd'].values
SUPERT_SLOPE = df['SUPERT_slope'].values
MACD = df['MACD'].values
MACD_SIG = df['MACD_signal'].values
BB_WIDTH = df['BB_width'].values

# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def get_nifty_lot_size(trade_date):
    if trade_date >= pd.Timestamp("2026-01-01").date(): return 65
    elif trade_date >= pd.Timestamp("2024-11-20").date(): return 75
    else: return 25

def calculate_options_charges(buy_price, sell_price, qty):
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    total_value = buy_value + sell_value
    brokerage = 40.0 
    stt = np.round(sell_value * 0.001)
    txn_charge = total_value * 0.0003503
    sebi_charge = total_value * 0.000001
    stamp_duty = np.round(buy_value * 0.00003)
    gst = (brokerage + txn_charge + sebi_charge) * 0.18
    return brokerage + stt + txn_charge + sebi_charge + stamp_duty + gst

# =====================================================================
# 4. UPGRADED EVALUATOR (Returns Stats)
# =====================================================================
def evaluate_and_report(signal_indices, stoplosses, targets, prices_high, prices_low, prices_close):
    stats = {
        'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
        'trades': 0, 'wins': 0, 'losses': 0
    }
    
    if len(signal_indices) == 0:
        return stats

    last_exit_idx = -1

    for i in range(len(signal_indices)):
        start_idx = signal_indices[i]
        
        if start_idx <= last_exit_idx:
            continue
        
        trade_date = DATES[start_idx]
        sl = stoplosses[i]
        tg = targets[i]
        entry_price = prices_close[start_idx]
        current_lot_size = get_nifty_lot_size(trade_date)
        
        curr_idx = start_idx + 1
        exit_price = 0.0
        
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            if high >= tg:
                exit_price = tg
                last_exit_idx = curr_idx
                break
            if low <= sl:
                exit_price = sl
                last_exit_idx = curr_idx
                break
            curr_idx += 1
            
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1]
            last_exit_idx = curr_idx - 1
            
        trade_gross_pnl = (exit_price - entry_price) * current_lot_size
        trade_charges = calculate_options_charges(entry_price, exit_price, current_lot_size)
        
        stats['gross_pnl'] += trade_gross_pnl
        stats['total_charges'] += trade_charges
        stats['net_pnl'] += (trade_gross_pnl - trade_charges)
        stats['trades'] += 1
        
        if trade_gross_pnl > trade_charges:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

    return stats

# =====================================================================
# 5. RUN THE REALITY CHECK
# =====================================================================
print("\nApplying rules to unseen data...")

# Build Masks using Best Params
ce_mask = (df['RSI'] > BEST_PARAMS['rsi_min']) & (df['ADX'] > BEST_PARAMS['adx_min'])
pe_mask = (df['RSI'] < (100 - BEST_PARAMS['rsi_min'])) & (df['ADX'] > BEST_PARAMS['adx_min'])

# EMA Filter
if BEST_PARAMS['use_ema']:
    ce_mask &= (SPOT_CLOSE > EMA_200)
    pe_mask &= (SPOT_CLOSE < EMA_200)
    
# Supertrend Direction Filter
if BEST_PARAMS['use_supertrend']:
    ce_mask &= (SUPERTD == 1)
    pe_mask &= (SUPERTD == -1)

# Active Slope Filter
if BEST_PARAMS['req_active_slope']:
    ce_mask &= (SUPERT_SLOPE > 0)
    pe_mask &= (SUPERT_SLOPE < 0)

# MACD Filter
if BEST_PARAMS['use_macd']:
    ce_mask &= (MACD > MACD_SIG)
    pe_mask &= (MACD < MACD_SIG)

# Bollinger Band Squeeze Filter
ce_mask &= (BB_WIDTH < BEST_PARAMS['bb_max_width'])
pe_mask &= (BB_WIDTH < BEST_PARAMS['bb_max_width'])

ce_indices = np.where(ce_mask)[0]
pe_indices = np.where(pe_mask)[0]

total_stats = {
    'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
    'trades': 0, 'wins': 0, 'losses': 0
}

# Evaluate CE
if len(ce_indices) > 0:
    ce_entry = CE_CLOSE[ce_indices]
    ce_atrs = df['ce_ATR'].values[ce_indices]
    ce_sl = ce_entry - (ce_atrs * BEST_PARAMS['sl_atr'])
    ce_tg = ce_entry + (ce_atrs * BEST_PARAMS['target_atr'])
    
    ce_stats = evaluate_and_report(ce_indices, ce_sl, ce_tg, CE_HIGH, CE_LOW, CE_CLOSE)
    for k in total_stats: total_stats[k] += ce_stats[k]

# Evaluate PE
if len(pe_indices) > 0:
    pe_entry = PE_CLOSE[pe_indices]
    pe_atrs = df['pe_ATR'].values[pe_indices]
    pe_sl = pe_entry - (pe_atrs * BEST_PARAMS['sl_atr'])
    pe_tg = pe_entry + (pe_atrs * BEST_PARAMS['target_atr'])
    
    pe_stats = evaluate_and_report(pe_indices, pe_sl, pe_tg, PE_HIGH, PE_LOW, PE_CLOSE)
    for k in total_stats: total_stats[k] += pe_stats[k]

# =====================================================================
# 6. THE REPORT
# =====================================================================
print("\n" + "="*50)
print("🎯 OUT-OF-SAMPLE VALIDATION REPORT 🎯")
print("="*50)
print(f"Total Trades Taken  : {total_stats['trades']}")

if total_stats['trades'] > 0:
    win_rate = (total_stats['wins'] / total_stats['trades']) * 100
    print(f"Winning Trades      : {total_stats['wins']}")
    print(f"Losing Trades       : {total_stats['losses']}")
    print(f"Win Rate            : {win_rate:.2f}%")
    print("-" * 50)
    print(f"Gross PnL           : ₹{total_stats['gross_pnl']:,.2f}")
    print(f"Taxes & Brokerage   : ₹{total_stats['total_charges']:,.2f}")
    print(f"NET PROFIT (REAL)   : ₹{total_stats['net_pnl']:,.2f}")
else:
    print("No trades triggered in the testing period.")
print("="*50)