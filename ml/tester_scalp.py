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
    'rsi_min': 45,
    'adx_min': 20,
    'use_ema': True,
    'use_supertrend': False,
    'req_active_slope': False,
    'use_macd': False,
    'bb_max_width': 0.06897211941654376,
    'sl_atr': 1.3,
    'target_atr': 2.0,
    'trailing_sl_atr': 0.3,
    'max_daily_trades': 50,
    'max_daily_profit': 18301.0
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
CE_NEXT_OPEN = df['ce_next_open'].values  # NEW: Latency simulator

# Extract PE arrays for bearish trades
PE_HIGH = df['pe_high'].values
PE_LOW = df['pe_low'].values
PE_CLOSE = df['pe_close'].values
PE_NEXT_OPEN = df['pe_next_open'].values  # NEW: Latency simulator

# Extract Trend & Filter arrays
SPOT_CLOSE = df['close'].values
EMA_200 = df['EMA_200'].values  
SUPERTD = df['SUPERTd'].values
SUPERT_SLOPE = df['SUPERT_slope'].values
MACD = df['MACD'].values
MACD_SIG = df['MACD_signal'].values
BB_WIDTH = df['BB_width'].values

# =====================================================================
# 1.5 DYNAMIC LOT SIZE & TAX CALCULATOR
# =====================================================================
SLIPPAGE = 0.5  # Constant Spread slippage applied to market orders

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
# 4. UNIFIED CHRONOLOGICAL EVALUATOR
# =====================================================================
def evaluate_and_report_unified(ce_indices, pe_indices, ce_sl_arr, ce_tg_arr, ce_trail_arr, pe_sl_arr, pe_tg_arr, pe_trail_arr, max_daily_trades, max_daily_profit):
    stats = {
        'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
        'trades': 0, 'wins': 0, 'losses': 0
    }
    daily_logs = {}

    # 1. Bundle all signals together (idx, type(0=CE, 1=PE), sl, target, trailing)
    signals = []
    for i, idx in enumerate(ce_indices):
        signals.append((idx, 0, ce_sl_arr[i], ce_tg_arr[i], ce_trail_arr[i]))
    for i, idx in enumerate(pe_indices):
        signals.append((idx, 1, pe_sl_arr[i], pe_tg_arr[i], pe_trail_arr[i]))

    # 2. Sort chronologically to permanently fix Time Leaks
    signals.sort(key=lambda x: x[0])

    last_exit_idx = -1
    current_date = None
    trades_today = 0
    profit_today = 0.0

    for sig in signals:
        start_idx, opt_type, sl, tg, trail_dist = sig
        
        # Position Lock: Prevents overlapping CE and PE trades
        if start_idx <= last_exit_idx:
            continue
        
        trade_date = DATES[start_idx]
        trade_date_str = str(trade_date)
        
        if trade_date != current_date:
            current_date = trade_date
            trades_today = 0
            profit_today = 0.0
            
        if trade_date_str not in daily_logs:
            daily_logs[trade_date_str] = {'trades': 0, 'net_pnl': 0.0}
            
        # Unified Daily limit enforcement
        if trades_today >= max_daily_trades:
            continue
        if profit_today >= max_daily_profit:
            continue
            
        # Select appropriate arrays based on option type
        if opt_type == 0:
            prices_high, prices_low, prices_close = CE_HIGH, CE_LOW, CE_CLOSE
            prices_next_open = CE_NEXT_OPEN
        else:
            prices_high, prices_low, prices_close = PE_HIGH, PE_LOW, PE_CLOSE
            prices_next_open = PE_NEXT_OPEN

        # LATENCY ENTRY: Enter exactly on the NEXT candle's open + spread penalty
        entry_price = prices_next_open[start_idx] + SLIPPAGE
        highest_seen = entry_price
        current_lot_size = get_nifty_lot_size(trade_date)
        
        curr_idx = start_idx + 1
        exit_price = 0.0
        
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            # --- THE PESSIMISTIC FLIP (Safety First) ---
            # Check Stoploss BEFORE Target to simulate the absolute worst-case scenario.
            if low <= sl:
                exit_price = sl - SLIPPAGE # Market Order: Pay Spread Penalty
                last_exit_idx = curr_idx
                break
            if high >= tg:
                exit_price = tg # Limit Order: Zero Slippage
                last_exit_idx = curr_idx
                break
                
            if high > highest_seen:
                highest_seen = high
                sl = max(sl, highest_seen - trail_dist)
                
            curr_idx += 1
            
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1] - SLIPPAGE # EOD Market Order
            last_exit_idx = curr_idx - 1
            
        trade_gross_pnl = (exit_price - entry_price) * current_lot_size
        trade_charges = calculate_options_charges(entry_price, exit_price, current_lot_size)
        trade_net_pnl = trade_gross_pnl - trade_charges
        
        # Live updates
        daily_logs[trade_date_str]['trades'] += 1
        daily_logs[trade_date_str]['net_pnl'] += trade_net_pnl
        
        stats['gross_pnl'] += trade_gross_pnl
        stats['total_charges'] += trade_charges
        stats['net_pnl'] += trade_net_pnl
        stats['trades'] += 1
        
        trades_today += 1
        profit_today += trade_net_pnl
        
        if trade_gross_pnl > trade_charges:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

    return stats, daily_logs

# =====================================================================
# 5. RUN THE REALITY CHECK
# =====================================================================
print("\nApplying Unified Chronological Scalping rules to unseen data...")

# Build Masks
ce_mask = (df['RSI'] > BEST_PARAMS['rsi_min']) & (df['ADX'] > BEST_PARAMS['adx_min'])
pe_mask = (df['RSI'] < (100 - BEST_PARAMS['rsi_min'])) & (df['ADX'] > BEST_PARAMS['adx_min'])

if BEST_PARAMS['use_ema']:
    ce_mask &= (SPOT_CLOSE > EMA_200)
    pe_mask &= (SPOT_CLOSE < EMA_200)
    
if BEST_PARAMS['use_supertrend']:
    ce_mask &= (SUPERTD == 1)
    pe_mask &= (SUPERTD == -1)

if BEST_PARAMS['req_active_slope']:
    ce_mask &= (SUPERT_SLOPE > 0)
    pe_mask &= (SUPERT_SLOPE < 0)

if BEST_PARAMS['use_macd']:
    ce_mask &= (MACD > MACD_SIG)
    pe_mask &= (MACD < MACD_SIG)

ce_mask &= (BB_WIDTH < BEST_PARAMS['bb_max_width'])
pe_mask &= (BB_WIDTH < BEST_PARAMS['bb_max_width'])

ce_indices = np.where(ce_mask)[0]
pe_indices = np.where(pe_mask)[0]

# --- EXTRACT ARRAYS AND PASS TO THE UNIFIED ENGINE IN ONE GO ---
# Build Arrays - Using REAL fill price (Next Open + Slippage) to set targets/stops
ce_sl, ce_tg, ce_trail = [], [], []
if len(ce_indices) > 0:
    ce_atrs = df['ce_ATR'].values[ce_indices]
    ce_fills = CE_NEXT_OPEN[ce_indices] + SLIPPAGE
    ce_sl = ce_fills - (ce_atrs * BEST_PARAMS['sl_atr'])
    ce_tg = ce_fills + (ce_atrs * BEST_PARAMS['target_atr'])
    ce_trail = ce_atrs * BEST_PARAMS['trailing_sl_atr'] 

pe_sl, pe_tg, pe_trail = [], [], []
if len(pe_indices) > 0:
    pe_atrs = df['pe_ATR'].values[pe_indices]
    pe_fills = PE_NEXT_OPEN[pe_indices] + SLIPPAGE
    pe_sl = pe_fills - (pe_atrs * BEST_PARAMS['sl_atr'])
    pe_tg = pe_fills + (pe_atrs * BEST_PARAMS['target_atr'])
    pe_trail = pe_atrs * BEST_PARAMS['trailing_sl_atr'] 

total_stats, daily_logs = evaluate_and_report_unified(
    ce_indices, pe_indices, ce_sl, ce_tg, ce_trail, pe_sl, pe_tg, pe_trail, 
    BEST_PARAMS['max_daily_trades'], BEST_PARAMS['max_daily_profit']
)

# =====================================================================
# 6. THE REPORT
# =====================================================================
print("\n" + "="*50)
print("🎯 OUT-OF-SAMPLE VALIDATION REPORT (UNIFIED SCALPING) 🎯")
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

if daily_logs:
    print("\n" + "="*50)
    print("📅 DAILY BREAKDOWN")
    print("="*50)
    print(f"{'Date':<15} | {'Trades':<8} | {'Net PnL'}")
    print("-" * 50)
    for d in sorted(daily_logs.keys()):
        day_stats = daily_logs[d]
        pnl_str = f"₹{day_stats['net_pnl']:,.2f}"
        print(f"{d:<15} | {day_stats['trades']:<8} | {pnl_str}")
    print("="*50)