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
# 1. OPTUNA VALIDATED PARAMETERS (5-MIN TIMEFRAME)
# =====================================================================
BEST_PARAMS = {
    "rsi_min": 60,
    "adx_min": 25,
    "adx_max": 75,
    "use_ema": True,
    "use_supertrend": False,
    "req_active_slope": False,
    "use_macd": False,
    "bb_max_width": 0.035,
    "sl_atr": 1.5,
    "target_atr": 5.0,
    "trailing_sl_atr": 1.2,
    "max_daily_trades": 3,
    "max_daily_profit": 12000,
}

if not TESTING_DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {TESTING_DATA_PATH}. Did you run database_builder.py?")

df = pd.read_parquet(TESTING_DATA_PATH)
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# Pre-extract base 1-minute arrays for execution simulation
DATES = df.index.date
CE_HIGH = df['ce_high'].values
CE_LOW = df['ce_low'].values
CE_CLOSE = df['ce_close'].values
CE_NEXT_OPEN = df['ce_next_open'].values

PE_HIGH = df['pe_high'].values
PE_LOW = df['pe_low'].values
PE_CLOSE = df['pe_close'].values
PE_NEXT_OPEN = df['pe_next_open'].values

SLIPPAGE = 0.25  # Constant Spread slippage on market orders

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
# 2. MULTI-TIMEFRAME RESAMPLING FOR STRATEGY SIGNALS
# =====================================================================
print("Resampling 1-minute data into 5-minute candles for signal generation...")

df_5m = df.resample('5min', origin='start_day').agg({
    'close': 'last',
    'RSI': 'last',
    'ADX': 'last',
    'EMA_200': 'last',
    'SUPERTd': 'last',
    'SUPERT_slope': 'last',
    'MACD': 'last',
    'MACD_signal': 'last',
    'BB_width': 'last',
    'ce_ATR': 'last',
    'pe_ATR': 'last',
}).dropna()

ce_mask_5m = (df_5m['RSI'] > BEST_PARAMS['rsi_min']) & (df_5m['ADX'] > BEST_PARAMS['adx_min']) & (df_5m['ADX'] < BEST_PARAMS['adx_max'])
pe_mask_5m = (df_5m['RSI'] < (100 - BEST_PARAMS['rsi_min'])) & (df_5m['ADX'] > BEST_PARAMS['adx_min']) & (df_5m['ADX'] < BEST_PARAMS['adx_max'])

if BEST_PARAMS['use_ema']:
    ce_mask_5m &= (df_5m['close'] > df_5m['EMA_200'])
    pe_mask_5m &= (df_5m['close'] < df_5m['EMA_200'])

if BEST_PARAMS['use_supertrend']:
    ce_mask_5m &= (df_5m['SUPERTd'] == 1)
    pe_mask_5m &= (df_5m['SUPERTd'] == -1)

if BEST_PARAMS['req_active_slope']:
    ce_mask_5m &= (df_5m['SUPERT_slope'] > 0)
    pe_mask_5m &= (df_5m['SUPERT_slope'] < 0)

if BEST_PARAMS['use_macd']:
    ce_mask_5m &= (df_5m['MACD'] > df_5m['MACD_signal'])
    pe_mask_5m &= (df_5m['MACD'] < df_5m['MACD_signal'])

ce_mask_5m &= (df_5m['BB_width'] < BEST_PARAMS['bb_max_width'])
pe_mask_5m &= (df_5m['BB_width'] < BEST_PARAMS['bb_max_width'])

# Map 5-minute signal timestamps back to exact 1-minute array indices
ce_signal_timestamps = df_5m.index[ce_mask_5m]
pe_signal_timestamps = df_5m.index[pe_mask_5m]

ce_indices = df.index.get_indexer(ce_signal_timestamps)
pe_indices = df.index.get_indexer(pe_signal_timestamps)

ce_indices = ce_indices[ce_indices != -1]
pe_indices = pe_indices[pe_indices != -1]

# Set targets/stops based on execution fills
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

# =====================================================================
# 3. UNIFIED CHRONOLOGICAL EVALUATOR (1-MIN STEPPING)
# =====================================================================
def evaluate_and_report_unified(ce_indices, pe_indices, ce_sl_arr, ce_tg_arr, ce_trail_arr, pe_sl_arr, pe_tg_arr, pe_trail_arr, max_daily_trades, max_daily_profit):
    stats = {
        'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
        'trades': 0, 'wins': 0, 'losses': 0
    }
    daily_logs = {}

    signals = []
    for i, idx in enumerate(ce_indices):
        signals.append((idx, 0, ce_sl_arr[i], ce_tg_arr[i], ce_trail_arr[i]))
    for i, idx in enumerate(pe_indices):
        signals.append((idx, 1, pe_sl_arr[i], pe_tg_arr[i], pe_trail_arr[i]))

    signals.sort(key=lambda x: x[0])

    last_exit_idx = -1
    current_date = None
    trades_today = 0
    profit_today = 0.0

    for sig in signals:
        start_idx, opt_type, sl, tg, trail_dist = sig
        
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
            
        if trades_today >= max_daily_trades or profit_today >= max_daily_profit:
            continue
            
        prices_high = CE_HIGH if opt_type == 0 else PE_HIGH
        prices_low = CE_LOW if opt_type == 0 else PE_LOW
        prices_close = CE_CLOSE if opt_type == 0 else PE_CLOSE
        prices_next_open = CE_NEXT_OPEN if opt_type == 0 else PE_NEXT_OPEN

        # Enter on the immediate next 1-minute open + slippage
        entry_price = prices_next_open[start_idx] + SLIPPAGE
        highest_seen = entry_price
        current_lot_size = get_nifty_lot_size(trade_date)
        
        curr_idx = start_idx + 1
        exit_price = 0.0
        
        # Step through 1-minute bars to manage target, SL, and trailing stoploss
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            # Pessimistic Check: Stop loss evaluated first
            if low <= sl:
                exit_price = sl - SLIPPAGE
                last_exit_idx = curr_idx
                break
            if high >= tg:
                exit_price = tg
                last_exit_idx = curr_idx
                break
                
            if high > highest_seen:
                highest_seen = high
                sl = max(sl, highest_seen - trail_dist)
                
            curr_idx += 1
            
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1] - SLIPPAGE
            last_exit_idx = curr_idx - 1
            
        trade_gross_pnl = (exit_price - entry_price) * current_lot_size
        trade_charges = calculate_options_charges(entry_price, exit_price, current_lot_size)
        trade_net_pnl = trade_gross_pnl - trade_charges
        
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

total_stats, daily_logs = evaluate_and_report_unified(
    ce_indices, pe_indices, ce_sl, ce_tg, ce_trail, pe_sl, pe_tg, pe_trail, 
    BEST_PARAMS['max_daily_trades'], BEST_PARAMS['max_daily_profit']
)

print("\n" + "="*50)
print("🎯 OUT-OF-SAMPLE VALIDATION REPORT (5M SIGNALS / 1M EXECUTION) 🎯")
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