from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import warnings
import logging
import optuna
import sys

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
logger = logging.getLogger(__name__)

# IMPORTING CUSTOM MODULES
from core.upstox_methods import UpstoxClient
from ml.database_builder import TRAINING_DATA_PATH
ustox = UpstoxClient()

if not TRAINING_DATA_PATH.exists():
     raise FileNotFoundError(f"Could not find {TRAINING_DATA_PATH}. Did you run the database_builder.py script?")
 
global_df = pd.read_parquet(TRAINING_DATA_PATH)

if not isinstance(global_df.index, pd.DatetimeIndex):
    global_df.index = pd.to_datetime(global_df.index)

# =====================================================================
# EXTRACT THE ARRAYS
# =====================================================================
DATES = global_df.index.date
CE_HIGH = global_df['ce_high'].values
CE_LOW = global_df['ce_low'].values
CE_CLOSE = global_df['ce_close'].values
CE_NEXT_OPEN = global_df['ce_next_open'].values  # Latency simulator

PE_HIGH = global_df['pe_high'].values
PE_LOW = global_df['pe_low'].values
PE_CLOSE = global_df['pe_close'].values
PE_NEXT_OPEN = global_df['pe_next_open'].values  # Latency simulator

SPOT_CLOSE = global_df['close'].values
EMA_200 = global_df['EMA_200'].values
SUPERTD = global_df['SUPERTd'].values
SUPERT_SLOPE = global_df['SUPERT_slope'].values
MACD = global_df['MACD'].values
MACD_SIG = global_df['MACD_signal'].values
BB_WIDTH = global_df['BB_width'].values

# =====================================================================
# 1.5 DYNAMIC LOT SIZE & TAX CALCULATOR
# =====================================================================
SLIPPAGE = 0.5  # Constant Spread slippage applied to market orders

def get_nifty_lot_size(trade_date):
    if trade_date >= datetime.date(2026, 1, 1): return 65
    elif trade_date >= datetime.date(2024, 11, 20): return 75
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
# 2. FAST UNIFIED PNL EVALUATOR
# =====================================================================
def fast_evaluate_unified(ce_indices, pe_indices, ce_sl_arr, ce_tg_arr, ce_trail_arr, pe_sl_arr, pe_tg_arr, pe_trail_arr, max_daily_trades, max_daily_profit):
    total_pnl = 0.0

    # Bundle and Sort chronologically
    signals = []
    for i, idx in enumerate(ce_indices):
        signals.append((idx, 0, ce_sl_arr[i], ce_tg_arr[i], ce_trail_arr[i]))
    for i, idx in enumerate(pe_indices):
        signals.append((idx, 1, pe_sl_arr[i], pe_tg_arr[i], pe_trail_arr[i]))

    # Sort strictly by the index (which represents time)
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

        if trade_date != current_date:
            current_date = trade_date
            trades_today = 0
            profit_today = 0.0

        if trades_today >= max_daily_trades:
            continue

        if profit_today >= max_daily_profit:
            continue

        if opt_type == 0:
            prices_high, prices_low, prices_close = CE_HIGH, CE_LOW, CE_CLOSE
            prices_next_open = CE_NEXT_OPEN
        else:
            prices_high, prices_low, prices_close = PE_HIGH, PE_LOW, PE_CLOSE
            prices_next_open = PE_NEXT_OPEN

        # LATENCY ENTRY: Enter exactly on the NEXT candle's open + spread penalty
        # (Entries are still Market Orders catching the breakout)
        entry_price = prices_next_open[start_idx] + SLIPPAGE
        highest_seen = entry_price
        current_lot_size = get_nifty_lot_size(trade_date)

        # Because we entered on next_open, execution starts on the NEXT candle
        curr_idx = start_idx + 1
        exit_price = 0.0

        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]

            # --- THE PESSIMISTIC FLIP (Safety First) ---
            # Check Stoploss BEFORE Target to simulate the absolute worst-case scenario.
            if low <= sl:
                # REVERTED TO MARKET ORDER: Safest way to guarantee an exit. 
                # We deduct SLIPPAGE because we are hitting the bid.
                exit_price = sl - SLIPPAGE 
                last_exit_idx = curr_idx
                break
            if high >= tg:
                # TARGET LIMIT ORDER: Placed in the book, hit precisely. Zero Slippage.
                exit_price = tg 
                last_exit_idx = curr_idx
                break

            if high > highest_seen:
                highest_seen = high
                sl = max(sl, highest_seen - trail_dist)

            curr_idx += 1

        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1] - SLIPPAGE # EOD Forced Market Order
            last_exit_idx = curr_idx - 1

        trade_gross_pnl = (exit_price - entry_price) * current_lot_size
        trade_charges = calculate_options_charges(entry_price, exit_price, current_lot_size)
        trade_net_pnl = trade_gross_pnl - trade_charges

        total_pnl += trade_net_pnl
        trades_today += 1
        profit_today += trade_net_pnl

    return total_pnl

# =====================================================================
# 3. THE OPTUNA OBJECTIVE
# =====================================================================
def objective(trial):
    rsi_min = trial.suggest_int("rsi_min", 50, 70)  
    adx_min = trial.suggest_int("adx_min", 25, 45)
    
    # NEW: Crash/Parabolic Prevention. Do not enter if ADX is dangerously high!
    adx_max = trial.suggest_int("adx_max", 50, 80)
    
    use_ema_filter = trial.suggest_categorical("use_ema", [True, False])
    use_supertrend_filter = trial.suggest_categorical("use_supertrend", [True, False])
    req_active_slope = trial.suggest_categorical("req_active_slope", [True, False])
    use_macd_filter = trial.suggest_categorical("use_macd", [True, False])
    
    bb_max_width = trial.suggest_float("bb_max_width", 0.01, 0.06, step=0.005)
    
    sl_atr = trial.suggest_float("sl_atr", 0.8, 2.0, step=0.2)
    target_atr = trial.suggest_float("target_atr", 2.0, 6.0, step=0.2)
    trailing_sl_atr = trial.suggest_float("trailing_sl_atr", 0.5, 2.0, step=0.2)
    
    max_daily_trades = trial.suggest_int("max_daily_trades", 2, 20)
    max_daily_profit = trial.suggest_int("max_daily_profit", 3000, 15000) 
    
    # Applied ADX Ceiling Protection
    ce_mask = (global_df['RSI'] > rsi_min) & (global_df['ADX'] > adx_min) & (global_df['ADX'] < adx_max)
    pe_mask = (global_df['RSI'] < (100 - rsi_min)) & (global_df['ADX'] > adx_min) & (global_df['ADX'] < adx_max)
    
    if use_ema_filter:
        ce_mask &= (SPOT_CLOSE > EMA_200)
        pe_mask &= (SPOT_CLOSE < EMA_200)
    if use_supertrend_filter:
        ce_mask &= (SUPERTD == 1)
        pe_mask &= (SUPERTD == -1)
    if req_active_slope:
        ce_mask &= (SUPERT_SLOPE > 0)
        pe_mask &= (SUPERT_SLOPE < 0)
    if use_macd_filter:
        ce_mask &= (MACD > MACD_SIG)
        pe_mask &= (MACD < MACD_SIG)
        
    ce_mask &= (BB_WIDTH < bb_max_width)
    pe_mask &= (BB_WIDTH < bb_max_width)
    
    ce_indices = np.where(ce_mask)[0]
    pe_indices = np.where(pe_mask)[0]
    
    if len(ce_indices) == 0 and len(pe_indices) == 0:
        return -99999.0

    ce_sl, ce_tg, ce_trail = [], [], []
    if len(ce_indices) > 0:
        ce_atrs = global_df['ce_ATR'].values[ce_indices]
        ce_fills = CE_NEXT_OPEN[ce_indices] + SLIPPAGE
        ce_sl = ce_fills - (ce_atrs * sl_atr)
        ce_tg = ce_fills + (ce_atrs * target_atr)
        ce_trail = ce_atrs * trailing_sl_atr

    pe_sl, pe_tg, pe_trail = [], [], []
    if len(pe_indices) > 0:
        pe_atrs = global_df['pe_ATR'].values[pe_indices]
        pe_fills = PE_NEXT_OPEN[pe_indices] + SLIPPAGE
        pe_sl = pe_fills - (pe_atrs * sl_atr)
        pe_tg = pe_fills + (pe_atrs * target_atr)
        pe_trail = pe_atrs * trailing_sl_atr

    net_profit = fast_evaluate_unified(
        ce_indices, pe_indices, ce_sl, ce_tg, ce_trail, pe_sl, pe_tg, pe_trail, 
        max_daily_trades, max_daily_profit
    )
    
    return net_profit

# =====================================================================
# 4. IGNITE THE ENGINE
# =====================================================================
if __name__ == "__main__":
    
    study_name = "nifty_scalp_robust"
    storage_name = "sqlite:///optuna_scalp_robust.db" 
    
    print(f"Igniting Robust Optuna Scalp Database: {storage_name}")
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize",
        load_if_exists=True 
    )
    
    print("Firing up parallel CPU cores for robust chronological optimization...")
    
    try:
        study.optimize(
            objective, 
            n_trials=2500, 
            n_jobs=24, 
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization Stopped Early by User.")
        
    print("\n" + "="*50)
    print("🏆 ROBUST SCALP OPTIMIZATION COMPLETE 🏆")
    print("="*50)
    print(f"Absolute Best Net PnL : ₹{study.best_value:,.2f}")
    print("Generalizable Parameter Combination:")
    for key, value in study.best_params.items():
        print(f"  --> {key}: {value}")