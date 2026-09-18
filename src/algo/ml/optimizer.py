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

DATES = global_df.index.date
CE_HIGH = global_df['ce_high'].values
CE_LOW = global_df['ce_low'].values
CE_CLOSE = global_df['ce_close'].values

# Extract PE arrays for bearish trades
PE_HIGH = global_df['pe_high'].values
PE_LOW = global_df['pe_low'].values
PE_CLOSE = global_df['pe_close'].values

SPOT_CLOSE = global_df['close'].values
EMA_200 = global_df['EMA_200'].values
SUPERTD = global_df['SUPERTd'].values
SUPERT_SLOPE = global_df['SUPERT_slope'].values
MACD = global_df['MACD'].values
MACD_SIG = global_df['MACD_signal'].values
BB_WIDTH = global_df['BB_width'].values

# =====================================================================
# 1.5 DYNAMIC LOT SIZE & REALISTIC BROKERAGE CALCULATOR
# =====================================================================
def get_nifty_lot_size(trade_date):
    """
    Dynamically returns the Nifty 50 lot size based on historical NSE circulars.
    - Before Nov 20, 2024: 25
    - Nov 20, 2024 to Dec 31, 2025: 75
    - Jan 1, 2026 onwards: 65
    """
    if trade_date >= datetime.date(2026, 1, 1):
        return 65
    elif trade_date >= datetime.date(2024, 11, 20):
        return 75
    else:
        return 25

def calculate_options_charges(buy_price, sell_price, qty):
    """
    Calculates exact Upstox F&O charges for a round-trip NSE options trade.
    Uses the latest STT (0.1%) and NSE (0.03503%) rate slabs.
    """
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    total_value = buy_value + sell_value

    brokerage = 40.0  # ₹20 Buy + ₹20 Sell
    
    # STT: 0.1% on Sell Side Premium
    stt = np.round(sell_value * 0.001)
    
    # Exchange Txn Charge: 0.03503% on total premium (NSE)
    txn_charge = total_value * 0.0003503
    
    # SEBI Turnover Charge: 0.0001% on total premium (₹10 per Crore)
    sebi_charge = total_value * 0.000001
    
    # Stamp Duty: 0.003% on Buy Side Premium
    stamp_duty = np.round(buy_value * 0.00003)
    
    # GST: 18% on (Brokerage + Exchange Charge + SEBI Charge)
    gst = (brokerage + txn_charge + sebi_charge) * 0.18
    
    total_charges = brokerage + stt + txn_charge + sebi_charge + stamp_duty + gst
    return total_charges


# =====================================================================
# 2. FAST PNL EVALUATOR
# =====================================================================
def fast_evaluate_trades(signal_indices, stoplosses, targets, prices_high, prices_low, prices_close):
    """
    Evaluates trades instantly using NumPy instead of slow Python loops.
    Now accepts specific option arrays (CE or PE) dynamically.
    """
    total_pnl = 0.0
    actual_trades_taken = 0  # Track actual trades, not just signals
    
    if len(signal_indices) == 0:
        return 0.0

    last_exit_idx = -1  # Tracks when the simulator is "free" to trade again

    for i in range(len(signal_indices)):
        start_idx = signal_indices[i]
        
        # The Position Lock
        # If this signal happens while we are already in an active trade, skip it!
        if start_idx <= last_exit_idx:
            continue
        
        trade_date = DATES[start_idx]
        sl = stoplosses[i]
        tg = targets[i]
        entry_price = prices_close[start_idx]
        
        # Dynamically fetch the correct lot size for this specific day
        current_lot_size = get_nifty_lot_size(trade_date)
        
        curr_idx = start_idx + 1
        exit_price = 0.0
        
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            # Check Target Hit
            if high >= tg:
                exit_price = tg
                last_exit_idx = curr_idx  # Record the exact minute we exited
                break
                
            # Check Stoploss Hit
            if low <= sl:
                exit_price = sl
                last_exit_idx = curr_idx  # Record the exact minute we exited
                break
                
            curr_idx += 1
            
        # If neither hit, square off at the last available price of the day
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1]
            last_exit_idx = curr_idx - 1  # Record EOD exit
            
        # --- NEW: Exact PnL & Brokerage Calculation using Dynamic Lot Size ---
        trade_gross_pnl = (exit_price - entry_price) * current_lot_size
        trade_charges = calculate_options_charges(entry_price, exit_price, current_lot_size)
        
        # Add net profit to total
        total_pnl += (trade_gross_pnl - trade_charges)
        actual_trades_taken += 1  # Increment actual trades taken

    return total_pnl


# =====================================================================
# 3. THE OPTUNA OBJECTIVE
# =====================================================================
def objective(trial):
    """
    The function Optuna tries to maximize. It guesses parameters, 
    filters the dataframe, evaluates trades, and returns the net profit.
    """
    
    # 1. Let Optuna guess your ENTRY logic (The Sliders)
    rsi_min = trial.suggest_int("rsi_min", 25, 50)
    adx_min = trial.suggest_int("adx_min", 60, 85)
    
    # Let Optuna guess your TREND FILTERS (The True/False Switches)
    # Optuna will decide if strictly following these indicators is actually profitable
    use_ema_filter = trial.suggest_categorical("use_ema", [True, False])
    use_supertrend_filter = trial.suggest_categorical("use_supertrend", [True, False])
    req_active_slope = trial.suggest_categorical("req_active_slope", [True, False])
    # Should the AI demand a Bullish/Bearish MACD cross?
    use_macd_filter = trial.suggest_categorical("use_macd", [True, False])
    # Let the AI guess what a "Squeeze" looks like (e.g., bands are within 0.1% to 1.5% of price)
    bb_max_width = trial.suggest_float("bb_max_width", 0.001, 0.015)

    
    # 2. Let Optuna guess your EXIT logic (ATR Multipliers)
    sl_atr = trial.suggest_float("sl_atr", 0.5, 3.0, step=0.1)
    target_atr = trial.suggest_float("target_atr", 1.0, 8.0, step=0.2)

    # 3. Vectorized Filtering for CE (Bullish) and PE (Bearish)
    # Start with the base Momentum (RSI) and Volatility (ADX) rules
    ce_mask = (global_df['RSI'] > rsi_min) & (global_df['ADX'] > adx_min)
    pe_mask = (global_df['RSI'] < (100 - rsi_min)) & (global_df['ADX'] > adx_min)
    
    # Dynamically apply the 200 EMA Filter if Optuna wants to
    if use_ema_filter:
        # Only buy CE if Spot is above 200 EMA. Only buy PE if below.
        ce_mask &= (SPOT_CLOSE > EMA_200)
        pe_mask &= (SPOT_CLOSE < EMA_200)
        
    # Dynamically apply the Supertrend Filter if Optuna wants to
    if use_supertrend_filter:
        # SUPERTd is 1 for Bullish, -1 for Bearish
        ce_mask &= (SUPERTD == 1)
        pe_mask &= (SUPERTD == -1)
    if req_active_slope:
        # A positive difference means the trailing stop is actively pushing higher
        ce_mask &= (SUPERT_SLOPE > 0)
        # A negative difference means the trailing stop is actively pushing lower
        pe_mask &= (SUPERT_SLOPE < 0)

    if use_macd_filter:
        ce_mask &= (MACD > MACD_SIG)
        pe_mask &= (MACD < MACD_SIG)
    
# Apply the Bollinger Band Squeeze filter to both (Requires volatility to be low BEFORE the breakout)
    ce_mask &= (BB_WIDTH < bb_max_width)
    pe_mask &= (BB_WIDTH < bb_max_width)
    
    
    # Get the raw integer indices where trades happened
    ce_indices = np.where(ce_mask)[0]
    pe_indices = np.where(pe_mask)[0]
    
    # If no trades happen with these parameters, penalize heavily
    if len(ce_indices) == 0 and len(pe_indices) == 0:
        return -99999.0

    net_profit = 0.0

    # 4. Evaluate CE Trades
    if len(ce_indices) > 0:
        ce_entry = CE_CLOSE[ce_indices]
        ce_atrs = global_df['ce_ATR'].values[ce_indices]
        ce_sl = ce_entry - (ce_atrs * sl_atr)
        ce_tg = ce_entry + (ce_atrs * target_atr)
        # Pass the CE arrays to the evaluator
        net_profit += fast_evaluate_trades(ce_indices, ce_sl, ce_tg, CE_HIGH, CE_LOW, CE_CLOSE)

    # 5. Evaluate PE Trades
    if len(pe_indices) > 0:
        pe_entry = PE_CLOSE[pe_indices]
        pe_atrs = global_df['pe_ATR'].values[pe_indices]
        pe_sl = pe_entry - (pe_atrs * sl_atr)
        pe_tg = pe_entry + (pe_atrs * target_atr)
        # Pass the PE arrays to the evaluator
        net_profit += fast_evaluate_trades(pe_indices, pe_sl, pe_tg, PE_HIGH, PE_LOW, PE_CLOSE)
    
    return net_profit


# =====================================================================
# 4. IGNITE THE ENGINE
# =====================================================================
if __name__ == "__main__":
    
    study_name = "nifty_ce_optimization"
    # This creates a local file 'optuna_study.db' to act as the brain
    storage_name = "sqlite:///optuna_study.db"
    
    print(f"Igniting Optuna Database: {storage_name}")
    
    # load_if_exists lets you stop the script and resume it later safely!
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize",
        load_if_exists=True 
    )
    
    print("Firing up parallel CPU cores...")
    
    try:
        # n_jobs=24 utilizes 24 of your 32 logical cores
        # This prevents your ASUS ROG from completely freezing up
        study.optimize(
            objective, 
            n_trials=5000, 
            n_jobs=24, 
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization Stopped Early by User.")
        
    # Print the final results
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print("="*50)
    print(f"Absolute Best Net PnL : ₹{study.best_value:,.2f}")
    print("Perfect Parameter Combination:")
    for key, value in study.best_params.items():
        print(f"  --> {key}: {value}")
