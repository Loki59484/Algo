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

from core.upstox_methods import UpstoxClient
from ml.database_builder import TRAINING_DATA_PATH
ustox = UpstoxClient()

if not TRAINING_DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {TRAINING_DATA_PATH}. Did you run database_builder.py?")

global_df = pd.read_parquet(TRAINING_DATA_PATH)
if not isinstance(global_df.index, pd.DatetimeIndex):
    global_df.index = pd.to_datetime(global_df.index)

# Pre-extract base 1-minute execution arrays
DATES = global_df.index.date
CE_HIGH = global_df['ce_high'].values
CE_LOW = global_df['ce_low'].values
CE_CLOSE = global_df['ce_close'].values
CE_ATR = global_df['ce_ATR'].values

PE_HIGH = global_df['pe_high'].values
PE_LOW = global_df['pe_low'].values
PE_CLOSE = global_df['pe_close'].values
PE_ATR = global_df['pe_ATR'].values

# Pre-resample 5-minute dataset for ultra-fast signal generation
df_5m = global_df.resample('5min', origin='start_day').agg({
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

SLIPPAGE = 0.25 # Only applied on market exits (SL)

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

def fast_evaluate_unified(ce_indices, pe_indices, sl_atr, target_atr, trailing_sl_atr, max_daily_trades, max_daily_profit):
    total_pnl = 0.0
    signals = [(idx, 0) for idx in ce_indices] + [(idx, 1) for idx in pe_indices]
    signals.sort(key=lambda x: x[0])

    last_exit_idx = -1
    current_date = None
    trades_today = 0
    profit_today = 0.0

    for sig in signals:
        start_idx, opt_type = sig

        # SHIFT TO THE 5M CLOSE
        signal_idx = start_idx + 4
        if signal_idx >= len(DATES) or DATES[signal_idx] != DATES[start_idx]:
            continue

        if signal_idx <= last_exit_idx:
            continue

        trade_date = DATES[signal_idx]

        if trade_date != current_date:
            current_date = trade_date
            trades_today = 0
            profit_today = 0.0

        if trades_today >= max_daily_trades or profit_today >= max_daily_profit:
            continue

        prices_high = CE_HIGH if opt_type == 0 else PE_HIGH
        prices_low = CE_LOW if opt_type == 0 else PE_LOW
        prices_close = CE_CLOSE if opt_type == 0 else PE_CLOSE
        atrs = CE_ATR if opt_type == 0 else PE_ATR

        # --- LIMIT ORDER PENETRATION LOGIC ---
        limit_price = prices_close[signal_idx] 
        fill_price = 0.0
        curr_idx = signal_idx
        
        for offset in range(1, 4): 
            check_idx = signal_idx + offset
            if check_idx < len(DATES) and DATES[check_idx] == trade_date:
                if prices_low[check_idx] < limit_price: # Strict fill
                    fill_price = limit_price
                    curr_idx = check_idx
                    break
                    
        if fill_price == 0.0:
            continue
            
        entry_price = fill_price
        highest_seen = entry_price
        current_lot_size = get_nifty_lot_size(trade_date)
        
        current_atr = atrs[signal_idx]
        sl = entry_price - (current_atr * sl_atr)
        tg = entry_price + (current_atr * target_atr)
        trail_dist = current_atr * trailing_sl_atr

        exit_price = 0.0

        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]

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

        total_pnl += trade_net_pnl
        trades_today += 1
        profit_today += trade_net_pnl

    return total_pnl

def objective(trial):
    rsi_min = trial.suggest_int("rsi_min", 50, 70, step=5)  
    adx_min = trial.suggest_int("adx_min", 20, 45, step=5)
    adx_max = trial.suggest_int("adx_max", 55, 80, step=5)
    
    use_ema_filter = trial.suggest_categorical("use_ema", [True, False])
    use_supertrend_filter = trial.suggest_categorical("use_supertrend", [True, False])
    req_active_slope = trial.suggest_categorical("req_active_slope", [True, False])
    use_macd_filter = trial.suggest_categorical("use_macd", [True, False])
    
    bb_max_width = trial.suggest_float("bb_max_width", 0.01, 0.06, step=0.01)
    
    sl_atr = trial.suggest_float("sl_atr", 0.8, 2.5, step=0.1)
    target_atr = trial.suggest_float("target_atr", 2.0, 7.0, step=0.5)
    trailing_sl_atr = trial.suggest_float("trailing_sl_atr", 0.5, 2.0, step=0.2)
    
    max_daily_trades = trial.suggest_int("max_daily_trades", 2, 6)
    max_daily_profit = trial.suggest_int("max_daily_profit", 3000, 20000, step=1000) 
    
    # 5-Minute Signal Evaluation
    ce_mask_5m = (df_5m['RSI'] > rsi_min) & (df_5m['ADX'] > adx_min) & (df_5m['ADX'] < adx_max)
    pe_mask_5m = (df_5m['RSI'] < (100 - rsi_min)) & (df_5m['ADX'] > adx_min) & (df_5m['ADX'] < adx_max)
    
    if use_ema_filter:
        ce_mask_5m &= (df_5m['close'] > df_5m['EMA_200'])
        pe_mask_5m &= (df_5m['close'] < df_5m['EMA_200'])
    if use_supertrend_filter:
        ce_mask_5m &= (df_5m['SUPERTd'] == 1)
        pe_mask_5m &= (df_5m['SUPERTd'] == -1)
    if req_active_slope:
        ce_mask_5m &= (df_5m['SUPERT_slope'] > 0)
        pe_mask_5m &= (df_5m['SUPERT_slope'] < 0)
    if use_macd_filter:
        ce_mask_5m &= (df_5m['MACD'] > df_5m['MACD_signal'])
        pe_mask_5m &= (df_5m['MACD'] < df_5m['MACD_signal'])
        
    ce_mask_5m &= (df_5m['BB_width'] < bb_max_width)
    pe_mask_5m &= (df_5m['BB_width'] < bb_max_width)
    
    ce_timestamps = df_5m.index[ce_mask_5m]
    pe_timestamps = df_5m.index[pe_mask_5m]

    ce_indices = global_df.index.get_indexer(ce_timestamps)
    pe_indices = global_df.index.get_indexer(pe_timestamps)

    ce_indices = ce_indices[ce_indices != -1]
    pe_indices = pe_indices[pe_indices != -1]
    
    if len(ce_indices) == 0 and len(pe_indices) == 0:
        return -99999.0

    net_profit = fast_evaluate_unified(
        ce_indices, pe_indices, sl_atr, target_atr, trailing_sl_atr, 
        max_daily_trades, max_daily_profit
    )
    
    return net_profit

if __name__ == "__main__":
    study_name = "nifty_5m_1m_mtf_optimization"
    storage_name = "sqlite:///optuna_5m_1m_mtf.db" 
    
    print(f"Igniting Optuna Database (5M Strategy / 1M Execution): {storage_name}")
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        direction="maximize",
        load_if_exists=True 
    )
    
    try:
        study.optimize(
            objective, 
            n_trials=3000, 
            n_jobs=24, 
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization Stopped Early by User.")
        
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE (5M / 1M MTF ENGINE) 🏆")
    print("="*50)
    print(f"Best Net PnL : ₹{study.best_value:,.2f}")
    for key, value in study.best_params.items():
        print(f"  --> {key}: {value}")