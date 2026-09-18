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
from core.methods import calculate_trade_charges

ustox = UpstoxClient()

# =====================================================================
# 1. OPTUNA VALIDATED PARAMETERS & REALITY CONSTRAINTS
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
    "capital_allocation_pct": 0.50,
    "max_lots_cap": 5,
}

INITIAL_CAPITAL = 70000.0
FIXED_1_LOT_BASELINE = False
SLIPPAGE = 0.25

REPORTS_DIR = ROOT_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not TESTING_DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {TESTING_DATA_PATH}. Did you run database_builder.py?")

df = pd.read_parquet(TESTING_DATA_PATH)
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

TIMESTAMPS = df.index
DATES = df.index.date
CE_HIGH = df['ce_high'].values
CE_LOW = df['ce_low'].values
CE_CLOSE = df['ce_close'].values
CE_ATR = df['ce_ATR'].values

PE_HIGH = df['pe_high'].values
PE_LOW = df['pe_low'].values
PE_CLOSE = df['pe_close'].values
PE_ATR = df['pe_ATR'].values

def get_nifty_lot_size(trade_date):
    if trade_date >= pd.Timestamp("2026-01-01").date(): return 65
    elif trade_date >= pd.Timestamp("2024-11-20").date(): return 75
    else: return 25

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

ce_signal_timestamps = df_5m.index[ce_mask_5m]
pe_signal_timestamps = df_5m.index[pe_mask_5m]

ce_indices = df.index.get_indexer(ce_signal_timestamps)
pe_indices = df.index.get_indexer(pe_signal_timestamps)

ce_indices = ce_indices[ce_indices != -1]
pe_indices = pe_indices[pe_indices != -1]

# =====================================================================
# 3. UNIFIED CHRONOLOGICAL EVALUATOR (CAPITAL-AWARE)
# =====================================================================
def evaluate_and_report_unified(
    ce_indices, pe_indices, params,
    initial_capital=70000.0,
    fixed_1_lot=False
):
    stats = {
        'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
        'trades': 0, 'wins': 0, 'losses': 0, 'skipped_no_capital': 0
    }
    daily_logs = {}
    trade_records = []

    signals = [(idx, 0) for idx in ce_indices] + [(idx, 1) for idx in pe_indices]
    signals.sort(key=lambda x: x[0])

    last_exit_idx = -1
    current_date = None
    trades_today = 0
    profit_today = 0.0
    running_capital = initial_capital
    trade_counter = 0

    for sig in signals:
        start_idx, opt_type = sig
        
        # --- FIX: SHIFT EXECUTION TO THE CLOSE OF THE 5M CANDLE ---
        signal_idx = start_idx + 4
        
        # Ensure we haven't crossed into the next day or out of bounds
        if signal_idx >= len(DATES) or DATES[signal_idx] != DATES[start_idx]:
            continue

        if signal_idx <= last_exit_idx:
            continue
        
        trade_date = DATES[signal_idx]
        trade_date_str = str(trade_date)
        
        if trade_date != current_date:
            current_date = trade_date
            trades_today = 0
            profit_today = 0.0
            
        if trade_date_str not in daily_logs:
            daily_logs[trade_date_str] = {'trades': 0, 'net_pnl': 0.0}
            
        if trades_today >= params['max_daily_trades'] or profit_today >= params['max_daily_profit']:
            continue
            
        prices_high = CE_HIGH if opt_type == 0 else PE_HIGH
        prices_low = CE_LOW if opt_type == 0 else PE_LOW
        prices_close = CE_CLOSE if opt_type == 0 else PE_CLOSE
        atrs = CE_ATR if opt_type == 0 else PE_ATR

        # --- FIX: STRICT PENETRATION FILL LOGIC ---
        limit_price = prices_close[signal_idx] 
        fill_price = 0.0
        curr_idx = signal_idx
        
        # Check the next 3 minutes to see if the market drops BELOW our limit
        for offset in range(1, 4): 
            check_idx = signal_idx + offset
            if check_idx < len(DATES) and DATES[check_idx] == trade_date:
                if prices_low[check_idx] < limit_price: # Must penetrate limit to fill
                    fill_price = limit_price
                    curr_idx = check_idx
                    break
                    
        if fill_price == 0.0:
            continue 
            
        entry_price = fill_price
        base_lot_size = get_nifty_lot_size(trade_date)
        cost_per_lot = entry_price * base_lot_size
        
        if fixed_1_lot:
            num_lots = 1
            if running_capital < cost_per_lot:
                stats['skipped_no_capital'] += 1
                continue
        else:
            allocated_cash = running_capital * params['capital_allocation_pct']
            num_lots = int(allocated_cash // cost_per_lot)
            
            if num_lots < 1:
                stats['skipped_no_capital'] += 1
                continue 
            
            if num_lots > params['max_lots_cap']:
                num_lots = params['max_lots_cap']
                
        total_quantity = num_lots * base_lot_size
        highest_seen = entry_price
        
        # Calculate trailing SL based on the ATR exactly at execution time
        current_atr = atrs[signal_idx]
        sl = entry_price - (current_atr * params['sl_atr'])
        tg = entry_price + (current_atr * params['target_atr'])
        trail_dist = current_atr * params['trailing_sl_atr']

        exit_price = 0.0
        remark = "EOD"
        exit_idx = curr_idx
        
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            if low <= sl:
                exit_price = sl - SLIPPAGE
                remark = "STOPLOSS"
                exit_idx = curr_idx
                last_exit_idx = curr_idx
                break
            if high >= tg:
                exit_price = tg
                remark = "TARGET"
                exit_idx = curr_idx
                last_exit_idx = curr_idx
                break
                
            if high > highest_seen:
                highest_seen = high
                sl = max(sl, highest_seen - trail_dist)
                
            curr_idx += 1
            
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1] - SLIPPAGE
            remark = "EOD"
            exit_idx = curr_idx - 1
            last_exit_idx = curr_idx - 1
            
        trade_gross_pnl = (exit_price - entry_price) * total_quantity
        trade_charges = calculate_trade_charges(entry_price, exit_price, total_quantity, instrument='O', trade_type='I')
        trade_net_pnl = trade_gross_pnl - trade_charges
        
        running_capital += trade_net_pnl
        trade_counter += 1
        
        trade_records.append({
            "trade_id": f"TEST_{trade_counter:04d}",
            "date": str(trade_date),
            "side": "CE" if opt_type == 0 else "PE",
            "buy_timestamp": str(TIMESTAMPS[signal_idx]),
            "sell_timestamp": str(TIMESTAMPS[exit_idx]),
            "buy_price": round(entry_price, 2),
            "sell_price": round(exit_price, 2),
            "quantity": total_quantity,
            "lots": num_lots,
            "gross_pnl": round(trade_gross_pnl, 2),
            "charges": round(trade_charges, 2),
            "net_pnl": round(trade_net_pnl, 2),
            "remark": remark,
            "running_capital": round(running_capital, 2)
        })
        
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

    df_trades_export = pd.DataFrame(trade_records)
    csv_path = REPORTS_DIR / "tester_trades.csv"
    parquet_path = REPORTS_DIR / "tester_trades.parquet"
    df_trades_export.to_csv(csv_path, index=False)
    df_trades_export.to_parquet(parquet_path, index=False)
    print(f"✅ Trade log exported to:\n  - {csv_path}\n  - {parquet_path}")

    return stats, daily_logs

# =====================================================================
# 4. EXECUTE BACKTEST
# =====================================================================
total_stats, daily_logs = evaluate_and_report_unified(
    ce_indices, 
    pe_indices, 
    BEST_PARAMS,
    initial_capital=INITIAL_CAPITAL,
    fixed_1_lot=FIXED_1_LOT_BASELINE
)