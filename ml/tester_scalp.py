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
}

# --- REALITY CHECK CONSTRAINTS ---
INITIAL_CAPITAL = 70000.0
CAPITAL_ALLOCATION_PCT = 0.50  # Max 40% of running equity per trade
MAX_LOTS_CAP = 5              # Prevent sweeping the order book (liquidity limit)
FIXED_1_LOT_BASELINE = False    # Set to True to evaluate raw strategy edge, False to compound

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
CE_ATR = df['ce_ATR'].values

PE_HIGH = df['pe_high'].values
PE_LOW = df['pe_low'].values
PE_CLOSE = df['pe_close'].values
PE_ATR = df['pe_ATR'].values

SLIPPAGE = 0.25  # Spread slippage (will be multiplied by quantity)

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

# =====================================================================
# 3. UNIFIED CHRONOLOGICAL EVALUATOR (CAPITAL-AWARE)
# =====================================================================
def evaluate_and_report_unified(
    ce_indices, pe_indices, sl_atr, target_atr, trailing_sl_atr, 
    max_daily_trades, max_daily_profit,
    initial_capital=70000.0,
    capital_allocation_pct=0.40,
    max_lots=3,
    fixed_1_lot=False
):
    stats = {
        'net_pnl': 0.0, 'gross_pnl': 0.0, 'total_charges': 0.0,
        'trades': 0, 'wins': 0, 'losses': 0, 'skipped_no_capital': 0
    }
    daily_logs = {}

    signals = [(idx, 0) for idx in ce_indices] + [(idx, 1) for idx in pe_indices]
    signals.sort(key=lambda x: x[0])

    last_exit_idx = -1
    current_date = None
    trades_today = 0
    profit_today = 0.0
    
    running_capital = initial_capital

    for sig in signals:
        start_idx, opt_type = sig
        
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
        atrs = CE_ATR if opt_type == 0 else PE_ATR

        # --- LIMIT ORDER ENTRY LOGIC ---
        limit_price = prices_close[start_idx] 
        fill_price = 0.0
        curr_idx = start_idx
        
        for offset in range(1, 4): 
            check_idx = start_idx + offset
            if check_idx < len(DATES) and DATES[check_idx] == trade_date:
                if prices_low[check_idx] <= limit_price:
                    fill_price = limit_price
                    curr_idx = check_idx
                    break
                    
        if fill_price == 0.0:
            continue 
            
        entry_price = fill_price
        base_lot_size = get_nifty_lot_size(trade_date)
        cost_per_lot = entry_price * base_lot_size
        
        # --- CAPITAL & POSITION SIZING LOGIC ---
        if fixed_1_lot:
            num_lots = 1
            if running_capital < cost_per_lot:
                stats['skipped_no_capital'] += 1
                continue
        else:
            allocated_cash = running_capital * capital_allocation_pct
            num_lots = int(allocated_cash // cost_per_lot)
            
            if num_lots < 1:
                stats['skipped_no_capital'] += 1
                continue 
            
            # Apply Hard Cap to prevent unrealistic liquidity assumptions
            if num_lots > max_lots:
                num_lots = max_lots
                
        total_quantity = num_lots * base_lot_size
        highest_seen = entry_price
        
        current_atr = atrs[start_idx]
        sl = entry_price - (current_atr * sl_atr)
        tg = entry_price + (current_atr * target_atr)
        trail_dist = current_atr * trailing_sl_atr

        exit_price = 0.0
        
        # --- 1-MINUTE EXECUTION EVALUATION ---
        while curr_idx < len(DATES) and DATES[curr_idx] == trade_date:
            high = prices_high[curr_idx]
            low = prices_low[curr_idx]
            
            if low <= sl:
                exit_price = sl - SLIPPAGE # Pay slippage on market stoploss
                last_exit_idx = curr_idx
                break
            if high >= tg:
                exit_price = tg # No slippage on limit target
                last_exit_idx = curr_idx
                break
                
            if high > highest_seen:
                highest_seen = high
                sl = max(sl, highest_seen - trail_dist)
                
            curr_idx += 1
            
        if exit_price == 0.0:
            exit_price = prices_close[curr_idx - 1] - SLIPPAGE
            last_exit_idx = curr_idx - 1
            
        trade_gross_pnl = (exit_price - entry_price) * total_quantity
        trade_charges = calculate_options_charges(entry_price, exit_price, total_quantity)
        trade_net_pnl = trade_gross_pnl - trade_charges
        
        running_capital += trade_net_pnl
        
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
# 4. COMPREHENSIVE QUANTITATIVE REPORTING
# =====================================================================
def print_comprehensive_report(daily_logs, total_stats, all_dataset_dates, initial_capital=70000.0):
    """
    Prints a detailed quantitative analytics report covering timeline coverage,
    daily activity distributions, risk metrics, and trade expectancy.
    """
    if not daily_logs:
        print("\nNo trades were logged. Report cannot be generated.")
        return

    # 1. Timeline & Date Aggregations
    all_unique_dates = pd.to_datetime(pd.Series(all_dataset_dates).unique()).sort_values()
    start_date = all_unique_dates.min()
    end_date = all_unique_dates.max()
    
    total_calendar_days = (end_date - start_date).days + 1
    total_dataset_trading_days = len(all_unique_dates)
    approx_months = max(1.0, total_calendar_days / 30.4375)

    # 2. DataFrame Construction across All Trading Days
    df_trades = pd.DataFrame.from_dict(daily_logs, orient='index')
    df_trades.index = pd.to_datetime(df_trades.index)

    # Reindex over all dataset dates so zero-trade days are accurately captured
    df = pd.DataFrame(index=all_unique_dates)
    df.index.name = 'Date'
    df['trades'] = df_trades['trades'].reindex(df.index).fillna(0).astype(int)
    df['net_pnl'] = df_trades['net_pnl'].reindex(df.index).fillna(0.0)

    # 3. Daily Activity Breakdown
    active_days_df = df[df['trades'] > 0]
    active_trading_days = len(active_days_df)
    idle_days = total_dataset_trading_days - active_trading_days
    
    winning_days_df = df[df['net_pnl'] > 0]
    losing_days_df = df[df['net_pnl'] < 0]
    breakeven_days_df = df[(df['net_pnl'] == 0) & (df['trades'] > 0)]

    winning_days = len(winning_days_df)
    losing_days = len(losing_days_df)
    daily_win_rate = (winning_days / active_trading_days * 100) if active_trading_days > 0 else 0.0

    avg_daily_pnl_all = df['net_pnl'].mean()
    avg_daily_pnl_active = active_days_df['net_pnl'].mean() if active_trading_days > 0 else 0.0
    avg_winning_day = winning_days_df['net_pnl'].mean() if winning_days > 0 else 0.0
    avg_losing_day = losing_days_df['net_pnl'].mean() if losing_days > 0 else 0.0

    best_day_pnl = df['net_pnl'].max()
    worst_day_pnl = df['net_pnl'].min()

    # 4. Equity Curve, Drawdowns & Profit Factor
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    df['equity'] = initial_capital + df['cumulative_pnl']
    df['hwm'] = df['equity'].cummax()
    df['drawdown_pct'] = (df['equity'] - df['hwm']) / df['hwm']
    df['drawdown_cash'] = df['equity'] - df['hwm']

    max_drawdown_pct = df['drawdown_pct'].min() * 100
    max_drawdown_cash = df['drawdown_cash'].min()

    gross_profit = total_stats.get('gross_pnl', 0.0)
    gross_loss = abs(total_stats.get('gross_pnl', 0.0) - total_stats.get('net_pnl', 0.0))  # Estimated loss pool
    profit_factor = (winning_days_df['net_pnl'].sum() / abs(losing_days_df['net_pnl'].sum())) if abs(losing_days_df['net_pnl'].sum()) > 0 else np.nan

    # 5. Annualized Risk Metrics
    df['daily_return_pct'] = df['net_pnl'] / initial_capital
    if len(df) > 1 and df['daily_return_pct'].std() != 0:
        sharpe_ratio = np.sqrt(252) * (df['daily_return_pct'].mean() / df['daily_return_pct'].std())
    else:
        sharpe_ratio = 0.0

    total_return = (total_stats['net_pnl'] / initial_capital) * 100
    win_loss_ratio = total_stats['wins'] / total_stats['losses'] if total_stats['losses'] > 0 else total_stats['wins']
    expectancy = total_stats['net_pnl'] / total_stats['trades'] if total_stats['trades'] > 0 else 0.0

    # =================================================================
    # TERMINAL REPORT RENDERING
    # =================================================================
    mode_str = "FIXED 1-LOT BASELINE" if FIXED_1_LOT_BASELINE else f"DYNAMIC COMPOUNDING (Max {MAX_LOTS_CAP} Lots)"
    
    print("\n" + "=" * 65)
    print(f"📈 COMPREHENSIVE QUANTITATIVE PERFORMANCE REPORT [{mode_str}] 📈")
    print("=" * 65)
    
    print("\n📅 1. TIMELINE & COVERAGE")
    print("-" * 65)
    print(f"Testing Period               : {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    print(f"Duration                     : {approx_months:.1f} Months ({total_calendar_days} Calendar Days)")
    print(f"Total Market Trading Days    : {total_dataset_trading_days} Days")
    print(f"Active Trading Days (Traded) : {active_trading_days} Days ({(active_trading_days / total_dataset_trading_days)*100:.1f}%)")
    print(f"Idle Days (No Signals Taken) : {idle_days} Days ({(idle_days / total_dataset_trading_days)*100:.1f}%)")

    print("\n💰 2. CAPITAL & RETURNS")
    print("-" * 65)
    print(f"Initial Capital              : ₹{initial_capital:,.2f}")
    print(f"Final Equity Base            : ₹{(initial_capital + total_stats['net_pnl']):,.2f}")
    print(f"Gross Profit                 : ₹{total_stats['gross_pnl']:,.2f}")
    print(f"Taxes & Brokerage Charges    : ₹{total_stats['total_charges']:,.2f}")
    print(f"Net Profit (Realized)        : ₹{total_stats['net_pnl']:,.2f}")
    print(f"Total Return                 : {total_return:.2f}%")
    print(f"Monthly Run Rate (Avg)       : ₹{(total_stats['net_pnl'] / approx_months):,.2f} / month")

    print("\n📊 3. DAILY PERFORMANCE METRICS")
    print("-" * 65)
    print(f"Winning Days vs Losing Days  : {winning_days} Wins / {losing_days} Losses")
    print(f"Daily Win Rate               : {daily_win_rate:.2f}%")
    print(f"Daily Profit Factor          : {profit_factor:.2f}")
    print(f"Avg Profit (Active Days)     : ₹{avg_daily_pnl_active:,.2f} / day")
    print(f"Avg Profit (All Market Days) : ₹{avg_daily_pnl_all:,.2f} / day")
    print(f"Average Winning Day          : ₹{avg_winning_day:,.2f}")
    print(f"Average Losing Day           : ₹{avg_losing_day:,.2f}")
    print(f"Best Single Day PnL          : ₹{best_day_pnl:,.2f}")
    print(f"Worst Single Day PnL         : ₹{worst_day_pnl:,.2f}")

    print("\n🎯 4. TRADE-LEVEL EXPECTANCY & RISK")
    print("-" * 65)
    print(f"Total Executed Trades        : {total_stats['trades']}")
    print(f"Skipped (No Margin Buffer)   : {total_stats.get('skipped_no_capital', 0)}")
    print(f"Avg Trades per Active Day    : {(total_stats['trades'] / active_trading_days):.2f}" if active_trading_days > 0 else "0.00")
    print(f"Win / Loss Ratio             : {win_loss_ratio:.2f}")
    print(f"Net Expectancy               : ₹{expectancy:,.2f} per trade")
    print(f"Max Drawdown (%)             : {max_drawdown_pct:.2f}%")
    print(f"Max Drawdown (Cash)          : ₹{max_drawdown_cash:,.2f}")
    print(f"Sharpe Ratio (Annualized)    : {sharpe_ratio:.2f}")
    print("=" * 65 + "\n")


# =====================================================================
# 5. EXECUTE BACKTEST
# =====================================================================
# =====================================================================
# 5. EXECUTE BACKTEST
# =====================================================================
total_stats, daily_logs = evaluate_and_report_unified(
    ce_indices, 
    pe_indices, 
    BEST_PARAMS['sl_atr'], 
    BEST_PARAMS['target_atr'], 
    BEST_PARAMS['trailing_sl_atr'], 
    BEST_PARAMS['max_daily_trades'], 
    BEST_PARAMS['max_daily_profit'],
    initial_capital=INITIAL_CAPITAL,
    capital_allocation_pct=CAPITAL_ALLOCATION_PCT,
    max_lots=MAX_LOTS_CAP,
    fixed_1_lot=FIXED_1_LOT_BASELINE
)

# Pass DATES as the third argument
print_comprehensive_report(daily_logs, total_stats, DATES, initial_capital=INITIAL_CAPITAL)